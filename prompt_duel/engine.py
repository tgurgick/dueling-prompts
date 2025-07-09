#!/usr/bin/env python3
"""
Prompt Duel Engine - Core functionality for comparing prompts
"""

import yaml
import json
import csv
from datetime import datetime
from pathlib import Path
import typing
from dataclasses import dataclass
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from .store import DuelStore
from .metrics import SafetyJudgeMetric, LLMJudgeMetric, ExactMatchMetric, ContainsMetric, RelevanceMetric, SemanticSimilarityMetric
from .hf_metrics import get_hf_metric, HF_METRICS

load_dotenv()

@dataclass
class TestCase:
    input: str
    expected: typing.Optional[str] = None

@dataclass
class DuelResult:
    case_num: int
    prompt_a_response: str
    prompt_b_response: str
    prompt_a_input_tokens: int
    prompt_a_output_tokens: int
    prompt_b_input_tokens: int
    prompt_b_output_tokens: int
    winner: str  # 'A', 'B', or 'Tie'
    score_a: float
    score_b: float

@dataclass
class MultiPromptResult:
    case_num: int
    responses: typing.Dict[str, str]  # prompt_name -> response
    input_tokens: typing.Dict[str, int]  # prompt_name -> input_tokens
    output_tokens: typing.Dict[str, int]  # prompt_name -> output_tokens
    scores: typing.Dict[str, typing.Dict[str, float]]  # metric_name -> {prompt_name: score}
    winners: typing.Dict[str, str]  # metric_name -> winner

class PromptDuel:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.client = OpenAI()
        self.store = DuelStore()
        
    def _load_config(self, config_path: str) -> typing.Dict[str, typing.Any]:
        """Load and validate YAML configuration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = ['experiment', 'model', 'prompts', 'cases']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        if len(config['prompts']) != 2:
            raise ValueError("Exactly two prompts (A and B) are required")
        
        return config
    
    def _create_metric(self, metric_name: str) -> typing.Any:
        """Create a metric instance with prompts from YAML if available."""
        # Check if it's a Hugging Face metric first
        if metric_name in HF_METRICS:
            # Get metric configuration from store
            metric_config = self.store.get_metric(metric_name)
            if metric_config:
                # Use parameters from YAML if available
                return get_hf_metric(metric_name, **metric_config.parameters)
            else:
                # Use default parameters
                return get_hf_metric(metric_name)
        
        # Get metric configuration from store
        metric_config = self.store.get_metric(metric_name)
        if not metric_config:
            # Fallback to default metrics
            if metric_name == 'exact_match':
                return ExactMatchMetric()
            elif metric_name == 'contains_check':
                return ContainsMetric()
            elif metric_name == 'relevance':
                return RelevanceMetric()
            elif metric_name == 'semantic_similarity':
                return SemanticSimilarityMetric()
            else:
                raise ValueError(f"Unknown metric: {metric_name}")
        
        # Create metric with parameters from YAML
        if metric_name == 'safety_judge':
            return SafetyJudgeMetric(
                client=self.client,
                model=metric_config.parameters.get('judge_model', 'gpt-4o'),
                temperature=metric_config.parameters.get('temperature', 0.0),
                prompts=metric_config.prompts
            )
        elif metric_name == 'llm_judge':
            return LLMJudgeMetric(
                client=self.client,
                model=metric_config.parameters.get('judge_model', 'gpt-4o'),
                temperature=metric_config.parameters.get('temperature', 0.0),
                prompts=metric_config.prompts
            )
        elif metric_name == 'exact_match':
            return ExactMatchMetric()
        elif metric_name == 'contains_check':
            return ContainsMetric()
        elif metric_name == 'relevance':
            return RelevanceMetric(
                model_name=metric_config.parameters.get('model_name', 'all-MiniLM-L6-v2')
            )
        elif metric_name == 'semantic_similarity':
            return SemanticSimilarityMetric(
                model_name=metric_config.parameters.get('model_name', 'all-MiniLM-L6-v2')
            )
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
    
    def _interpolate_template(self, template: str, case: TestCase, vars_dict: typing.Dict[str, typing.Any]) -> str:
        """Replace template variables with actual values."""
        result = template
        
        # Replace {input} with case input
        result = result.replace('{input}', case.input)
        
        # Replace {vars.xxx} with global variables
        if vars_dict:
            for key, value in vars_dict.items():
                result = result.replace(f'{{vars.{key}}}', str(value))
        
        return result
    
    def _call_model(self, prompt: str, system_prompt: typing.Optional[str] = None) -> typing.Tuple[str, int, int]:
        """Call the OpenAI model and return response with token counts."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.config['model'],
            messages=messages,
            temperature=0
        )
        
        return (
            response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )
    
    def _score_exact(self, response: str, expected: str) -> float:
        """Score based on exact match (case-insensitive)."""
        return 1.0 if response.lower().strip() == expected.lower().strip() else 0.0
    
    def _score_contains(self, response: str, expected: str) -> float:
        """Score based on whether expected text is contained in response."""
        return 1.0 if expected.lower() in response.lower() else 0.0
    
    def _score_judge(self, response_a: str, response_b: str, expected: str) -> typing.Tuple[float, float]:
        """Score using LLM judge."""
        judge_model = self.config.get('judge_model', self.config['model'])
        
        judge_prompt = f"""
You are an impartial judge comparing two AI responses to a given input.

Input: {expected}
Response A: {response_a}
Response B: {response_b}

Which response is better? Respond with exactly one word: A, B, or Tie.

Your judgment:"""
        
        judge_response, _, _ = self._call_model(judge_prompt)
        judge_decision = judge_response.strip().upper()
        
        if judge_decision == 'A':
            return 1.0, 0.0
        elif judge_decision == 'B':
            return 0.0, 1.0
        else:  # Tie or invalid response
            return 0.5, 0.5
    
    def _determine_winner(self, score_a: float, score_b: float) -> str:
        """Determine the winner based on scores."""
        if score_a > score_b:
            return 'A'
        elif score_b > score_a:
            return 'B'
        else:
            return 'Tie'
    
    def run_duel(self) -> typing.List[DuelResult]:
        """Run the prompt duel and return results."""
        results = []
        cases = [TestCase(**case) for case in self.config['cases']]
        
        print(f"🏆 Running: {self.config['experiment']}")
        print(f"🤖 Model: {self.config['model']}")
        print(f"📊 Cases: {len(cases)}")
        print()
        
        for i, case in enumerate(tqdm(cases, desc="Testing cases")):
            # Prepare prompts
            prompt_a = self._interpolate_template(self.config['prompts']['A'], case, self.config.get('vars', {}))
            prompt_b = self._interpolate_template(self.config['prompts']['B'], case, self.config.get('vars', {}))
            
            # Call models
            response_a, tokens_a_prompt, tokens_a_completion = self._call_model(
                prompt_a, self.config.get('system_prompt')
            )
            response_b, tokens_b_prompt, tokens_b_completion = self._call_model(
                prompt_b, self.config.get('system_prompt')
            )
            
            # Score responses using modular metrics
            metric_name = self.config.get('metric', 'exact_match')
            metric = self._create_metric(metric_name)
            
            if hasattr(metric, 'score_comparative'):
                # Use comparative scoring if available
                score_a, score_b = metric.score_comparative(response_a, response_b, case.expected or "")
            else:
                # Fallback to individual scoring
                score_a = metric.score(response_a, case.expected or "")
                score_b = metric.score(response_b, case.expected or "")
            
            winner = self._determine_winner(score_a, score_b)
            
            result = DuelResult(
                case_num=i + 1,
                prompt_a_response=response_a,
                prompt_b_response=response_b,
                prompt_a_input_tokens=tokens_a_prompt,
                prompt_a_output_tokens=tokens_a_completion,
                prompt_b_input_tokens=tokens_b_prompt,
                prompt_b_output_tokens=tokens_b_completion,
                winner=winner,
                score_a=score_a,
                score_b=score_b
            )
            results.append(result)
        
        return results
    
    def print_results(self, results: typing.List[DuelResult]):
        """Print formatted results."""
        print("\n" + "="*50)
        print("📋 RESULTS")
        print("="*50)
        
        a_wins = sum(1 for r in results if r.winner == 'A')
        b_wins = sum(1 for r in results if r.winner == 'B')
        ties = sum(1 for r in results if r.winner == 'Tie')
        
        total_input_tokens_a = sum(r.prompt_a_input_tokens for r in results)
        total_output_tokens_a = sum(r.prompt_a_output_tokens for r in results)
        total_input_tokens_b = sum(r.prompt_b_input_tokens for r in results)
        total_output_tokens_b = sum(r.prompt_b_output_tokens for r in results)
        
        # Show case-by-case results
        for result in results:
            emoji = "✅" if result.winner != 'Tie' else "🤝"
            print(f"\nCase {result.case_num}: {emoji} Prompt {result.winner} wins")
            print(f"  Score A: {result.score_a:.3f} | Score B: {result.score_b:.3f}")
            print(f"  Tokens A: {result.prompt_a_input_tokens} input + {result.prompt_a_output_tokens} output = {result.prompt_a_input_tokens + result.prompt_a_output_tokens} total")
            print(f"  Tokens B: {result.prompt_b_input_tokens} input + {result.prompt_b_output_tokens} output = {result.prompt_b_input_tokens + result.prompt_b_output_tokens} total")
            
            # Show truncated responses
            print(f"  Response A: {result.prompt_a_response[:150]}{'...' if len(result.prompt_a_response) > 150 else ''}")
            print(f"  Response B: {result.prompt_b_response[:150]}{'...' if len(result.prompt_b_response) > 150 else ''}")
        
        print("\n" + "-"*30)
        print(f"A wins: {a_wins} | B wins: {b_wins} | Ties: {ties}")
        print(f"Total tokens - A: {total_input_tokens_a} input + {total_output_tokens_a} output = {total_input_tokens_a + total_output_tokens_a}")
        print(f"Total tokens - B: {total_input_tokens_b} input + {total_output_tokens_b} output = {total_input_tokens_b + total_output_tokens_b}")
        
        if a_wins > b_wins:
            print("🏆 Overall winner: Prompt A")
        elif b_wins > a_wins:
            print("🏆 Overall winner: Prompt B")
        else:
            print("🤝 Overall result: Tie")
        
        # Add AI-powered analysis
        analysis = self._analyze_results(results)
        return analysis
    
    def _analyze_results(self, results: typing.List[DuelResult]) -> str:
        """Analyze results using AI to provide insights and return the analysis text."""
        if not results:
            return ""
        
        # Prepare analysis data
        prompt_a = self.config['prompts']['A']
        prompt_b = self.config['prompts']['B']
        
        a_wins = sum(1 for r in results if r.winner == 'A')
        b_wins = sum(1 for r in results if r.winner == 'B')
        ties = sum(1 for r in results if r.winner == 'Tie')
        
        total_tokens_a = sum(r.prompt_a_input_tokens + r.prompt_a_output_tokens for r in results)
        total_tokens_b = sum(r.prompt_b_input_tokens + r.prompt_b_output_tokens for r in results)
        
        # Create sample responses for analysis
        sample_responses = []
        for i, result in enumerate(results[:3]):  # Analyze first 3 cases
            sample_responses.append(f"Case {i+1}:\nA: {result.prompt_a_response[:200]}...\nB: {result.prompt_b_response[:200]}...")
        
        analysis_prompt = f"""
You are an expert prompt engineer analyzing the results of a prompt A/B test. Provide insights about the performance of two prompts.

PROMPT A: "{prompt_a}"
PROMPT B: "{prompt_b}"

RESULTS:
- A wins: {a_wins}
- B wins: {b_wins} 
- Ties: {ties}
- Total tokens - A: {total_tokens_a}, B: {total_tokens_b}

SAMPLE RESPONSES:
{chr(10).join(sample_responses)}

Please provide a concise analysis (2-3 paragraphs) covering:
1. Key differences in prompt approaches
2. Performance insights (wins, token usage, response quality)
3. Recommendations for improvement

Focus on actionable insights that would help improve prompt engineering.
"""
        
        try:
            analysis_response, _, _ = self._call_model(analysis_prompt)
            print("\n" + "="*50)
            print("🧠 AI ANALYSIS")
            print("="*50)
            print(analysis_response)
            return analysis_response
        except Exception as e:
            error_msg = f"Could not generate AI analysis: {e}"
            print(f"\n⚠️  {error_msg}")
            return error_msg
    
    def save_csv(self, results: typing.List[DuelResult]):
        """Save results to CSV file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['case', 'winner', 'score_a', 'score_b', 'tokens_a_input', 'tokens_a_output', 'tokens_b_input', 'tokens_b_output', 'response_a', 'response_b']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow({
                    'case': result.case_num,
                    'winner': result.winner,
                    'score_a': result.score_a,
                    'score_b': result.score_b,
                    'tokens_a_input': result.prompt_a_input_tokens,
                    'tokens_a_output': result.prompt_a_output_tokens,
                    'tokens_b_input': result.prompt_b_input_tokens,
                    'tokens_b_output': result.prompt_b_output_tokens,
                    'response_a': result.prompt_a_response,
                    'response_b': result.prompt_b_response
                })
        
        print(f"\n💾 Results saved to: {filename}")

class MultiPromptDuel:
    """Enhanced engine for multiple prompts and metrics.
    Adds a global win_threshold (default 0.01) for winner logic on all metrics.
    A prompt only wins if its score exceeds the next best by at least win_threshold; otherwise, it's a tie.
    """
    
    def __init__(self, config_path: str, win_threshold: float = 0.01):
        self.config = self._load_config(config_path)
        self.client = OpenAI()
        self.store = DuelStore()
        self.win_threshold = win_threshold
    
    def _load_config(self, config_path: str) -> typing.Dict[str, typing.Any]:
        """Load and validate YAML configuration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = ['experiment', 'model', 'prompts', 'cases']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        if len(config['prompts']) < 2:
            raise ValueError("At least two prompts are required")
        
        return config
    
    def _create_metric(self, metric_name: str) -> typing.Any:
        """Create a metric instance with prompts from YAML if available."""
        # Check if it's a Hugging Face metric first
        if metric_name in HF_METRICS:
            # Get metric configuration from store
            metric_config = self.store.get_metric(metric_name)
            if metric_config:
                # Use parameters from YAML if available
                return get_hf_metric(metric_name, **metric_config.parameters)
            else:
                # Use default parameters
                return get_hf_metric(metric_name)
        
        # Get metric configuration from store
        metric_config = self.store.get_metric(metric_name)
        if not metric_config:
            # Fallback to default metrics
            if metric_name == 'exact_match':
                return ExactMatchMetric()
            elif metric_name == 'contains_check':
                return ContainsMetric()
            elif metric_name == 'relevance':
                return RelevanceMetric()
            elif metric_name == 'semantic_similarity':
                return SemanticSimilarityMetric()
            else:
                raise ValueError(f"Unknown metric: {metric_name}")
        
        # Create metric with parameters from YAML
        if metric_name == 'safety_judge':
            return SafetyJudgeMetric(
                client=self.client,
                model=metric_config.parameters.get('judge_model', 'gpt-4o'),
                temperature=metric_config.parameters.get('temperature', 0.0),
                prompts=metric_config.prompts
            )
        elif metric_name == 'llm_judge':
            return LLMJudgeMetric(
                client=self.client,
                model=metric_config.parameters.get('judge_model', 'gpt-4o'),
                temperature=metric_config.parameters.get('temperature', 0.0),
                prompts=metric_config.prompts
            )
        elif metric_name == 'exact_match':
            return ExactMatchMetric()
        elif metric_name == 'contains_check':
            return ContainsMetric()
        elif metric_name == 'relevance':
            return RelevanceMetric(
                model_name=metric_config.parameters.get('model_name', 'all-MiniLM-L6-v2')
            )
        elif metric_name == 'semantic_similarity':
            return SemanticSimilarityMetric(
                model_name=metric_config.parameters.get('model_name', 'all-MiniLM-L6-v2')
            )
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
    
    def _interpolate_template(self, template: str, case: TestCase, vars_dict: typing.Dict[str, typing.Any]) -> str:
        """Replace template variables with actual values."""
        result = template
        
        # Replace {input} with case input
        result = result.replace('{input}', case.input)
        
        # Replace {vars.xxx} with global variables
        if vars_dict:
            for key, value in vars_dict.items():
                result = result.replace(f'{{vars.{key}}}', str(value))
        
        return result
    
    def _call_model(self, prompt: str, system_prompt: typing.Optional[str] = None) -> typing.Tuple[str, int, int]:
        """Call the OpenAI model and return response with token counts."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.config['model'],
            messages=messages,
            temperature=0
        )
        
        return (
            response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )
    
    def _determine_winner(self, scores: typing.Dict[str, float], metric_name: str = None) -> str:
        """Determine the winner based on scores and metric-specific thresholds."""
        if not scores:
            return "Tie"
        
        # Find the highest score and the next highest
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_scores) < 2:
            return sorted_scores[0][0]
        top_name, top_score = sorted_scores[0]
        second_score = sorted_scores[1][1]
        
        # Use metric-specific thresholds
        threshold = self._get_metric_threshold(metric_name)
        
        if abs(top_score - second_score) < threshold:
            return "Tie"
        else:
            return top_name
    
    def _get_metric_threshold(self, metric_name: str) -> float:
        """Get metric-specific win threshold."""
        if not metric_name:
            return self.win_threshold
        
        # Metric-specific thresholds based on typical score ranges and sensitivity
        thresholds = {
            # Binary metrics - very strict
            'exact_match': 0.001,
            'contains_check': 0.001,
            
            # Similarity metrics - moderate sensitivity
            'relevance': 0.05,
            'semantic_similarity': 0.05,
            
            # NLP metrics - higher sensitivity due to lower typical scores
            'bleu': 0.02,
            'rouge': 0.01,  # Lower threshold for ROUGE due to 0.5 default scores
            'meteor': 0.03,
            'bertscore': 0.02,
            'bleurt': 0.02,
            'ter': 0.05,  # Lower is better for TER
            
            # LLM judge metrics - moderate sensitivity
            'llm_judge': 0.1,
            'safety_judge': 0.1,
        }
        
        return thresholds.get(metric_name.lower(), self.win_threshold)
    
    def run_duel(self) -> typing.List[MultiPromptResult]:
        """Run the multi-prompt duel and return results."""
        results = []
        cases = [TestCase(**case) for case in self.config['cases']]
        prompt_names = list(self.config['prompts'].keys())
        metrics = self.config.get('metrics', ['exact_match'])
        
        # Add automatic random control prompt if not already present
        if 'Random' not in self.config['prompts']:
            self.config['prompts']['Random'] = "Generate a random response about: {input}"
            print("🎲 Added automatic random control prompt")
        
        print(f"🏆 Running: {self.config['experiment']}")
        print(f"🤖 Model: {self.config['model']}")
        print(f"📝 Prompts: {', '.join(list(self.config['prompts'].keys()))}")
        print(f"📊 Metrics: {', '.join(metrics)}")
        print(f"📋 Cases: {len(cases)}")
        print()
        
        for i, case in enumerate(tqdm(cases, desc="Testing cases")):
            # Prepare all prompts
            prompts = {}
            for name, template in self.config['prompts'].items():
                prompts[name] = self._interpolate_template(template, case, self.config.get('vars', {}))
            
            # Call models for all prompts
            responses = {}
            input_tokens = {}
            output_tokens = {}
            
            for name, prompt in prompts.items():
                response, in_tokens, out_tokens = self._call_model(
                    prompt, self.config.get('system_prompt')
                )
                responses[name] = response
                input_tokens[name] = in_tokens
                output_tokens[name] = out_tokens
            
            # Score responses using all metrics
            scores = {}
            winners = {}
            
            for metric_name in metrics:
                metric = self._create_metric(metric_name)
                metric_scores = {}
                
                # For now, we'll use individual scoring for each prompt
                # In the future, we could implement multi-way comparative scoring
                for name, response in responses.items():
                    metric_scores[name] = metric.score(response, case.expected or "")
                
                scores[metric_name] = metric_scores
                winners[metric_name] = self._determine_winner(metric_scores, metric_name)
            
            result = MultiPromptResult(
                case_num=i + 1,
                responses=responses,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                scores=scores,
                winners=winners
            )
            results.append(result)
        
        return results
    
    def print_results(self, results: typing.List[MultiPromptResult]):
        """Print formatted results for multi-prompt duel."""
        print("\n" + "="*50)
        print("📋 RESULTS")
        print("="*50)
        
        prompt_names = list(results[0].responses.keys()) if results else []
        metrics = list(results[0].scores.keys()) if results else []
        
        # Print case-by-case results
        for result in results:
            print(f"\nCase {result.case_num}:")
            for metric_name in metrics:
                winner = result.winners[metric_name]
                scores = result.scores[metric_name]
                print(f"  {metric_name.upper()}: {winner} wins")
                for prompt_name in prompt_names:
                    score = scores.get(prompt_name, 0.0)
                    tokens = result.input_tokens.get(prompt_name, 0) + result.output_tokens.get(prompt_name, 0)
                    print(f"    {prompt_name}: {score:.3f} ({tokens} tokens)")
            
            # Show first response as example
            if prompt_names:
                first_prompt = prompt_names[0]
                response = result.responses[first_prompt]
                print(f"  Example response ({first_prompt}): {response[:100]}...")
        
        # Print summary statistics
        print("\n" + "-" * 50)
        print("📊 SUMMARY")
        print("-" * 50)
        
        for metric_name in metrics:
            print(f"\n{metric_name.upper()} Results:")
            wins = {}
            total_scores = {}
            total_tokens = {}
            
            for prompt_name in prompt_names:
                wins[prompt_name] = 0
                total_scores[prompt_name] = 0.0
                total_tokens[prompt_name] = 0
            
            for result in results:
                winner = result.winners[metric_name]
                if winner != "Tie":
                    wins[winner] += 1
                
                scores = result.scores[metric_name]
                for prompt_name in prompt_names:
                    total_scores[prompt_name] += scores.get(prompt_name, 0.0)
                    total_tokens[prompt_name] += (result.input_tokens.get(prompt_name, 0) + 
                                                 result.output_tokens.get(prompt_name, 0))
            
            # Print wins
            win_str = " | ".join([f"{name}: {count}" for name, count in wins.items()])
            print(f"  Wins: {win_str}")
            
            # Print average scores
            avg_scores = {name: total_scores[name] / len(results) for name in prompt_names}
            score_str = " | ".join([f"{name}: {score:.3f}" for name, score in avg_scores.items()])
            print(f"  Avg Scores: {score_str}")
            
            # Print total tokens
            token_str = " | ".join([f"{name}: {total_tokens[name]}" for name in prompt_names])
            print(f"  Total Tokens: {token_str}")
        
        # Determine overall winner
        print(f"\n🏆 Overall Analysis:")
        overall_wins = {}
        for prompt_name in prompt_names:
            overall_wins[prompt_name] = 0
        
        for result in results:
            for metric_name in metrics:
                winner = result.winners[metric_name]
                if winner != "Tie":
                    overall_wins[winner] += 1
        
        max_wins = max(overall_wins.values())
        overall_winners = [name for name, wins in overall_wins.items() if wins == max_wins]
        
        if len(overall_winners) == 1:
            print(f"  Overall Winner: {overall_winners[0]}")
        else:
            print(f"  Overall Result: Tie between {', '.join(overall_winners)}")
        
        # AI Analysis
        analysis = self._analyze_multi_results(results)
        return analysis
    
    def _analyze_multi_results(self, results: typing.List[MultiPromptResult]) -> str:
        """Generate AI analysis for multi-prompt results using GPT-4o."""
        print("\n" + "="*50)
        print("🧠 AI ANALYSIS")
        print("="*50)
        
        if not results:
            return ""
        
        prompt_names = list(results[0].responses.keys())
        metrics = list(results[0].scores.keys())
        
        # Collect detailed data for analysis
        analysis_data = {
            'prompt_names': prompt_names,
            'metrics': metrics,
            'total_cases': len(results),
            'wins_by_prompt': {},
            'avg_scores_by_prompt': {},
            'avg_scores_by_metric': {},
            'case_details': [],
            'metric_insights': {}
        }
        
        # Calculate wins and scores
        for prompt_name in prompt_names:
            analysis_data['wins_by_prompt'][prompt_name] = 0
            analysis_data['avg_scores_by_prompt'][prompt_name] = 0.0
        
        for metric_name in metrics:
            analysis_data['avg_scores_by_metric'][metric_name] = {}
            analysis_data['metric_insights'][metric_name] = {
                'wins_by_prompt': {name: 0 for name in prompt_names},
                'score_ranges': {name: [] for name in prompt_names}
            }
            for prompt_name in prompt_names:
                analysis_data['avg_scores_by_metric'][metric_name][prompt_name] = 0.0
        
        # Collect detailed case data
        for result in results:
            case_data = {
                'case_num': result.case_num,
                'responses': result.responses,
                'scores': result.scores,
                'winners': result.winners,
                'tokens': {name: result.input_tokens.get(name, 0) + result.output_tokens.get(name, 0) 
                          for name in prompt_names}
            }
            analysis_data['case_details'].append(case_data)
            
            for metric_name in metrics:
                winner = result.winners[metric_name]
                if winner != "Tie":
                    analysis_data['wins_by_prompt'][winner] += 1
                    analysis_data['metric_insights'][metric_name]['wins_by_prompt'][winner] += 1
                
                scores = result.scores[metric_name]
                for prompt_name in prompt_names:
                    score = scores.get(prompt_name, 0.0)
                    analysis_data['avg_scores_by_prompt'][prompt_name] += score
                    analysis_data['avg_scores_by_metric'][metric_name][prompt_name] += score
                    analysis_data['metric_insights'][metric_name]['score_ranges'][prompt_name].append(score)
        
        # Calculate averages
        for prompt_name in prompt_names:
            analysis_data['avg_scores_by_prompt'][prompt_name] /= len(results)
        
        for metric_name in metrics:
            for prompt_name in prompt_names:
                analysis_data['avg_scores_by_metric'][metric_name][prompt_name] /= len(results)
        
        # Generate detailed analysis prompt
        analysis_prompt = f"""You are an expert prompt engineer analyzing the results of a multi-prompt evaluation experiment. 

EXPERIMENT DATA:
- Prompts tested: {', '.join(prompt_names)}
- Metrics used: {', '.join(metrics)}
- Number of test cases: {len(results)}

DETAILED RESULTS:
{self._format_results_for_analysis(analysis_data)}

Please provide a comprehensive analysis that includes:

1. **Executive Summary**: Overall winner and key findings
2. **Metric-Specific Analysis**: What each metric tells us about prompt performance
3. **Prompt Strengths & Weaknesses**: Detailed analysis of each prompt's performance
4. **Control Prompt Analysis**: How the random control performed and what it tells us
5. **Practical Recommendations**: Specific, actionable insights for prompt improvement
6. **Statistical Insights**: Notable patterns, outliers, or interesting findings

Focus on providing specific, actionable insights rather than generic advice. Analyze what the metrics are actually telling us about the prompts' effectiveness.
"""
        
        try:
            # Use GPT-4o for analysis (different from experiment model)
            analysis_client = OpenAI()
            messages = [
                {"role": "system", "content": "You are an expert prompt engineer and data analyst. Provide detailed, specific analysis based on the data provided."},
                {"role": "user", "content": analysis_prompt}
            ]
            
            analysis_response = analysis_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.1
            )
            analysis_text = analysis_response.choices[0].message.content
            print(analysis_text)
            return analysis_text
        except Exception as e:
            print(f"Warning: AI analysis failed: {e}")
            # Fallback to basic analysis
            return self._generate_basic_analysis(analysis_data)
    
    def _format_results_for_analysis(self, analysis_data: dict) -> str:
        """Format the results data for the AI analysis prompt."""
        formatted = ""
        
        # Overall statistics
        formatted += f"OVERALL STATISTICS:\n"
        for prompt_name in analysis_data['prompt_names']:
            wins = analysis_data['wins_by_prompt'][prompt_name]
            avg_score = analysis_data['avg_scores_by_prompt'][prompt_name]
            formatted += f"- {prompt_name}: {wins} wins, avg score {avg_score:.3f}\n"
        
        # Metric-specific results
        formatted += f"\nMETRIC-SPECIFIC RESULTS:\n"
        for metric_name in analysis_data['metrics']:
            formatted += f"\n{metric_name.upper()}:\n"
            metric_scores = analysis_data['avg_scores_by_metric'][metric_name]
            wins = analysis_data['metric_insights'][metric_name]['wins_by_prompt']
            for prompt_name in analysis_data['prompt_names']:
                score = metric_scores[prompt_name]
                win_count = wins[prompt_name]
                formatted += f"  {prompt_name}: {score:.3f} (avg), {win_count} wins\n"
        
        # Case-by-case details
        formatted += f"\nCASE-BY-CASE DETAILS:\n"
        for case in analysis_data['case_details']:
            formatted += f"\nCase {case['case_num']}:\n"
            for prompt_name in analysis_data['prompt_names']:
                response = case['responses'][prompt_name][:100] + "..." if len(case['responses'][prompt_name]) > 100 else case['responses'][prompt_name]
                tokens = case['tokens'][prompt_name]
                formatted += f"  {prompt_name}: {response} ({tokens} tokens)\n"
            
            formatted += f"  Winners by metric:\n"
            for metric_name in analysis_data['metrics']:
                winner = case['winners'][metric_name]
                scores = case['scores'][metric_name]
                score_str = ", ".join([f"{name}: {score:.3f}" for name, score in scores.items()])
                formatted += f"    {metric_name}: {winner} wins ({score_str})\n"
        
        return formatted
    
    def _generate_basic_analysis(self, analysis_data: dict) -> str:
        """Generate basic analysis as fallback."""
        analysis = f"""### Multi-Prompt Analysis

**Experiment Overview:**
- {len(analysis_data['prompt_names'])} prompts tested: {', '.join(analysis_data['prompt_names'])}
- {len(analysis_data['metrics'])} metrics used: {', '.join(analysis_data['metrics'])}
- {analysis_data['total_cases']} test cases evaluated

**Performance Summary:**
"""
        
        # Add performance insights
        best_prompt = max(analysis_data['wins_by_prompt'].items(), key=lambda x: x[1])
        analysis += f"- Most winning prompt: {best_prompt[0]} ({best_prompt[1]} wins)\n"
        
        best_avg_score = max(analysis_data['avg_scores_by_prompt'].items(), key=lambda x: x[1])
        analysis += f"- Highest average score: {best_avg_score[0]} ({best_avg_score[1]:.3f})\n"
        
        analysis += "\n**Metric-Specific Insights:**\n"
        for metric_name in analysis_data['metrics']:
            metric_scores = analysis_data['avg_scores_by_metric'][metric_name]
            best_metric_prompt = max(metric_scores.items(), key=lambda x: x[1])
            analysis += f"- {metric_name}: {best_metric_prompt[0]} performs best ({best_metric_prompt[1]:.3f})\n"
        
        analysis += "\n**Recommendations:**\n"
        analysis += "- Consider combining the best aspects of different prompts\n"
        analysis += "- Test with larger datasets for more reliable conclusions\n"
        analysis += "- Analyze specific use cases where each prompt excels\n"
        
        return analysis
    
    def save_csv(self, results: typing.List[MultiPromptResult], filename: typing.Optional[str] = None):
        """Save multi-prompt results to CSV."""
        if not results:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multi_results_{timestamp}.csv"
        
        prompt_names = list(results[0].responses.keys())
        metrics = list(results[0].scores.keys())
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['case', 'prompt', 'response', 'input_tokens', 'output_tokens']
            for metric_name in metrics:
                fieldnames.append(f'{metric_name}_score')
            fieldnames.append('winner')
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                for prompt_name in prompt_names:
                    row = {
                        'case': result.case_num,
                        'prompt': prompt_name,
                        'response': result.responses[prompt_name],
                        'input_tokens': result.input_tokens[prompt_name],
                        'output_tokens': result.output_tokens[prompt_name]
                    }
                    
                    for metric_name in metrics:
                        row[f'{metric_name}_score'] = result.scores[metric_name][prompt_name]
                    
                    # Determine winner for this prompt across all metrics
                    wins = sum(1 for metric_name in metrics if result.winners[metric_name] == prompt_name)
                    if wins > len(metrics) / 2:
                        row['winner'] = 'Yes'
                    elif wins == len(metrics) / 2:
                        row['winner'] = 'Tie'
                    else:
                        row['winner'] = 'No'
                    
                    writer.writerow(row)
        
        print(f"\n💾 Results saved to: {filename}") 