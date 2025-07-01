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

load_dotenv()

print('DEBUG: list:', list)
print('DEBUG: dict:', dict)
print('DEBUG: tuple:', tuple)

@dataclass
class TestCase:
    input: str
    expected: typing.Optional[str] = None

print('DEBUG: TestCase after definition:', TestCase)

@dataclass
class DuelResult:
    case_num: int
    prompt_a_response: str
    prompt_b_response: str
    prompt_a_tokens: int
    prompt_b_tokens: int
    winner: str  # 'A', 'B', or 'Tie'
    score_a: float
    score_b: float


class PromptDuel:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.client = OpenAI()
        
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
        print('DEBUG: TestCase in run_duel:', TestCase)
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
            
            # Score responses
            metric = self.config.get('metric', 'exact')
            
            if metric == 'exact' and case.expected:
                score_a = self._score_exact(response_a, case.expected)
                score_b = self._score_exact(response_b, case.expected)
            elif metric == 'contains' and case.expected:
                score_a = self._score_contains(response_a, case.expected)
                score_b = self._score_contains(response_b, case.expected)
            elif metric == 'judge' and case.expected:
                score_a, score_b = self._score_judge(response_a, response_b, case.expected)
            else:
                # Default to exact match if no expected value or unknown metric
                score_a = 0.0
                score_b = 0.0
            
            winner = self._determine_winner(score_a, score_b)
            
            result = DuelResult(
                case_num=i + 1,
                prompt_a_response=response_a,
                prompt_b_response=response_b,
                prompt_a_tokens=tokens_a_prompt + tokens_a_completion,
                prompt_b_tokens=tokens_b_prompt + tokens_b_completion,
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
        
        total_tokens_a = sum(r.prompt_a_tokens for r in results)
        total_tokens_b = sum(r.prompt_b_tokens for r in results)
        
        for result in results:
            emoji = "✅" if result.winner != 'Tie' else "🤝"
            print(f"Case {result.case_num}: {emoji} Prompt {result.winner} wins")
        
        print("\n" + "-"*30)
        print(f"A wins: {a_wins} | B wins: {b_wins} | Ties: {ties}")
        print(f"Total tokens - A: {total_tokens_a}, B: {total_tokens_b}")
        
        if a_wins > b_wins:
            print("🏆 Overall winner: Prompt A")
        elif b_wins > a_wins:
            print("🏆 Overall winner: Prompt B")
        else:
            print("🤝 Overall result: Tie")
    
    def save_csv(self, results: typing.List[DuelResult]):
        """Save results to CSV file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['case', 'winner', 'score_a', 'score_b', 'tokens_a', 'tokens_b', 'response_a', 'response_b']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow({
                    'case': result.case_num,
                    'winner': result.winner,
                    'score_a': result.score_a,
                    'score_b': result.score_b,
                    'tokens_a': result.prompt_a_tokens,
                    'tokens_b': result.prompt_b_tokens,
                    'response_a': result.prompt_a_response,
                    'response_b': result.prompt_b_response
                })
        
        print(f"\n💾 Results saved to: {filename}") 