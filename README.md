# Duel - Modular Prompt Testing & Management

A powerful CLI tool for A/B testing prompts with OpenAI models, featuring modular architecture, versioning, comprehensive analytics, and AI-powered insights.

## 🚀 Features

- **Modular Architecture**: Separate management of prompts, metrics, and experiments
- **Prompt Versioning**: Track prompt evolution with full history and diffing
- **Multiple Metrics**: Exact match, contains check, LLM judge, semantic similarity, relevance, and safety evaluation
- **Multi-Prompt Experiments**: Compare more than two prompts simultaneously
- **AI-Powered Analysis**: Automatic generation and saving of insights about prompt performance
- **Configurable Metrics**: Customize model parameters for different scoring methods
- **Experiment Management**: Run, track, and analyze prompt comparison experiments
- **Enhanced Analytics**: Detailed token usage (input/output), response comparison, and AI-powered insights
- **History Tracking**: Complete audit trail of prompt changes and experiment results

## 📦 Installation

```bash
pip install prompt-duel
```

Or install from source:
```bash
git clone https://github.com/tgurgick/dueling-prompts.git
cd dueling-prompts
pip install -e .
```

## 🏗️ Architecture

The tool uses a modular directory structure:

```
.duel/
├── prompts/           # Prompt versions and metadata
│   ├── summarize/
│   │   ├── v1.yaml    # Prompt version
│   │   ├── v2.yaml    # Updated version
│   │   └── metadata.yaml
├── metrics/           # Available scoring metrics
│   ├── exact_match.yaml
│   ├── llm_judge.yaml
│   └── ...
├── experiments/       # Experiment results
│   ├── exp_1234567890/
│   │   ├── config.yaml
│   │   ├── results.json
│   │   └── analysis.txt    # AI-generated insights
└── history/          # Audit trail
    ├── prompt_evolution.json
    └── experiment_history.json
```

## 🎯 Quick Start

### Initialize Workspace
```bash
duel prompt save summarize "Summarize this: {input}" --tags summarization
```

### List Prompts
```bash
duel prompt list
duel prompt list --tags summarization
```

### Version Management
```bash
duel prompt save summarize "Summarize in one sentence: {input}" --changes "More concise"
duel prompt version summarize
duel prompt diff summarize summarize --version1 v1 --version2 v2
```

### Run Experiments
```bash
duel experiment run config.yaml
duel experiment list
duel experiment view exp_1234567890
duel experiment analysis exp_1234567890  # View AI analysis
```

## 📋 CLI Commands

### Prompt Management
```bash
# Save prompts with versioning
duel prompt save <name> <prompt> [--version <v>] [--tags <tags>] [--changes <description>]

# List and filter prompts
duel prompt list [--tags <filter>]

# Version management
duel prompt version <name> [--show <version>]
duel prompt diff <name1> <name2> [--version1 <v1>] [--version2 <v2>]
duel prompt history <name>

# Delete prompts
duel prompt delete <name>
```

### Metric Management
```bash
# List available metrics
duel metric list

# Configure metric parameters
duel metric configure <name> [--model <model>] [--judge-model <model>] [--temperature <temp>]

# Show available models for metrics
duel metric models [<metric_name>]

# Create custom metrics
duel metric create <name> --type <type> --description <desc> --implementation <impl> [--params <json>]

# Test metrics
duel metric test <name> <response1> <response2> [--expected <expected>]
```

### Experiment Management
```bash
# Run experiments
duel experiment run <config.yaml>

# List experiments
duel experiment list [--status <pending|completed>]

# View experiment details (includes AI analysis)
duel experiment view <experiment_id>

# View just the AI analysis
duel experiment analysis <experiment_id>

# Create templates
duel experiment template <name> [--from <existing>]
```

## 📊 Available Metrics

- **exact_match**: Case-insensitive exact matching
- **contains_check**: Check if expected text is contained in response
- **llm_judge**: Use LLM to compare two responses (configurable prompts)
- **relevance**: Semantic relevance using sentence transformers (configurable models)
- **semantic_similarity**: Cosine similarity of text embeddings (configurable models)
- **safety_judge**: Safety evaluation using LLM judge (configurable prompts)

### Configurable Models
- **Sentence Transformers**: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `paraphrase-multilingual-MiniLM-L12-v2`, etc.
- **OpenAI Models**: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`

## 🔬 Experiment Configuration

### Single Prompt Comparison
Create a YAML file for your experiment:

```yaml
experiment: "My Prompt Comparison"
model: "gpt-4o-mini"
prompts:
  A: "Summarize the following: {input}"
  B: "Summarize in one sentence: {input}"
cases:
  - input: "The quick brown fox jumps over the lazy dog..."
    expected: "A quick brown fox jumps over a lazy dog..."
  - input: "Machine learning is a subset of artificial intelligence..."
    expected: "Machine learning enables computers to learn..."
metric: "exact_match"  # or "contains_check", "llm_judge", "relevance", etc.
system_prompt: "You are a helpful assistant."
```

### Multi-Prompt Comparison
Compare multiple prompts with multiple metrics:

```yaml
experiment: "Multi-Prompt Analysis"
model: "gpt-4o-mini"
prompts:
  formal: "Please provide a formal summary: {input}"
  casual: "TL;DR: {input}"
  technical: "Summarize technically: {input}"
  creative: "Create a creative summary: {input}"
metrics:
  - exact_match
  - relevance
  - llm_judge
cases:
  - input: "The quick brown fox jumps over the lazy dog..."
    expected: "A quick brown fox jumps over a lazy dog..."
  - input: "Machine learning is a subset of artificial intelligence..."
    expected: "Machine learning enables computers to learn..."
```

## 📈 Enhanced Results with AI Analysis

The tool now provides detailed analysis with AI-powered insights:

```
Case 1: ✅ Prompt A wins
  Score A: 1.000 | Score B: 0.000
  Tokens A: 53 input + 31 output = 84 total
  Tokens B: 54 input + 29 output = 83 total
  Response A: The sentence "The quick brown fox jumps over the lazy dog"...
  Response B: The sentence "The quick brown fox jumps over the lazy dog"...

------------------------------
A wins: 3 | B wins: 0 | Ties: 0
Total tokens - A: 106 input + 59 output = 165
Total tokens - B: 108 input + 57 output = 165
🏆 Overall winner: Prompt A

==================================================
🧠 AI ANALYSIS
==================================================
### Key Differences in Prompt Approaches

The primary distinction between Prompt A and Prompt B lies in their...

### Performance Insights

Both prompts resulted in...

### Recommendations for Improvement

To enhance the effectiveness of these prompts...
```

## 🔧 Configuration

### Metric Configuration
Configure models and parameters for different metrics:

```bash
# Configure semantic similarity to use a different model
duel metric configure semantic_similarity --model all-mpnet-base-v2

# Configure LLM judge to use a different model
duel metric configure llm_judge --judge-model gpt-4o-mini --temperature 0.1

# View available models for a metric
duel metric models relevance
```

### Custom Metric Prompts
Create YAML files for LLM-based metrics with custom prompts:

```yaml
# .duel/metrics/llm_judge.yaml
name: llm_judge
type: llm_judge
description: "LLM-based response comparison"
implementation: "LLMJudgeMetric"
parameters:
  judge_model: gpt-4o
  temperature: 0.0
prompts:
  system: "You are an expert evaluator comparing two responses..."
  user: "Compare these responses for quality and accuracy..."
```

## 🔄 Migration from Legacy

If you have existing YAML files, the legacy commands still work:

```bash
# Legacy commands (still supported)
duel save <name> <prompt>
duel list
duel test <prompt1> <prompt2> --inputs <inputs>
duel run <config.yaml>
```

## 🛠️ Development

### Adding Custom Metrics

Create a new metric by extending the `Metric` class:

```python
from prompt_duel.metrics import Metric

class MyCustomMetric(Metric):
    def score(self, response: str, expected: str) -> float:
        # Your scoring logic here
        return score
    
    def score_comparative(self, response_a: str, response_b: str, expected: str) -> Tuple[float, float]:
        # Your comparative scoring logic here
        return score_a, score_b
```

### Directory Structure

- **Prompts**: Each prompt gets its own directory with versioned YAML files
- **Metrics**: YAML definitions for scoring methods with configurable parameters
- **Experiments**: Results stored with timestamps, metadata, and AI analysis
- **History**: JSON logs for audit trails and analytics

## 📝 Examples

### Complex Multi-Prompt Comparison
```bash
# Save multiple prompt styles
duel prompt save technical "You are an expert technical writer..." --tags technical formal
duel prompt save engaging "Imagine you are explaining to high school students..." --tags engaging educational
duel prompt save concise "Provide a brief summary..." --tags concise

# Run multi-prompt comparison
duel experiment run multi_prompt_test.yaml

# View results with AI analysis
duel experiment view exp_1234567890

# View just the AI analysis
duel experiment analysis exp_1234567890
```

### Metric Testing and Configuration
```bash
# Test semantic similarity
duel metric test semantic_similarity "Response A" "Response B" --expected "Expected output"

# Configure metrics for better performance
duel metric configure relevance --model all-mpnet-base-v2
duel metric configure safety_judge --judge-model gpt-4o-mini
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 Links

- GitHub: https://github.com/tgurgick/dueling-prompts
- Issues: https://github.com/tgurgick/dueling-prompts/issues

---

**Duel** - Making prompt engineering systematic, measurable, and insightful with AI-powered analysis. 