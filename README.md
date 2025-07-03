# Duel - Modular Prompt Testing & Management

A powerful CLI tool for A/B testing prompts with OpenAI models, featuring modular architecture, versioning, and comprehensive analytics.

## 🚀 Features

- **Modular Architecture**: Separate management of prompts, metrics, and experiments
- **Prompt Versioning**: Track prompt evolution with full history and diffing
- **Multiple Metrics**: Exact match, contains check, LLM judge, semantic similarity, relevance, and safety evaluation
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
│   │   └── results.json
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

# View experiment details
duel experiment view <experiment_id>

# Create templates
duel experiment template <name> [--from <existing>]
```

## 📊 Available Metrics

- **exact_match**: Case-insensitive exact matching
- **contains_check**: Check if expected text is contained in response
- **llm_judge**: Use LLM to compare two responses
- **relevance**: Semantic relevance using sentence transformers
- **semantic_similarity**: Cosine similarity of text embeddings
- **safety_judge**: Safety evaluation using LLM judge

## 🔬 Experiment Configuration

Create a YAML file for your experiment:

```yaml
experiment: "My Prompt Comparison"
model: "gpt-4o-mini"
judge_model: "gpt-4o"  # Optional: different model for judging
prompts:
  A: "Summarize the following: {input}"
  B: "Summarize in one sentence: {input}"
cases:
  - input: "The quick brown fox jumps over the lazy dog..."
    expected: "A quick brown fox jumps over a lazy dog..."
  - input: "Machine learning is a subset of artificial intelligence..."
    expected: "Machine learning enables computers to learn..."
metric: "judge"  # or "exact", "contains", "relevance", etc.
system_prompt: "You are a helpful assistant."
```

## 📈 Enhanced Results

The tool now provides detailed analysis:

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
- **Metrics**: YAML definitions for scoring methods
- **Experiments**: Results stored with timestamps and metadata
- **History**: JSON logs for audit trails and analytics

## 📝 Examples

### Complex Prompt Comparison
```bash
# Save two different prompt styles
duel prompt save technical "You are an expert technical writer..." --tags technical formal
duel prompt save engaging "Imagine you are explaining to high school students..." --tags engaging educational

# Run comparison with LLM judge
duel experiment run complex_comparison.yaml
```

### Metric Testing
```bash
# Test semantic similarity
duel metric test semantic_similarity "Response A" "Response B" --expected "Expected output"
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

**Duel** - Making prompt engineering systematic and measurable. 