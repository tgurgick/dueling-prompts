# 🏆 Duel - Prompt Comparison Tool

A lightweight CLI tool for comparing two prompts against test cases using YAML configuration. Perfect for A/B testing prompts and validating prompt engineering experiments.

## Features

- **YAML Configuration**: Simple, readable experiment specs
- **Multiple Metrics**: Exact match, contains, or LLM judge
- **Token Tracking**: Monitor usage and costs
- **CSV Export**: Save detailed results for analysis
- **Progress Bars**: Visual feedback during testing
- **Template Variables**: Dynamic prompt interpolation
- **Unified CLI**: Single `duel` command for all operations
- **AI-Powered Analysis**: Intelligent insights about prompt performance
- **Prompt Management**: Save, list, and delete prompts

## Quick Start

### 1. Install the Package

```bash
# Clone and install
git clone <your-repo-url>
cd dueling-prompts
pip install -e .

# Or install directly from the directory
pip install -e .
```

### 2. Set Up OpenAI API Key

Create a `.env` file in your project directory:
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

Or set it as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Run Your First Duel

```bash
# Initialize workspace
duel init

# Save some prompts
duel save summarize "Summarize: {input}"
duel save tldr "TL;DR: {input}"

# Test them
duel test summarize tldr -i "Hello world"

# Or run from config file
duel run example_duel.yaml
```

## CLI Commands

The `duel` command provides a unified interface for all operations:

### Main Help
```bash
duel --help
```

### Available Commands

#### `duel init`
Initialize the duel workspace (creates `.duel/` directory and `prompts.yaml`).

#### `duel save <name> <prompt>`
Save a prompt template for later use.

**Examples:**
```bash
duel save summarize "Summarize the following text: {input}"
duel save translate "Translate to Spanish: {input}"
duel save classify "Classify this text as positive or negative: {input}"
```

#### `duel list`
List all saved prompts.

#### `duel delete <name>`
Delete a saved prompt by name.

**Examples:**
```bash
duel delete summarize
duel delete old_prompt
```

#### `duel test <prompt1> <prompt2> [options]`
Test two prompts against inputs. Prompts can be saved names or inline templates.

**Required Arguments:**
- `prompt1`: First prompt (saved name or template)
- `prompt2`: Second prompt (saved name or template)
- `-i, --inputs`: Test inputs (one or more)

**Optional Arguments:**
- `-e, --expected`: Expected outputs for scoring
- `-m, --model`: OpenAI model to use (default: `gpt-4o-mini`)
- `--metric`: Scoring metric: `exact`, `contains`, or `judge` (default: `exact`)
- `--save-csv`: Save results to CSV file

**Examples:**
```bash
# Test saved prompts
duel test summarize tldr -i "Hello world" "Another test"

# Test inline prompts
duel test "What is {input}?" "Calculate {input}" -i "2+2" "3+3" "5*2"

# Use different model
duel test summarize tldr -i "Long text here" -m gpt-3.5-turbo

# Use judge metric with expected outputs
duel test "Write a story about {input}" "Create a narrative about {input}" \
  -i "a robot" "a time traveler" \
  -e "creative story" "engaging narrative" \
  --metric judge

# Save results to CSV
duel test summarize tldr -i "test1" "test2" --save-csv
```

#### `duel run <config.yaml> [options]`
Run a duel from a YAML configuration file.

**Arguments:**
- `config`: Path to YAML configuration file
- `--save-csv`: Save results to CSV file

**Examples:**
```bash
duel run example_duel.yaml
duel run advanced_duel.yaml --save-csv
```

## Model Selection

You can specify different OpenAI models using the `-m` or `--model` flag:

### Available Models

**GPT-4 Models:**
- `gpt-4o` - Latest GPT-4 model (most capable)
- `gpt-4o-mini` - Faster, more efficient GPT-4 (default)
- `gpt-4-turbo` - Previous GPT-4 Turbo model
- `gpt-4` - Standard GPT-4 model

**GPT-3.5 Models:**
- `gpt-3.5-turbo` - Fast and cost-effective
- `gpt-3.5-turbo-16k` - Higher context window

**Examples:**
```bash
# Use GPT-4o for highest quality
duel test summarize tldr -i "test" -m gpt-4o

# Use GPT-3.5-turbo for faster/cheaper testing
duel test summarize tldr -i "test" -m gpt-3.5-turbo

# Use in YAML config
duel run config.yaml  # where config.yaml specifies model: gpt-4o
```

### Model Considerations

- **GPT-4o**: Best quality, higher cost, slower
- **GPT-4o-mini**: Good balance of quality, cost, and speed (recommended)
- **GPT-3.5-turbo**: Fastest and cheapest, good for initial testing

## Scoring Metrics

### Exact Match (`exact`) - Default
- Case-insensitive string comparison
- Perfect for deterministic outputs
- Requires expected outputs to be specified

### Contains (`contains`)
- Checks if expected text is contained in response
- Good for flexible but specific requirements
- Requires expected outputs to be specified

### LLM Judge (`judge`)
- Uses another LLM to compare responses
- Best for subjective quality assessment
- Can work without expected outputs
- Uses the same model by default, or specify `judge_model` in YAML

**Examples:**
```bash
# Exact match (requires expected outputs)
duel test "What is {input}?" "Calculate {input}" \
  -i "2+2" "3+3" \
  -e "4" "6" \
  --metric exact

# Contains check
duel test "Summarize: {input}" "TL;DR: {input}" \
  -i "Long text here" \
  -e "summary" \
  --metric contains

# LLM judge (no expected outputs needed)
duel test "Write a story about {input}" "Create a narrative about {input}" \
  -i "a robot" "a time traveler" \
  --metric judge
```

## AI-Powered Analysis

Every test run includes intelligent analysis of your results:

### What the AI Analyzes:
- **Prompt differences** - How your prompts approach the task differently
- **Performance insights** - Analysis of wins, token usage, and response quality
- **Recommendations** - Actionable suggestions for improving your prompts

### Example Analysis Output:
```
==================================================
🧠 AI ANALYSIS
==================================================
The key difference between Prompt A and Prompt B lies in their approach...
[Intelligent analysis of prompt differences, performance, and recommendations]
```

The AI analysis helps you understand not just which prompt won, but why, and how to improve your prompt engineering skills.

## Configuration Format

Create a YAML file with your experiment configuration:

```yaml
experiment: "My Prompt Test"
model: "gpt-4o-mini"
prompts:
  A: "Your first prompt template: {input}"
  B: "Your second prompt template: {input}"
cases:
  - input: "Test input 1"
    expected: "Expected output 1"
  - input: "Test input 2"
    expected: "Expected output 2"
metric: "exact"  # or "contains" or "judge"
system_prompt: "Optional system prompt"
vars:
  custom_var: "value"  # Access with {vars.custom_var}
```

## Configuration Options

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `experiment` | string | ✅ | Human-readable experiment name |
| `model` | string | ✅ | OpenAI model ID (e.g., `gpt-4o-mini`) |
| `prompts` | map | ✅ | Exactly two prompts (`A` and `B`) |
| `cases` | list | ✅ | Test cases with `input` and optional `expected` |
| `metric` | string | ❌ | Scoring method: `exact`, `contains`, or `judge` |
| `judge_model` | string | ❌ | Model for judging (if `metric: judge`) |
| `system_prompt` | string | ❌ | System prompt prepended to both prompts |
| `vars` | map | ❌ | Global variables for template interpolation |

## Template Variables

Use `{input}` for test case input and `{vars.variable_name}` for global variables:

```yaml
prompts:
  A: "Write a {vars.genre} story about {input} in {vars.word_count} words."
vars:
  genre: "science fiction"
  word_count: "100"
```

## Usage Examples

### Basic Usage
```bash
# Run from YAML config
duel run config.yaml

# Save results to CSV
duel run config.yaml --save-csv
```

### Quick Testing with CLI
```bash
# Test saved prompts
duel test summarize tldr -i "Hello world"

# Test inline prompts
duel test "What is {input}?" "Calculate {input}" -i "2+2" "3+3"

# Use different model
duel test summarize tldr -i "Long text here" -m gpt-3.5-turbo

# Use judge metric
duel test "Write a story about {input}" "Create a narrative featuring {input}" \
  -i "a robot" "a time traveler" --metric judge

# Test with expected outputs for scoring
duel test "What is {input}?" "Calculate {input}" \
  -i "2+2" "3+3" "5*2" \
  -e "4" "6" "10" \
  --metric exact

# Save detailed results to CSV
duel test summarize tldr -i "test1" "test2" --save-csv
```

### Prompt Management
```bash
# Initialize workspace
duel init

# Save prompts for reuse
duel save summarize "Summarize the following text: {input}"
duel save translate "Translate to Spanish: {input}"
duel save classify "Classify as positive/negative: {input}"

# List saved prompts
duel list

# Delete a prompt
duel delete old_prompt

# Use saved prompts in tests
duel test summarize translate -i "Hello world"
```

### Example Output
```
🏆 Running: Summarization Duel
🤖 Model: gpt-4o-mini
📊 Cases: 3

Testing cases: 100%|██████████| 3/3 [00:05<00:00]

==================================================
📋 RESULTS
==================================================
Case 1: ✅ Prompt A wins
Case 2: ✅ Prompt B wins
Case 3: 🤝 Tie

------------------------------
A wins: 1 | B wins: 1 | Ties: 1
Total tokens - A: 245, B: 198
🤝 Overall result: Tie
```

### CSV Output Format

When using `--save-csv`, results are saved to a timestamped file (e.g., `results_20250701_161759.csv`) with the following columns:

| Column | Description |
|--------|-------------|
| `case` | Test case number |
| `winner` | Winner: A, B, or Tie |
| `score_a` | Score for prompt A (0.0-1.0) |
| `score_b` | Score for prompt B (0.0-1.0) |
| `tokens_a` | Total tokens used by prompt A |
| `tokens_b` | Total tokens used by prompt B |
| `response_a` | Full response from prompt A |
| `response_b` | Full response from prompt B |

**Example CSV:**
```csv
case,winner,score_a,score_b,tokens_a,tokens_b,response_a,response_b
1,Tie,0.0,0.0,22,20,2 + 2 equals 4.,2 + 2 equals 4.
2,Tie,0.0,0.0,22,20,3 + 3 equals 6.,3 + 3 equals 6.
3,Tie,0.0,0.0,23,20,5 multiplied by 2 equals 10.,5 * 2 = 10.
```

## Package Structure

```
dueling-prompts/
├── prompt_duel/           # Package source code
│   ├── __init__.py
│   ├── cli.py            # Command-line interface
│   └── engine.py         # Core duel engine
├── setup.py              # Package configuration
├── requirements.txt      # Dependencies
├── example_duel.yaml     # Example configuration
├── advanced_duel.yaml    # Advanced example
└── .duel/               # User workspace (created by duel init)
    └── prompts.yaml     # Saved prompts
```

## Example Configurations

### Simple Summarization Test
See `example_duel.yaml` for a basic summarization comparison.

### Creative Writing with Judge
See `advanced_duel.yaml` for an example using the judge metric and template variables.

## Requirements

- Python 3.7+
- OpenAI API key
- Internet connection

## Dependencies

All dependencies are automatically installed when you install the package:
- `pyyaml` - YAML configuration parsing
- `openai` - OpenAI API client
- `tqdm` - Progress bars
- `click` - CLI interface

## Installation

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

## Tips

1. **Start Simple**: Use `exact` metric for deterministic outputs
2. **Use Judge Sparingly**: LLM judging costs more and may be subjective
3. **Template Variables**: Keep prompts DRY with `{vars.xxx}`
4. **CSV Export**: Use `--save-csv` for detailed analysis
5. **Token Monitoring**: Watch token usage to manage costs

## Troubleshooting

### Common Issues

**Missing API Key**
```
❌ Error: OpenAI API key not found
```
Solution: Set `OPENAI_API_KEY` environment variable

**Invalid YAML**
```
❌ Error: Missing required field: prompts
```
Solution: Check your YAML syntax and required fields

**Model Not Found**
```
❌ Error: Model not found
```
Solution: Verify the model name in your configuration

## Contributing

This is a minimal MVP. Potential enhancements:
- Support for more than 2 prompts
- Custom scoring functions
- Cost estimation
- Parallel execution
- Web interface

---

Happy dueling! 🥊 