# 🚀 Installing Duel

## Quick Install

```bash
# Clone the repository
git clone <your-repo-url>
cd dueling-prompts

# Install in development mode
pip install -e .

# That's it! Now you can use 'duel' from anywhere
duel --help
```

## What This Does

The `pip install -e .` command:

1. **Installs the package** in "editable" mode
2. **Creates a `duel` command** available system-wide
3. **Links to your source code** so changes are immediately available
4. **Installs dependencies** from `requirements.txt`

## Usage

Now you can use `duel` from any directory:

```bash
# Initialize a new workspace
duel init

# Save prompts
duel save summarize "Summarize: {input}"
duel save tldr "TL;DR: {input}"

# List saved prompts
duel list

# Test prompts
duel test summarize tldr -i "Hello world"

# Run from config file
duel run example_duel.yaml
```

## Uninstall

To remove the package:

```bash
pip uninstall prompt-duel
```

## Development

Since it's installed in editable mode, any changes you make to the code are immediately available without reinstalling.

## Alternative: Global Install

If you want to install it globally (not recommended for development):

```bash
pip install .
```

## Requirements

- Python 3.7+
- OpenAI API key: `export OPENAI_API_KEY="your-key"`

---

**That's it!** No more `./` needed - just `duel` from anywhere! 🎯 