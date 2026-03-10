# CLI Quick Reference Guide

## Installation

```bash
# Standard install (creates hobbit-scholar and hobbit-init commands)
pip install .

# Editable install for development
pip install -e .

# Isolated global CLI install
pipx install .
```

## Quick Start

```bash
# 1. Initialize database
hobbit-init          # or: python vector.py

# 2. Run chatbot
hobbit-scholar       # or: python main.py
```

## Command Cheat Sheet

### Chatbot Commands

```bash
# Interactive mode (default)
hobbit-scholar
python main.py

# Single question
hobbit-scholar -q "Who is Bilbo?"
python main.py -q "Who is Bilbo?"

# Change model
hobbit-scholar --model llama3.2

# Adjust creativity
hobbit-scholar -q "Describe Smaug" --temperature 0.5

# Quiet mode (answer only)
hobbit-scholar -q "What is Sting?" --quiet

# Full configuration
hobbit-scholar -q "Tell me about Gandalf" \
  --model llama3.1 \
  --temperature 0.3 \
  --context-size 16000 \
  --quiet
```

### Database Commands

```bash
# Initialize database
hobbit-init
python vector.py

# Force rebuild
hobbit-init --force
python vector.py --force

# Custom source file
hobbit-init --source MyBook.md --db ./my_db
python vector.py --source MyBook.md --db ./my_db

# Show book stats
hobbit-init --info
python vector.py --info
```

## Common Use Cases

### Development & Testing

```bash
# Quick test query
hobbit-scholar -q "List the dwarves" --quiet

# Benchmark different models
time hobbit-scholar -q "Who is Thorin?" --model llama3.1
time hobbit-scholar -q "Who is Thorin?" --model llama3.2

# Save responses
hobbit-scholar -q "Summarize Chapter 1" --quiet > chapter1_summary.txt
```

### Production Pipeline

```bash
# Batch processing (example)
while IFS= read -r question; do
  hobbit-scholar -q "$question" --quiet
done < questions.txt > answers.txt
```

### Interactive Exploration

```bash
hobbit-scholar

# Then in the interactive session:
Question: Who is Bilbo Baggins?
# [Answer appears...]

Question: Tell me more about his home
# [Follow-up answer using conversation history...]

Question: What chapter introduces him?
# [Chapter-aware answer...]

Question: exit
```

## Options Reference

### Main Chatbot Options

| Short | Long | Default | Description |
|-------|------|---------|-------------|
| `-q` | `--query` | Interactive | Single question mode |
| `-m` | `--model` | llama3.1 | Ollama model name |
| `-t` | `--temperature` | 0.1 | Creativity (0.0-1.0) |
| `-c` | `--context-size` | 12000 | Context window |
| | `--quiet` | False | Minimal output |
| `-v` | `--version` | | Show version |
| `-h` | `--help` | | Show help |

### Database Init Options

| Short | Long | Default | Description |
|-------|------|---------|-------------|
| `-s` | `--source` | TheHobbit.md | Source file |
| `-d` | `--db` | ./chroma_db | Database path |
| `-f` | `--force` | False | Force rebuild |
| `-i` | `--info` | False | Show metadata only |
| `-v` | `--version` | | Show version |
| `-h` | `--help` | | Show help |

## Troubleshooting

```bash
# Check Ollama is running
ollama list
ollama serve  # if not running

# Verify models are installed
ollama pull llama3.1
ollama pull mxbai-embed-large

# Rebuild database if issues
hobbit-init --force

# Test with minimal query
hobbit-scholar -q "test" --quiet
```

## Tips & Tricks

1. **Speed up responses:** Use `--context-size 8000` for faster processing
2. **More creative answers:** Increase `--temperature` to 0.5-0.7
3. **Scripting:** Use `--quiet` flag for clean output in scripts
4. **Chapter-specific questions:** Mention chapter numbers (e.g., "What happens in Chapter 5?")
5. **Follow-up questions:** Use interactive mode for multi-turn conversations
6. **Save responses:** Redirect output with `> filename.txt`

## Environment Variables (Optional)

You can also set defaults via environment variables:

```bash
# Windows PowerShell
$env:HOBBIT_MODEL = "llama3.2"
$env:HOBBIT_TEMPERATURE = "0.3"
$env:HOBBIT_DB_PATH = "./my_custom_db"

# Linux/Mac
export HOBBIT_MODEL=llama3.2
export HOBBIT_TEMPERATURE=0.3
export HOBBIT_DB_PATH=./my_custom_db
```

(Note: This would require code modifications to read these env vars)
