# The Hobbit Scholar - RAG Chatbot CLI

A sophisticated command-line RAG (Retrieval-Augmented Generation) chatbot for answering questions about J.R.R. Tolkien's "The Hobbit". Powered by Ollama, LangChain, and ChromaDB with advanced features including:

- 🔍 Hybrid retrieval (BM25 + vector search)
- 🎯 Contextual compression with FlashRank reranking
- 📖 Chapter-aware filtering
- 💬 Conversation history tracking
- ⚡ Streaming responses

## Prerequisites

1. **Python 3.9+** installed
2. **Ollama** installed and running with:
   - LLM model (e.g., `llama3.1`, `llama3.2`)
   - Embedding model (e.g., `mxbai-embed-large`)

Install Ollama models:
```bash
ollama pull llama3.1
ollama pull mxbai-embed-large
```

## Installation

### Option 1: Install From Source (Recommended for Development)

```bash
pip install .
```

For editable development installs:

```bash
pip install -e .
```

This creates two commands:
- `hobbit-scholar` - Main chatbot interface
- `hobbit-init` - Initialize vector database

### Option 2: Install Isolated With pipx (Recommended for End Users)

```bash
pipx install .
```

This keeps the CLI isolated from system Python packages while still exposing the same commands globally.

### Option 3: Build Once, Install Anywhere (Best for Multiple Devices)

Build distributable artifacts:

```bash
python -m build
```

This generates files in `dist/`:
- `*.whl` (wheel, preferred)
- `*.tar.gz` (source distribution)

Install the wheel on any target device:

```bash
pip install dist/hobbit_scholar-1.0.0-py3-none-any.whl
```

### First Run

After installation:

```bash
hobbit-init
hobbit-scholar
```

## Usage

### Database Initialization

Initialize the vector database before first use:

```bash
# Using script directly
python vector.py

# Or using installed CLI command
hobbit-init
```

**Advanced initialization options:**

```bash
# Force rebuild without confirmation
python vector.py --force

# Use a different source file
python vector.py --source MyCustomBook.md

# Specify custom database location
python vector.py --db ./custom_db

# View book metadata without rebuilding
python vector.py --info
```

### Running the Chatbot

**Interactive Mode (Default):**

```bash
# Using script
python main.py

# Or using installed command
hobbit-scholar
```

This starts an interactive session where you can ask multiple questions about The Hobbit.

**Single Query Mode:**

Ask a single question and exit:

```bash
# Using script
python main.py -q "Who is Bilbo Baggins?"

# Or with installed command
hobbit-scholar -q "What is the One Ring?"
```

**Advanced Options:**

```bash
# Use a different Ollama model
python main.py --model llama3.2

# Adjust creativity (temperature 0.0-1.0)
python main.py -q "Describe Smaug" --temperature 0.5

# Increase context window
python main.py --context-size 16000

# Quiet mode (minimal output, answer only)
python main.py -q "What is Sting?" --quiet

# Combine options
python main.py -q "Tell me about Gandalf" --model llama3.1 --temperature 0.3 --quiet
```

### CLI Command Reference

#### Main Chatbot (`main.py` or `hobbit-scholar`)

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--query` | `-q` | Ask a single question and exit | Interactive mode |
| `--model` | `-m` | Ollama model to use | `llama3.1` |
| `--temperature` | `-t` | Model temperature (0.0-1.0) | `0.1` |
| `--context-size` | `-c` | Context window size | `12000` |
| `--quiet` | | Minimal output (answer only) | Verbose |
| `--version` | `-v` | Show version | |
| `--help` | `-h` | Show help message | |

#### Database Initialization (`vector.py` or `hobbit-init`)

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--source` | `-s` | Source markdown file | `TheHobbit.md` |
| `--db` | `-d` | Database directory path | `./chroma_db` |
| `--force` | `-f` | Force rebuild without confirmation | Prompt |
| `--info` | `-i` | Display book metadata and exit | |
| `--version` | `-v` | Show version | |
| `--help` | `-h` | Show help message | |

## Examples

### Interactive Sessions

```bash
# Start interactive mode
python main.py

# Example interaction:
Question: Who is Thorin Oakenshield?
Response: [Detailed answer about Thorin...]

Question: Tell me more about his quest
Response: [Follow-up answer using conversation history...]

Question: exit
```

### Single Queries

```bash
# Quick question
python main.py -q "What is the Arkenstone?"

# Production pipelines (quiet mode for parsing)
python main.py -q "List all the dwarves" --quiet > dwarves.txt

# Using different models for comparison
python main.py -q "Describe Bilbo" --model llama3.1 > answer1.txt
python main.py -q "Describe Bilbo" --model llama3.2 > answer2.txt
```

### Database Management

```bash
# Initial setup
python vector.py

# Rebuild after updating TheHobbit.md
python vector.py --force

# Use different source
python vector.py --source "TheLordOfTheRings.md" --db ./lotr_db

# Check book statistics
python vector.py --info
```

## Features

### Conversation History
The chatbot maintains the last 4 Q&A turns for context, allowing natural follow-up questions like "tell me more" or "what happened next?"

### Chapter-Aware Filtering
When you mention a specific chapter (e.g., "What happens in Chapter 5?"), the chatbot automatically filters results to that chapter.

### Streaming Responses
Responses stream in real-time for better interactivity, with timing information displayed after completion.

### Smart Retrieval
- **Hybrid search:** Combines BM25 keyword matching with semantic vector search
- **Parent-document retrieval:** Retrieves small chunks but returns larger context
- **Reranking:** Uses FlashRank to reorder results by relevance

### Fast Startup ⚡
- **Lazy loading:** Knowledge base loads on first query, not at startup (~1-2s startup time)
- **Cached retriever:** Once loaded, subsequent queries are instant
- **Progress indicators:** See what's happening during initialization
- See [PERFORMANCE.md](PERFORMANCE.md) for optimization details

## Troubleshooting

**"Failed to connect to Ollama"**
- Make sure Ollama is running: `ollama serve`
- Verify model is installed: `ollama list`

**"Source file not found"**
- Ensure `TheHobbit.md` exists in the project directory
- Or specify a custom path: `python vector.py --source /path/to/file.md`

**Database errors**
- Try rebuilding: `python vector.py --force`
- Check disk space and permissions

**Slow responses**
- Reduce context size: `python main.py --context-size 8000`
- Use a smaller/faster model: `python main.py --model llama3.2`

## Project Structure

```
.
├── main.py                 # Main chatbot CLI
├── vector.py              # Vector database initialization
├── setup.py               # Package installation script
├── requirements.txt       # Python dependencies
├── TheHobbit.md          # Source text
├── chroma_db/            # Vector database (generated)
└── README.md             # This file
```

## Development

Run tests:
```bash
pytest test_rag_chatbot.py
pytest test_rag_integration.py
```

## License

MIT License (or specify your license)

## Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Powered by [Ollama](https://ollama.ai/)
- Vector storage by [ChromaDB](https://www.trychroma.com/)
- Reranking by [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank)
