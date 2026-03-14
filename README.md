# The Hobbit Scholar - RAG Chatbot CLI

A sophisticated command-line RAG (Retrieval-Augmented Generation) chatbot for answering questions about J.R.R. Tolkien's "The Hobbit". Powered by Ollama, LangChain, and ChromaDB with advanced features including:

-  Hybrid retrieval (BM25 + vector search)
- Contextual compression with FlashRank reranking
- Chapter-aware filtering
- Conversation history tracking
- Streaming responses

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
pip install hobbit-scholar
```


### First Run

After installation:

```bash
hobbit-init
hobbit-scholar
```

## License

MIT License (or specify your license)

## Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Powered by [Ollama](https://ollama.ai/)
- Vector storage by [ChromaDB](https://www.trychroma.com/)
- Reranking by [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank)
