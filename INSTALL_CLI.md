# Installing The CLI Across Devices

This guide focuses on package-based installation so `hobbit-scholar` and `hobbit-init` work the same way on Windows, macOS, and Linux.

## Recommended Install Methods

### 1. Install directly from source

From the project root:

```bash
pip install .
```

For development mode (live code edits):

```bash
pip install -e .
```

### 2. Install with pipx (isolated global CLI)

```bash
pipx install .
```

Use this for clean, user-level CLI installs without affecting other Python projects.

## Best Method for Multiple Devices

Build the package once and install the wheel everywhere.

### On your build machine

```bash
python -m build
```

Artifacts are created in `dist/`:
- `hobbit_scholar-1.0.0-py3-none-any.whl`
- `hobbit_scholar-1.0.0.tar.gz`

### On each target machine

Copy the wheel file, then run:

```bash
pip install hobbit_scholar-1.0.0-py3-none-any.whl
```

## Verify Installation

```bash
hobbit-scholar --help
hobbit-init --help
```

## First-Time Runtime Setup

Install and run Ollama plus required models on each machine:

```bash
ollama pull llama3.1
ollama pull mxbai-embed-large
```

Initialize the local vector DB:

```bash
hobbit-init
```

Run the chatbot:

```bash
hobbit-scholar
```

## Troubleshooting

`command not found` or `not recognized`:
- Open a new terminal after install.
- Ensure the Python scripts directory is on PATH (pip usually configures this).

`Failed to connect to Ollama`:
- Start Ollama service with `ollama serve`.
- Confirm models are installed with `ollama list`.

Database issues:
- Rebuild local DB with `hobbit-init --force`.
