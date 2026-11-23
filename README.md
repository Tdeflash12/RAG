# Local RAG with LangChain and Ollama

A simple Retrieval-Augmented Generation (RAG) example built with LangChain. The
project splits a sample AI document into chunks, creates embeddings with
Hugging Face, stores them in ChromaDB, and uses a local Ollama model to answer
questions with source documents.

## Features

- Recursive document chunking
- Hugging Face embeddings using `all-MiniLM-L6-v2`
- Persistent ChromaDB vector store
- Local LLM inference with Ollama
- Similarity search with source document retrieval
- Multiple example questions about AI, NLP, computer vision, and reinforcement learning

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/)
- An Ollama model named `llama3.1:8b`

Install and start Ollama, then download the model:

```bash
ollama pull llama3.1:8b
```

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>

python -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the RAG pipeline:

```bash
python rag.py
```

The script will:

1. Prepare the sample document.
2. Split the document into overlapping chunks.
3. Generate embeddings.
4. Create or update the local `chroma_db/` vector store.
5. Run a similarity search.
6. Ask the local Ollama model several questions.
7. Print each answer and its source documents.

The first run may download the embedding model from Hugging Face and can take
longer than subsequent runs.

## Project Structure

```text
.
├── rag.py             # Main RAG pipeline
├── requirements.txt   # Python dependencies
├── chroma_db/         # Persisted ChromaDB data
└── README.md          # Project documentation
```

## Configuration

The main settings are defined in `rag.py`:

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Ollama model: `llama3.1:8b`
- Chunk size: `200`
- Chunk overlap: `50`
- Retrieved documents: `2`

To use a different Ollama model, update the `model` value in the `ChatOllama`
configuration and pull that model with Ollama first.

## Troubleshooting

### Ollama connection error

Make sure Ollama is installed and running:

```bash
ollama serve
```

Then confirm that the required model is available:

```bash
ollama list
```

### Embedding model download issues

The Hugging Face embedding model is downloaded on first use. Check your
internet connection and ensure the Python environment can access Hugging Face.

## License

Add a license to this repository before publishing if you want to specify how
others may use, modify, and distribute the code.
