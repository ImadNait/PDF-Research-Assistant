# PDF Research Assistant

AI-powered PDF research assistant using Retrieval-Augmented Generation (RAG). Upload your documents and ask questions in natural language.

## Features

- Semantic search across multiple PDF documents
- Source citations with page numbers
- Persistent vector database
- Optional AI-powered answers with Ollama

## Quick Start

### Installation
```bash
git clone https://github.com/yourusername/documind.git
cd documind
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Usage

1. Add PDF files to the `pdfs/` folder
2. Run the assistant:
```bash
   python pdf_rag.py
```
3. Ask questions about your documents

### Example
```
Your question: What is the main contribution of this paper?

Result 1:
Source: research_paper.pdf (Page 3)
Content: The Transformer architecture eliminates recurrence and relies
entirely on attention mechanisms...

Sources: research_paper.pdf (p.3, p.7)
```
## Optional: AI Answers

For natural language answers, install Ollama:
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.2
pip install langchain-ollama
```

Then select mode 2 when running the assistant.

## How It Works

1. PDFs are loaded and split into chunks
2. Chunks are converted to embeddings
3. Embeddings are stored in ChromaDB
4. User queries are matched against stored chunks
5. Relevant context is retrieved and displayed

## License

MIT
