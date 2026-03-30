# Retrieval-Augmented-Generation-RAG-

# RAG PDF Q&A — Retrieval-Augmented Generation with LangChain, Groq & ChromaDB

A Jupyter Notebook implementation of a Retrieval-Augmented Generation (RAG) pipeline that lets you ask natural language questions against a PDF document. The project uses a HuggingFace embedding model for semantic search, ChromaDB as the vector store, and Groq's LLaMA 3.1 as the LLM — all wired together with LangChain. A Gradio UI is included for interactive Q&A.

---

## How It Works

```
PDF Document
     │
     ▼
PyPDFLoader  ──►  CharacterTextSplitter  ──►  HuggingFace Embeddings
                                                        │
                                                        ▼
                                                  ChromaDB (persisted)
                                                        │
                                          User Question ──► Retriever (top-k=3)
                                                        │
                                                        ▼
                                              Groq LLaMA 3.1 8B Instant
                                                        │
                                                        ▼
                                                    Answer
```

1. **Load** — Downloads and loads a PDF (Python OOPs tutorial) using `PyPDFLoader`.
2. **Split** — Splits the 111-page document into chunks (`chunk_size=1000`, `chunk_overlap=200`).
3. **Embed** — Generates embeddings with `all-MiniLM-L6-v2` from HuggingFace.
4. **Store** — Persists vectors in a local ChromaDB instance (`./chroma_db`).
5. **Retrieve** — At query time, fetches the top 3 most relevant chunks.
6. **Generate** — Passes context + question to Groq LLaMA 3.1 8B Instant.
7. **UI** — Serves the full pipeline through a Gradio web interface.

---

## Tech Stack

| Component        | Library / Model                          |
|------------------|------------------------------------------|
| Framework        | LangChain, LangChain Community           |
| Document Loader  | `PyPDFLoader` (via `langchain_community`)|
| Text Splitting   | `CharacterTextSplitter`                  |
| Embeddings       | `all-MiniLM-L6-v2` (HuggingFace)        |
| Vector Store     | ChromaDB (`langchain-chroma`)            |
| LLM              | Groq — `llama-3.1-8b-instant`            |
| UI               | Gradio                                   |
| Python Version   | 3.13                                     |

---

## Prerequisites

- Python 3.10+
- Anaconda (or any virtual environment)
- A **Groq API key** — get one free at [console.groq.com](https://console.groq.com)

---

## Installation

Install all required packages:

```bash
pip install --upgrade langchain
pip install langchain_community
pip install "Unstructured[pdf]"
pip install -U langchain-text-splitters
pip install langchain_groq
pip install langchain-chroma
pip install -qU langchain-huggingface
pip install gradio
```

---

## Configuration

**Important:** The notebook sets the Groq API key inline. For security, replace the hardcoded key with an environment variable or a `.env` file:

```python
# Recommended: use a .env file
from dotenv import load_dotenv
load_dotenv()

# Or set it via terminal before launching Jupyter:
# export GROQ_API_KEY="your_key_here"   (Linux/Mac)
# set GROQ_API_KEY=your_key_here        (Windows CMD)
```

---

## Usage

1. Clone or download the notebook.
2. Set your `GROQ_API_KEY` (see above).
3. Open `RAG_code.ipynb` in Jupyter and run all cells.
4. The notebook will:
   - Download the source PDF automatically.
   - Build and persist the ChromaDB vector store to `./chroma_db`.
   - Launch a Gradio UI at `http://127.0.0.1:7861`.
5. Type any question about the PDF content into the Gradio interface.

### Example questions

- *What are the types of inheritance in Python?*
- *What is encapsulation?*
- *Explain polymorphism with an example.*

---

## Project Structure

```
.
├── RAG_code.ipynb       # Main notebook
├── python_oops.pdf      # Downloaded PDF (auto-created on first run)
└── chroma_db/           # Persisted vector store (auto-created on first run)
```

---

## Key Parameters

| Parameter        | Value            | Location                    |
|------------------|------------------|-----------------------------|
| `chunk_size`     | 1000             | `CharacterTextSplitter`     |
| `chunk_overlap`  | 200              | `CharacterTextSplitter`     |
| `k` (retriever)  | 3                | `vectorstore.as_retriever`  |
| `temperature`    | 1                | `ChatGroq`                  |
| Embedding model  | all-MiniLM-L6-v2 | `HuggingFaceEmbeddings`     |
| LLM model        | llama-3.1-8b-instant | `ChatGroq`              |

---

## Notes

- The vector store is **persisted** to `./chroma_db`, so embeddings are only computed once. Delete the folder to force a rebuild.
- The LLM is instructed to answer **only from the retrieved context**, reducing hallucinations.
- The source PDF is the [TutorialsPoint Object-Oriented Python Tutorial](https://www.tutorialspoint.com/object_oriented_python/object_oriented_python_tutorial.pdf).
- A `langchain-google-genai` version conflict may appear in pip output — this does not affect the RAG pipeline.
