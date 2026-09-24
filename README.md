# AI Document Intelligence Assistant

An AI-powered document intelligence assistant that uses Retrieval-Augmented Generation (RAG) to answer questions from uploaded PDF documents with grounded answers and source attribution.

## Technology Stack

- **Python**
- **Streamlit**
- **LangChain**
- **LangGraph**
- **Google Gemini**
- **Gemini Embeddings**
- **FAISS**
- **PyPDF**
- **python-dotenv**
- **unittest**
- **Git / GitHub**

## Architecture

PDF Document
→ PDF Loading
→ Text Chunking
→ Gemini Embeddings
→ FAISS Vector Store
→ Semantic Retrieval
→ Grounded Gemini Answer
→ Source Attribution
→ LangGraph Orchestration
→ Streamlit UI

## Key Features

- Upload and process PDF documents
- Extract and split document content into chunks
- Generate Gemini embeddings
- Build a FAISS vector store
- Perform semantic document retrieval
- Generate grounded answers using Google Gemini
- Display source document and page information
- Orchestrate the RAG workflow using LangGraph
- Validate application-level RAG behavior
- Streamlit-based user interface
- Application and UI boundary tests using Python `unittest`

## Project Structure

```text
ai-document-intelligence-assistant/
│
├── app.py
├── requirements.txt
├── .env.example
├── README.md
│
├── chat/
│   └── service.py
├── config/
│   └── settings.py
├── document/
│   └── service.py
├── embeddings/
│   └── gemini_embeddings.py
├── graph/
│   ├── nodes.py
│   ├── state.py
│   └── workflow.py
├── ingestion/
│   ├── pdf_loader.py
│   └── text_splitter.py
├── llm/
│   └── gemini_client.py
├── retrieval/
│   └── retriever.py
├── vectorstore/
│   └── faiss_store.py
├── evaluation/
│   └── ...
└── tests/
    └── ...