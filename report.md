# Project Report

## 1. Project Description

This project implements a document-aware conversational assistant using Python. It allows users to upload documents or provide URLs, which are then processed and indexed for retrieval-augmented generation (RAG). The assistant uses a generative language model to answer user queries, incorporating relevant information from the uploaded sources. The system is designed for users who need a conversational interface capable of leveraging custom documents for context, such as researchers, students, or professionals.

The application features a Streamlit-based web interface for managing chats, uploading files, and adding URLs. It integrates a local SQLite database for persistence, a Chroma vector store for document retrieval, and Google Gemini APIs for generative responses and embeddings.

## 2. Functions Implemented

### Major Functional Components

1. **Document Ingestion and Indexing**
   - **Files**: Uploaded files are saved locally, parsed using `llama-parse`, and split into chunks for indexing in the Chroma vector store.
   - **URLs**: Web pages are fetched, cleaned using the LLM, and similarly indexed.

2. **Chat Management**
   - Users can create, delete, and manage chats. Each chat can have specific sources (files or URLs) enabled or disabled.

3. **Retrieval-Augmented Generation (RAG)**
   - Queries are processed by retrieving relevant document chunks from the vector store and augmenting the LLM prompt with this context. The assistant generates responses that cite the sources used.

4. **Streamlit Web Interface**
   - Provides a user-friendly interface for chat interactions, file uploads, URL management, and source summarization.

5. **Database Persistence**
   - SQLite is used to store chat metadata, messages, and file information. Relationships between chats and sources are managed via SQLAlchemy ORM.

### Core Logic and Functions

- **`VectorStoreHelper`** (in `vector_store.py`):
  - Handles file and URL ingestion, text extraction, chunking, and vector store operations.
  - Functions: `add_files`, `add_urls`, `similarity_search`.

- **`Agent`** (in `agent.py`):
  - Manages the LLM and vector store interactions.
  - Functions: `new_prompt`, `summarize`.

- **Streamlit Pages** (in `streamlit_app.py` and `chat.py`):
  - `chat_page`: Renders the chat interface and handles user interactions.
  - `sources_dialog`: Manages source addition and removal.

- **Database Models** (in `database.py`):
  - Models: `Chat`, `Message`, `FileItem`.
  - Functions: `new_chat`, `get_chats`, `get_sources`, `delete_chat`.

## 3. Technical Details

### Database/Persistence
- **SQLite**: Used for relational data storage, including chats, messages, and file metadata.
- **Chroma**: A local vector store for document embeddings and similarity search.
- **File Storage**: Uploaded files are saved to the `uploads` directory.

### Software/Frameworks
- **Streamlit**: Provides the web interface for the application.
- **LangChain**: Used for LLM integration, document processing, and vector store management.
- **SQLAlchemy**: ORM for database interactions.
- **Requests**: For fetching web pages.
- **Llama-Parse**: For parsing uploaded files.

### Programming Languages & Scripts
- **Python**: The primary programming language for the application.
- **Nix**: Used for managing the development environment.

### Technical Stack
- **Backend**: Python with LangChain, SQLAlchemy, and Chroma.
- **Frontend**: Streamlit for the user interface.
- **Database**: SQLite for relational data and Chroma for vector storage.

## 4. Highlighted Features

1. **Document and Webpage Ingestion**
   - Supports multiple file types and URLs.
   - Uses `llama-parse` for file parsing and the LLM for cleaning web pages.

2. **Retrieval-Augmented Generation**
   - Retrieves relevant document chunks and integrates them into the LLM prompt.
   - Responses include source citations for transparency.

3. **Streamed Responses**
   - Uses the LLM's streaming API to provide real-time responses in the chat interface.

4. **Source Management**
   - Allows users to enable or disable specific sources for each chat.
   - Provides a dialog for managing files and URLs.

5. **Local Persistence**
   - Combines SQLite and Chroma for efficient storage and retrieval of data.

## 5. Discussion and Reflection

### Technique Effectiveness
- **Python**: Well-suited for rapid development and integration of machine learning libraries.
- **Streamlit**: Provides an intuitive interface but may not scale well for production use.
- **Chroma**: Effective for local vector storage but may require external solutions for larger-scale deployments.

### Comparative Analysis
- **Vector Store**: Chroma is simple to set up locally but lacks the scalability of cloud-based solutions like Pinecone or Weaviate.
- **File Storage**: Storing raw file bytes in the database is convenient but not scalable. Using object storage (e.g., S3) would be more efficient.
- **Webpage Cleaning**: Using the LLM for cleaning is flexible but introduces additional API costs. Deterministic methods could be more cost-effective.

### Tradeoffs
- **Local vs. Cloud**: The project prioritizes local persistence for simplicity, which limits scalability.
- **Generative Model**: Google Gemini provides high-quality responses but introduces vendor lock-in and API costs.

## Conclusion

This project demonstrates a well-rounded implementation of a document-aware conversational assistant. It effectively combines modern LLM capabilities with local persistence and a user-friendly interface. While suitable for small-scale use cases, scaling the application would require addressing limitations in storage, concurrency, and API costs.