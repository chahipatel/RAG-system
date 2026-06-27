🤖 RAG System (PDF + URL + AI Assistant)

📘 Project Overview

Multi source RAG system is a Retrieval Augmented Generation (RAG) application that enables intelligent question answering over custom knowledge sources including PDF documents and web URLs.

It combines LangChain, ChromaDB, HuggingFace Embeddings, and Mistral AI to build an end to end AI pipeline that converts unstructured data into searchable vector knowledge and generates context aware responses using an LLM.

✨ Key Features

📚 Multi Source Ingestion
PDF document processing
Web URL content extraction
Unified preprocessing pipeline

🔍 Semantic Search Engine
ChromaDB vector database
HuggingFace embeddings (all-MiniLM-L6-v2)
MMR based retrieval for better context diversity

🤖 AI-Powered Answer Generation
Mistral LLM integration
Retrieval grounded responses
Reduced hallucination through context enforcement

💬 Conversation Memory
Chat history support
Follow up question handling
Context-aware responses

🧰 Tech Stack
Framework: LangChain
LLM: Mistral AI
Vector DB: ChromaDB
Embeddings: HuggingFace Sentence Transformers
Data Processing: PyPDF, BeautifulSoup, Requests
Environment: python-dotenv

⚙️ Installation
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt

🔑 Environment Setup
Create a .env file:

MISTRAL_API_KEY=your_api_key_here

▶️ Run Application
python app.py

🧪 Workflow
Load PDF or URL
Extract & clean text
Split into chunks
Generate embeddings
Store in ChromaDB
User asks question
Retrieve relevant context
Mistral generates final answer

🧠 Architecture
User Query
   ↓
Embedding Model (HuggingFace)
   ↓
ChromaDB Vector Search
   ↓
Context Retrieval
   ↓
Mistral LLM
   ↓
Final Answer

📌 Project Goal
This project demonstrates a complete modern RAG pipeline, combining semantic search and LLM reasoning to build an AI system capable of answering questions from external documents and web sources.






