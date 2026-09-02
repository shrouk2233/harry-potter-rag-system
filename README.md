# ⚡ Harry Potter RAG System

A Production-ready Retrieval-Augmented Generation (RAG) system designed to answer questions based strictly on the Harry Potter books knowledge base. Built with dynamic query routing, vector similarity search, and a modern wizarding-themed UI.

## 🌟 Key Features

* **Intent-based Query Routing:** Automatically classifies user input into `retrieve` (book lore), `chitchat` (friendly greetings), or `off-topic` responses using Groq.
* **Vector Search Engine:** Leverages Qdrant Vector Database with normalized embeddings (`intfloat/multilingual-e5-large`) for precise document retrieval.
* **Context-Grounded Generation:** Uses Google Gemini (`gemini-1.5-flash`) to generate concise answers strictly grounded in retrieved book pages.
* **Traceable Sources:** Displays book titles, page numbers, and relevance scores for every generated answer.
* **Modern Web Interface:** Sleek HTML/CSS/JS frontend featuring API health monitoring and typing animation responses.

## 🛠️ Tech Stack

* **Backend Framework:** FastAPI, Uvicorn, Pydantic
* **LLMs & Orchestration:** LangChain, Google Gemini, Groq, HuggingFace SentenceTransformers
* **Vector Database:** Qdrant Cloud
* **Frontend:** HTML5, CSS3, JavaScript (Fetch API)

## 📁 Project Structure

```text
├── backend/
│   └── app/
│       ├── main.py        # FastAPI endpoints & RAG logic pipeline
│       └── ...
├── frontend/
│   └── index.html      # Responsive UI interface
├── .env                # API Keys & Configurations
└── README.md