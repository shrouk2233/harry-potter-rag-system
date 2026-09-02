"""Simple Harry Potter RAG API exercise.

Task: complete every TODO in this file, then run the API and test /query.
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


# TASK
# Complete TODO 1, TODO 2, TODO 3, and TODO 4.
# Then run the API and test retrieve, chitchat, and off-topic questions.


# ============================= Setup =============================

import os
from pathlib import Path
from dotenv import load_dotenv

# تحديد مسار .env المتواجد في فولدر backend
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)




app = FastAPI(title="Harry Potter RAG API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO 1: Add these values to your .env file and load them here.
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-large",
)
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOP_K = int(os.getenv("TOP_K", 3))

model = SentenceTransformer(EMBEDDING_MODEL) # HERE WE NEED TO LOAD THE SAME EMBEDDING MODEL THAT WE USED TO CREATE THE VECTOR DATABASE, WITH THE SAME DIMENSIONALITY.

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# IF YOU WILL USE ANOTHER LLM FROM ANOTHER PROVIDER, USE THE CORRECT CLASS FROM LANGCHAIN AND PROVIDE THE REQUIRED PARAMETERS.
gemini_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    api_key=GEMINI_API_KEY,
    temperature=0,
)

groq_llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0,
)


# =========================== Schemas ===========================

class QueryRequest(BaseModel):
    query: str


class Source(BaseModel):
    book_name: str
    page_number: int
    score: float


class QueryResponse(BaseModel):
    query: str
    route: str
    answer: str
    sources: list[Source]


# =========================== Endpoints ===========================

@app.get("/")
def root():
    return {"name": "Harry Potter RAG API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):


    # TODO 2: Write a prompt that returns only one of these words:
    # retrieve, chitchat, or off-topic.
    ROUTER_SYSTEM_PROMPT = """You classify user questions for a Harry Potter RAG system.
You MUST reply with ONLY one word from this exact list:
- retrieve: if the user asks a question about Harry Potter books, characters, places, spells, or lore.
- chitchat: if the user greets you or makes polite conversation (e.g. hi, hello, how are you).
- off-topic: if the user asks about unrelated topics (e.g. math, coding, general knowledge).

Do NOT include any additional punctuation or explanation."""



    route = groq_llm.invoke([
        SystemMessage(content=(ROUTER_SYSTEM_PROMPT)),
        HumanMessage(content=request.query),
    ]).text.strip().lower()

    if route not in {"retrieve", "chitchat", "off-topic"}:
        route = "off-topic"

    if route == "chitchat":

        # TODO 3: Write a short, friendly prompt for Harry Potter chitchat.
        CHITCHAT_SYSTEM_PROMPT = """You are a helpful and friendly Harry Potter library assistant.
Respond warmly and concisely to general greetings or small talk."""

        response = groq_llm.invoke([
            SystemMessage(content=CHITCHAT_SYSTEM_PROMPT),
            HumanMessage(content=request.query),
        ])

        return QueryResponse(
            query=request.query,
            route=route,
            answer=response.text,
            sources=[],
        )

    if route == "off-topic":
        return QueryResponse(
            query=request.query,
            route=route,
            answer="I can only answer questions about the Harry Potter books.",
            sources=[],
        )

    query_vector = model.encode(
        [f"query: {request.query}"],
        normalize_embeddings=True,
    )[0].tolist()

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=TOP_K,
        with_payload=True,
    ).points

    context = "\n\n".join(
        f"Book: {result.payload['book_name']}\n"
        f"Page: {result.payload['page_number']}\n"
        f"Content: {result.payload['content']}"
        for result in results
    )

    # TODO 4: Write a prompt that answers only from the provided context.
    # Tell the model to say "I do not know" when the context is not enough.
    RAG_SYSTEM_PROMPT = """You are an expert Harry Potter assistant.
Answer the user's question using ONLY the provided Context below.
If the answer cannot be found in the context, explicitly reply: "I do not know."
Keep the response concise, accurate, and directly to the point."""

    response = gemini_llm.invoke([
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Context:\n{context}\n\nQuestion:\n{request.query}"
        ),
    ])

    return QueryResponse(
        query=request.query,
        route=route,
        answer=response.text,
        sources=[
            Source(
                book_name=result.payload["book_name"],
                page_number=result.payload["page_number"],
                score=result.score,
            )
            for result in results
        ],
    )
