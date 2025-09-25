# from fastapi import FastAPI, Request
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import openai
# import os
# from knowledge import *

# from config import OPENAI_API_KEY

# openai.api_key = OPENAI_API_KEY

# # Load KB (simple example from file)
# with open("../data/knowledge.txt") as f:
#     text = f.read()
# # Split text into chunks
# chunks = [text[i:i+500] for i in range(0, len(text), 500)]
# index, text_chunks = build_index(chunks)


# app = FastAPI()

# # CORS so frontend can call backend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class ChatRequest(BaseModel):
#     message: str
#     conversation: list
#     MAX_MESSAGES = 20

#     if len(conversation) > MAX_MESSAGES:
#         conversation = conversation[-MAX_MESSAGES:]
# #     conversation = [
# #     {"role": "system", "content": "You are a helpful chatbot."},
# #     {"role": "user", "content": "Hello"},
# #     {"role": "assistant", "content": "Hi! How can I help you today?"},
# # ]


# @app.post("/chat")
# async def chat(req: ChatRequest):
#     # Retrieve relevant KB docs
#     context_chunks = query_index(index, text_chunks, req.message, top_k=3)
#     context = "\n".join(context_chunks)

#     # Prompt with context
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant that uses the provided knowledge base."},
#         {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.message}"}
#     ]

#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=messages
#     )
#     reply = response.choices[0].message["content"].strip()
#     return {"reply": reply}


import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.responses import FileResponse
from rag_faiss_pipeline import conv_chain, get_memory
import logging
from config import MODEL_NAME, TEMPERATURE, TOP_P, TOP_K, INDEX_DIR


# --- Setup logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="Chatbot API with LangChain + FAISS")

# Enable CORS for all origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve static files (css, js) from ../frontend/static
frontend_path = Path(__file__).parent.parent / "frontend"
# app.mount("/static", StaticFiles(directory=frontend_path), name="static")


load_dotenv(dotenv_path=".env")
# --- Verify OpenAI key is set ---
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "❌ OPENAI_API_KEY environment variable is not set. "
        "Please export it before starting the server."
    )


# --- Log config at startup ---
logger.info(f"✅ Chatbot starting with configuration:")
logger.info(f"Model: {MODEL_NAME}")
logger.info(f"Temperature: {TEMPERATURE}")
logger.info(f"Top-p: {TOP_P}")
logger.info(f"Vectorstore index_dir: {INDEX_DIR}")
logger.info(f"Retriever top_k: {TOP_K}")


# --- Request schema ---
class ChatRequest(BaseModel):
    message: str

# --- Response schema ---
class ChatResponse(BaseModel):
    reply: str

# --- Health check endpoint ---
@app.get("/health")
async def health():
    return {"status": "ok"}

# Serve index.html at root
@app.get("/")
async def root():
    return FileResponse(frontend_path / "index.html")

# --- Chat endpoint ---
@app.post("/chat")
async def chat(req: dict):
    try:
        session_id = req.get("session_id", "default")
        user_message = req.get("message", "")

        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Attach persistent memory for this session
        memory = get_memory(session_id)

        print(f"Session ID: {session_id}")
        conv_chain.memory = memory

        # Run the chain
        result = conv_chain({"question": user_message})
        bot_reply = result["answer"]

        # Explicitly store interaction into memory (user + AI)
        memory.chat_memory.add_user_message(user_message)
        memory.chat_memory.add_ai_message(bot_reply)

        return {"reply": bot_reply}
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))