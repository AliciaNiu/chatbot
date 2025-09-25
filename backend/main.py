from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.qdrant_pipeline import rag
from pathlib import Path
from dotenv import load_dotenv

# Serve static files (css, js) from ../frontend/static
frontend_path = Path(__file__).parent.parent / "frontend"



app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods
    allow_headers=["*"],  # allow all headers
)

# Define request schema
class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.get("/health")
async def health():
    return {"status": "ok"}

# @app.post("/add_doc")
# async def add_doc(doc: str):
#     add_document(doc)
#     return {"status": "document added"}
# Serve index.html at root
@app.get("/")
async def root():
    return FileResponse(frontend_path / "index.html")

@app.post("/chat")
async def chat(request: ChatRequest):
    question = request.message
    answer = rag(question)
    print("Answer from RAG:", answer)
    return {"reply": answer}
