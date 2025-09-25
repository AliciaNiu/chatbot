from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from sqlalchemy import create_engine
from dotenv import load_dotenv
from config import OPENAI_API_KEY, MODEL_NAME, TEMPERATURE, TOP_P, INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K

load_dotenv(dotenv_path=".env")

# --- Create SQLAlchemy engine ---
engine = create_engine("sqlite:///chat_memory.db", connect_args={"check_same_thread": False})

# Each user/session can have its own history
def get_memory(session_id: str):
    chat_history = SQLChatMessageHistory(
        session_id=session_id,
        connection=engine  # <-- use connection instead of connection_string
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=chat_history
    )
    return memory


# --- Load KB (resolve path relative to project root) ---
data_file = Path(__file__).parent.parent / "data" / "knowledge.txt"
# index_dir = Path(__file__).parent.parent / "faiss_index"
with open(data_file, encoding="utf-8") as f:
    text = f.read()
    # print(text)

# --- Split text into chunks ---
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_text(text)

# --- Build FAISS vectorstore ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# --- If index exists, load it. Otherwise, create & save. ---
if INDEX_DIR.exists():
    print("🔹 Loading existing FAISS index...")
    vectorstore = FAISS.load_local(
        str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )
else:
    print("⚡ Building new FAISS index (this will call OpenAI API)...")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local(str(INDEX_DIR))

# --- Retriever ---
# Too low k → may miss useful context.
# Too high k → increases cost (longer prompt sent to LLM), and might add noise.
# common k=3 or k=5
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})

# --- LLM ---
"""
🔹 Temperature scale
temperature=0 → most deterministic
The model always tries to return the same answer for the same input.
Great for factual Q&A, retrieval-augmented generation (RAG), or when consistency is important.
temperature=0.7 (default in many setups) → balanced randomness
Slight variation in responses, but still mostly relevant.
temperature=1 or higher → very creative
Useful for brainstorming, storytelling, or generating varied text.
Can be less factual or even drift off-topic.
"""
# For accuracy → lower temperature, higher top_p.
# For creativity → higher temperature, slightly lower top_p.
llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE, top_p=TOP_P)


# --- Memory (conversation history) ---
# memory = ConversationBufferMemory(
#     memory_key="chat_history",
#     return_messages=True
# )

# --- Conversational Retrieval Chain ---
conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=None
)
