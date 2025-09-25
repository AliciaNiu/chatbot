# rag_pipeline_weaviate.py

import weaviate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain_community.document_loaders import TextLoader

# --- Embeddings ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- Connect to Weaviate ---
weaviate_client = weaviate.Client("http://localhost:8080")

# --- Index name ---
INDEX_NAME = "DatabricksDocs"

# --- Load documents ---
docs = TextLoader("knowledge_base/databricks.txt").load()

# --- Create or load vector store ---
try:
    vectorstore = Weaviate(
        client=weaviate_client,
        index_name=INDEX_NAME,
        text_key="text",   # attribute where docs will be stored
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
except Exception:
    # If index doesn't exist, create and add documents
    vectorstore = Weaviate.from_documents(
        docs,
        embeddings,
        client=weaviate_client,
        index_name=INDEX_NAME,
        text_key="text"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("Weaviate retriever ready ✅")
