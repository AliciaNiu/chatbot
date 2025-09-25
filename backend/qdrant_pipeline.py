from qdrant_client import QdrantClient
# from qdrant_client.models import VectorParams, Distance, PointStruct
import openai
# import uuid
import json
from qdrant_client import QdrantClient, models
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")
# --- Verify OpenAI key is set ---
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "❌ OPENAI_API_KEY environment variable is not set. "
        "Please export it before starting the server."
    )



def read_doc():
    from pathlib import Path
    # --- Load KB (resolve path relative to project root) ---
    data_file = Path(__file__).parent.parent / "data" / "documents.json"
    with open(data_file, 'rt') as f_in:
        docs_raw = json.load(f_in)

    documents = []

    for course_dict in docs_raw:
        for doc in course_dict['documents']:
            doc['course'] = course_dict['course']
            documents.append(doc)

    return documents


def build_client(collection_name, documents, client, model_handle):
    from fastembed import TextEmbedding
    TextEmbedding.list_supported_models()

    EMBEDDING_DIMENSIONALITY = 512

    # for model in TextEmbedding.list_supported_models():
    #     if model["dim"] == EMBEDDING_DIMENSIONALITY:
    #         print(json.dumps(model, indent=2))


    # client.delete_collection(collection_name=collection_name) # delete the collection if it exists
    if client.get_collection(collection_name=collection_name):
        print(f"Collection '{collection_name}' already exists.")
        # Do not create again, just return client
        return client
    # Create the collection with specified vector parameters
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIMENSIONALITY,  # Dimensionality of the vectors
            distance=models.Distance.COSINE  # Distance metric for similarity search
        )
    )

    points = []
    id = 0

    for doc in documents:
        point = models.PointStruct(
            id=id,
            vector=models.Document(text=doc['text'], model=model_handle), #embed text locally with "jinaai/jina-embeddings-v2-small-en" from FastEmbed
            payload={
                "text": doc['text'],
                "section": doc['section'],
                "question": doc['question'],
                "course": doc['course']
            } #save all needed metadata fields
        )
        points.append(point)
        id += 1

    client.upsert(
        collection_name=collection_name,
        points=points
    )
    return client


def search(collection_name, client, model_handle, query, limit=1):
    results = client.query_points(
        collection_name=collection_name,
        query=models.Document( #embed the query text locally with "jinaai/jina-embeddings-v2-small-en"
            text=query,
            model=model_handle 
        ),
        limit=limit, # top closest matches
        with_payload=True #to get metadata in the results
    )
    return results


def search_in_course(collection_name, client, model_handle, query, course="mlops-zoomcamp", limit=5):
    client.create_payload_index(
        collection_name=collection_name,
        field_name="course",
        field_schema="keyword" # exact matching on string metadata fields
    )

    points = client.query_points(
        collection_name=collection_name,
        query=models.Document( #embed the query text locally with "jinaai/jina-embeddings-v2-small-en"
            text=query,
            model=model_handle
        ),
        query_filter=models.Filter( # filter by course name
            must=[
                models.FieldCondition(
                    key="course",
                    match=models.MatchValue(value=course)
                )
            ]
        ),
        limit=limit, # top closest matches
        with_payload=True #to get metadata in the results
    )
    results = []
    for point in points.points:
        results.append(point.payload)
    return results


def build_prompt(query, search_results):
    prompt_template = """
        You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
        Use only the facts from the CONTEXT when answering the QUESTION.

        QUESTION: {question}

        CONTEXT: 
        {context}
        """.strip()

    context = ""
    
    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


def build_prompt(query, search_results):
    prompt_template = """
        You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
        Use only the facts from the CONTEXT when answering the QUESTION.

        QUESTION: {question}

        CONTEXT: 
        {context}
        """.strip()

    context = ""
    
    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt



def llm(prompt):
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

def rag(query, course="mlops-zoomcamp", limit=5):
    documents= read_doc()
    
    client = QdrantClient("http://localhost:6333")
    client.get_collections()

    # Define the collection name
    collection_name = "qdrant-rag"
    model_handle = "jinaai/jina-embeddings-v2-small-en"
    client_b = build_client(collection_name, documents, client, model_handle)
    search_results = search_in_course(collection_name, client_b, model_handle, query, course, limit=limit)

    if not search_results:
        return "No relevant information found."
    
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    print(answer)
    
    return answer

# rag("What if I submit homeworks late?", course="mlops-zoomcamp", limit=5)
# rag("What is the refund policy?", course="mlops-zoomcamp", limit=5)