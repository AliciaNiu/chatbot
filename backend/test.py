import requests
# from qdrant_utils import add_document

def fetch_wikipedia_summary(topic: str) -> str:
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
    headers = {"User-Agent": "rag-app/1.0"}
    
    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        if resp.status_code == 200:
            print('get here')
            data = resp.json()
            return data.get("extract", "")
    except Exception as e:
        print(f"Error fetching summary: {e}")
        return ""

# Example: fetch and insert into Qdrant
topic = "Databricks"
summary = fetch_wikipedia_summary(topic)
if summary:
    print(len(summary))
else:
    print("No summary found.")

