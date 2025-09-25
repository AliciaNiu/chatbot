import os
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Use environment variable if set
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or config["openai"].get("api_key")
MODEL_NAME = config["openai"]["model"]
TEMPERATURE = config["openai"]["temperature"]
TOP_P = config["openai"]["top_p"]

INDEX_DIR = Path(__file__).parent.parent / config["vectorstore"]["index_dir"]
CHUNK_SIZE = config["vectorstore"]["chunk_size"]
CHUNK_OVERLAP = config["vectorstore"]["chunk_overlap"]
TOP_K = config["vectorstore"]["top_k"]
