import os
from pathlib import Path

# 1. SET ENV VARS FIRST - Before any haiku imports
os.environ["HAIKU_EMBEDDING_MODEL"] = "mxbai-embed-large:latest"
os.environ["HAIKU_EMBEDDING_VECTOR_DIM"] = "1024"
# Ensure the key is exactly what the library expects (check if it's HAIKU_RAG_CONFIG)
os.environ["HAIKU_RAG_CONFIG"] = "/data/nkhajehn/watcher-mcp-server/haiku.rag.yaml"

# 2. NOW IMPORT
from haiku.rag.client import HaikuRAG

DB_PATH = Path(os.getenv("DB", "/data/nkhajehn/watcher-mcp-server/data/haiku_mxbai.rag.lancedb"))

async def retrieve_context(target: str, interface: str, top_k: int = 5):
    query = f"You are a fuzzing engine. Your target is {target}. Retrieve information..."
    chunks = []
    
    # 3. PASS THE CONFIG EXPLICITLY
    # If the class allows a config_path, pass it here to override defaults.
    async with HaikuRAG(db_path=DB_PATH, read_only=True) as client:
        results = await client.search(query) # Use the actual query, not "test"
        
        for r in results[:top_k]:
            text = getattr(r, "text", None) or getattr(r, "content", None) or str(r)
            chunks.append(text)
    return "\n\n".join(chunks)