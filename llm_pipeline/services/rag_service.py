
from pathlib import Path
import os

os.environ["HAIKU_EMBEDDING_MODEL"] = "mxbai-embed-large:latest"
os.environ["HAIKU_EMBEDDING_VECTOR_DIM"] = "1024"
os.environ["CONFIG"] = "/home/nkhajehn/watcher-mcp-server/haiku.rag.yaml"

from haiku.rag.client import HaikuRAG

DB_PATH = Path(os.getenv("DB", "/home/nkhajehn/watcher-mcp-server/data/haiku_mxbai.rag.lancedb"))

async def retrieve_context(target: str, interface: str, top_k: int = 5):
    
    query = "You are a fuzzing engine. Your target is " + target + " and you are fuzzing the " + interface + " interface. \
            Retrieve information and documentation for additional commands that the target would accept."
    chunks = []
    async with HaikuRAG(DB_PATH, read_only=True) as client:
        results = await client.search(query)
        
        for r in results[:top_k]:
            text = getattr(r, "text", None) or getattr(r, "content", None) or str(r)
            chunks.append(text)
            print("chunks", r)
    return "\n\n".join(chunks)

#async def parse_context(response): 
    