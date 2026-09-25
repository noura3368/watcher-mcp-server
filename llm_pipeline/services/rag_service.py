import sys
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_QUERY = "Commands, syntax and parameters for {target}"


def retrieve_context(target: str, interface: str, top_k: int = 5, db_path: Optional[str] = None,
                     query: str = "") -> Tuple[str, List[dict]]:
    """Retrieve the top_k pages for the query from the ColQwen index.

    `query` may use {target} and {interface}. Returns (context text, page hits);
    the text is each page's markdown under a `[file p.N]` header.
    """
    from colrag.store import search  # imported lazily so no-RAG runs don't need torch

    query = (query or DEFAULT_QUERY).format(target=target, interface=interface)
    hits = search(query, k=top_k, db_path=Path(db_path) if db_path else None)
    text = "\n\n".join(f"[{Path(h['source_file']).name} p.{h['page_num']}]\n{h['text']}" for h in hits)
    for h in hits:
        h["query"] = query
    return text, hits
