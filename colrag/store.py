"""ColQwen page index shared by the watcher (indexing) and the LLM pipeline (search).

Each PDF page is one row. The page image is embedded by ColQwen into one 128-dim
vector per image patch; those vectors are used only for search. The page text
(pymupdf4llm markdown, tables kept) is what gets returned as context.

Pages are scored with exact late interaction (MaxSim, as in ColBERT/ColPali):
for every query token take its best-matching page vector, then sum over tokens.
The corpus is a few manuals, so scoring every page exactly is cheap.
"""

import os
import re
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = Path(os.getenv("COLRAG_DB", str(REPO_ROOT / "data" / "colqwen.lancedb")))
MODEL_NAME = os.getenv("COLRAG_MODEL", "vidore/colqwen2.5-v0.2")
TABLE = "pages"
DIM = 128
RENDER_DPI = 150
BATCH_SIZE = 4


# ---------------------------------------------------------------- model

class Encoder:
    """Loads ColQwen on first use; call close() to give the GPU memory back (e.g. to Ollama)."""

    def __init__(self, model_name: str = MODEL_NAME):
        import torch
        from colpali_engine import models

        # The checkpoint name picks the colpali-engine class (ColQwen2.5 by default).
        name = model_name.lower()
        if "qwen3.5" in name or "qwen3_5" in name:
            model_cls, proc_cls = models.ColQwen3_5, models.ColQwen3_5Processor
        elif "qwen3" in name:
            model_cls, proc_cls = models.ColQwen3, models.ColQwen3Processor
        else:
            model_cls, proc_cls = models.ColQwen2_5, models.ColQwen2_5_Processor

        self.torch = torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        self.model = model_cls.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device, attn_implementation="sdpa",
        ).eval()
        self.processor = proc_cls.from_pretrained(model_name)

    def _embed(self, batch) -> List[np.ndarray]:
        batch = batch.to(self.model.device)
        with self.torch.no_grad():
            out = self.model(**batch)
        # Padding positions come back as zero vectors; drop them with the attention mask.
        mask = batch["attention_mask"].bool()
        return [out[i][mask[i]].float().cpu().numpy() for i in range(out.shape[0])]

    def embed_images(self, images) -> List[np.ndarray]:
        return self._embed(self.processor.process_images(images))

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed(self.processor.process_queries([query]))[0]

    def close(self) -> None:
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


# ---------------------------------------------------------------- pages

def _clean(md: str) -> str:
    md = re.sub(r'!\[.*?\]\(.*?\)', '', md)
    md = re.sub(r'<img[^>]*>', '', md, flags=re.IGNORECASE)
    return re.sub(r'\n{3,}', '\n\n', md).strip()


def iter_pages(pdf: Path) -> Iterator[tuple]:
    """Yield (page_num (1-based), PIL image, markdown text) for every page of the PDF."""
    import pymupdf
    import pymupdf4llm
    from PIL import Image

    texts = pymupdf4llm.to_markdown(str(pdf), page_chunks=True, write_images=False)
    with pymupdf.open(str(pdf)) as doc:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=RENDER_DPI)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            text = _clean(texts[i]["text"]) if i < len(texts) else ""
            yield i + 1, image, text


# ---------------------------------------------------------------- storage

def _schema():
    import pyarrow as pa
    return pa.schema([
        pa.field("source_file", pa.string()),
        pa.field("file_sha256", pa.string()),
        pa.field("page_num", pa.int32()),
        pa.field("text", pa.string()),
        pa.field("n_vectors", pa.int32()),
        pa.field("vectors", pa.binary()),   # float32, shape (n_vectors, DIM)
    ])


def open_table(db_path: Optional[Path] = None, create: bool = False):
    import lancedb
    db_path = Path(db_path or DEFAULT_DB)
    if not create and not db_path.exists():
        raise FileNotFoundError(f"ColQwen index not found: {db_path}")
    db = lancedb.connect(str(db_path))
    if TABLE in db.table_names():
        return db.open_table(TABLE)
    if not create:
        raise FileNotFoundError(f"ColQwen index {db_path} has no '{TABLE}' table; index some PDFs first")
    return db.create_table(TABLE, schema=_schema())


def _quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def delete_source(source_file: str, db_path: Optional[Path] = None) -> None:
    open_table(db_path, create=True).delete(f"source_file = {_quote(source_file)}")


def index_pdf(pdf: Path, sha256: str, db_path: Optional[Path] = None,
              encoder: Optional[Encoder] = None) -> int:
    """Embed every page of the PDF and replace its rows in the index. Returns pages indexed.

    Pages without extractable text are skipped: the generator models only see text,
    so such a page could be retrieved but would add nothing to the prompt.
    """
    table = open_table(db_path, create=True)
    own = encoder is None
    encoder = encoder or Encoder()
    rows = []
    try:
        batch = []
        for page_num, image, text in iter_pages(pdf):
            if not text:
                print(f"[colrag] {pdf.name} p.{page_num}: no text, skipped", flush=True)
                continue
            batch.append((page_num, image, text))
            if len(batch) == BATCH_SIZE:
                rows += _embed_rows(encoder, pdf, sha256, batch)
                batch = []
        if batch:
            rows += _embed_rows(encoder, pdf, sha256, batch)
    finally:
        if own:
            encoder.close()
    table.delete(f"source_file = {_quote(str(pdf))}")
    if rows:
        table.add(rows)
    return len(rows)


def _embed_rows(encoder: Encoder, pdf: Path, sha256: str, batch: list) -> list:
    vecs = encoder.embed_images([image for _, image, _ in batch])
    return [{
        "source_file": str(pdf),
        "file_sha256": sha256,
        "page_num": page_num,
        "text": text,
        "n_vectors": int(v.shape[0]),
        "vectors": v.astype(np.float32).tobytes(),
    } for (page_num, _, text), v in zip(batch, vecs)]


# ---------------------------------------------------------------- search

def maxsim(query: np.ndarray, page: np.ndarray) -> float:
    """Late-interaction score: sum over query vectors of the best dot product with a page vector."""
    return float((query @ page.T).max(axis=1).sum())


def search(query: str, k: int = 5, db_path: Optional[Path] = None,
           encoder: Optional[Encoder] = None) -> List[dict]:
    """Return the top-k pages as dicts with source_file, page_num, score and text."""
    rows = open_table(db_path).to_arrow().to_pylist()
    if not rows:
        return []
    own = encoder is None
    encoder = encoder or Encoder()
    try:
        q = encoder.embed_query(query)
    finally:
        if own:
            encoder.close()
    hits = []
    for r in rows:
        page = np.frombuffer(r["vectors"], dtype=np.float32).reshape(r["n_vectors"], DIM)
        hits.append({"source_file": r["source_file"], "page_num": r["page_num"],
                     "score": maxsim(q, page), "text": r["text"]})
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:k]
