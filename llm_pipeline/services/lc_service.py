import re
from pathlib import Path


def load_lc_content(pdf_path: str) -> str:
    """Convert a PDF to markdown, strip image references, and return the text."""
    try:
        import pymupdf4llm
    except ImportError:
        raise ImportError(
            "pymupdf4llm is required for LC feature. Install with: pip install pymupdf4llm"
        )

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"LC PDF file not found: {path}")

    md_content = pymupdf4llm.to_markdown(str(path), write_images=False)

    # Remove markdown image syntax: ![alt](src)
    md_content = re.sub(r'!\[.*?\]\(.*?\)', '', md_content)
    # Remove HTML <img> tags
    md_content = re.sub(r'<img[^>]*>', '', md_content, flags=re.IGNORECASE)
    # Collapse runs of blank lines left by removed images
    md_content = re.sub(r'\n{3,}', '\n\n', md_content)

    return md_content.strip()
