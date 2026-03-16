
import html
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import ollama

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = "qwen2.5:7b"
MAX_RETRIES = 2
SLEEP_BETWEEN_CALLS = 0.2

SECTION_HEADING_RE = re.compile(r"(?m)^##\s+.+$")
client = ollama.Client(host=OLLAMA_URL)

SYSTEM_PROMPT = """You extract command definitions from device-manual markdown.

Return valid JSON only.

Interpret HTML-escaped symbols correctly:
- &lt; means <
- &gt; means >
- &amp; means &

You must return either:
1. a single JSON object, or
2. a JSON array of objects

Each object must use exactly these keys:
- entry_name
- syntax
- command_type
- description
- response
- parameters
- notes
- examples
- section_title

Rules:
- Do not invent commands or fields not supported by the text.
- Preserve command syntax exactly when possible.
- If a field is missing, use:
  - "" for strings
  - {} for parameters
  - [] for notes/examples
- command_type must be one of:
  - "query"
  - "set"
  - "test"
  - "execute"
  - ""
- If the chunk does not describe any command, return [].
- Return JSON only. No markdown fences. No explanation.
"""

USER_PROMPT_TEMPLATE = """Extract command record(s) from this markdown chunk.

Markdown:
{chunk}
"""


def read_markdown_file(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8")
    return html.unescape(text)


def split_into_sections(md: str) -> List[str]:
    matches = list(SECTION_HEADING_RE.finditer(md))
    if not matches:
        return [md.strip()] if md.strip() else []

    sections: List[str] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        chunk = md[start:end].strip()
        if chunk:
            sections.append(chunk)
    return sections


def infer_section_title_from_chunk(chunk: str) -> str:
    first_line = chunk.strip().splitlines()[0].strip() if chunk.strip() else ""
    return re.sub(r"^##\s*", "", first_line).strip()


def clean_json_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def normalize_command_type(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    v = value.strip().lower()
    allowed = {"query", "set", "test", "execute", ""}
    if v in allowed:
        return v
    if "query" in v:
        return "query"
    if "test" in v:
        return "test"
    if "set" in v:
        return "set"
    if "exec" in v or "run" in v:
        return "execute"
    return ""


def validate_record(record: Dict[str, Any], fallback_section_title: str) -> Optional[Dict[str, Any]]:
    if not isinstance(record, dict):
        return None

    out = {
        "entry_name": "",
        "syntax": "",
        "command_type": "",
        "description": "",
        "response": "",
        "parameters": {},
        "notes": [],
        "examples": [],
        "section_title": fallback_section_title or "",
        "neighbours": [],
    }

    for key in out.keys():
        if key in record:
            out[key] = record[key]

    out["entry_name"] = str(out["entry_name"]).strip()
    out["syntax"] = str(out["syntax"]).strip()
    out["description"] = str(out["description"]).strip()
    out["response"] = str(out["response"]).strip()
    out["section_title"] = str(out["section_title"]).strip() or fallback_section_title
    out["command_type"] = normalize_command_type(out["command_type"])

    if not isinstance(out["parameters"], dict):
        out["parameters"] = {}

    if not isinstance(out["notes"], list):
        out["notes"] = [str(out["notes"]).strip()] if str(out["notes"]).strip() else []

    if not isinstance(out["examples"], list):
        out["examples"] = [str(out["examples"]).strip()] if str(out["examples"]).strip() else []

    if not isinstance(out["neighbours"], list):
        out["neighbours"] = []

    out["notes"] = [str(x).strip() for x in out["notes"] if str(x).strip()]
    out["examples"] = [str(x).strip() for x in out["examples"] if str(x).strip()]

    if not out["entry_name"] and out["syntax"]:
        m = re.match(r"^([A-Za-z_*][A-Za-z0-9_+*?-]*)", out["syntax"])
        if m:
            out["entry_name"] = m.group(1)

    if not out["entry_name"] and not out["syntax"]:
        return None

    return out


def deduplicate_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []

    for r in records:
        key = (
            r.get("entry_name", "").strip(),
            r.get("syntax", "").strip(),
            r.get("section_title", "").strip(),
        )
        if key not in seen:
            seen.add(key)
            out.append(r)

    return out


def add_surrounding_neighbours(records: List[Dict[str, Any]], window: int = 5) -> List[Dict[str, Any]]:
    def neighbour_obj(rec: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "entry_name": rec.get("entry_name", "").strip(),
            "syntax": rec.get("syntax", "").strip(),
            "description": rec.get("description", "").strip(),
            "section_title": rec.get("section_title", "").strip(),
        }

    for i, rec in enumerate(records):
        neighbours = []
        start = max(0, i - window)
        end = min(len(records), i + window + 1)

        for j in range(start, end):
            if j == i:
                continue
            neighbours.append(neighbour_obj(records[j]))

        rec["neighbours"] = neighbours

    return records


def call_llm(chunk: str, model_name: str) -> str:
    response = client.generate(
        model=model_name,
        system=SYSTEM_PROMPT,
        prompt=USER_PROMPT_TEMPLATE.format(chunk=chunk),
        format="json",
        options={"temperature": 0},
    )
    return response["response"]


def parse_llm_output(raw_text: str) -> Any:
    cleaned = clean_json_text(raw_text)
    return json.loads(cleaned)


def extract_records_from_chunk(chunk: str, model_name=MODEL_NAME) -> List[Dict[str, Any]]:
    fallback_section_title = infer_section_title_from_chunk(chunk)

    for attempt in range(MAX_RETRIES + 1):
        try:
            raw = call_llm(chunk, model_name)
            parsed = parse_llm_output(raw)

            if isinstance(parsed, dict):
                parsed = [parsed]
            elif not isinstance(parsed, list):
                parsed = []

            validated: List[Dict[str, Any]] = []
            for item in parsed:
                rec = validate_record(item, fallback_section_title)
                if rec is not None:
                    validated.append(rec)

            return validated

        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(SLEEP_BETWEEN_CALLS)
            else:
                print(f"[WARN] Failed chunk: {fallback_section_title}", flush=True)
                print(f"       Error: {e}", flush=True)

    return []


def extract_records_from_markdown(
    md: str,
    model_name: str = MODEL_NAME,
    sleep_between_calls: float = SLEEP_BETWEEN_CALLS,
) -> List[Dict[str, Any]]:
    md = html.unescape(md)
    sections = split_into_sections(md)

    print(f"Found {len(sections)} sections", flush=True)

    all_records: List[Dict[str, Any]] = []

    for idx, section in enumerate(sections, start=1):
        title = infer_section_title_from_chunk(section)
        print(f"[{idx}/{len(sections)}] Processing: {title}", flush=True)

        records = extract_records_from_chunk(section, model_name)
        all_records.extend(records)

        time.sleep(sleep_between_calls)

    all_records = deduplicate_records(all_records)
    all_records = add_surrounding_neighbours(all_records, window=5)
    return all_records


def extract_records_from_markdown_file(
    path: str | Path,
    model_name: str = MODEL_NAME,
) -> List[Dict[str, Any]]:
    md = read_markdown_file(path)
    return extract_records_from_markdown(md, model_name=model_name)


def write_records_json(records: List[Dict[str, Any]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    input_md = Path("docling_test/output.md")
    output_json = Path("commands_extracted.json")

    records = extract_records_from_markdown_file(input_md, model_name=MODEL_NAME)
    write_records_json(records, output_json)

    print(f"Wrote {len(records)} records to {output_json}", flush=True)


if __name__ == "__main__":
    main()