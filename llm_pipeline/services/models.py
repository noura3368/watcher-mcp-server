#!/usr/bin/env python3
"""
Model list loading and Ollama model download.

CLI:
    python models.py --csv models.csv --pull-all [--workers 3] [--host URL]
"""

import argparse
import csv
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

import ollama

log = logging.getLogger("models")


@dataclass
class ModelSpec:
    name: str
    row: int  # 1-based data row in the CSV


def load_models(csv_path: str, start_row: int = 0, end_row: int = 0) -> List[ModelSpec]:
    """Read the 'Model Name' column. Rows are 1-based; start_row <= 1 means from the top,
    end_row <= 0 means to the end (inclusive bounds). Duplicate names are dropped."""
    models, seen = [], set()
    with open(csv_path, "r", encoding="utf-8") as f:
        for row_num, row in enumerate(csv.DictReader(f), 1):
            name = (row.get("Model Name") or "").strip()
            if not name or row_num < start_row or (end_row > 0 and row_num > end_row):
                continue
            if name in seen:
                log.warning("Duplicate model %s at row %d skipped", name, row_num)
                continue
            seen.add(name)
            models.append(ModelSpec(name, row_num))
    return models


def is_present(client: ollama.Client, name: str) -> bool:
    try:
        client.show(name)
        return True
    except ollama.ResponseError as e:
        if e.status_code == 404:
            return False
        raise


def ensure_model(client: ollama.Client, name: str, retries: int = 3) -> bool:
    """Make sure the model exists locally, pulling it if needed. Returns False if it could not be pulled."""
    try:
        if is_present(client, name):
            return True
    except Exception as e:
        log.error("Could not query model %s: %s", name, e)
        return False

    for attempt in range(1, retries + 1):
        log.info("Pulling %s (attempt %d/%d)", name, attempt, retries)
        try:
            last_pct = -10
            for p in client.pull(name, stream=True):
                if p.total and p.completed:
                    pct = int(100 * p.completed / p.total)
                    if pct >= last_pct + 10:
                        log.info("  %s: %s %d%%", name, p.status, pct)
                        last_pct = pct
            if is_present(client, name):
                log.info("Pulled %s", name)
                return True
        except Exception as e:
            log.warning("Pull of %s failed: %s", name, e)
            time.sleep(5 * attempt)
    log.error("Giving up on %s after %d attempts", name, retries)
    return False


def remove_model(client: ollama.Client, name: str) -> None:
    try:
        client.delete(name)
        log.info("Removed %s", name)
    except Exception as e:
        log.warning("Could not remove %s: %s", name, e)


def main() -> int:
    ap = argparse.ArgumentParser(description="Download all models listed in a CSV with Ollama.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pull-all", action="store_true", help="Pull every missing model")
    ap.add_argument("--workers", type=int, default=2, help="Concurrent downloads")
    ap.add_argument("--host", default=None, help="Ollama host, e.g. http://localhost:11434")
    ap.add_argument("--start-row", type=int, default=0)
    ap.add_argument("--end-row", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    specs = load_models(args.csv, args.start_row, args.end_row)
    client = ollama.Client(host=args.host)

    missing = [s.name for s in specs if not is_present(client, s.name)]
    log.info("%d models listed, %d missing", len(specs), len(missing))
    if not args.pull_all:
        for m in missing:
            print(m)
        return 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        results = list(ex.map(lambda n: (n, ensure_model(client, n)), missing))
    failed = [n for n, ok in results if not ok]
    if failed:
        log.error("Failed to pull: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
