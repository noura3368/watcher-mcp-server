#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Set, Union


def extract_commands_from_response(response: Union[str, list], return_params: bool = False) -> Iterable:
    """Yield command strings (or {command: parameters} dicts when return_params) from a response.

    Items whose command is missing, not a string, or blank are skipped; the
    remaining items of the response are still processed.
    """
    try:
        parsed = json.loads(response) if isinstance(response, str) else response
    except Exception:
        return
    if not isinstance(parsed, list):
        return

    for item in parsed:
        if not isinstance(item, dict):
            continue
        cmd = item.get("command")
        if not isinstance(cmd, str) or not cmd.strip():
            continue
        cmd_str = cmd.strip()
        if return_params:
            params = item.get("parameters", {}) or {}
            yield {cmd_str: params}
        else:
            yield cmd_str


def format_foundsofar(commands: Iterable[str]) -> str:
    """Render the FOUNDSOFAR prompt string: sorted, each followed by ',', space-separated.

    Matches the historical `get_commands.py <model> | tr '\\n' ' '` output.
    """
    return " ".join(sorted({c + "," for c in commands if c}))


def collect_commands(paths: Iterable[Path], model: Optional[str] = None) -> Set[str]:
    """Collect unique commands from result JSON files, optionally only for an exact model name."""
    commands: Set[str] = set()
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if model is not None and data.get("model") != model:
            continue
        response = data.get("response")
        if response is None:
            continue
        commands.update(extract_commands_from_response(response, False))
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract unique commands from model JSON outputs.")
    parser.add_argument("model", help="Exact model name (matched against the 'model' field in each JSON)")
    parser.add_argument("--out-dir", dest="out_dir", default=None,
                        help="Directory containing JSON files (default: <script_dir>/results)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else (script_dir / "results")
    for cmd in sorted(c + "," for c in collect_commands(sorted(out_dir.glob("*.json")), args.model)):
        print(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
