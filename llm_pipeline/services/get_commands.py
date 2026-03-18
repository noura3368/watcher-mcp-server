#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Set, List, Union


def iter_input_files(model_prefix: str, out_dir: Path) -> Iterable[Path]:
    # Handle the actual filename pattern: prompt*.jinja_{model}*.json
    pattern = f"*{model_prefix}*.json" if model_prefix else "*.json"
    yield from sorted(out_dir.glob(pattern))

def extract_commands_from_response(response: Union[str, list], return_params:False) -> Iterable[str]:
    try:
        if isinstance(response, str):
            parsed = json.loads(response)
        else:
            parsed = response
    except Exception:
        return []

    if not isinstance(parsed, list):
        return []
    for item in parsed:
        if isinstance(item, dict):
            cmd = item.get("command")
            if cmd.strip() == "":
                return [] 
            if isinstance(cmd, str) and cmd.strip():
                cmd_str = cmd.strip()
            if return_params:
                params = item.get("parameters", {}) or {}
                # yield a fresh dict for each command (avoid mutating/returning the same object)
                yield {cmd_str: params}
            else:
                yield cmd_str


def collect_commands(paths: Iterable[Path]) -> Set[str]:
    commands: Set[str] = set()
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
        except Exception:
            continue

        response = data.get("response")
        if response is None:
            continue

        for cmd in extract_commands_from_response(response, False):
            if len(cmd) > 0:
                commands.add(cmd + ",")

    return commands


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract unique commands from model JSON outputs.")
    parser.add_argument("model", help="Model prefix to match files like out/{model}*.json")
    parser.add_argument("--out-dir", dest="out_dir", default=None, help="Directory containing JSON files (default: <script_dir>/out)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else (script_dir / "results")
    files = list(iter_input_files(args.model, out_dir))

    commands = collect_commands(files)
    for cmd in sorted(commands):
        print(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


