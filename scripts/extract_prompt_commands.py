#!/usr/bin/env python3

import json
import os
import glob
from typing import List, Dict, Any


def find_root_dir() -> str:
    """Return repository root (one level above this script)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def load_outer_json(path: str) -> Dict[str, Any]:
    """Load a single JSON file, returning {} on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def parse_response_entries(outer: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse the top-level `response` field, which is expected to be
    a JSON-encoded string representing a list of entries.
    """
    raw = outer.get("response")
    if not isinstance(raw, str):
        return []

    try:
        entries = json.loads(raw)
    except Exception:
        return []

    if not isinstance(entries, list):
        return []

    return [e for e in entries if isinstance(e, dict)]


def combine_command_and_params(command: str, parameters: List[Any]) -> str:
    """
    Combine command and parameters into a single string using the rules:
    - Parameters joined with ', '.
    - If command already embeds parameters (contains '=' and does not end with '='),
      return it as-is.
    - Else if command ends with '=', append joined parameters directly.
    - Otherwise, append '=' followed by joined parameters.
    """
    command = command.strip()

    # Normalize parameters to trimmed strings and drop empties
    norm_params: List[str] = []
    for p in parameters or []:
        s = str(p).strip()
        if s:
            norm_params.append(s)

    if not norm_params:
        return command

    # Skip entries where all parameters look like placeholder variables
    # (heuristic: every parameter has no digit characters).
    if all(not any(ch.isdigit() for ch in param) for param in norm_params):
        return ""

    # Heuristic: command already includes parameters
    if "=" in command and not command.rstrip().endswith("="):
        return command

    joined = ", ".join(norm_params)
    if command.rstrip().endswith("="):
        return f"{command}{joined}"

    return f"{command}={joined}"


def main() -> None:
    root = find_root_dir()

    pattern = os.path.join(root, "prompt*", "**", "*.json")
    json_paths = sorted(glob.glob(pattern, recursive=True))

    seen = set()
    combined_commands: List[str] = []

    # Per-prompt grouping (e.g. prompt4, prompt6, prompt8, prompt9)
    prompt_names = {"prompt4", "prompt6", "prompt8", "prompt9"}
    per_prompt_seen: Dict[str, set] = {name: set() for name in prompt_names}
    per_prompt_commands: Dict[str, List[str]] = {name: [] for name in prompt_names}

    for path in json_paths:
        # Determine prompt group from top-level directory, e.g. prompt4-run1 -> prompt4
        rel = os.path.relpath(path, root)
        top_component = rel.split(os.sep, 1)[0]
        prompt_prefix = top_component.split("-", 1)[0] if "-" in top_component else top_component

        outer = load_outer_json(path)
        if not outer:
            continue

        entries = parse_response_entries(outer)
        if not entries:
            continue

        for entry in entries:
            cmd = entry.get("command")
            if not isinstance(cmd, str):
                continue

            params = entry.get("parameters", [])
            if not isinstance(params, list):
                params = []

            combined = combine_command_and_params(cmd, params).strip()
            if not combined:
                continue

            if combined in seen:
                continue

            seen.add(combined)
            combined_commands.append(combined)

            # Also track per-prompt, if this file belongs to one of the requested prompts
            if prompt_prefix in prompt_names:
                if combined in per_prompt_seen[prompt_prefix]:
                    continue
                per_prompt_seen[prompt_prefix].add(combined)
                per_prompt_commands[prompt_prefix].append(combined)

    # Global combined output
    output_path = os.path.join(root, "prompt_commands.txt")
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        for line in combined_commands:
            f.write(line + "\n")

    # Per-prompt outputs
    for prompt_name in prompt_names:
        out_path = os.path.join(root, f"{prompt_name}_commands.txt")
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            for line in per_prompt_commands[prompt_name]:
                f.write(line + "\n")


if __name__ == "__main__":
    main()

