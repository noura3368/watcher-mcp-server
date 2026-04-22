#!/usr/bin/env python3
"""
Compute the total number of unique commands each LLM generates across all runs,
then average that across all experimental folders.
"""

import json
import csv
import re
from pathlib import Path
from collections import defaultdict

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def extract_model_size_from_name(model_name):
    """Extract model size (in billions) from the model name string."""
    patterns = [
        r'(\d+(?:\.\d+)?)b',  # e.g., "7b", "1.5b"
        r'(\d+(?:\.\d+)?)m',  # e.g., "270m", "1m"
    ]
    for pattern in patterns:
        match = re.search(pattern, model_name)
        if match:
            size = float(match.group(1))
            if 'm' in pattern:
                size = size / 1000.0  # convert millions to billions
            return size
    return None


def load_fallback_sizes():
    """Load model sizes from an existing summary CSV file."""
    fallback_sizes = {}
    csv_path = "/home/nkhajehn/MCP-Command-Generation/hypothesis_testing/kruskal_tests/multi_folder_commands_summary_all_folders.csv"
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row["model"]
                size = float(row["size_billions"]) if row["size_billions"] else None
                if size is not None:
                    fallback_sizes[model] = size
        print(f"Loaded fallback sizes for {len(fallback_sizes)} models.")
    except Exception as e:
        print(f"Warning: Could not load fallback sizes: {e}")
    return fallback_sizes


def extract_commands_from_json(json_file):
    """Extract commands from one JSON run file."""
    try:
        data = json.load(open(json_file, "r"))
        model = data.get("model", "")
        response_raw = data.get("response", "[]")
        try:
            response_data = json.loads(response_raw)
            if isinstance(response_data, list):
                cmds = {
                    entry.get("command", "").strip()
                    for entry in response_data
                    if isinstance(entry, dict) and entry.get("command")
                }
                return model, cmds
        except (json.JSONDecodeError, TypeError):
            pass
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
    return None, set()


# ----------------------------------------------------------------------
# Main computation
# ----------------------------------------------------------------------

def main():
    folders = [
        
        #"/home/nkhajehn/MCP-Command-Generation/results_oct14/dynamic-004",
        #"/home/nkhajehn/MCP-Command-Generation/results_oct14/dynamic-005",
        #"/home/nkhajehn/MCP-Command-Generation/results_oct14/dynamic-006",
        #"/home/nkhajehn/MCP-Command-Generation/results_oct14/dynamic-007",
        "/home/nkhajehn/MCP-Command-Generation/results_oct14/dynamic-0104"
        #"/home/nkhajehn/MCP-Command-Generation/results_oct14/dynamic-008",
        #"/home/nkhajehn/MCP-Command-Generation/results_oct14/dynamic-009"
    ]

    fallback_sizes = load_fallback_sizes()

    # Per-folder unique command sets
    folder_unique_counts = defaultdict(lambda: defaultdict(int))
    model_sizes = {}

    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"Warning: {folder} not found, skipping.")
            continue

        print(f"\nProcessing folder: {folder}")
        model_to_commands = defaultdict(set)
        json_files = list(folder_path.glob("*.json"))
        print(f"  Found {len(json_files)} JSON files")

        for json_file in json_files:
            model, cmds = extract_commands_from_json(json_file)
            if not model:
                continue
            model_to_commands[model].update(cmds)
            # Record size
            if model not in model_sizes:
                size = extract_model_size_from_name(model) or fallback_sizes.get(model)
                model_sizes[model] = size

        # Store total unique commands per model for this folder
        for model, cmd_set in model_to_commands.items():
            folder_unique_counts[model][folder] = len(cmd_set)

    # ------------------------------------------------------------------
    # Aggregate across folders
    # ------------------------------------------------------------------
    output_data = []
    for model, folder_counts in folder_unique_counts.items():
        counts = list(folder_counts.values())
        avg_unique = sum(counts) / len(counts) if counts else 0
        total_files = len(counts)
        output_data.append({
            "model": model,
            "size_billions": model_sizes.get(model),
            "avg_commands": round(avg_unique, 2),
            "num_files": total_files
        })

    # Sort by model size for clarity
    output_data.sort(key=lambda x: (x["size_billions"] or 0))

    # ------------------------------------------------------------------
    # Write output CSV
    # ------------------------------------------------------------------
    output_path = "/home/nkhajehn/MCP-Command-Generation/hypothesis_testing/kruskal_tests/avg_total_unique_commands_per_model.csv"
    with open(output_path, "w", newline="") as f:
        fieldnames = ["model", "size_billions", "avg_commands", "num_files"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_data)

    print(f"\n✅ Results written to: {output_path}")
    print(f"Processed {len(output_data)} models.")
    print("\nExample rows:")
    for row in output_data[:10]:
        print(row)


if __name__ == "__main__":
    main()