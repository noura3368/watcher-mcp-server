#!/usr/bin/env python3
"""
Extract model statistics from NO_RAG_KORAD_results.csv.

For each prompt, creates a CSV file with:
- model: Model name
- size_billions: Parameter count in billions (from ollama show)
- number of base commands: Average number of base commands across all runs
"""

import os
import re
import ast
import csv
import subprocess
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from collections import defaultdict


ROOT_DIR = "/home/nkhajehn/MCP-Command-Generation/"
INPUT_CSV = os.path.join(ROOT_DIR, "post_processing", "plots", "RAG_KORAD_results.csv")
OUTPUT_DIR = os.path.join(ROOT_DIR, "hypothesis_testing", "csv_files")


def parse_set_string(set_str: str) -> int:
    """Parse Python set string representation and return count of commands.
    
    Examples:
        "set()" -> 0
        "{'VSET'}" -> 1
        "{'ISET', 'VSET'}" -> 2
    """
    if not set_str or set_str.strip() == "set()":
        return 0
    
    # Remove quotes if present and strip whitespace
    set_str = set_str.strip().strip('"').strip("'")
    
    # Handle empty set
    if set_str == "set()":
        return 0
    
    # Try to parse as a set literal
    try:
        # Use ast.literal_eval to safely parse the set
        parsed_set = ast.literal_eval(set_str)
        if isinstance(parsed_set, set):
            return len(parsed_set)
        return 0
    except (ValueError, SyntaxError):
        # Fallback: count items between curly braces
        # Remove outer braces and split by comma
        content = set_str.strip('{}')
        if not content:
            return 0
        # Count items (handling quoted strings)
        items = [item.strip().strip("'\"") for item in content.split(',')]
        items = [item for item in items if item]  # Remove empty strings
        return len(items)


def get_model_parameters(model: str, cache: Dict[str, Optional[float]]) -> Optional[float]:
    """Run ollama show and extract parameter count in billions.
    
    Args:
        model: Model name (e.g., "aya-expanse:8b")
        cache: Dictionary to cache results
        
    Returns:
        Parameter count in billions, or None if model not found
    """
    if model in cache:
        return cache[model]
    
    try:
        # Run ollama show -v
        result = subprocess.run(
            ["ollama", "show", model, "-v"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"Warning: Failed to get parameters for {model}: {result.stderr}")
            cache[model] = None
            return None
        
        # Parse output to find "parameters" field
        # Look for line like: "    parameters          8.0B"
        output = result.stdout
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith('parameters'):
                # Extract the value (e.g., "8.0B")
                parts = line.split()
                if len(parts) >= 2:
                    param_str = parts[1]
                    # Remove 'B' suffix and convert to float
                    param_str = param_str.rstrip('B').strip()
                    try:
                        param_billions = float(param_str)
                        cache[model] = param_billions
                        return param_billions
                    except ValueError:
                        pass
        
        # Alternative: look for "general.parameter_count" in metadata
        # Format: "general.parameter_count                   8.028033024e+09"
        for line in output.split('\n'):
            if 'general.parameter_count' in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        param_count = float(parts[-1])
                        # Convert to billions
                        param_billions = param_count / 1e9
                        cache[model] = param_billions
                        return param_billions
                    except ValueError:
                        pass
        
        print(f"Warning: Could not parse parameters for {model}")
        cache[model] = None
        return None
        
    except subprocess.TimeoutExpired:
        print(f"Warning: Timeout getting parameters for {model}")
        cache[model] = None
        return None
    except Exception as e:
        print(f"Warning: Error getting parameters for {model}: {e}")
        cache[model] = None
        return None


def calculate_average_commands(rows: List[Dict[str, str]], model: str, prompt: str) -> float:
    """Calculate average number of base commands across all runs for a (model, prompt) combination.
    
    Args:
        rows: List of dictionaries representing CSV rows
        model: Model name
        prompt: Prompt name
        
    Returns:
        Average number of base commands
    """
    # Filter for this model and prompt
    model_prompt_rows = [
        row for row in rows
        if row.get('model') == model and row.get('prompt') == prompt
    ]
    
    if not model_prompt_rows:
        return 0.0
    
    # Group by run_number
    runs_dict: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in model_prompt_rows:
        try:
            run_num = int(row.get('run_number', 0))
            runs_dict[run_num].append(row)
        except (ValueError, TypeError):
            continue
    
    command_counts = []
    
    for run_num, run_rows in runs_dict.items():
        # Find the row with maximum iteration
        max_iteration = -1
        last_iteration_row = None
        
        for row in run_rows:
            try:
                iteration = int(row.get('iteration', -1))
                if iteration > max_iteration:
                    max_iteration = iteration
                    last_iteration_row = row
            except (ValueError, TypeError):
                continue
        
        if last_iteration_row:
            base_commands_str = last_iteration_row.get('base_commands_seen_so_far', 'set()')
            count = parse_set_string(base_commands_str)
            command_counts.append(count)
    
    if not command_counts:
        return 0.0
    
    return sum(command_counts) / len(command_counts)


def main():
    """Main function to process CSV and generate output files."""
    # Check if input CSV exists
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input CSV file not found at {INPUT_CSV}")
        return
    
    print(f"Reading CSV file: {INPUT_CSV}")
    
    # Read CSV file
    rows = []
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Loaded {len(rows)} rows")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Get unique prompts
    prompts = sorted(set(row.get('prompt', '') for row in rows if row.get('prompt')))
    print(f"Found prompts: {prompts}")
    
    # Cache for model parameters
    param_cache: Dict[str, Optional[float]] = {}
    
    # Process each prompt
    for prompt in prompts:
        print(f"\nProcessing prompt: {prompt}")
        
        # Filter data for this prompt
        prompt_rows = [row for row in rows if row.get('prompt') == prompt]
        
        # Get unique models for this prompt
        models = sorted(set(row.get('model', '') for row in prompt_rows if row.get('model')))
        print(f"  Found {len(models)} models")
        
        # Collect results for this prompt
        results = []
        
        for model in models:
            # Calculate average base commands
            avg_commands = calculate_average_commands(rows, model, prompt)
            
            # Get model parameters
            size_billions = get_model_parameters(model, param_cache)
            
            results.append({
                'model': model,
                'size_billions': size_billions if size_billions is not None else '',
                'number of base commands': avg_commands
            })
            
            size_str = f"{size_billions}B" if size_billions is not None else "N/A"
            print(f"  {model}: avg_commands={avg_commands:.2f}, size={size_str}")
        
        # Sort by model name
        results.sort(key=lambda x: x['model'])
        
        # Generate output filename
        output_filename = f"{prompt}_model_stats.csv"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Write CSV file
        fieldnames = ['model', 'size_billions', 'number of base commands']
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"  Wrote {len(results)} rows to {output_path}")
    
    print(f"\nDone! Output files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
