#!/usr/bin/env python3

import os
import sys
import json
import csv
import re
from pathlib import Path
from collections import defaultdict


def extract_model_name_from_filename(filename):
    """Extract model name from filename (everything before the + character)."""
    return filename.split('+')[0]


def parse_model_size_from_name(model_name):
    """Extract parameter size from model name.
    
    Examples:
    - aya-expanse:32b -> 32.0
    - qwen2.5:0.5b -> 0.5
    - smollm:135m -> 0.135
    """
    # Look for patterns like :Xb, :X.Yb, :Xm, :X.Ym
    match = re.search(r':(\d+(?:\.\d+)?)([bm])', model_name)
    if match:
        size = float(match.group(1))
        unit = match.group(2)
        if unit == 'm':  # millions
            return size / 1000.0  # convert to billions
        else:  # billions
            return size
    return None


def load_fallback_sizes(csv_path):
    """Load model sizes from the fallback CSV file."""
    model_sizes = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row['model'].strip()
                size_billions = float(row['size_billions']) if row['size_billions'] else None
                if size_billions is not None:
                    model_sizes[model] = size_billions
        print(f"Loaded fallback sizes for {len(model_sizes)} models")
        return model_sizes
    except Exception as e:
        print(f"Error loading fallback sizes: {e}")
        return {}


def count_commands_in_response(response_text):
    """Count the number of commands in a JSON response."""
    try:
        # Parse the response as JSON
        response_data = json.loads(response_text)
        if isinstance(response_data, list):
            return len(response_data)
        else:
            return 0
    except (json.JSONDecodeError, TypeError):
        return 0


def process_multi_folder_data(folder_paths):
    """Process all JSON files from multiple dynamic folders."""
    model_commands = defaultdict(list)
    model_sizes = {}
    model_file_counts = defaultdict(int)
    
    # Load fallback sizes
    fallback_sizes = load_fallback_sizes("/home/nkhajehn/MCP-Command-Generation/model_commands_combined.csv")
    
    total_files_processed = 0
    
    # Process each folder
    for folder_path in folder_paths:
        print(f"Processing folder: {folder_path}")
        json_files = list(Path(folder_path).glob("*.json"))
        print(f"  Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract model name from filename
                model_name = extract_model_name_from_filename(json_file.name)
                
                # Count commands in response
                response_text = data.get('response', '')
                command_count = count_commands_in_response(response_text)
                
                # Store command count for this model
                model_commands[model_name].append(command_count)
                model_file_counts[model_name] += 1
                total_files_processed += 1
                
                # Extract model size if not already done
                if model_name not in model_sizes:
                    # Try to extract from model name first
                    size = parse_model_size_from_name(model_name)
                    if size is not None:
                        model_sizes[model_name] = size
                    else:
                        # Fallback to CSV data
                        model_sizes[model_name] = fallback_sizes.get(model_name)
                        
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
    
    print(f"Total files processed: {total_files_processed}")
    
    # Calculate averages and sums
    model_data = {}
    for model_name, command_counts in model_commands.items():
        if command_counts:  # Only include models with data
            avg_commands = sum(command_counts) / len(command_counts)
            sum_commands = sum(command_counts)
            model_data[model_name] = {
                'avg_commands': avg_commands,
                'sum_commands': sum_commands,
                'size_billions': model_sizes.get(model_name),
                'num_files': model_file_counts[model_name]
            }
    
    print(f"Processed data for {len(model_data)} models")
    return model_data


def create_csv_output(model_data, output_path):
    """Create CSV file with model data."""
    # Create hypothesis_testing directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['model', 'size_billions', 'avg_commands', 'sum_commands', 'num_files']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        # Sort by model size for better readability
        sorted_models = sorted(model_data.items(), key=lambda x: x[1]['size_billions'] or 0)
        
        for model_name, data in sorted_models:
            writer.writerow({
                'model': model_name,
                'size_billions': data['size_billions'],
                'avg_commands': data['avg_commands'],
                'sum_commands': data['sum_commands'],
                'num_files': data['num_files']
            })
    
    print(f"Created CSV file: {output_path}")


def main():
    # Define the four dynamic folders to process
    folder_paths = [
        #"/home/nkhajehn/MCP-Command-Generation/dynamic-004",
        #"/home/nkhajehn/MCP-Command-Generation/dynamic-005", 
        #"/home/nkhajehn/MCP-Command-Generation/dynamic-006",
        #"/home/nkhajehn/MCP-Command-Generation/dynamic-007",
        "/home/nkhajehn/MCP-Command-Generation/results_oct1/dynamic-0104"
    ]
    
    output_path = "/home/nkhajehn/MCP-Command-Generation/hypothesis_testing/multi_folder_commands_summary_dynamic-0104.csv"

    # Process multi-folder data
    model_data = process_multi_folder_data(folder_paths)
    if not model_data:
        print("Failed to process multi-folder data; aborting.")
        sys.exit(1)

    # Create CSV output
    create_csv_output(model_data, output_path)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total models: {len(model_data)}")
    
    models_with_size = [m for m in model_data.values() if m['size_billions'] is not None]
    print(f"Models with size data: {len(models_with_size)}")
    
    total_commands = sum(m['sum_commands'] for m in model_data.values())
    print(f"Total commands across all models: {total_commands:,}")
    
    avg_commands_overall = sum(m['avg_commands'] for m in model_data.values()) / len(model_data)
    print(f"Overall average commands per model: {avg_commands_overall:.2f}")


if __name__ == "__main__":
    main()




