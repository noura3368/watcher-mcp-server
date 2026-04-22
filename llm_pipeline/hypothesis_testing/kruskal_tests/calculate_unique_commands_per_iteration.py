#!/usr/bin/env python3
"""
Calculate average number of new unique commands per iteration for each model.
"""

import json
import os
import csv
from collections import defaultdict
from pathlib import Path

def load_size_parameters():
    """Load model size parameters from CSV file."""
    size_params = {}
    csv_path = "/home/nkhajehn/MCP-Command-Generation/hypothesis_testing/kruskal_tests/multi_folder_commands_summary_all_folders.csv"
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row['model']
            size_billions = float(row['size_billions'])
            size_params[model] = size_billions
    
    return size_params

def process_json_file(file_path):
    """Extract model, commands from response, and ITERATION from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        model = data.get('model', '')
        iteration = data.get('params', {}).get('ITERATION', 0)
        response_raw = data.get('response', '[]')
        
        # Parse response JSON to extract commands
        commands = []
        try:
            response_data = json.loads(response_raw)
            if isinstance(response_data, list):
                commands = [entry.get('command', '') for entry in response_data if isinstance(entry, dict)]
                # Filter out empty commands
                # perhaps not necessary but just in case
                commands = [cmd for cmd in commands if cmd]
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Warning: Could not parse response JSON in {file_path}: {e}")
            commands = []
        
        return {
            'model': model,
            'iteration': iteration,
            'commands': commands,
            'file_path': file_path
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_folder(folder_path):
    """Process all JSON files in a folder."""
    results = []
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Folder {folder_path} does not exist")
        return results
    
    for json_file in folder.glob("*.json"):
        result = process_json_file(json_file)
        if result:
            results.append(result)
    
    print(f"Processed {len(results)} files from {folder_path}")
    return results

def calculate_avg_new_commands_per_folder(folder_results):
    # if model has less than two files, we can't calculate the average
    """Calculate average number of new commands per iteration for each model in a folder."""
    # Group by model within this folder
    model_groups = defaultdict(list)
    for result in folder_results:
        model_groups[result['model']].append(result)
    
    folder_stats = []
    
    for model, model_data in model_groups.items():
        '''
        if len(model_data) < 2:
            folder_stats.append({
                'model': model,
                'avg_new_commands': 0.0,
                'total_files': len(model_data)
            })
            continue
        '''
        # Sort by iteration
        model_data.sort(key=lambda x: x['iteration'])
        
        # Track cumulative set of commands seen so far
        seen_commands = set()
        new_commands_per_iteration = []
        for data_point in model_data:
            current_commands = set(data_point['commands'])
            # Count commands that are NEW (not seen in previous iterations)
            new_commands = current_commands - seen_commands
            new_commands_per_iteration.append(len(new_commands))
            # Add new commands to the cumulative set
            seen_commands.update(new_commands)

        # Calculate average (excluding first iteration which has no "previous" to compare against)
        if len(new_commands_per_iteration) > 1:
            # First iteration is always "new" so we include it
            avg_new_commands = sum(new_commands_per_iteration) / len(new_commands_per_iteration)
        else:
            avg_new_commands = new_commands_per_iteration[0] if new_commands_per_iteration else 0.0
        
        folder_stats.append({
            'model': model,
            'avg_new_commands': avg_new_commands,
            'total_files': len(model_data)
        })
    
    return folder_stats

def main():
    """Main function to process all folders and generate output CSV."""
    
    # Define folders to process
    folders = [
        "/home/nkhajehn/MCP-Command-Generation/results_oct14/dynamic-004",
        "/home/nkhajehn/MCP-Command-Generation/results_oct14/dynamic-005", 
        "/home/nkhajehn/MCP-Command-Generation/results_oct14/dynamic-006",
        "/home/nkhajehn/MCP-Command-Generation/results_oct14/dynamic-007",
        "/home/nkhajehn/MCP-Command-Generation/results_oct14/dynamic-0104",
        '/home/nkhajehn/MCP-Command-Generation/results_oct14/dynamic-008',
        '/home/nkhajehn/MCP-Command-Generation/results_oct14/dynamic-009'
    ]
    
    # Load size parameters
    print("Loading size parameters...")
    size_params = load_size_parameters()
    print(f"Loaded size parameters for {len(size_params)} models")
    
    # Process each folder separately and collect per-folder statistics
    all_folder_stats = []
    total_files_processed = 0
    
    for folder in folders:
        print(f"\nProcessing folder: {folder}")
        folder_results = process_folder(folder)
        total_files_processed += len(folder_results)
        
        if folder_results:
            folder_stats = calculate_avg_new_commands_per_folder(folder_results)
            all_folder_stats.extend(folder_stats)
            print(f"Total models processed: {len(all_folder_stats)}")
            print(f"Found {len(folder_stats)} models in {folder}")
    
    print(f"\nTotal files processed: {total_files_processed}")
    
    # Aggregate results across all folders using simple averages
    model_aggregates = defaultdict(lambda: {'avg_sum': 0.0, 'folder_count': 0, 'total_files': 0})
    
    for stat in all_folder_stats:
        model = stat['model']
        avg_commands = stat['avg_new_commands']
        files = stat['total_files']
        
        # Simple average calculation: sum(avg) / count(folders)
        model_aggregates[model]['avg_sum'] += avg_commands
        model_aggregates[model]['folder_count'] += 1
        model_aggregates[model]['total_files'] += files
    
    # Calculate final simple averages and prepare output
    output_data = []
    for model, aggregate in model_aggregates.items():
        if aggregate['folder_count'] > 0:
            simple_avg = aggregate['avg_sum'] / aggregate['folder_count']
        else:
            simple_avg = 0.0
        
        size_billions = size_params.get(model, 0.0)  # Default to 0 if not found
        
        output_data.append({
            'model': model,
            'size_billions': size_billions,
            'avg_commands': round(simple_avg, 4),
            'num_files': aggregate['total_files']
        })
    
    # Sort by size_billions
    output_data.sort(key=lambda x: x['size_billions'])
    
    # Write output CSV
    output_path = "/home/nkhajehn/MCP-Command-Generation/hypothesis_testing/kruskal_tests/unique_commands_per_iteration.csv"
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['model', 'size_billions', 'avg_commands', 'num_files']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_data)
    
    print(f"\nResults written to: {output_path}")
    print(f"Processed {len(output_data)} models")
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Models with size parameters: {sum(1 for x in output_data if x['size_billions'] > 0)}")
    print(f"Models without size parameters: {sum(1 for x in output_data if x['size_billions'] == 0)}")
    
    # Show top 10 models by average new commands
    top_models = sorted(output_data, key=lambda x: x['avg_commands'], reverse=True)[:10]
    print("\nTop 10 models by average new commands per iteration:")
    for model in top_models:
        print(f"  {model['model']}: {model['avg_commands']:.4f} (size: {model['size_billions']}B)")

if __name__ == "__main__":
    main()
