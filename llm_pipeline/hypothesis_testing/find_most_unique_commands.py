#!/usr/bin/env python3
"""
Find which model has the most unique commands across all dynamic folders in results_oct14/.
"""

import json
import os
import csv
from collections import defaultdict
from pathlib import Path

def process_json_file(file_path):
    """Extract model and commands from response field."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        model = data.get('model', '')
        response_raw = data.get('response', '[]')
        
        # Parse response JSON to extract commands
        commands = []
        try:
            response_data = json.loads(response_raw)
            if isinstance(response_data, list):
                commands = [entry.get('command', '') for entry in response_data if isinstance(entry, dict)]
                # Filter out empty commands
                commands = [cmd for cmd in commands if cmd]
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Warning: Could not parse response JSON in {file_path}: {e}")
            commands = []
        
        return {
            'model': model,
            'commands': commands,
            'file_path': file_path
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_model_sizes():
    """Load model parameter sizes from the existing CSV file."""
    model_sizes = {}
    csv_path = "/home/nkhajehn/MCP-Command-Generation/hypothesis_testing/kruskal_tests/multi_folder_commands_summary_all_folders.csv"
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row['model']
                size_billions = float(row['size_billions'])
                model_sizes[model] = size_billions
    except FileNotFoundError:
        print(f"Warning: Could not find {csv_path}, parameter sizes will be missing")
    except Exception as e:
        print(f"Warning: Error reading model sizes: {e}")
    
    return model_sizes

def analyze_all_dynamic_folders():
    """Analyze all dynamic folders in results_oct14/ to find model with highest average unique commands per folder."""
    base_path = "/home/nkhajehn/MCP-Command-Generation/results_oct14"
    
    # Load model parameter sizes
    model_sizes = load_model_sizes()
    dynamic_folders = [
        #"dynamic-004",
        #"dynamic-005", 
        #"dynamic-006",
        #"dynamic-007",
        #"dynamic-008",
        "dynamic-0104"
    ]
    
    # Track unique commands per model per folder
    model_folder_commands = defaultdict(lambda: defaultdict(set))
    model_folder_file_counts = defaultdict(lambda: defaultdict(int))
    total_files_processed = 0
    
    for folder_name in dynamic_folders:
        folder_path = os.path.join(base_path, folder_name)
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"Warning: Folder {folder_path} does not exist, skipping...")
            continue
        
        # Process all JSON files in this folder
        folder_results = []
        for json_file in folder.glob("*.json"):
            result = process_json_file(json_file)
            if result:
                folder_results.append(result)
        
        print(f"Processed {len(folder_results)} files from {folder_path}")
        total_files_processed += len(folder_results)
        
        # Group by model within this folder
        for result in folder_results:
            model = result['model']
            commands = result['commands']
            model_folder_file_counts[model][folder_name] += 1
            
            # Add commands to the model's unique set for this folder
            for cmd in commands:
                model_folder_commands[model][folder_name].add(cmd)
    
    print(f"\nTotal files processed across all folders: {total_files_processed}")
    
    # Calculate statistics for each model
    model_stats = []
    for model, folder_commands in model_folder_commands.items():
        # Calculate average unique commands per folder for this model
        unique_commands_per_folder = [len(commands) for commands in folder_commands.values()]
        avg_unique_commands_per_folder = sum(unique_commands_per_folder) / len(unique_commands_per_folder) if unique_commands_per_folder else 0
        
        # Calculate total files across all folders for this model
        total_files = sum(model_folder_file_counts[model].values())
        
        # Get all unique commands across all folders (for detailed view)
        all_unique_commands = set()
        for commands in folder_commands.values():
            all_unique_commands.update(commands)
        
        # Create per-folder breakdown
        folder_breakdown = {}
        for folder_name, commands in folder_commands.items():
            folder_breakdown[folder_name] = {
                'unique_commands_count': len(commands),
                'files': model_folder_file_counts[model][folder_name],
                'commands': sorted(list(commands))
            }
        
        model_stats.append({
            'model': model,
            'avg_commands': avg_unique_commands_per_folder,
            'total_folders': len(folder_commands),
            'num_files': total_files,
            'folder_breakdown': folder_breakdown,
            'all_unique_commands': sorted(list(all_unique_commands)),
            'size_billions': model_sizes.get(model, None)
        })
    
    # Sort by average unique commands per folder (descending)
    model_stats.sort(key=lambda x: x['avg_commands'], reverse=True)
    
    # Calculate overall statistics
    total_models = len(model_stats)
    overall_avg_unique_commands_per_folder = sum(stat['avg_commands'] for stat in model_stats) / total_models if total_models > 0 else 0
    
    # Print results
    print(f"\n{'='*80}")
    print(f"ANALYSIS RESULTS FOR ALL DYNAMIC FOLDERS IN RESULTS_OCT14/")
    print(f"ANALYSIS: Average Unique Commands Per Folder (uniqueness within each folder)")
    print(f"{'='*80}")
    print(f"Total models found: {total_models}")
    print(f"Total files processed: {total_files_processed}")
    print(f"Overall average unique commands per folder across all models: {overall_avg_unique_commands_per_folder:.2f}")
    
    print(f"\n{'='*80}")
    print(f"TOP 10 MODELS BY AVERAGE UNIQUE COMMANDS PER FOLDER")
    print(f"{'='*80}")
    print(f"{'Rank':<4} {'Model':<30} {'Avg/Folder':<12} {'Folders':<8} {'Files':<8} {'Avg/File':<8}")
    print(f"{'-'*4} {'-'*30} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
    
    for i, stat in enumerate(model_stats[:10], 1):
        avg_per_file = stat['avg_commands'] * stat['total_folders'] / stat['num_files'] if stat['num_files'] > 0 else 0
        print(f"{i:<4} {stat['model']:<30} {stat['avg_commands']:<12.2f} {stat['total_folders']:<8} {stat['num_files']:<8} {avg_per_file:<8.2f}")
    
    # Show the winner in detail
    if model_stats:
        winner = model_stats[0]
        print(f"\n{'='*80}")
        print(f"WINNER: {winner['model']}")
        print(f"{'='*80}")
        print(f"Average Unique Commands per Folder: {winner['avg_commands']:.2f}")
        print(f"Total Folders: {winner['total_folders']}")
        print(f"Total Files: {winner['num_files']}")
        print(f"Average Commands per File: {winner['avg_commands'] * winner['total_folders'] / winner['num_files']:.2f}")
        
        print(f"\nPer-Folder Breakdown:")
        print(f"{'-'*50}")
        for folder_name, breakdown in winner['folder_breakdown'].items():
            print(f"{folder_name}: {breakdown['unique_commands_count']} unique commands ({breakdown['files']} files)")
        
        print(f"\nAll Unique Commands Found (across all folders):")
        print(f"{'-'*50}")
        for i, cmd in enumerate(winner['all_unique_commands'], 1):
            print(f"{i:3d}. {cmd}")
    
    # Save detailed results to file
    output_file = "/home/nkhajehn/MCP-Command-Generation/hypothesis_testing/results_oct14_avg_unique_commands_per_folder_analysis.txt"
    with open(output_file, 'w') as f:
        f.write("RESULTS_OCT14 ALL DYNAMIC FOLDERS AVERAGE UNIQUE COMMANDS PER FOLDER ANALYSIS\n")
        f.write("="*80 + "\n")
        f.write("Note: Uniqueness is determined within each folder independently\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total models: {total_models}\n")
        f.write(f"Total files: {total_files_processed}\n")
        f.write(f"Overall average unique commands per folder: {overall_avg_unique_commands_per_folder:.2f}\n\n")
        
        f.write("ALL MODELS RANKED BY AVERAGE UNIQUE COMMANDS PER FOLDER:\n")
        f.write("-" * 70 + "\n")
        for i, stat in enumerate(model_stats, 1):
            f.write(f"{i:3d}. {stat['model']:<30} {stat['avg_commands']:6.2f} avg/folder ({stat['total_folders']} folders, {stat['num_files']} files)\n")
        
        f.write(f"\nDETAILED PER-FOLDER BREAKDOWN FOR TOP 5 MODELS:\n")
        f.write("=" * 60 + "\n")
        for i, stat in enumerate(model_stats[:5], 1):
            f.write(f"\n{i}. {stat['model']} (avg: {stat['avg_commands']:.2f} unique commands/folder):\n")
            f.write("-" * 50 + "\n")
            for folder_name, breakdown in stat['folder_breakdown'].items():
                f.write(f"  {folder_name}: {breakdown['unique_commands_count']} unique commands ({breakdown['files']} files)\n")
            f.write(f"\n  All unique commands across folders:\n")
            for j, cmd in enumerate(stat['all_unique_commands'], 1):
                f.write(f"    {j:3d}. {cmd}\n")
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Save CSV results
    csv_output_file = "/home/nkhajehn/MCP-Command-Generation/hypothesis_testing/results_oct15_avg_unique_commands_per_folder.csv"
    with open(csv_output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'size_billions', 'avg_commands', 'sum_commands', 'num_files'])
        
        for stat in model_stats:
            # Calculate sum_commands (total unique commands across all folders)
            sum_commands = len(stat['all_unique_commands'])
            writer.writerow([
                stat['model'],
                stat['size_billions'] if stat['size_billions'] is not None else '',
                round(stat['avg_commands'], 4),
                sum_commands,
                stat['num_files']
            ])
    
    print(f"CSV results saved to: {csv_output_file}")

if __name__ == "__main__":
    analyze_all_dynamic_folders()
