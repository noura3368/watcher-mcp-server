import json
import csv
import re
from pathlib import Path
from collections import defaultdict

def extract_model_size_from_name(model_name):
    """Extract model size in billions from model name."""
    # Common patterns for model sizes
    patterns = [
        r'(\d+(?:\.\d+)?)b',  # e.g., "7b", "1.5b", "13b"
        r'(\d+(?:\.\d+)?)m',  # e.g., "270m", "1m"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, model_name)
        if match:
            size = float(match.group(1))
            # Convert millions to billions
            if 'm' in pattern:
                size = size / 1000
            return size
    
    return None

def load_fallback_sizes():
    """Load model sizes from the existing CSV file as fallback."""
    fallback_sizes = {}
    csv_path = '/home/nkhajehn/MCP-Command-Generation/hypothesis_testing/multi_folder_commands_summary_all_folders.csv'
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row['model']
                size_billions = float(row['size_billions'])
                fallback_sizes[model] = size_billions
    except Exception as e:
        print(f"Warning: Could not load fallback sizes from {csv_path}: {e}")
    
    return fallback_sizes

def extract_commands_from_file(file_path):
    """Extract model and unique command count from a JSON file."""
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
        
        # Return model and count of unique commands
        unique_commands = set(commands)
        return model, len(unique_commands)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, 0

def main():
    # Define folders to process
    folders = [
        "/home/nkhajehn/MCP-Command-Generation/dynamic-004",
        "/home/nkhajehn/MCP-Command-Generation/dynamic-005", 
        "/home/nkhajehn/MCP-Command-Generation/dynamic-006",
        "/home/nkhajehn/MCP-Command-Generation/dynamic-007",
        "/home/nkhajehn/MCP-Command-Generation/results_oct1/dynamic-0104"
    ]
    
    # Load fallback sizes
    fallback_sizes = load_fallback_sizes()
    
    # Group unique command counts by model
    model_command_counts = defaultdict(list)
    model_sizes = {}
    total_files_processed = 0
    
    # Process each folder
    for folder_path in folders:
        folder = Path(folder_path)
        if not folder.exists():
            print(f"Warning: Folder {folder_path} does not exist, skipping...")
            continue
            
        print(f"Processing folder: {folder_path}")
        json_files = list(folder.glob("*.json"))
        print(f"  Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            model, unique_count = extract_commands_from_file(json_file)
            if model:  # Only process if we successfully extracted a model
                model_command_counts[model].append(unique_count)
                total_files_processed += 1
                
                # Extract model size if not already done
                if model not in model_sizes:
                    # Try to extract from model name first
                    size = extract_model_size_from_name(model)
                    if size is not None:
                        model_sizes[model] = size
                    else:
                        # Fallback to CSV data
                        model_sizes[model] = fallback_sizes.get(model)
    
    # Calculate and display statistics per model
    print(f"\n{'='*80}")
    print(f"AVERAGE UNIQUE COMMANDS PER FILE BY MODEL")
    print(f"{'='*80}")
    print(f"Total files processed: {total_files_processed}")
    print(f"Total folders processed: {len([f for f in folders if Path(f).exists()])}")
    print(f"Total models found: {len(model_command_counts)}")
    print(f"\n{'Model':<30} {'Size(B)':<8} {'Files':<8} {'Avg Unique Commands':<20}")
    print(f"{'-'*70}")
    
    # Prepare data for CSV output
    csv_data = []
    
    # Sort models by name for consistent output
    for model in sorted(model_command_counts.keys()):
        counts = model_command_counts[model]
        avg_unique_commands = sum(counts) / len(counts)
        model_size = model_sizes.get(model, 'N/A')
        size_str = f"{model_size:.1f}" if model_size is not None else "N/A"
        print(f"{model:<30} {size_str:<8} {len(counts):<8} {avg_unique_commands:<20.2f}")
        
        # Add to CSV data
        csv_data.append({
            'model': model,
            'size_billions': model_size if model_size is not None else '',
            'num_files': len(counts),
            'avg_unique_commands': round(avg_unique_commands, 2)
        })
    
    # Write to CSV file
    csv_filename = '/home/nkhajehn/MCP-Command-Generation/hypothesis_testing/avg_unique_commands_per_model.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['model', 'size_billions', 'num_files', 'avg_unique_commands']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"\nResults saved to: {csv_filename}")

if __name__ == "__main__":
    main()