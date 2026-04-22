#!/usr/bin/env python3
"""
Filter models with fewer than 5 billion parameters from the CSV file.
"""

import csv
import sys

def filter_small_models(input_csv, output_csv, max_size=5.0):
    """
    Filter models with size_billions < max_size from the input CSV.
    
    Args:
        input_csv (str): Path to input CSV file
        output_csv (str): Path to output CSV file
        max_size (float): Maximum size in billions (default: 5.0)
    """
    small_models = []
    
    try:
        with open(input_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                size_billions = float(row['size_billions']) if row['size_billions'] else 0.0
                if size_billions < max_size:
                    small_models.append(row)
        
        # Write filtered results to output CSV
        with open(output_csv, 'w', newline='') as f:
            if small_models:
                fieldnames = small_models[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(small_models)
        
        print(f"✅ Found {len(small_models)} models with < {max_size}B parameters")
        print(f"📁 Results written to: {output_csv}")
        
        # Display summary
        print(f"\n📊 Summary of small models (< {max_size}B parameters):")
        print("-" * 60)
        for model in small_models:
            print(f"{model['model']:<30} {model['size_billions']:>8}B {model['avg_unique_commands']:>8} avg commands")
        
        return small_models
        
    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_csv}' not found")
        return []
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return []

def main():
    input_csv = "/home/nkhajehn/MCP-Command-Generation/hypothesis_testing/avg_unique_commands_per_model_no_iteration.csv"
    output_csv = "/home/nkhajehn/MCP-Command-Generation/hypothesis_testing/kruskal_tests/small_models_under_5b.csv"
    
    print("🔍 Filtering models with < 5 billion parameters...")
    small_models = filter_small_models(input_csv, output_csv, max_size=5.0)
    
    if small_models:
        print(f"\n📈 Statistics:")
        sizes = [float(model['size_billions']) for model in small_models]
        commands = [float(model['avg_unique_commands']) for model in small_models]
        
        print(f"   Size range: {min(sizes):.3f}B - {max(sizes):.3f}B")
        print(f"   Avg commands range: {min(commands):.1f} - {max(commands):.1f}")
        print(f"   Total models: {len(small_models)}")

if __name__ == "__main__":
    main()
