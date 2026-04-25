#!/usr/bin/env python

import os
import sys
import subprocess
import shutil
import time
import datetime
import asyncio
import json
import csv
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from structured_output import generate_with_timing
from rag_service import retrieve_context
from lc_service import load_lc_content

RAG_ENABLED = os.getenv("RAG_ENABLED", "1").lower() not in ("0", "false", "no")
LC_ENABLED = os.getenv("LC_ENABLED", "0").lower() not in ("0", "false", "no")

def info(message):
    """Print info message with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [INFO] {message}")

def run_command(cmd, shell=True, capture_output=False, text=True):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=capture_output, text=text, check=False)
        print("RESULT", result)
        return result
    except Exception as e:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Error running command: {cmd}")
        print(f"[{timestamp}] Error: {e}")
        return None

def load_models_from_csv(csv_path, model_start):
    """Load model names from CSV file"""
    models = []
    index = 0
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                model_name = row['Model Name'].strip()
                index += 1
                if model_name and index >= model_start:  # Skip empty rows
                    models.append(model_name)
        info(f"Loaded {len(models)} models from {csv_path}")
        return models
    except Exception as e:
        info(f"Error loading models from CSV: {e}")
        return []

def get_available_templates(templates_dir):
    """Get list of available Jinja templates"""
    templates = []
    try:
        templates_path = Path(templates_dir)
        if templates_path.exists():
            for file in templates_path.glob("*.jinja"):
                templates.append(file.name)
        info(f"Found {len(templates)} templates: {templates}")
        return templates
    except Exception as e:
        info(f"Error getting templates: {e}")
        return []

def load_config(config_file="/data/nkhajehn/watcher-mcp-server/llm_pipeline/services/config.txt"):
    """Load configuration from text file"""
    config = {
        "models_csv": "models_leq5b_20251028_203636.csv",
        "templates_dir": "templates",
        "template": [],  # Empty list means use all templates
        "start_iteration": 5,
        "num_iterations": 55,
        "trials_per_model": 50,
        "target": "KORAD KA3005P Power Supply Unit",
        "interface": "RS232 interface",
        "folder_prefix": "dynamic"  # Prefix for output folder names
    }
    
    try:
        print(os.path.exists(config_file))
        if not os.path.exists(config_file):
            info(f"Config file {config_file} not found, using defaults")
            return config
            
        with open(config_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse key=value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    # Convert to appropriate types
                    if key == "start_iteration":
                        config[key] = int(value)
                    elif key == "num_iterations":
                        config[key] = int(value)
                    elif key == "trials_per_model":
                        config[key] = int(value)
                    elif key == "template":
                        # Handle comma-separated templates
                        if value.lower() in ['all', '']:
                            config[key] = []
                        else:
                            config[key] = [t.strip() for t in value.split(',')]
                    else:
                        config[key] = value
                else:
                    info(f"Warning: Invalid config line {line_num}: {line}")
        
        info(f"Loaded configuration from {config_file}")
        return config
        
    except Exception as e:
        info(f"Error loading config file {config_file}: {e}")
        info("Using default configuration")
        return config


def render_template_and_generate(template_path, data_path, model, ollama_host, params, output_path, failed_dir,timeout, csv_file):
    """Render template and generate structured response in-process"""
    # Build template environment
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    jinja_template = env.get_template(template_path.name)
    print(jinja_template)
    # Render the template
    try:
        rendered_prompt = jinja_template.render(**params)
    except Exception as e:
        print(f"Error rendering template: {e}", file=sys.stderr)
        return False
    
    # Load structured data
    with open(data_path, "r") as f:
        structured_text = json.load(f)
    
    # Base prompt built purely from your Jinja template
    composed_prompt = rendered_prompt.strip()

    
    print("Generated prompt:")
    print(composed_prompt)
    print("\n" + "="*50 + "\n")

    try:
        # Generate structured response with timing
        structured_response, started_at, ended_at, elapsed_ms = generate_with_timing(
            model_name=model,
            prompt=composed_prompt,
            ollama_host=ollama_host,
            timeout=timeout, 
            csv_file=csv_file
        )
        # Convert to the format expected by the original system
        output_text = json.dumps([cmd.model_dump() for cmd in structured_response.commands], indent=2)
        
        # Create response JSON
        response_json = {
            "template": template_path.name,
            "data": data_path.name,
            "params": params,
            "model": model,
            "prompt": composed_prompt,
            "time": elapsed_ms,
            "response": output_text,
            "structured": True,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat()
        }
        
        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(response_json, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Output written to: {output_path}")
        print("✅ Used structured output with outlines")
        
        return True
        
    except Exception as e:
        # --- SAVE FAILURE DETAILS ---
        timestamp = str(int(time.time() * 1000000000))
        fail_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fail_file = failed_dir / f"failed_{model}_{timestamp}.json"

        fail_data = {
            "template": template_path.name if 'template_path' in locals() else None,
            "data": data_path.name if 'data_path' in locals() else None,
            "params": params if 'params' in locals() else None,
            "model": model,
            "prompt": composed_prompt if 'composed_prompt' in locals() else None,
            "time" : elapsed_ms if 'elapsed_ms' in locals() else None,
            "error": str(e),
            "response": structured_response if 'structured_response' in locals() else None,
            "fail_time": fail_time,
            "structured": False,
            "started_at": started_at.isoformat() if 'started_at' in locals() else None,
            "ended_at": ended_at.isoformat() if 'ended_at' in locals() else None
        }

        # Write failed run info
        with open(fail_file, "w", encoding="utf-8") as f:
            json.dump(fail_data, f, ensure_ascii=False, indent=2)

        print(f"❌ Error calling Ollama generate: {e}", file=sys.stderr)
        print(f"⚠️  Saved failed output to: {fail_file}", file=sys.stderr)
        return False
def main():
    # Load configuration from file
    config = load_config()
    # Load models from CSV file
    models = load_models_from_csv(config["models_csv"], int(config['model_row']))
    if not models:
        info("No models loaded from CSV file. Exiting.")
        sys.exit(1)
    context = ""
    if RAG_ENABLED:
        try:
            context = asyncio.run(retrieve_context(config["target"], config["interface"]))
        except Exception as e:
            print(f"[RAG] Failed to retrieve context: {e}", file=sys.stderr)
    elif LC_ENABLED:
        try:
            context = load_lc_content(config["LC_file"])
        except Exception as e:
            print(f"[LC] Failed to load LC file: {e}", file=sys.stderr)
    # Get available templates
    available_templates = get_available_templates(config["templates_dir"])
    if not available_templates:
        info(f"No templates found in {config['templates_dir']}. Exiting.")
        sys.exit(1)
    
    # Determine which templates to use
    if config["template"]:
        templates_to_use = []
        for template_name in config["template"]:
            if template_name in available_templates:
                templates_to_use.append(template_name)
            else:
                info(f"Template '{template_name}' not found. Available templates: {available_templates}")
        if not templates_to_use:
            info("No valid templates specified. Exiting.")
            sys.exit(1)
    else:
        templates_to_use = available_templates
    
    info(f"Using templates: {templates_to_use}")
    
    # Create output directory
    os.makedirs("results", exist_ok=True)
    failed_dir = Path("failed_outputs")
    failed_dir.mkdir(parents=True, exist_ok=True)

    # Setup data path
    data_path = Path(f"{config['templates_dir']}/command.json").expanduser().resolve()
    if not data_path.exists():
        print(f"Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)
    
    # Main execution loop
    for j in range(config["start_iteration"], config["num_iterations"] + 1):
        info(f"Starting iteration {j}")
        
        # Loop over templates
        for template_name in templates_to_use:
            info(f"Processing template: {template_name}")
            template_path = Path(f"{config['templates_dir']}/{template_name}").expanduser().resolve()
            
            if not template_path.exists():
                info(f"Template not found: {template_path}")
                continue
            
            # Main loop over models
            for model in models:
                info(f"Processing model: {model} with template: {template_name}")
                
                # Keep FOUNDSOFAR in memory instead of file
                foundsofar_str = ""
                
                # Run trials per model
                for i in range(1, config["trials_per_model"] + 1):
                    info(f"Run {j}/{i} of model {model} with template {template_name}")
                    
                    # Generate timestamp (nanoseconds)
                    timestamp = str(int(time.time() * 1000000000))
                    
                    # Prepare parameters
                    params = {
                        "TARGET": config["target"],
                        "INTERFACE": config["interface"],
                        "FOUNDSOFAR": foundsofar_str,
                        "ITERATION": i,
                        "RAG": context
                    }
                    
                    # Run template and generate in-process
                    output_path = Path(f"./results/{template_name}_{model}_{timestamp}.json")
                    success = render_template_and_generate(
                        template_path, data_path, model, None, params, output_path, failed_dir, float(config['timeout']), config["models_csv"])
                    
                    if not success:
                        info(f"Model returned unstructured response, saved in failed_outputs")
                        continue
                    
                    # Get identified commands so far and update foundsofar_str
                    script_dir_res = Path(__file__).resolve().parent.parent.parent
                    script_dir = Path(__file__).resolve().parent
                    get_commands_cmd = (
                        f"python {script_dir}/get_commands.py {model} --out-dir {script_dir_res}/results | tr '\\n' ' '"
                    )
                    result = run_command(get_commands_cmd, capture_output=True)
                    if result and result.returncode == 0:
                        foundsofar_str = result.stdout.strip()
                    else:
                        info(f"Failed to get commands for {model}")
            
            # Move output directory for this template and run
            template_base_name = template_name.replace('.jinja', '')
            run_number = j - config["start_iteration"] + 1  # Convert iteration to run number (1, 2, 3...)
            target_dir = f"{template_base_name}-run{run_number}"
            

            if os.path.exists("results"):
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                shutil.move("./results", target_dir)
                info(f"Moved outputs to: {target_dir}")
            
            # Move failed outputs for this template and run
            failed_target_dir = f"failed_outputs-{template_base_name}-run{run_number}"
            if os.path.exists("failed_outputs"):
                if os.path.exists(failed_target_dir):
                    shutil.rmtree(failed_target_dir)
                shutil.move("failed_outputs", failed_target_dir)
                info(f"Moved failed outputs to: {failed_target_dir}")
            
            # Create new output directories for next template
            os.makedirs("results", exist_ok=True)
            os.makedirs("failed_outputs", exist_ok=True)
    
    info("Script completed successfully")

if __name__ == "__main__":
    try:
        info("Script starting with structured output")
        main()
    except KeyboardInterrupt:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Script failed with error: {e}")
        sys.exit(1)

