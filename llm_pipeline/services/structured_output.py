#!/usr/bin/env python3
"""
Structured output classes for LLM responses using outlines package.
"""

from typing import List, Optional, Dict, Tuple, Any
from pydantic import BaseModel, Field
import outlines
from outlines import models
from outlines.generator import Generator
from datetime import datetime, timezone
from time import perf_counter
import requests
import json
import csv
from pathlib import Path
import ollama


class SecurityTestCommand(BaseModel):
    """
    A single security test command with justification and parameters.
    """
    justification: str = Field(
        description="A one sentence summary explaining why this command is useful for security testing"
    )
    command: str = Field(
        description="The command string to be sent to the target device"
    )
    parameters: List[str] = Field(
        default_factory=list,
        description="List of concrete parameter values to test with the command (provide actual numbers/values, not descriptions)"
    )


class SecurityTestResponse(BaseModel):
    """
    A list of security test commands for testing a target device.
    """
    commands: List[SecurityTestCommand] = Field(
        description="List of security test commands to execute against the target"
    )


# Module-level cache for generators to avoid re-initialization
_generator_cache: Dict[Tuple[str, str], Generator] = {}

_num_predict_cache: Optional[Dict[str, int]] = None
_DEFAULT_NUM_PREDICT = 4096

def load_num_predict_mapping(csv_path: Optional[str] = None) -> Dict[str, int]:
    """
    Load num_predict values from CSV file and create a model name to num_predict mapping.
    
    Args:
        csv_path: Optional path to CSV file. If None, uses models_combined_with_num_predict.csv
                  in the same directory as this script.
    
    Returns:
        Dictionary mapping model names to num_predict values
    """
    global _num_predict_cache
    
    # Return cached mapping if available
    if _num_predict_cache is not None:
        return _num_predict_cache
    
    # Determine CSV file path
    if csv_path is None:
        # Use the same directory as this script
        script_dir = Path(__file__).parent
        csv_path = script_dir / "models_combined_with_num_predict.csv"
    else:
        csv_path = Path(csv_path)
    
    mapping: Dict[str, int] = {}
    
    try:
        if not csv_path.exists():
            print(f"Warning: CSV file not found at {csv_path}, using default num_predict={_DEFAULT_NUM_PREDICT}")
            _num_predict_cache = {}
            return mapping
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_name = row.get('Model Name', '').strip()
                num_predict_str = row.get('num_predict', '').strip()
                
                if model_name and num_predict_str:
                    try:
                        num_predict = int(num_predict_str)
                        mapping[model_name] = num_predict
                    except (ValueError, TypeError):
                        # Skip invalid num_predict values
                        continue
        
        # Cache the mapping
        _num_predict_cache = mapping
        print(f"Loaded num_predict mapping for {len(mapping)} models from {csv_path}")
        
    except Exception as e:
        print(f"Error loading num_predict CSV from {csv_path}: {e}")
        print(f"Using default num_predict={_DEFAULT_NUM_PREDICT} for all models")
        _num_predict_cache = {}
    
    return mapping

def get_num_predict_for_model(model_name: str, csv_file: str) -> int:
    """
    Get the num_predict value for a given model name.
    
    Args:
        model_name: The name of the model
    
    Returns:
        The num_predict value for the model, or default if not found
    """
    mapping = load_num_predict_mapping(csv_file)
    return mapping.get(model_name, _DEFAULT_NUM_PREDICT)


def get_structured_generator_cached(model_name: str, ollama_host: str = None, timeout=500):
    """
    Get a cached structured generator for the given model and host.
    Creates and caches the generator if it doesn't exist.
    
    Args:
        model_name: The name of the Ollama model to use
        ollama_host: Optional Ollama host URL (e.g., http://localhost:11434)
        
    Returns:
        A cached generator that enforces the SecurityTestResponse schema
    """
    # Use "default" as key when ollama_host is None
    cache_key = (model_name, ollama_host or "default")
    
    if cache_key not in _generator_cache:
        
        client = ollama.Client(host=ollama_host, timeout=timeout)
        
        # Create the model using the correct API
        model = models.ollama.from_ollama(client, model_name)
        
        # Create a structured generator using the correct API
        gen = Generator(model, SecurityTestResponse)
        
        # Cache the generator
        _generator_cache[cache_key] = gen
    
    return _generator_cache[cache_key]


def generate_with_timing(model_name: str, prompt: str, ollama_host: str = None, timeout:float=500, csv_file: Optional[str] = None):
    """
    Generate structured response with timing information.
    
    Args:
        model_name: The name of the Ollama model to use
        prompt: The prompt to send to the model
        ollama_host: Ollama host URL (e.g., http://localhost:11434)
        
    Returns:
        Tuple of (SecurityTestResponse, started_at, ended_at, elapsed_ms)
        where started_at and ended_at are datetime objects in UTC
    """

    # Get cached generator
    gen = get_structured_generator_cached(model_name, ollama_host, timeout)
    started_at = datetime.now(timezone.utc)
    
    # Get num_predict value for this model from CSV
    num_predict = get_num_predict_for_model(model_name, csv_file)
    
    # Generate structured response
    response = gen(prompt, options={"num_predict": num_predict})
    
    ended_at = datetime.now(timezone.utc)
    elapsed_ms = int((ended_at - started_at).total_seconds() * 1000)
  
    
    # Convert to the expected format
    if hasattr(response, 'commands'):
        # Response is already a SecurityTestResponse object
        structured_response = response
    elif isinstance(response, str):
        try:
            parsed_response = json.loads(response)
            structured_response = SecurityTestResponse(**parsed_response)
        except Exception:
            print("Incorrect model format...")
            structured_response = response
    elif isinstance(response, dict):
        # Response might be a dict, convert it
        structured_response = SecurityTestResponse(**response)

   # output_tokens = count_tokens(model_name, structured_response.json())
        # 5. Compute tokens/sec
   

    return structured_response, started_at, ended_at, elapsed_ms