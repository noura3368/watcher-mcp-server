import os
import csv, ast
import re
from typing import Dict, Tuple, List, Set, Optional
from collections import defaultdict
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from collections import Counter
import pandas as pd
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

def write_html(fig: go.Figure, out_path: str) -> None:
    """Write a Plotly figure to an HTML file.
    
    Args:
        fig: Plotly figure object
        out_path: Path to output HTML file
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)


# Colors for coding vs non-coding model bars when code-command-analysis is used
CODING_MODEL_COLOR = "#1f77b4"
NON_CODING_MODEL_COLOR = "#7f7f7f"


def load_model_to_is_coding(csv_path: str) -> Optional[Dict[str, bool]]:
    """Load model -> is_coding_model mapping from CSV.
    
    Returns dict mapping model name to True/False if CSV has is_coding_model column;
    returns None if column is missing or on error.
    """
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if "is_coding_model" not in fieldnames:
                return None
            result: Dict[str, bool] = {}
            for row in reader:
                model = (row.get("model") or "").strip()
                if not model:
                    continue
                if model in result:
                    continue
                raw = row.get("is_coding_model", "").strip()
                if isinstance(raw, bool):
                    result[model] = raw
                else:
                    result[model] = str(raw).lower() in ("true", "1", "on", "yes")
            return result if result else None
    except Exception as e:
        print(f"Warning: Could not load is_coding_model from {csv_path}: {e}")
        return None


def bar_colors_for_models(
    models: List[str],
    model_to_is_coding: Dict[str, bool],
    coding_color: str = CODING_MODEL_COLOR,
    non_coding_color: str = NON_CODING_MODEL_COLOR,
) -> List[str]:
    """Return a list of colors, one per model: coding_color where is_coding_model else non_coding_color."""
    return [
        coding_color if model_to_is_coding.get(m, False) else non_coding_color
        for m in models
    ]


def load_model_size_mapping(root_dir: str = "/data2/nkhajehn/watcher-mcp-server/llm_pipeline/services") -> Dict[str, float]:
    """Load model parameter sizes from models_combined_with_num_predict.csv.
    
    Args:
        root_dir: Root directory where the models CSV file is located
        
    Returns:
        Dictionary mapping model names to parameter sizes in billions (float)
    """
    model_size_map = {}
    models_csv_path = os.path.join(root_dir, "models_combined_with_num_predict.csv")
    
    try:
        with open(models_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_name = row.get("Model Name", "").strip()
                try:
                    param_size = float(row.get("Parameter Size (Billions)", 0))
                except (ValueError, TypeError):
                    param_size = 0.0
                
                if model_name:
                    model_size_map[model_name] = param_size
    except Exception as e:
        print(f"Warning: Could not load model size mapping from {models_csv_path}: {e}")
    
    return model_size_map


# Metadata columns excluded when detecting blank rows (all value columns must be 0/empty)
_BLANK_METADATA_COLUMNS = {"prompt", "model", "run_number", "iteration", "is_coding_model", "iteration_duration_seconds"}


def _is_empty_value(val: str) -> bool:
    """True if val is 0, empty string, or empty list/set representation."""
    if val is None or val == "":
        return True
    s = str(val).strip()
    if s in ("0", "0.0", "[]", "{}", "set()", 0, 0.0, [], {}, set()):
        return True
    try:
        return float(s) == 0
    except (ValueError, TypeError):
        return False


def _is_blank_row(row: Dict[str, str], value_columns: List[str]) -> bool:
    """True if all value columns (excluding metadata) are empty."""
    #print(value_columns)
    return all(_is_empty_value(row.get(col, "")) for col in value_columns)


def create_average_valid_per_model_plots(csv_path: str, output_dir: str, model_to_is_coding: Optional[Dict[str, bool]] = None) -> None:
    """Create plots showing average valid_commands per model across runs (one file per prompt).

    For each prompt, compute the average of valid_commands for each model across all runs
    and write a single bar chart (models on X, average valid commands on Y).
    average valid commands = (sum of valid_commands across runs) / (number of runs)
    # how many valid commands were generate on average across the ALL the runs. 
    # add all the valid commands / number of total iterations across the runs 
    """
    avg_dir = os.path.join(output_dir, "average_valid_per_model")
    os.makedirs(avg_dir, exist_ok=True)

    # Collect valid_commands values per (prompt, model)
    data_by_prompt_model: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    prompts = set()

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row.get("prompt", "").strip()
                model = row.get("model", "").strip()
                try:
                    valid_commands = int(row.get("valid_commands", 0))
                except ValueError:
                    valid_commands = 0

                if prompt and model:
                    data_by_prompt_model[(prompt, model)].append(valid_commands)
                    prompts.add(prompt)
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return

    for prompt in sorted(prompts):
        # gather models for this prompt
        models = sorted([m for (p, m) in data_by_prompt_model.keys() if p == prompt])
        if not models:
            continue

        averages = []
        for model in models:
            vals = data_by_prompt_model.get((prompt, model), [])
            avg = (sum(vals) / len(vals)) if vals else 0.0
            averages.append(avg)

        # build figure
        fig = go.Figure()
        bar_kw: Dict[str, object] = dict(
            x=models,
            y=averages,
            text=[f"{v:.2f}" for v in averages],
            textposition="outside",
            hovertemplate="Model=%{x}<br>Avg Valid Commands=%{y:.2f}<extra></extra>",
            showlegend=False,
        )
        if model_to_is_coding and models:
            bar_kw["marker_color"] = bar_colors_for_models(models, model_to_is_coding)
        fig.add_trace(go.Bar(**bar_kw))

        fig.update_xaxes(title_text="Model", tickangle=-45)
        fig.update_yaxes(title_text="Average Valid Commands")
        fig.update_layout(
            title_text=f"Average Valid Commands per Model (Across Runs): {prompt}",
            template="plotly_white",
            height=700,
        )

        output_path = os.path.join(avg_dir, f"{prompt}.html")
        write_html(fig, output_path)
        print(f"Created average-valid plot: {output_path}")


def create_validity_ratio_plots(csv_path: str, output_dir: str, model_to_is_coding: Optional[Dict[str, bool]] = None) -> None:
    # finds the avg validity ratio per model per prompt across runs, where validity ratio = (unique_valid_commands from last iteration) / (cumulative_commands from last iteration)
    # add all the valid commands of the run up / add all the cumulative commands of the run up = avg validity ratio for that model for that run
    ratio_dir = os.path.join(output_dir, "validity_ratio")
    os.makedirs(ratio_dir, exist_ok=True)

    # Track last unique_valid_commands per (prompt, run_number, model) - prefer iteration 50 or max
    last_unique_valid_by_prompt_run_model: Dict[Tuple[str, int, str], Tuple[int, int]] = {}
    # Track last cumulative_commands per (prompt, run_number, model) by max iteration
    last_cum_by_prompt_run_model: Dict[Tuple[str, int, str], Tuple[int, float]] = {}
    prompts: Set[str] = set()
    runs: Set[int] = set()

    try:
        # Try the provided CSV, but if it doesn't contain the expected
        # 'base_commands_seen_so_far' column try a few sensible fallbacks
        tried_paths = [csv_path]
        selected_path = None

        def _has_base_column(path: str) -> bool:
            try:
                with open(path, "r", encoding="utf-8") as tf:
                    rdr = csv.DictReader(tf)
                    fns = rdr.fieldnames or []
                    return "base_commands_seen_so_far" in fns
            except Exception:
                return False

        if _has_base_column(csv_path):
            selected_path = csv_path
        if selected_path is None:
            print(
                "Warning: CSV does not contain 'base_commands_seen_so_far' column."
                " Tried files: {}".format(
                    ", ".join(p for p in tried_paths if os.path.exists(p)) or ", ".join(tried_paths)
                )
            )
            # Nothing to do
            return

        # Open the selected CSV file and iterate rows
        with open(selected_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row.get("prompt", "").strip()
                if not prompt:
                    continue

                try:
                    run_number = int(row.get("run_number", 0))
                except ValueError:
                    run_number = 0

                model = row.get("model", "").strip()
                if not model:
                    continue

                try:
                    iteration = int(row.get("iteration", 0))
                except ValueError:
                    iteration = 0

                try:
                    cumulative_commands = float(row.get("cumulative_commands", 0.0))
                except ValueError:
                    cumulative_commands = 0.0

                try:
                    unique_valid_commands = int(row.get("unique_valid_commands", 0))
                except ValueError:
                    unique_valid_commands = 0

                key = (prompt, run_number, model)

                # Track last iteration's unique_valid_commands (prefer iteration 50 or max)
                if key not in last_unique_valid_by_prompt_run_model:
                    last_unique_valid_by_prompt_run_model[key] = (iteration, unique_valid_commands)
                else:
                    prev_iter, _ = last_unique_valid_by_prompt_run_model[key]
                    # prefer iteration 50 if seen, otherwise keep the max
                    if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                        last_unique_valid_by_prompt_run_model[key] = (iteration, unique_valid_commands)

                # store last cumulative_commands by max iteration
                if key not in last_cum_by_prompt_run_model or iteration > last_cum_by_prompt_run_model[key][0]:
                    last_cum_by_prompt_run_model[key] = (iteration, cumulative_commands)
             
                prompts.add(prompt)
         
                runs.add(run_number)
                
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return

    # For each prompt build ratio per model
    for prompt in sorted(prompts):
        # gather all models for this prompt
        models_set: Set[str] = set()
        for (p, r, m) in list(last_unique_valid_by_prompt_run_model.keys()) + list(last_cum_by_prompt_run_model.keys()):
            if p == prompt:
                models_set.add(m)

        models = sorted(models_set)
        if not models:
            continue

        ratios: List[float] = []
        sum_unique_valids_list: List[float] = []
        sum_cums_list: List[float] = []

        for model in models:
            # Sum unique_valid_commands across runs
            sum_unique_valid = 0
            sum_cum = 0.0
            for run in sorted(runs):
                v_key = (prompt, run, model)
                # add last unique valid for this run if present
                if v_key in last_unique_valid_by_prompt_run_model:
                    _, unique_val = last_unique_valid_by_prompt_run_model[v_key]
                    sum_unique_valid += unique_val
                # add last cumulative for this run if present
                if v_key in last_cum_by_prompt_run_model:
                    _, cum_val = last_cum_by_prompt_run_model[v_key]
                    sum_cum += float(cum_val)
        
            # compute ratio safely
            ratio = (sum_unique_valid / sum_cum) if sum_cum > 0 else 0.0
            ratios.append(ratio)
            sum_unique_valids_list.append(sum_unique_valid)
            sum_cums_list.append(sum_cum)

        # Build figure
        fig = go.Figure()
        bar_kw: Dict[str, object] = dict(
            x=models,
            y=ratios,
            text=[f"{r:.3f}" for r in ratios],
            textposition="outside",
            hovertemplate=(
                "Model=%{x}<br>Unique Valid Sum=%{customdata[0]}<br>Cumulative Sum=%{customdata[1]:.2f}<br>"
                "Validity Ratio=%{y:.3f}<extra></extra>"
            ),
            customdata=list(zip(sum_unique_valids_list, sum_cums_list)),
            showlegend=False,
        )
        if model_to_is_coding and models:
            bar_kw["marker_color"] = bar_colors_for_models(models, model_to_is_coding)
        fig.add_trace(go.Bar(**bar_kw))

        fig.update_xaxes(title_text="Model", tickangle=-45)
        fig.update_yaxes(title_text="Validity Ratio (sum unique valid / sum cumulative)", range=[0, 1])
        fig.update_layout(
            title_text=f"Validity Ratio per Model (sum unique valid / sum cumulative across runs): {prompt}",
            template="plotly_white",
            height=700,
        )

        output_path = os.path.join(ratio_dir, f"{prompt}.html")
        write_html(fig, output_path)
        print(f"Created validity-ratio plot: {output_path}")

def create_valid_commands_barchart(csv_path: str, output_dir: str, model_to_is_coding: Optional[Dict[str, bool]] = None) -> None:
    """Create histogram plots showing total valid commands per model per run.
    
    Creates 8 HTML files (one per prompt), each with 3 subplots (one per run).
    Each subplot shows models on X-axis and total valid commands (sum across all iterations) on Y-axis.
    
    Args:
        csv_path: Path to command_statistics.csv
        output_dir: Base output directory (files will be saved in valid_commands_per_model subdirectory)
        model_to_is_coding: If provided, bar colors reflect coding vs non-coding model
    """
    dir = os.path.join(output_dir, "valid_commands_per_model")
    os.makedirs(dir, exist_ok=True)
    
    # Read CSV and group by (prompt, run_number, model)
    data_by_prompt_run_model: Dict[Tuple[str, int, str], int] = defaultdict(int)
    prompts = set()
    runs = set()
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row.get("prompt", "").strip()
                run_number = int(row.get("run_number", 0))
                model = row.get("model", "").strip()
                valid_commands = int(row.get("valid_commands", 0))
    
                if prompt and model:
                    data_by_prompt_run_model[(prompt, run_number, model)] += valid_commands
                    prompts.add(prompt)
                    runs.add(run_number)
               
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return
    
    # Create one HTML file per prompt
    for prompt in sorted(prompts):
        # Get all runs for this prompt
        prompt_runs = sorted([r for r in runs if any(p == prompt for p, _, _ in data_by_prompt_run_model.keys())])
        
        # Create subplots: 3 rows, 1 column
        fig = make_subplots(
            rows=len(prompt_runs),
            cols=1,
            subplot_titles=[f"{prompt} - Run {run}" for run in prompt_runs],
            vertical_spacing=0.15,
        )
        models_data_avg = defaultdict(int)
        # Process each run
        for run_idx, run_number in enumerate(prompt_runs, start=1):
            # Get all models for this (prompt, run) combination
            models_data = {}
            for (p, r, model), total_valid in data_by_prompt_run_model.items():
                if p == prompt and r == run_number:
                    models_data[model] = total_valid
                    models_data_avg[model] += total_valid

            if models_data:
                # Sort models by name for consistent ordering
                models = sorted(models_data.keys())
                counts = [models_data[model] for model in models]
            else:
                models = []
                counts = []
            
            # Add bar trace to the appropriate subplot
            bar_kw: Dict[str, object] = dict(
                x=models,
                y=counts,
                text=counts if counts else [],
                textposition="outside",
                hovertemplate="Model=%{x}<br>Total Valid Commands=%{y}<extra></extra>",
                showlegend=False,
            )
            if model_to_is_coding and models:
                bar_kw["marker_color"] = bar_colors_for_models(models, model_to_is_coding)
            fig.add_trace(
                go.Bar(**bar_kw),
                row=run_idx,
                col=1,
            )
        for model in models_data_avg:
            models_data_avg[model] = round(models_data_avg[model] / len(prompt_runs)) 
                    # Add bar trace to the appropriate subplot

        # Update layout
        if len(prompt_runs) > 0:
            mid_row = (len(prompt_runs) + 1) // 2
            fig.update_xaxes(title_text="Model", tickangle=-45, row=mid_row, col=1)
            fig.update_xaxes(tickangle=-45)  # Rotate all x-axis labels
            fig.update_yaxes(title_text="Total Valid Commands", row=mid_row, col=1)
        
        fig.update_layout(
            title_text=f"Valid Commands per Model: {prompt} (All Runs)",
            template="plotly_white",
            height=500 * len(prompt_runs),
        )
        
        # Write HTML file
        output_path = os.path.join(dir, f"{prompt}.html")
        write_html(fig, output_path)
        print(f"Created bar chart: {output_path}")


def create_valid_commands_avg_perrun(csv_path: str, output_dir: str, model_to_is_coding: Optional[Dict[str, bool]] = None) -> None:
    dir = os.path.join(output_dir, "valid_commands_per_model")
    os.makedirs(dir, exist_ok=True)
    
    # Read CSV and group by (prompt, run_number, model)
    data_by_prompt_run_model: Dict[Tuple[str, int, str], int] = defaultdict(int)
    prompts = set()
    runs = set()
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row.get("prompt", "").strip()
                run_number = int(row.get("run_number", 0))
                model = row.get("model", "").strip()
                valid_commands = int(row.get("valid_commands", 0))
    
                if prompt and model:
                    data_by_prompt_run_model[(prompt, run_number, model)] += valid_commands
                    prompts.add(prompt)
                    runs.add(run_number)
               
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return
    
    # Create one HTML file per prompt
    for prompt in sorted(prompts):
        # Get all runs for this prompt
        prompt_runs = sorted([r for r in runs if any(p == prompt for p, _, _ in data_by_prompt_run_model.keys())])
        

        models_data_avg = defaultdict(int)
        # Process each run
        for run_idx, run_number in enumerate(prompt_runs, start=1):
            # Get all models for this (prompt, run) combination
            for (p, r, model), total_valid in data_by_prompt_run_model.items():
                if p == prompt and r == run_number:
                    models_data_avg[model] += total_valid

        for model in models_data_avg:
            models_data_avg[model] = round(models_data_avg[model] / len(prompt_runs)) 
                    # Add bar trace to the appropriate subplot
        models = list(models_data_avg.keys())
        yvals = list(models_data_avg.values())
        bar_kw: Dict[str, object] = dict(
            x=models,
            y=yvals,
            text=yvals,
            textposition="outside",
            hovertemplate="Model=%{x}<br>Avg Valid Commands=%{y}<extra></extra>",
            showlegend=False,
        )
        if model_to_is_coding and models:
            # Assumes you already have this helper in your codebase
            bar_kw["marker_color"] = bar_colors_for_models(models, model_to_is_coding)

        fig = go.Figure(data=[go.Bar(**bar_kw)])
        fig.update_layout(
            title_text=f"Average Valid Commands per Model (across {len(prompt_runs)} run(s)): {prompt}",
            template="plotly_white",
            height=700,
            margin=dict(t=120),
        )
        fig.update_xaxes(title_text="Model", tickangle=-45)
        fig.update_yaxes(title_text="Avg Valid Commands")

        output_path = os.path.join(dir, f"{prompt}_average.html")
        write_html(fig, output_path)
        print(f"Created bar chart: {output_path}")

def create_unique_valid_avg_perrun(csv_path: str, output_dir: str, model_to_is_coding: Optional[Dict[str, bool]] = None) -> None:
    dir = os.path.join(output_dir, "unique_valid_commands_per_model")
    os.makedirs(dir, exist_ok=True)
    
    # Read CSV and track the last iteration's unique_valid_commands for each (prompt, run_number, model)
    # We need to find the maximum iteration for each combination
    max_iteration_by_prompt_run_model: Dict[Tuple[str, int, str], int] = {}
    data_by_prompt_run_model: Dict[Tuple[str, int, str], int] = {}
    prompts = set()
    runs = set()
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row.get("prompt", "").strip()
                run_number = int(row.get("run_number", 0))
                model = row.get("model", "").strip()
                iteration = int(row.get("iteration", 0))
                unique_valid_commands = int(row.get("unique_valid_commands", 0))
                
                if prompt and model and iteration > 0:
                    key = (prompt, run_number, model)
                    
                    # Track maximum iteration for this combination
                    if key not in max_iteration_by_prompt_run_model or iteration > max_iteration_by_prompt_run_model[key]:
                        max_iteration_by_prompt_run_model[key] = iteration
                        data_by_prompt_run_model[key] = unique_valid_commands
                    
                    prompts.add(prompt)
                    runs.add(run_number)
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return
    
       # Create one HTML file per prompt
    for prompt in sorted(prompts):
        # Get all runs for this prompt
        prompt_runs = sorted([r for r in runs if any(p == prompt for p, _, _ in data_by_prompt_run_model.keys())])
        

        models_data_avg = defaultdict(int)
        # Process each run
        for run_idx, run_number in enumerate(prompt_runs, start=1):
            # Get all models for this (prompt, run) combination
            for (p, r, model), unique_total_valid in data_by_prompt_run_model.items():
                if p == prompt and r == run_number:
                    models_data_avg[model] += unique_total_valid

        for model in models_data_avg:
            models_data_avg[model] = round(models_data_avg[model] / len(prompt_runs)) 
                    
        models = list(models_data_avg.keys())
        yvals = list(models_data_avg.values())
        bar_kw: Dict[str, object] = dict(
            x=models,
            y=yvals,
            text=yvals,
            textposition="outside",
            hovertemplate="Model=%{x}<br>Avg Valid Unique Commands=%{y}<extra></extra>",
            showlegend=False,
        )
        if model_to_is_coding and models:
            # Assumes you already have this helper in your codebase
            bar_kw["marker_color"] = bar_colors_for_models(models, model_to_is_coding)

        fig = go.Figure(data=[go.Bar(**bar_kw)])
        fig.update_layout(
            title_text=f"Average Valid Unique Commands per Model (across {len(prompt_runs)} run(s)): {prompt}",
            template="plotly_white",
            height=700,
            margin=dict(t=120),
        )
        fig.update_xaxes(title_text="Model", tickangle=-45)
        fig.update_yaxes(title_text="Avg Valid Unique Commands")

        output_path = os.path.join(dir, f"{prompt}_average.html")
        write_html(fig, output_path)
        print(f"Created bar chart: {output_path}")


def create_unique_valid_barchart(csv_path: str, output_dir: str, model_to_is_coding: Optional[Dict[str, bool]] = None) -> None:
    """Create histogram plots showing unique valid commands per model per run (from last iteration).
    
    Creates 8 HTML files (one per prompt), each with 3 subplots (one per run).
    Each subplot shows models on X-axis and unique valid commands (from last iteration) on Y-axis.
    
    Args:
        csv_path: Path to command_statistics.csv
        output_dir: Base output directory (files will be saved in unique_valid subdirectory)
        model_to_is_coding: If provided, bar colors reflect coding vs non-coding model
    """
    dir = os.path.join(output_dir, "unique_valid_commands_per_model")
    os.makedirs(dir, exist_ok=True)
    
    # Read CSV and track the last iteration's unique_valid_commands for each (prompt, run_number, model)
    # We need to find the maximum iteration for each combination
    max_iteration_by_prompt_run_model: Dict[Tuple[str, int, str], int] = {}
    data_by_prompt_run_model: Dict[Tuple[str, int, str], int] = {}
    prompts = set()
    runs = set()
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row.get("prompt", "").strip()
                run_number = int(row.get("run_number", 0))
                model = row.get("model", "").strip()
                iteration = int(row.get("iteration", 0))
                unique_valid_commands = int(row.get("unique_valid_commands", 0))
                
                if prompt and model and iteration > 0:
                    key = (prompt, run_number, model)
                    
                    # Track maximum iteration for this combination
                    if key not in max_iteration_by_prompt_run_model or iteration > max_iteration_by_prompt_run_model[key]:
                        max_iteration_by_prompt_run_model[key] = iteration
                        data_by_prompt_run_model[key] = unique_valid_commands
                    
                    prompts.add(prompt)
                    runs.add(run_number)
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return
    
    # Create one HTML file per prompt
    for prompt in sorted(prompts):
        # Get all runs for this prompt
        prompt_runs = sorted([r for r in runs if any(p == prompt for p, _, _ in data_by_prompt_run_model.keys())])
        
        # Create subplots: 3 rows, 1 column
        fig = make_subplots(
            rows=len(prompt_runs),
            cols=1,
            subplot_titles=[f"{prompt} - Run {run}" for run in prompt_runs],
            vertical_spacing=0.15,
        )
        
        # Process each run
        for run_idx, run_number in enumerate(prompt_runs, start=1):
            # Get all models for this (prompt, run) combination
            models_data = {}
            for (p, r, model), unique_valid in data_by_prompt_run_model.items():
                if p == prompt and r == run_number:
                    models_data[model] = unique_valid
            
            if models_data:
                # Sort models by name for consistent ordering
                models = sorted(models_data.keys())
                counts = [models_data[model] for model in models]
            else:
                models = []
                counts = []
            
            # Add bar trace to the appropriate subplot
            bar_kw: Dict[str, object] = dict(
                x=models,
                y=counts,
                text=counts if counts else [],
                textposition="outside",
                hovertemplate="Model=%{x}<br>Unique Valid Commands=%{y}<extra></extra>",
                showlegend=False,
            )
            if model_to_is_coding and models:
                bar_kw["marker_color"] = bar_colors_for_models(models, model_to_is_coding)
            fig.add_trace(
                go.Bar(**bar_kw),
                row=run_idx,
                col=1,
            )
        
        # Update layout
        if len(prompt_runs) > 0:
            mid_row = (len(prompt_runs) + 1) // 2
            fig.update_xaxes(title_text="Model", tickangle=-45, row=mid_row, col=1)
            fig.update_xaxes(tickangle=-45)  # Rotate all x-axis labels
            fig.update_yaxes(title_text="Unique Valid Commands (Last Iteration)", row=mid_row, col=1)
        
        fig.update_layout(
            title_text=f"Unique Valid Commands per Model: {prompt} (All Runs)",
            template="plotly_white",
            height=500 * len(prompt_runs),
        )
        
        # Write HTML file
        output_path = os.path.join(dir, f"{prompt}.html")
        write_html(fig, output_path)
        print(f"Created unique valid histogram plot: {output_path}")


def create_average_valid_vs_cumulative_plots(csv_path: str, output_dir: str, model_to_is_coding: Optional[Dict[str, bool]] = None) -> None:
    """Create bar plots showing average total valid_commands vs average cumulative unique_valid_commands per model.
    
    For each prompt, creates a bar chart comparing:
    - Average total valid_commands (sum across all iterations, averaged across runs)
    - Average cumulative unique_valid_commands (from last iteration, averaged across runs)
    
    Creates 8 HTML files (one per prompt).
    
    Args:
        csv_path: Path to command_statistics CSV file
        output_dir: Base output directory (files will be saved in average_valid_vs_cumulative subdirectory)
    """
    output_subdir = os.path.join(output_dir, "avg_valid_commands_vs_unique_commands")
    os.makedirs(output_subdir, exist_ok=True)
    
    # Track total valid_commands per (prompt, run, model) - sum across all iterations
    total_valid_by_prompt_run_model: Dict[Tuple[str, int, str], int] = defaultdict(int)
    # Track last iteration's unique_valid_commands per (prompt, run, model)
    last_unique_valid_by_prompt_run_model: Dict[Tuple[str, int, str], Tuple[int, int]] = {}
    prompts: Set[str] = set()
    runs: Set[int] = set()
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row.get("prompt", "").strip()
                if not prompt:
                    continue
                
                try:
                    run_number = int(row.get("run_number", 0))
                except ValueError:
                    run_number = 0
                
                model = row.get("model", "").strip()
                if not model:
                    continue
                
                try:
                    iteration = int(row.get("iteration", 0))
                except ValueError:
                    iteration = 0
                
                try:
                    valid_commands = int(row.get("valid_commands", 0))
                except ValueError:
                    valid_commands = 0
                
                try:
                    unique_valid_commands = int(row.get("unique_valid_commands", 0))
                except ValueError:
                    unique_valid_commands = 0
                
                key = (prompt, run_number, model)
                
                # Sum valid_commands across all iterations
                total_valid_by_prompt_run_model[key] += valid_commands
                
                # Track last iteration's unique_valid_commands
                if key not in last_unique_valid_by_prompt_run_model or iteration > last_unique_valid_by_prompt_run_model[key][0]:
                    last_unique_valid_by_prompt_run_model[key] = (iteration, unique_valid_commands)
                
                prompts.add(prompt)
                runs.add(run_number)
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return
    
    # Load model size mapping
    model_size_map = load_model_size_mapping()
    
    # Create one HTML file per prompt
    for prompt in sorted(prompts):
        # Get all models for this prompt
        models_set: Set[str] = set()
        for (p, r, m) in list(total_valid_by_prompt_run_model.keys()) + list(last_unique_valid_by_prompt_run_model.keys()):
            if p == prompt:
                models_set.add(m)
        
        models = sorted(models_set)
        if not models:
            continue
        
        # Split models by parameter size
        models_gt_4b = []
        models_gt_10b = []
        models_le_4b = []
        
        
        for model in models:
            param_size = model_size_map.get(model, 0.0)
            if param_size >= 4.0 and param_size < 10.0:
                models_gt_4b.append(model)
            elif param_size >= 10.0:
                models_gt_10b.append(model)
            elif param_size < 4.0:
                models_le_4b.append(model)
            # Skip models not found in mapping
        
        # Helper function to create plot for a group of models
        def create_plot_for_models(model_list: List[str], size_label: str, filename_suffix: str) -> None:
            if not model_list:
                return
            
            # Calculate averages for each model
            avg_total_valid = []
            avg_cumulative_unique = []
            
            for model in model_list:
                # Sum total valid_commands across runs
                total_valid_sum = 0
                cumulative_unique_sum = 0
                run_count = 0
                
                for run in sorted(runs):
                    key = (prompt, run, model)
                    
                    # Count runs that have data (either total_valid or cumulative_unique)
                    has_data = False
                    
                    # Add total valid for this run
                    if key in total_valid_by_prompt_run_model:
                        total_valid_sum += total_valid_by_prompt_run_model[key]
                        has_data = True
                    
                    # Add cumulative unique for this run (from last iteration)
                    if key in last_unique_valid_by_prompt_run_model:
                        _, cum_unique = last_unique_valid_by_prompt_run_model[key]
                        cumulative_unique_sum += cum_unique
                        has_data = True
                    
                    if has_data:
                        run_count += 1
                
                # Calculate averages
                avg_total = (total_valid_sum / run_count) if run_count > 0 else 0.0
                avg_cumulative = (cumulative_unique_sum / run_count) if run_count > 0 else 0.0
                
                avg_total_valid.append(avg_total)
                avg_cumulative_unique.append(avg_cumulative)
            
            # Create grouped bar chart
            fig = go.Figure()
            bar_colors = bar_colors_for_models(model_list, model_to_is_coding) if (model_to_is_coding and model_list) else None
            
            kw1: Dict[str, object] = dict(
                name="Avg Total Valid Commands",
                x=model_list,
                y=avg_total_valid,
                text=[f"{v:.2f}" for v in avg_total_valid],
                textposition="outside",
                hovertemplate="Model=%{x}<br>Avg Total Valid Commands=%{y:.2f}<extra></extra>",
            )
            if bar_colors:
                kw1["marker_color"] = bar_colors
            fig.add_trace(go.Bar(**kw1))
            
            kw2: Dict[str, object] = dict(
                name="Avg Unique Valid Commands",
                x=model_list,
                y=avg_cumulative_unique,
                text=[f"{v:.2f}" for v in avg_cumulative_unique],
                textposition="outside",
                hovertemplate="Model=%{x}<br>Avg Cumulative Unique Valid=%{y:.2f}<extra></extra>",
            )
            if bar_colors:
                kw2["marker_color"] = bar_colors
            fig.add_trace(go.Bar(**kw2))
            
            fig.update_xaxes(title_text="Model", tickangle=-45)
            fig.update_yaxes(title_text="Average Commands")
            fig.update_layout(
                title_text=f"Average Valid Commands vs Unique Valid Commands: {prompt} ({size_label})",
                template="plotly_white",
                barmode="group",
                height=700,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            
            output_path = os.path.join(output_subdir, f"{prompt}{filename_suffix}.html")
            write_html(fig, output_path)
            print(f"Created average valid vs cumulative plot: {output_path}")
        
        # Create plots for both groups'
        create_plot_for_models(models_gt_4b, ">=4B and <10b", "_gt4b")
        create_plot_for_models(models_gt_10b, ">=10B", "_gt10b")
        create_plot_for_models(models_le_4b, "<4B", "_le4b")


def create_cumulative_valid_unique_plots(csv_path: str, output_dir: str, model_to_is_coding: Optional[Dict[str, bool]] = None) -> None:
    """Create bar plots showing average cumulative_commands, valid_commands, and unique_valid_commands per model.
     z
    For each prompt, creates a grouped bar chart with three bars per model:
    - Average Cumulative Commands (from last iteration, averaged across runs)
    - Average Valid Commands (sum across all iterations, averaged across runs)
    - Average Unique Valid Commands (from last iteration, averaged across runs)
    
    Creates 8 HTML files (one per prompt).
    
    Args:
        csv_path: Path to command_statistics CSV file
        output_dir: Base output directory (files will be saved in plots/cumulative_valid_unique subdirectory)
    """
    output_subdir = os.path.join(output_dir, "plots", "cumulative_valid_unique")
    os.makedirs(output_subdir, exist_ok=True)
    
    # Track total valid_commands per (prompt, run, model) - sum across all iterations
    total_valid_by_prompt_run_model: Dict[Tuple[str, int, str], int] = defaultdict(int)
    # Track last iteration's cumulative_commands per (prompt, run, model) - prefer iteration 50 or max
    last_cumulative_by_prompt_run_model: Dict[Tuple[str, int, str], Tuple[int, float]] = {}
    # Track last iteration's unique_valid_commands per (prompt, run, model) - prefer iteration 50 or max
    last_unique_valid_by_prompt_run_model: Dict[Tuple[str, int, str], Tuple[int, int]] = {}
    prompts: Set[str] = set()
    runs: Set[int] = set()
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row.get("prompt", "").strip()
                if not prompt:
                    continue
                
                try:
                    run_number = int(row.get("run_number", 0))
                except ValueError:
                    run_number = 0
                
                model = row.get("model", "").strip()
                if not model:
                    continue
                
                try:
                    iteration = int(row.get("iteration", 0))
                except ValueError:
                    iteration = 0
                
                try:
                    valid_commands = int(row.get("valid_commands", 0))
                except ValueError:
                    valid_commands = 0
                
                try:
                    cumulative_commands = float(row.get("cumulative_commands", 0))
                except ValueError:
                    cumulative_commands = 0.0
                
                try:
                    unique_valid_commands = int(row.get("unique_valid_commands", 0))
                except ValueError:
                    unique_valid_commands = 0
                
                key = (prompt, run_number, model)
                
                # Sum valid_commands across all iterations
                total_valid_by_prompt_run_model[key] += valid_commands
                
                # Track last iteration's cumulative_commands (prefer iteration 50 or max)
                if key not in last_cumulative_by_prompt_run_model:
                    last_cumulative_by_prompt_run_model[key] = (iteration, cumulative_commands)
                else:
                    prev_iter, _ = last_cumulative_by_prompt_run_model[key]
                    # prefer iteration 50 if seen, otherwise keep the max
                    if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                        last_cumulative_by_prompt_run_model[key] = (iteration, cumulative_commands)
                
                # Track last iteration's unique_valid_commands (prefer iteration 50 or max)
                if key not in last_unique_valid_by_prompt_run_model:
                    last_unique_valid_by_prompt_run_model[key] = (iteration, unique_valid_commands)
                else:
                    prev_iter, _ = last_unique_valid_by_prompt_run_model[key]
                    # prefer iteration 50 if seen, otherwise keep the max
                    if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                        last_unique_valid_by_prompt_run_model[key] = (iteration, unique_valid_commands)
                
                prompts.add(prompt)
                runs.add(run_number)
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return
    
    # Load model size mapping
    model_size_map = load_model_size_mapping()
    
    # Create one HTML file per prompt
    for prompt in sorted(prompts):
        # Get all models for this prompt
        models_set: Set[str] = set()
        for (p, r, m) in list(total_valid_by_prompt_run_model.keys()) + list(last_cumulative_by_prompt_run_model.keys()) + list(last_unique_valid_by_prompt_run_model.keys()):
            if p == prompt:
                models_set.add(m)
        
        models = sorted(models_set)
        if not models:
            continue
        
        # Split models by parameter size
        models_gt_4b = []
        models_le_4b = []
        models_gt_10b = []
        
        for model in models:
            param_size = model_size_map.get(model, 0.0)
            if param_size > 4.0 and param_size < 10.0:
                models_gt_4b.append(model)
            elif param_size >= 10.0:
                models_gt_10b.append(model)
            elif param_size <= 4.0:
                models_le_4b.append(model)
            # Skip models not found in mapping
        
        # Helper function to create plot for a group of models
        def create_plot_for_models(model_list: List[str], size_label: str, filename_suffix: str) -> None:
            if not model_list:
                return
            
            # Calculate averages for each model
            avg_cumulative = []
            avg_total_valid = []
            avg_unique_valid = []
            
            for model in model_list:
                # Sum values across runs
                cumulative_sum = 0.0
                total_valid_sum = 0
                unique_valid_sum = 0
                run_count = 0
                
                for run in sorted(runs):
                    key = (prompt, run, model)
                    
                    # Count runs that have data
                    has_data = False
                    
                    # Add cumulative for this run (from last iteration)
                    if key in last_cumulative_by_prompt_run_model:
                        _, cum_val = last_cumulative_by_prompt_run_model[key]
                        cumulative_sum += cum_val
                        has_data = True
                    
                    # Add total valid for this run
                    if key in total_valid_by_prompt_run_model:
                        total_valid_sum += total_valid_by_prompt_run_model[key]
                        has_data = True
                    
                    # Add unique valid for this run (from last iteration)
                    if key in last_unique_valid_by_prompt_run_model:
                        _, unique_val = last_unique_valid_by_prompt_run_model[key]
                        unique_valid_sum += unique_val
                        has_data = True
                    
                    if has_data:
                        run_count += 1
                
                # Calculate averages
                avg_cum = round(cumulative_sum / run_count, 2) if run_count > 0 else 0.0
                avg_valid = round(total_valid_sum / run_count, 2) if run_count > 0 else 0.0
                avg_unique = round(unique_valid_sum / run_count,2) if run_count > 0 else 0.0
                
                avg_cumulative.append(avg_cum)
                avg_total_valid.append(avg_valid)
                avg_unique_valid.append(avg_unique)
                          
            # Create grouped bar chart
            fig = go.Figure()
            bar_colors = bar_colors_for_models(model_list, model_to_is_coding) if (model_to_is_coding and model_list) else None
            
            kw1: Dict[str, object] = dict(name="Avg Cumulative Commands", x=model_list, y=avg_cumulative, text=[f"{v:.2f}" for v in avg_cumulative], textposition="outside", hovertemplate="Model=%{x}<br>Avg Cumulative Commands=%{y:.2f}<extra></extra>")
            if bar_colors:
                kw1["marker_color"] = bar_colors
            fig.add_trace(go.Bar(**kw1))
            
            kw2: Dict[str, object] = dict(name="Avg Valid Commands", x=model_list, y=avg_total_valid, text=[f"{v:.2f}" for v in avg_total_valid], textposition="outside", hovertemplate="Model=%{x}<br>Avg Valid Commands (sum across all iterations)=%{y:.2f}<extra></extra>")
            if bar_colors:
                kw2["marker_color"] = bar_colors
            fig.add_trace(go.Bar(**kw2))
            
            kw3: Dict[str, object] = dict(name="Avg Unique Valid Commands", x=model_list, y=avg_unique_valid, text=[f"{v:.2f}" for v in avg_unique_valid], textposition="outside", hovertemplate="Model=%{x}<br>Avg Unique Valid Commands (from last iteration)=%{y:.2f}<extra></extra>")
            if bar_colors:
                kw3["marker_color"] = bar_colors
            fig.add_trace(go.Bar(**kw3))
            
            fig.update_xaxes(title_text="Model", tickangle=-45)
            fig.update_yaxes(title_text="Average Commands")
            fig.update_layout(
                title_text=f"Cumulative vs Valid vs Unique Valid Commands: {prompt} ({size_label})",
                template="plotly_white",
                barmode="group",
                height=700,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            
            output_path = os.path.join(output_subdir, f"{prompt}{filename_suffix}.html")
            write_html(fig, output_path)
            print(f"Created cumulative valid unique plot: {output_path}")
        
        # Create plots for both groups
        create_plot_for_models(models_gt_4b, ">4B and <10b", "_gt4b")
        create_plot_for_models(models_gt_10b, "<=10b", "_gt10b")
        create_plot_for_models(models_le_4b, "<=4B", "_le4b")


def create_cumulative_valid_unique_stacked_plots(csv_path: str, output_dir: str, model_to_is_coding: Optional[Dict[str, bool]] = None) -> None:
    """Create bar plots showing average cumulative_commands, valid_commands, and unique_valid_commands per model.
    
    For each prompt, creates a stacked bar chart with three segments per model:
    - Average Cumulative Commands (from last iteration, averaged across runs)
    - Average Valid Commands (sum across all iterations, averaged across runs)
    - Average Unique Valid Commands (from last iteration, averaged across runs)
    
    Creates 8 HTML files (one per prompt).
    
    Args:
        csv_path: Path to command_statistics CSV file
        output_dir: Base output directory (files will be saved in plots/cumulative_valid_unique_stacked subdirectory)
    """
    output_subdir = os.path.join(output_dir, "plots", "cumulative_valid_unique_stacked")
    os.makedirs(output_subdir, exist_ok=True)
    
    # Track total valid_commands per (prompt, run, model) - sum across all iterations
    total_valid_by_prompt_run_model: Dict[Tuple[str, int, str], int] = defaultdict(int)
    # Track last iteration's cumulative_commands per (prompt, run, model) - prefer iteration 50 or max
    last_cumulative_by_prompt_run_model: Dict[Tuple[str, int, str], Tuple[int, float]] = {}
    # Track last iteration's unique_valid_commands per (prompt, run, model) - prefer iteration 50 or max
    last_unique_valid_by_prompt_run_model: Dict[Tuple[str, int, str], Tuple[int, int]] = {}
    prompts: Set[str] = set()
    runs: Set[int] = set()
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row.get("prompt", "").strip()
                if not prompt:
                    continue
                
                try:
                    run_number = int(row.get("run_number", 0))
                except ValueError:
                    run_number = 0
                
                model = row.get("model", "").strip()
                if not model:
                    continue
                
                try:
                    iteration = int(row.get("iteration", 0))
                except ValueError:
                    iteration = 0
                
                try:
                    valid_commands = int(row.get("valid_commands", 0))
                except ValueError:
                    valid_commands = 0
                
                try:
                    cumulative_commands = float(row.get("cumulative_commands", 0))
                except ValueError:
                    cumulative_commands = 0.0
                
                try:
                    unique_valid_commands = int(row.get("unique_valid_commands", 0))
                except ValueError:
                    unique_valid_commands = 0
                
                key = (prompt, run_number, model)
                # Sum valid_commands across all iterations
        
                total_valid_by_prompt_run_model[key] += valid_commands
                
                # Track last iteration's cumulative_commands (prefer iteration 50 or max)
                if key not in last_cumulative_by_prompt_run_model:
                    last_cumulative_by_prompt_run_model[key] = (iteration, cumulative_commands)
                else:
                    prev_iter, _ = last_cumulative_by_prompt_run_model[key]
                    # prefer iteration 50 if seen, otherwise keep the max
                    if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                        last_cumulative_by_prompt_run_model[key] = (iteration, cumulative_commands)
                
                # Track last iteration's unique_valid_commands (prefer iteration 50 or max)
                if key not in last_unique_valid_by_prompt_run_model:
                    last_unique_valid_by_prompt_run_model[key] = (iteration, unique_valid_commands)
                else:
                    prev_iter, _ = last_unique_valid_by_prompt_run_model[key]
                    # prefer iteration 50 if seen, otherwise keep the max
                    if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                        last_unique_valid_by_prompt_run_model[key] = (iteration, unique_valid_commands)
                
                prompts.add(prompt)
                runs.add(run_number)
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return
    
    # Load model size mapping
    model_size_map = load_model_size_mapping()
    
    # Create one HTML file per prompt
    for prompt in sorted(prompts):
        # Get all models for this prompt
        models_set: Set[str] = set()
        for (p, r, m) in list(total_valid_by_prompt_run_model.keys()) + list(last_cumulative_by_prompt_run_model.keys()) + list(last_unique_valid_by_prompt_run_model.keys()):
            if p == prompt:
                models_set.add(m)
        
        models = sorted(models_set)
        if not models:
            continue
        
        # Split models by parameter size
        models_gt_4b = []
        models_le_4b = []
        
        for model in models:
            param_size = model_size_map.get(model, 0.0)
            if param_size > 4.0:
                models_gt_4b.append(model)
            elif param_size <= 4.0:
                models_le_4b.append(model)
            # Skip models not found in mapping
        
        # Helper function to create plot for a group of models
        def create_plot_for_models(model_list: List[str], size_label: str, filename_suffix: str) -> None:
            if not model_list:
                return
            # Calculate averages for each model
            avg_cumulative = []
            avg_total_valid = []
            avg_unique_valid = []
            
            for model in model_list:
                # Sum values across runs
                cumulative_sum = 0.0
                total_valid_sum = 0
                unique_valid_sum = 0
                run_count = 0
                
                for run in sorted(runs):
                    key = (prompt, run, model)
                    
                    # Count runs that have data
                    has_data = False
                    
                    # Add cumulative for this run (from last iteration)
                    if key in last_cumulative_by_prompt_run_model:
                        _, cum_val = last_cumulative_by_prompt_run_model[key]
                        cumulative_sum += cum_val
                        has_data = True
                    
                    # Add total valid for this run
                    if key in total_valid_by_prompt_run_model:
                        total_valid_sum += total_valid_by_prompt_run_model[key]
                        has_data = True
                    cumulative_sum -= total_valid_sum 
                    # Add unique valid for this run (from last iteration)
                    if key in last_unique_valid_by_prompt_run_model:
                        _, unique_val = last_unique_valid_by_prompt_run_model[key]
                        unique_valid_sum += unique_val
                        has_data = True
                    
                    if has_data:
                        run_count += 1
                
                # Calculate averages
                avg_cum = (cumulative_sum / run_count) if run_count > 0 else 0.0
                avg_valid = (total_valid_sum / run_count) if run_count > 0 else 0.0
                avg_unique = (unique_valid_sum / run_count) if run_count > 0 else 0.0
                
                avg_cumulative.append(avg_cum)
                avg_total_valid.append(avg_valid)
                avg_unique_valid.append(avg_unique)
            
            # Create stacked bar chart
            fig = go.Figure()
            
            bar_colors = bar_colors_for_models(model_list, model_to_is_coding) if (model_to_is_coding and model_list) else None
            kw1: Dict[str, object] = dict(name="Avg Non-valid Commands", x=model_list, y=avg_cumulative, text=[f"{v:.2f}" for v in avg_cumulative], textposition="inside", hovertemplate="Model=%{x}<br>Avg Non-Valid Commands=%{y:.2f}<extra></extra>")
            if bar_colors:
                kw1["marker_color"] = bar_colors
            fig.add_trace(go.Bar(**kw1))
            
            kw2: Dict[str, object] = dict(name="Avg Valid Commands", x=model_list, y=avg_total_valid, text=[f"{v:.2f}" for v in avg_total_valid], textposition="inside", hovertemplate="Model=%{x}<br>Avg Valid Commands (sum across all iterations)=%{y:.2f}<extra></extra>")
            if bar_colors:
                kw2["marker_color"] = bar_colors
            fig.add_trace(go.Bar(**kw2))
            
            kw3: Dict[str, object] = dict(name="Avg Unique Valid Commands", x=model_list, y=avg_unique_valid, text=[f"{v:.2f}" for v in avg_unique_valid], textposition="inside", hovertemplate="Model=%{x}<br>Avg Unique Valid Commands (from last iteration)=%{y:.2f}<extra></extra>")
            if bar_colors:
                kw3["marker_color"] = bar_colors
            fig.add_trace(go.Bar(**kw3))
            
            fig.update_xaxes(title_text="Model", tickangle=-45)
            fig.update_yaxes(title_text="Average Commands")
            fig.update_layout(
                title_text=f"Cumulative vs Valid vs Unique Valid Commands (Stacked): {prompt} ({size_label})",
                template="plotly_white",
                barmode="stack",
                height=700,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            
            output_path = os.path.join(output_subdir, f"{prompt}{filename_suffix}.html")
            write_html(fig, output_path)
            print(f"Created cumulative valid unique stacked plot: {output_path}")
        
        # Create plots for both groups
        create_plot_for_models(models_gt_4b, ">4B", "_gt4b")
        create_plot_for_models(models_le_4b, "<=4B", "_le4b")


def create_base_commands_bar_plots(csv_path: str, output_dir: str, model_to_is_coding: Optional[Dict[str, bool]] = None) -> None:
    """Create per-prompt folders containing bar charts of final base commands seen per model.

    For each prompt, create a subdirectory under output_dir/base_commands_per_model/<prompt>/
    and write an index.html that contains subplots (one per run). Each bar is a model;
    bar height = number of base commands seen so far at the final iteration (prefer iteration 50 or max),
    hover shows the final set of base commands for that model/run.
    """
    base_dir = os.path.join(output_dir, "base_commands_per_model")
    os.makedirs(base_dir, exist_ok=True)
    param_size_map = load_model_size_mapping()

    # Map (prompt, run, model) -> (iteration, set_string)
    final_base_by_prompt_run_model: Dict[Tuple[str, int, str], Tuple[int, str]] = {}
    prompts: Set[str] = set()
    runs: Set[int] = set()

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row.get("prompt", "").strip()
                if not prompt:
                    continue
                try:
                    run_number = int(row.get("run_number", 0))
                except Exception:
                    run_number = 0
                try:
                    iteration = int(row.get("iteration", 0))
                except Exception:
                    iteration = 0
                model = row.get("model", "").strip()
                if not model:
                    continue

                # read raw set string
                base_set_str = row.get("base_commands_seen_so_far", "")

                key = (prompt, run_number, model)
                # prefer exact iteration 50; otherwise take the max iteration
                if key not in final_base_by_prompt_run_model:
                    final_base_by_prompt_run_model[key] = (iteration, base_set_str)
                else:
                    prev_iter, _ = final_base_by_prompt_run_model[key]
                    # prefer iteration 50 if seen, otherwise keep the max
                    if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                        final_base_by_prompt_run_model[key] = (iteration, base_set_str)

                prompts.add(prompt)
                runs.add(run_number)
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return

    # For each prompt, create a folder and a single HTML with subplots per run
    for prompt in sorted(prompts):
        prompt_runs = sorted([r for r in runs if any(p == prompt for p, _, _ in final_base_by_prompt_run_model.keys())])
        if not prompt_runs:
            continue

        # create figure with one row per run
        fig = make_subplots(rows=len(prompt_runs), cols=1, subplot_titles=[f"{prompt} - Run {r}" for r in prompt_runs], vertical_spacing=0.12)

        for row_idx, run_number in enumerate(prompt_runs, start=1):
            # collect models and counts
            models = []
            counts = []
            hover_texts = []

            for (p, r, model), (iter_val, set_str) in final_base_by_prompt_run_model.items():
                if p != prompt or r != run_number:
                    continue
                # parse set_str to count elements
                s = (set_str or "").strip()
                if s == "set()" or s == "{}" or s == "":
                    parsed = set()
                else:
                    # extract quoted items
                    items = re.findall(r"'([^']*)'|\"([^\"]*)\"", s)
                    parsed = set([a or b for a, b in items])
                
                models.append(model)
                counts.append(len(parsed))
                # show the original string (or a cleaned join) in hover
                hover_texts.append(
                    ", ".join(sorted(parsed)) if parsed else "set()"
                )

            # ensure consistent ordering by model size (ascending), then alphabetically for ties
            if models:
                order = sorted(range(len(models)), key=lambda i: (param_size_map.get(models[i], float('inf')), models[i]))
                models_sorted = [models[i] for i in order]
                counts_sorted = [counts[i] for i in order]
                hover_sorted = [hover_texts[i] for i in order]
            else:
                models_sorted = []
                counts_sorted = []
                hover_sorted = []

            bar_kw: Dict[str, object] = dict(
                x=models_sorted,
                y=counts_sorted,
                text=counts_sorted if counts_sorted else [],
                textposition="outside",
                customdata=[[h] for h in hover_sorted],
                hovertemplate="Model=%{x}<br># Base Commands=%{y}<br>Base set=%{customdata[0]}<extra></extra>",
                showlegend=False,
            )
            if model_to_is_coding and models_sorted:
                bar_kw["marker_color"] = bar_colors_for_models(models_sorted, model_to_is_coding)
            fig.add_trace(
                go.Bar(**bar_kw),
                row=row_idx,
                col=1,
            )

            fig.update_xaxes(tickangle=-45, row=row_idx, col=1)

        fig.update_layout(
            title_text=f"Final Base Commands Seen per Model: {prompt}",
            template="plotly_white",
            height=500 * len(prompt_runs),
        )

        # make prompt-specific folder and write index.html
        prompt_dir = os.path.join(base_dir, prompt)
        os.makedirs(prompt_dir, exist_ok=True)
        output_path = os.path.join(prompt_dir, "index.html")
        write_html(fig, output_path)
        print(f"Created base-commands bar charts: {output_path}")


def create_average_base_commands_plots(csv_path: str, output_dir: str, model_to_is_coding: Optional[Dict[str, bool]] = None) -> None:
    """Create bar charts showing average number of base commands per model (averaged across 3 runs).
    
    For each prompt, creates a single bar chart showing:
    - X-axis: models (sorted)
    - Y-axis: average number of base commands (across 3 runs)
    - Hover: model name and total base set (union across all 3 runs)
    
    Creates 8 HTML files (one per prompt).
    
    Args:
        csv_path: Path to command_statistics CSV file
        output_dir: Base output directory (files will be saved in base_commands_per_model subdirectory)
    """
    base_dir = os.path.join(output_dir, "base_commands_per_model")
    os.makedirs(base_dir, exist_ok=True)
    param_size_map = load_model_size_mapping()
    
    # Map (prompt, run, model) -> (iteration, set_string)
    final_base_by_prompt_run_model: Dict[Tuple[str, int, str], Tuple[int, str]] = {}
    prompts: Set[str] = set()
    runs: Set[int] = set()
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row.get("prompt", "").strip()
                if not prompt:
                    continue
                try:
                    run_number = int(row.get("run_number", 0))
                except Exception:
                    run_number = 0
                try:
                    iteration = int(row.get("iteration", 0))
                except Exception:
                    iteration = 0
                model = row.get("model", "").strip()
                if not model:
                    continue
                
                # read raw set string
                base_set_str = row.get("base_commands_seen_so_far", "")
                
                key = (prompt, run_number, model)
                # prefer exact iteration 50; otherwise take the max iteration
                if key not in final_base_by_prompt_run_model:
                    final_base_by_prompt_run_model[key] = (iteration, base_set_str)
                else:
                    prev_iter, _ = final_base_by_prompt_run_model[key]
                    # prefer iteration 50 if seen, otherwise keep the max
                    if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                        final_base_by_prompt_run_model[key] = (iteration, base_set_str)
                
                prompts.add(prompt)
                runs.add(run_number)
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return
    
    # Helper function to parse set string
    def parse_base_set(set_str: str) -> Set[str]:
        """Parse base_commands_seen_so_far string into a set of commands."""
        s = (set_str or "").strip()
        if s == "set()" or s == "{}" or s == "":
            return set()
        else:
            # extract quoted items
            items = re.findall(r"'([^']*)'|\"([^\"]*)\"", s)
            return set([a or b for a, b in items])
    
    # For each prompt, calculate averages and create bar chart
    for prompt in sorted(prompts):
        # Get all models for this prompt
        models_set: Set[str] = set()
        for (p, r, m) in final_base_by_prompt_run_model.keys():
            if p == prompt:
                models_set.add(m)
        
        models = sorted(models_set, key=lambda m: (param_size_map.get(m, float('inf')), m))
        if not models:
            continue
        
        # Calculate averages and total sets for each model
        avg_counts = []
        total_sets = []
        
        for model in models:
            # Collect base command sets and counts for all runs
            run_counts = []
            all_base_commands = set()
            
            for run in sorted(runs):
                key = (prompt, run, model)
                if key in final_base_by_prompt_run_model:
                    _, set_str = final_base_by_prompt_run_model[key]
                    parsed_set = parse_base_set(set_str)
                    run_counts.append(len(parsed_set))
                    all_base_commands.update(parsed_set)
            
            # Calculate average across runs
            if run_counts:
                avg_count = sum(run_counts) / len(run_counts)
            else:
                avg_count = 0.0
            
            avg_counts.append(avg_count)
            
            # Format total base set for hover (sorted, comma-separated)
            if all_base_commands:
                total_set_str = ", ".join(sorted(all_base_commands))
            else:
                total_set_str = "set()"
            total_sets.append(total_set_str)
        
        # Create bar chart
        fig = go.Figure()
        bar_kw: Dict[str, object] = dict(
            x=models,
            y=avg_counts,
            text=[f"{v:.2f}" for v in avg_counts],
            textposition="outside",
            customdata=[[ts] for ts in total_sets],
            hovertemplate=(
                "Model=%{x}<br>"
                "Avg Base Commands=%{y:.2f}<br>"
                "Total Base Set (across all runs)=%{customdata[0]}<extra></extra>"
            ),
            showlegend=False,
        )
        if model_to_is_coding and models:
            bar_kw["marker_color"] = bar_colors_for_models(models, model_to_is_coding)
        fig.add_trace(go.Bar(**bar_kw))
        
        fig.update_xaxes(title_text="Model", tickangle=-45)
        fig.update_yaxes(title_text="Average Number of Base Commands (Across all Runs)")
        fig.update_layout(
            title_text=f"Average Base Commands per Model: {prompt}",
            template="plotly_white",
            height=700,
        )
        
        # Write HTML file
        output_path = os.path.join(base_dir, f"{prompt}_average.html")
        write_html(fig, output_path)
        print(f"Created average base-commands plot: {output_path}")


def create_base_commands_comparison_plot(csv_path: str, output_dir: str, model_to_is_coding: Optional[Dict[str, bool]] = None) -> None:
    """Create a grouped bar chart comparing average base commands for prompt2, prompt6, and prompt7.
    
    Creates a single HTML file with a grouped bar chart showing:
    - X-axis: models (sorted)
    - Y-axis: average number of base commands (across all runs)
    - 3 bars per model: one for prompt2, one for prompt6, one for prompt7
    
    Args:
        csv_path: Path to command_statistics CSV file
        output_dir: Base output directory (file will be saved in plots/base_commands_comparison subdirectory)
    """
    output_subdir = os.path.join(output_dir, "plots", "base_commands_comparison")
    os.makedirs(output_subdir, exist_ok=True)
    
    # Target prompts
    target_prompts = {"prompt2", "prompt6", "prompt7"}
    
    # Map (prompt, run, model) -> (iteration, set_string)
    final_base_by_prompt_run_model: Dict[Tuple[str, int, str], Tuple[int, str]] = {}
    prompts: Set[str] = set()
    runs: Set[int] = set()
    
    # Helper function to parse set string
    def parse_base_set(set_str: str) -> Set[str]:
        """Parse base_commands_seen_so_far string into a set of commands."""
        s = (set_str or "").strip()
        if s == "set()" or s == "{}" or s == "":
            return set()
        else:
            # extract quoted items
            items = re.findall(r"'([^']*)'|\"([^\"]*)\"", s)
            return set([a or b for a, b in items])
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row.get("prompt", "").strip()
                # Filter for only target prompts
                if prompt not in target_prompts:
                    continue
                
                try:
                    run_number = int(row.get("run_number", 0))
                except Exception:
                    run_number = 0
                try:
                    iteration = int(row.get("iteration", 0))
                except Exception:
                    iteration = 0
                model = row.get("model", "").strip()
                if not model:
                    continue
                
                # read raw set string
                base_set_str = row.get("base_commands_seen_so_far", "")
                
                key = (prompt, run_number, model)
                # prefer exact iteration 50; otherwise take the max iteration
                if key not in final_base_by_prompt_run_model:
                    final_base_by_prompt_run_model[key] = (iteration, base_set_str)
                else:
                    prev_iter, _ = final_base_by_prompt_run_model[key]
                    # prefer iteration 50 if seen, otherwise keep the max
                    if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                        final_base_by_prompt_run_model[key] = (iteration, base_set_str)
                
                prompts.add(prompt)
                runs.add(run_number)
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return
    
    # Get all models that appear in any of the target prompts
    models_set: Set[str] = set()
    for (p, r, m) in final_base_by_prompt_run_model.keys():
        if p in target_prompts:
            models_set.add(m)
    
    models = sorted(models_set)
    if not models:
        print("No models found for prompt2, prompt6, or prompt7.")
        return
    
    # Calculate averages for each model and each prompt
    avg_prompt2 = []
    avg_prompt6 = []
    avg_prompt7 = []
    
    for model in models:
        # Calculate average for prompt2
        prompt7_counts = []
        for run in sorted(runs):
            key = ("prompt7", run, model)
            if key in final_base_by_prompt_run_model:
                _, set_str = final_base_by_prompt_run_model[key]
                parsed_set = parse_base_set(set_str)
                prompt7_counts.append(len(parsed_set))
        avg_prompt7.append(sum(prompt7_counts) / len(prompt7_counts) if prompt7_counts else 0.0)
    
        prompt2_counts = []
        for run in sorted(runs):
            key = ("prompt2", run, model)
            if key in final_base_by_prompt_run_model:
                _, set_str = final_base_by_prompt_run_model[key]
                parsed_set = parse_base_set(set_str)
                prompt2_counts.append(len(parsed_set))
        avg_prompt2.append(sum(prompt2_counts) / len(prompt2_counts) if prompt2_counts else 0.0)
        
        # Calculate average for prompt6
        prompt6_counts = []
        for run in sorted(runs):
            key = ("prompt6", run, model)
            if key in final_base_by_prompt_run_model:
                _, set_str = final_base_by_prompt_run_model[key]
                parsed_set = parse_base_set(set_str)
                prompt6_counts.append(len(parsed_set))
        avg_prompt6.append(sum(prompt6_counts) / len(prompt6_counts) if prompt6_counts else 0.0)
        
        # Calculate average for prompt7
    # Create grouped bar chart
    fig = go.Figure()
    bar_colors = bar_colors_for_models(models, model_to_is_coding) if (model_to_is_coding and models) else None
    kw1: Dict[str, object] = dict(name="Zero-Shot (prompt7)", x=models, y=avg_prompt7, text=[f"{v:.2f}" for v in avg_prompt7], textposition="outside", hovertemplate="Model=%{x}<br>Prompt=Zero-shot<br>Avg Base Commands=%{y:.2f}<extra></extra>")
    if bar_colors:
        kw1["marker_color"] = bar_colors
    fig.add_trace(go.Bar(**kw1))
    kw2: Dict[str, object] = dict(name="1-shot (prompt2)", x=models, y=avg_prompt2, text=[f"{v:.2f}" for v in avg_prompt2], textposition="outside", hovertemplate="Model=%{x}<br>Prompt=1-shot<br>Avg Base Commands=%{y:.2f}<extra></extra>")
    if bar_colors:
        kw2["marker_color"] = bar_colors
    fig.add_trace(go.Bar(**kw2))
    kw3: Dict[str, object] = dict(name="2-shot (prompt6)", x=models, y=avg_prompt6, text=[f"{v:.2f}" for v in avg_prompt6], textposition="outside", hovertemplate="Model=%{x}<br>Prompt=2-shot<br>Avg Base Commands=%{y:.2f}<extra></extra>")
    if bar_colors:
        kw3["marker_color"] = bar_colors
    fig.add_trace(go.Bar(**kw3))
    
    fig.update_xaxes(title_text="Model", tickangle=-45)
    fig.update_yaxes(title_text="Average Number of Base Commands (Across All Runs)")
    fig.update_layout(
        title_text="Average Base Commands Comparison: zero-shot, one-shot, two-shot",
        template="plotly_white",
        barmode="group",
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    output_path = os.path.join(output_subdir, "prompt2_prompt6_prompt7_comparison.html")
    write_html(fig, output_path)
    print(f"Created base commands comparison plot: {output_path}")


def create_shot_type_comparison_plot(csv_path: str, output_dir: str, model_to_is_coding: Optional[Dict[str, bool]] = None) -> None:
    """Create a grouped bar chart comparing average base commands for 0-shot, 1-shot, and 2-shot prompting.
    
    Groups prompts by shot type:
    - 0-shot: prompt5, prompt7, security-incremental-1 (if exists)
    - 1-shot: prompt1, prompt2, prompt3, prompt4
    - 2-shot: prompt6
    
    For each model, collects all base command counts from all runs and all prompts in each group,
    then averages those values to get a single average per shot type.
    
    Creates a single HTML file with a grouped bar chart showing:
    - X-axis: models (sorted)
    - Y-axis: average number of base commands (across all runs and prompts in group)
    - 3 bars per model: one for 0-shot, one for 1-shot, one for 2-shot
    
    Args:
        csv_path: Path to command_statistics CSV file
        output_dir: Base output directory (file will be saved in plots/shot_type_comparison subdirectory)
    """
    output_subdir = os.path.join(output_dir, "plots", "shot_type_comparison")
    os.makedirs(output_subdir, exist_ok=True)
    
    # Define prompt groups by shot type
    zero_shot_prompts = {"prompt5", "prompt7", "security-incremental-1"}
    one_shot_prompts = {"prompt1", "prompt2", "prompt3", "prompt4"}
    two_shot_prompts = {"prompt6"}
    
    # Map (prompt, run, model) -> (iteration, set_string)
    final_base_by_prompt_run_model: Dict[Tuple[str, int, str], Tuple[int, str]] = {}
    prompts: Set[str] = set()
    runs: Set[int] = set()
    
    # Helper function to parse set string
    def parse_base_set(set_str: str) -> Set[str]:
        """Parse base_commands_seen_so_far string into a set of commands."""
        s = (set_str or "").strip()
        if s == "set()" or s == "{}" or s == "":
            return set()
        else:
            # extract quoted items
            items = re.findall(r"'([^']*)'|\"([^\"]*)\"", s)
            return set([a or b for a, b in items])
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row.get("prompt", "").strip()
                if not prompt:
                    continue
                
                try:
                    run_number = int(row.get("run_number", 0))
                except Exception:
                    run_number = 0
                try:
                    iteration = int(row.get("iteration", 0))
                except Exception:
                    iteration = 0
                model = row.get("model", "").strip()
                if not model:
                    continue
                
                # read raw set string
                base_set_str = row.get("base_commands_seen_so_far", "")
                
                key = (prompt, run_number, model)
                # prefer exact iteration 50; otherwise take the max iteration
                if key not in final_base_by_prompt_run_model:
                    final_base_by_prompt_run_model[key] = (iteration, base_set_str)
                else:
                    prev_iter, _ = final_base_by_prompt_run_model[key]
                    # prefer iteration 50 if seen, otherwise keep the max
                    if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                        final_base_by_prompt_run_model[key] = (iteration, base_set_str)
                
                prompts.add(prompt)
                runs.add(run_number)
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return
    
    # Get all models that appear in any of the target prompts
    models_set: Set[str] = set()
    all_target_prompts = zero_shot_prompts | one_shot_prompts | two_shot_prompts
    for (p, r, m) in final_base_by_prompt_run_model.keys():
        if p in all_target_prompts:
            models_set.add(m)
    
    models = sorted(models_set)
    if not models:
        print("No models found for shot type comparison prompts.")
        return
    
    # Load model size mapping
    model_size_map = load_model_size_mapping()
    
    # Split models by parameter size
    models_le_4b = []
    models_gt_4b = []
    
    for model in models:
        param_size = model_size_map.get(model, 0.0)
        if param_size <= 4.0:
            models_le_4b.append(model)
        elif param_size > 4.0:
            models_gt_4b.append(model)
        # Skip models not found in mapping
    
    # Helper function to create plot for a group of models
    def create_plot_for_models(model_list: List[str], size_label: str, filename_suffix: str) -> None:
        if not model_list:
            return
        
        # Calculate averages for each model and each shot type group
        # For each model, collect ALL base command counts from ALL runs and ALL prompts in each group,
        # then average all those values together
        avg_zero_shot = []
        avg_one_shot = []
        avg_two_shot = []
        
        for model in model_list:
            # Collect all base command counts for 0-shot group (all runs, all prompts in group)
            zero_shot_counts = []
            for prompt in zero_shot_prompts:
                for run in sorted(runs):
                    key = (prompt, run, model)
                    if key in final_base_by_prompt_run_model:
                        _, set_str = final_base_by_prompt_run_model[key]
                        parsed_set = parse_base_set(set_str)
                        zero_shot_counts.append(len(parsed_set))
            
            # Calculate average across all collected values
            avg_zero = sum(zero_shot_counts) / len(zero_shot_counts) if zero_shot_counts else 0.0
            avg_zero_shot.append(avg_zero)
            
            # Collect all base command counts for 1-shot group (all runs, all prompts in group)
            one_shot_counts = []
            for prompt in one_shot_prompts:
                for run in sorted(runs):
                    key = (prompt, run, model)
                    if key in final_base_by_prompt_run_model:
                        _, set_str = final_base_by_prompt_run_model[key]
                        parsed_set = parse_base_set(set_str)
                        one_shot_counts.append(len(parsed_set))
            
            # Calculate average across all collected values
            avg_one = sum(one_shot_counts) / len(one_shot_counts) if one_shot_counts else 0.0
            avg_one_shot.append(avg_one)
            
            # Collect all base command counts for 2-shot group (all runs, all prompts in group)
            two_shot_counts = []
            for prompt in two_shot_prompts:
                for run in sorted(runs):
                    key = (prompt, run, model)
                    if key in final_base_by_prompt_run_model:
                        _, set_str = final_base_by_prompt_run_model[key]
                        parsed_set = parse_base_set(set_str)
                        two_shot_counts.append(len(parsed_set))
            
            # Calculate average across all collected values
            avg_two = sum(two_shot_counts) / len(two_shot_counts) if two_shot_counts else 0.0
            avg_two_shot.append(avg_two)
        
        # Create grouped bar chart
        fig = go.Figure()
        bar_colors = bar_colors_for_models(model_list, model_to_is_coding) if (model_to_is_coding and model_list) else None
        kw1: Dict[str, object] = dict(name="0-shot", x=model_list, y=avg_zero_shot, text=[f"{v:.2f}" for v in avg_zero_shot], textposition="outside", hovertemplate="Model=%{x}<br>Shot Type=0-shot<br>Avg Base Commands=%{y:.2f}<extra></extra>")
        if bar_colors:
            kw1["marker_color"] = bar_colors
        fig.add_trace(go.Bar(**kw1))
        kw2: Dict[str, object] = dict(name="1-shot", x=model_list, y=avg_one_shot, text=[f"{v:.2f}" for v in avg_one_shot], textposition="outside", hovertemplate="Model=%{x}<br>Shot Type=1-shot<br>Avg Base Commands=%{y:.2f}<extra></extra>")
        if bar_colors:
            kw2["marker_color"] = bar_colors
        fig.add_trace(go.Bar(**kw2))
        kw3: Dict[str, object] = dict(name="2-shot", x=model_list, y=avg_two_shot, text=[f"{v:.2f}" for v in avg_two_shot], textposition="outside", hovertemplate="Model=%{x}<br>Shot Type=2-shot<br>Avg Base Commands=%{y:.2f}<extra></extra>")
        if bar_colors:
            kw3["marker_color"] = bar_colors
        fig.add_trace(go.Bar(**kw3))
        
        fig.update_xaxes(title_text="Model", tickangle=-45)
        fig.update_yaxes(title_text="Average Number of Base Commands (Across All Runs and Prompts in Group)")
        fig.update_layout(
            title_text=f"Shot Type Comparison: Average Base Commands (0-shot vs 1-shot vs 2-shot) ({size_label})",
            template="plotly_white",
            barmode="group",
            height=700,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        
        output_path = os.path.join(output_subdir, f"shot_type_comparison{filename_suffix}.html")
        write_html(fig, output_path)
        print(f"Created shot type comparison plot: {output_path}")
    
    # Create plots for both groups
    create_plot_for_models(models_le_4b, "<=4B", "_le4b")
    create_plot_for_models(models_gt_4b, ">4B", "_gt4b")


def create_cumulative_time_series(csv_path: str, output_dir: str) -> None:
    """Create cumulative time series plots averaged across runs.
    
    Creates 8 HTML files (one per prompt), each showing average cumulative commands
    across 3 runs for each model over iterations.
    
    Args:
        csv_path: Path to command_statistics.csv
        output_dir: Base output directory (files will be saved in cumulative_time_series subdirectory)
    """
    cumulative_dir = os.path.join(output_dir, "cumulative_time_series")
    os.makedirs(cumulative_dir, exist_ok=True)
    
    # Read CSV and group by (prompt, model, iteration, run_number)
    data_by_prompt_model_iter_run: Dict[Tuple[str, str, int, int], float] = {}
    prompts = set()
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row.get("prompt", "").strip()
                model = row.get("model", "").strip()
                iteration = int(row.get("iteration", 0))
                run_number = int(row.get("run_number", 0))
                cumulative_commands = float(row.get("cumulative_commands", 0))
                
                if prompt and model and iteration > 0:
                    data_by_prompt_model_iter_run[(prompt, model, iteration, run_number)] = cumulative_commands
                    prompts.add(prompt)
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return
    
    # Create one HTML file per prompt
    for prompt in sorted(prompts):
        # Group by (model, iteration) and calculate averages
        by_model_iter: Dict[Tuple[str, int], List[float]] = defaultdict(list)
        models = set()
        iterations = set()
        
        for (p, model, iteration, run_number), cumulative in data_by_prompt_model_iter_run.items():
            if p == prompt:
                by_model_iter[(model, iteration)].append(cumulative)
                models.add(model)
                iterations.add(iteration)
        
        # Create figure
        fig = go.Figure()
        
        # Add trace for each model
        for model in sorted(models):
            # Get all iterations for this model
            model_iterations = sorted([it for mod, it in by_model_iter.keys() if mod == model])
            
            if not model_iterations:
                continue
            
            # Calculate average for each iteration
            x_vals = []
            y_vals = []
            for iteration in sorted(iterations):
                key = (model, iteration)
                if key in by_model_iter:
                    values = by_model_iter[key]
                    avg_cumulative = sum(values) / len(values) if values else 0.0
                    x_vals.append(iteration)
                    y_vals.append(avg_cumulative)
                else:
                    # Forward-fill: use previous value if available
                    if y_vals:
                        x_vals.append(iteration)
                        y_vals.append(y_vals[-1])
            
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines+markers",
                    name=str(model),
                    hovertemplate=(
                        "Model=%{text}<br>Iteration=%{x}<br>Avg Cumulative Commands=%{y:.2f}<extra></extra>"
                    ),
                    text=[str(model)] * len(x_vals),
                )
            )
        
        fig.update_layout(
            title=f"Average Cumulative Commands by Model: {prompt}",
            xaxis_title="Iteration",
            yaxis_title="Average Cumulative Commands (Across Runs)",
            legend_title="Model",
            template="plotly_white",
            height=800,
        )
        
        # Write HTML file
        output_path = os.path.join(cumulative_dir, f"{prompt}.html")
        write_html(fig, output_path)
        print(f"Created cumulative time series: {output_path}")
    
    # Create combined file with all prompts
    sorted_prompts = sorted(prompts)
    if len(sorted_prompts) > 0:
        # Create subplots: 4 rows x 2 columns for 8 prompts
        fig = make_subplots(
            rows=4,
            cols=2,
            subplot_titles=[prompt for prompt in sorted_prompts],
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )
        
        # Add traces for each prompt
        for prompt_idx, prompt in enumerate(sorted_prompts, start=1):
            # Calculate subplot position
            row = (prompt_idx - 1) // 2 + 1
            col = (prompt_idx - 1) % 2 + 1
            
            # Group by (model, iteration) and calculate averages
            by_model_iter: Dict[Tuple[str, int], List[float]] = defaultdict(list)
            models = set()
            iterations = set()
            
            for (p, model, iteration, run_number), cumulative in data_by_prompt_model_iter_run.items():
                if p == prompt:
                    by_model_iter[(model, iteration)].append(cumulative)
                    models.add(model)
                    iterations.add(iteration)
            
            # Add trace for each model
            for model in sorted(models):
                # Calculate average for each iteration
                x_vals = []
                y_vals = []
                for iteration in sorted(iterations):
                    key = (model, iteration)
                    if key in by_model_iter:
                        values = by_model_iter[key]
                        avg_cumulative = sum(values) / len(values) if values else 0.0
                        x_vals.append(iteration)
                        y_vals.append(avg_cumulative)
                    else:
                        # Forward-fill: use previous value if available
                        if y_vals:
                            x_vals.append(iteration)
                            y_vals.append(y_vals[-1])
                
                # Only show legend on first subplot
                show_legend = (prompt_idx == 1)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="lines+markers",
                        name=str(model),
                        hovertemplate=(
                            "Model=%{text}<br>Iteration=%{x}<br>Avg Cumulative Commands=%{y:.2f}<extra></extra>"
                        ),
                        text=[str(model)] * len(x_vals),
                        showlegend=show_legend,
                        legendgroup=str(model),
                    ),
                    row=row,
                    col=col,
                )
        
        # Update axes labels
        fig.update_xaxes(title_text="Iteration", row=4, col=1)
        fig.update_xaxes(title_text="Iteration", row=4, col=2)
        fig.update_yaxes(title_text="Avg Cumulative Commands", row=2, col=1)
        
        fig.update_layout(
            title_text="Average Cumulative Commands by Model: All Prompts",
            template="plotly_white",
            height=2000,  # Tall enough for 4 rows
            legend_title="Model",
        )
        
        # Write combined HTML file
        combined_output_path = os.path.join(cumulative_dir, "all_prompts.html")
        write_html(fig, combined_output_path)
        print(f"Created combined cumulative time series: {combined_output_path}")


def create_per_iteration_time_series(csv_path: str, output_dir: str) -> None:
    """Create per-iteration time series plots averaged across runs.
    
    Creates 8 HTML files (one per prompt), each showing average number_of_commands
    across 3 runs for each of the 79 models over iterations.
    
    Args:
        csv_path: Path to command_statistics.csv
        output_dir: Base output directory (files will be saved in per_iteration_time_series subdirectory)
    """
    per_iter_dir = os.path.join(output_dir, "per_iteration_time_series")
    os.makedirs(per_iter_dir, exist_ok=True)
    
    # First, get all unique models from CSV
    all_models = set()
    prompts = set()
    data_by_prompt_model_iter_run: Dict[Tuple[str, str, int, int], float] = {}
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row.get("prompt", "").strip()
                model = row.get("model", "").strip()
                iteration = int(row.get("iteration", 0))
                run_number = int(row.get("run_number", 0))
                number_of_commands = float(row.get("number_of_commands", 0))
                
                if prompt and model and iteration > 0:
                    all_models.add(model)
                    prompts.add(prompt)
                    data_by_prompt_model_iter_run[(prompt, model, iteration, run_number)] = number_of_commands
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return
    
    # Create one HTML file per prompt
    for prompt in sorted(prompts):
        # Group by (model, iteration) and calculate averages
        by_model_iter: Dict[Tuple[str, int], List[float]] = defaultdict(list)
        iterations = set()
        
        for (p, model, iteration, run_number), num_commands in data_by_prompt_model_iter_run.items():
            if p == prompt:
                by_model_iter[(model, iteration)].append(num_commands)
                iterations.add(iteration)
        
        # Create figure
        fig = go.Figure()
        
        # Add trace for each model (all 79 models, even if some have 0 commands)
        for model in sorted(all_models):
            # Get all iterations for this model in this prompt
            model_iterations = sorted([it for mod, it in by_model_iter.keys() if mod == model])
            
            # Calculate average for each iteration
            x_vals = []
            y_vals = []
            for iteration in sorted(iterations):
                key = (model, iteration)
                if key in by_model_iter:
                    values = by_model_iter[key]
                    avg_commands = sum(values) / len(values) if values else 0.0
                else:
                    # Model didn't have data for this iteration, use 0
                    avg_commands = 0.0
                
                x_vals.append(iteration)
                y_vals.append(avg_commands)
            
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines+markers",
                    name=str(model),
                    hovertemplate=(
                        "Model=%{text}<br>Iteration=%{x}<br>Avg Commands=%{y:.2f}<extra></extra>"
                    ),
                    text=[str(model)] * len(x_vals),
                )
            )
        
        fig.update_layout(
            title=f"Average Commands per Iteration by Model: {prompt}",
            xaxis_title="Iteration",
            yaxis_title="Average Number of Commands (Across Runs)",
            legend_title="Model",
            template="plotly_white",
            height=1000,
        )
        
        # Write HTML file
        output_path = os.path.join(per_iter_dir, f"{prompt}.html")
        write_html(fig, output_path)
        print(f"Created per-iteration time series: {output_path}")
    
    # Create combined file with all prompts
    sorted_prompts = sorted(prompts)
    if len(sorted_prompts) > 0:
        # Create subplots: 4 rows x 2 columns for 8 prompts
        fig = make_subplots(
            rows=4,
            cols=2,
            subplot_titles=[prompt for prompt in sorted_prompts],
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )
        
        # Add traces for each prompt
        for prompt_idx, prompt in enumerate(sorted_prompts, start=1):
            # Calculate subplot position
            row = (prompt_idx - 1) // 2 + 1
            col = (prompt_idx - 1) % 2 + 1
            
            # Group by (model, iteration) and calculate averages
            by_model_iter: Dict[Tuple[str, int], List[float]] = defaultdict(list)
            iterations = set()
            
            for (p, model, iteration, run_number), num_commands in data_by_prompt_model_iter_run.items():
                if p == prompt:
                    by_model_iter[(model, iteration)].append(num_commands)
                    iterations.add(iteration)
            
            # Add trace for each model (all models, even if some have 0 commands)
            for model in sorted(all_models):
                # Calculate average for each iteration
                x_vals = []
                y_vals = []
                for iteration in sorted(iterations):
                    key = (model, iteration)
                    if key in by_model_iter:
                        values = by_model_iter[key]
                        avg_commands = sum(values) / len(values) if values else 0.0
                    else:
                        # Model didn't have data for this iteration, use 0
                        avg_commands = 0.0
                    
                    x_vals.append(iteration)
                    y_vals.append(avg_commands)
                
                # Only show legend on first subplot
                show_legend = (prompt_idx == 1)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="lines+markers",
                        name=str(model),
                        hovertemplate=(
                            "Model=%{text}<br>Iteration=%{x}<br>Avg Commands=%{y:.2f}<extra></extra>"
                        ),
                        text=[str(model)] * len(x_vals),
                        showlegend=show_legend,
                        legendgroup=str(model),
                    ),
                    row=row,
                    col=col,
                )
        
        # Update axes labels
        fig.update_xaxes(title_text="Iteration", row=4, col=1)
        fig.update_xaxes(title_text="Iteration", row=4, col=2)
        fig.update_yaxes(title_text="Avg Commands", row=2, col=1)
        
        fig.update_layout(
            title_text="Average Commands per Iteration by Model: All Prompts",
            template="plotly_white",
            height=2000,  # Tall enough for 4 rows
            legend_title="Model",
        )
        
        # Write combined HTML file
        combined_output_path = os.path.join(per_iter_dir, "all_prompts.html")
        write_html(fig, combined_output_path)
        print(f"Created combined per-iteration time series: {combined_output_path}")


def create_model_analysis_text(csv_path: str, output_dir: str) -> None:
    """Create a text file analyzing models based on validity ratio and base commands.
    
    For each prompt, identifies models that meet:
    - Validity ratio > 30% (sum valid_commands / sum cumulative_commands across runs)
    - Average base commands > 4 (average count across runs)
    - Both criteria (separate category)
    
    Args:
        csv_path: Path to command_statistics CSV file
        output_dir: Base output directory (file will be saved as analysis_of_models.txt)
    """
    output_file = os.path.join(output_dir, "analysis_of_models.txt")
    
    # Track total valid commands per (prompt, run_number, model) - sum across all iterations
    total_valid_by_prompt_run_model: Dict[Tuple[str, int, str], int] = defaultdict(int)
    # Track last cumulative_commands per (prompt, run_number, model) by max iteration
    last_cum_by_prompt_run_model: Dict[Tuple[str, int, str], Tuple[int, float]] = {}
    # Track final base_commands_seen_so_far per (prompt, run_number, model) - prefer iteration 50 or max
    final_base_by_prompt_run_model: Dict[Tuple[str, int, str], Tuple[int, str]] = {}
    prompts: Set[str] = set()
    runs: Set[int] = set()
    
    # Helper function to parse set string
    def parse_base_set(set_str: str) -> Set[str]:
        """Parse base_commands_seen_so_far string into a set of commands."""
        s = (set_str or "").strip()
        if s == "set()" or s == "{}" or s == "":
            return set()
        else:
            # extract quoted items
            items = re.findall(r"'([^']*)'|\"([^\"]*)\"", s)
            return set([a or b for a, b in items])
    
    try:
        # Try the provided CSV, but if it doesn't contain the expected
        # 'base_commands_seen_so_far' column try a few sensible fallbacks
        tried_paths = [csv_path]
        selected_path = None
        
        def _has_base_column(path: str) -> bool:
            try:
                with open(path, "r", encoding="utf-8") as tf:
                    rdr = csv.DictReader(tf)
                    fns = rdr.fieldnames or []
                    return "base_commands_seen_so_far" in fns
            except Exception:
                return False
        
        if _has_base_column(csv_path):
            selected_path = csv_path
        
        if selected_path is None:
            print(
                "Warning: CSV does not contain 'base_commands_seen_so_far' column."
                " Tried files: {}".format(
                    ", ".join(p for p in tried_paths if os.path.exists(p)) or ", ".join(tried_paths)
                )
            )
            return
        
        # Open the selected CSV file and iterate rows
        with open(selected_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row.get("prompt", "").strip()
                if not prompt:
                    continue
                
                try:
                    run_number = int(row.get("run_number", 0))
                except ValueError:
                    run_number = 0
                
                model = row.get("model", "").strip()
                if not model:
                    continue
                
                try:
                    iteration = int(row.get("iteration", 0))
                except ValueError:
                    iteration = 0
                
                try:
                    valid_commands = int(row.get("valid_commands", 0))
                except ValueError:
                    valid_commands = 0
                
                try:
                    cumulative_commands = float(row.get("cumulative_commands", 0.0))
                except ValueError:
                    cumulative_commands = 0.0
                
                # read raw set string
                base_set_str = row.get("base_commands_seen_so_far", "")
                
                key = (prompt, run_number, model)
                
                # accumulate valid commands across iterations
                total_valid_by_prompt_run_model[key] += valid_commands
                
                # store last cumulative_commands by max iteration
                if key not in last_cum_by_prompt_run_model or iteration > last_cum_by_prompt_run_model[key][0]:
                    last_cum_by_prompt_run_model[key] = (iteration, cumulative_commands)
                
                # track final base_commands_seen_so_far (prefer iteration 50 or max)
                if key not in final_base_by_prompt_run_model:
                    final_base_by_prompt_run_model[key] = (iteration, base_set_str)
                else:
                    prev_iter, _ = final_base_by_prompt_run_model[key]
                    # prefer iteration 50 if seen, otherwise keep the max
                    if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                        final_base_by_prompt_run_model[key] = (iteration, base_set_str)
                
                prompts.add(prompt)
                runs.add(run_number)
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return
    
    # Write analysis to text file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL ANALYSIS: VALIDITY RATIO AND BASE COMMANDS\n")
        f.write("=" * 80 + "\n")
        f.write("Validity Ratio = (Sum of valid_commands across all iterations) / (Sum of cumulative_commands from last iteration)\n")
        f.write("Average Base Commands = Average count of base commands across runs\n")
        f.write("Thresholds: Validity Ratio > 30%, Average Base Commands > 3\n")
        f.write("\n")
        
        # For each prompt, analyze models
        for prompt in sorted(prompts):
            f.write("=" * 80 + "\n")
            f.write(f"Prompt: {prompt}\n")
            f.write("=" * 80 + "\n")
            
            # Gather all models for this prompt
            models_set: Set[str] = set()
            for (p, r, m) in list(total_valid_by_prompt_run_model.keys()) + list(last_cum_by_prompt_run_model.keys()) + list(final_base_by_prompt_run_model.keys()):
                if p == prompt:
                    models_set.add(m)
            
            models = sorted(models_set)
            if not models:
                f.write("No models found for this prompt.\n\n")
                continue
            
            # Calculate metrics for each model
            model_metrics: Dict[str, Dict[str, float]] = {}
            
            for model in models:
                # Calculate validity ratio
                sum_valid = 0
                sum_cum = 0.0
                for run in sorted(runs):
                    v_key = (prompt, run, model)
                    # add total valid for this run if present
                    sum_valid += total_valid_by_prompt_run_model.get(v_key, 0)
                    # add last cumulative for this run if present
                    if v_key in last_cum_by_prompt_run_model:
                        _, cum_val = last_cum_by_prompt_run_model[v_key]
                        sum_cum += float(cum_val)
                
                # compute ratio safely
                validity_ratio = (sum_valid / sum_cum) if sum_cum > 0 else 0.0
                
                # Calculate average base commands
                run_counts = []
                for run in sorted(runs):
                    key = (prompt, run, model)
                    if key in final_base_by_prompt_run_model:
                        _, set_str = final_base_by_prompt_run_model[key]
                        parsed_set = parse_base_set(set_str)
                        run_counts.append(len(parsed_set))
                
                avg_base_commands = (sum(run_counts) / len(run_counts)) if run_counts else 0.0
                
                model_metrics[model] = {
                    "validity_ratio": validity_ratio,
                    "avg_base_commands": avg_base_commands
                }
            
            # Categorize models
            validity_ratio_models: List[Tuple[str, float]] = []
            base_commands_models: List[Tuple[str, float]] = []
            both_models: List[Tuple[str, float, float]] = []
            
            for model in models:
                ratio = model_metrics[model]["validity_ratio"]
                avg_base = model_metrics[model]["avg_base_commands"]
                
                meets_validity = ratio > 0.30
                meets_base = avg_base >= 3.0
                
                if meets_validity and meets_base:
                    both_models.append((model, ratio, avg_base))
                elif meets_validity:
                    validity_ratio_models.append((model, ratio))
                elif meets_base:
                    base_commands_models.append((model, avg_base))
            
            # Write results
            # Validity Ratio > 30% section
            f.write(f"\nModels with Validity Ratio > 30% ({len(validity_ratio_models)} models):\n")
            if validity_ratio_models:
                for model, ratio in sorted(validity_ratio_models, key=lambda x: x[1], reverse=True):
                    f.write(f"  - {model}: {ratio * 100:.2f}%\n")
            else:
                f.write("  (none)\n")
            
            # Average Base Commands > 4 section
            f.write(f"\nModels with Average Base Commands > 3 ({len(base_commands_models)} models):\n")
            if base_commands_models:
                for model, avg_base in sorted(base_commands_models, key=lambda x: x[1], reverse=True):
                    f.write(f"  - {model}: {avg_base:.2f}\n")
            else:
                f.write("  (none)\n")
            
            # Both Criteria section
            f.write(f"\nModels meeting Both Criteria ({len(both_models)} models):\n")
            if both_models:
                for model, ratio, avg_base in sorted(both_models, key=lambda x: (x[1], x[2]), reverse=True):
                    f.write(f"  - {model}: Validity Ratio = {ratio * 100:.2f}%, Avg Base Commands = {avg_base:.2f}\n")
            else:
                f.write("  (none)\n")
            
            f.write("\n")
    
    print(f"Created model analysis text file: {output_file}")


def create_average_iteration_duration_plots(csv_path: str, output_dir: str, model_to_is_coding: Optional[Dict[str, bool]] = None) -> None:
    """Create plots showing average iteration duration per (prompt, run, model) combination.
    
    For each prompt, creates one HTML file with subplots (one per run). Each subplot shows
    models on X-axis and average iteration duration (averaged across iterations for that
    specific prompt/run/model combination) on Y-axis.
    
    This calculates something like 80 models × 3 runs × 8 prompts = 1,920 individual averages.
    
    Args:
        csv_path: Path to command_statistics CSV file
        output_dir: Base output directory (files will be saved in average_iteration_duration subdirectory)
    """
    avg_dir = os.path.join(output_dir, "average_iteration_duration")
    os.makedirs(avg_dir, exist_ok=True)

    # Collect iteration_duration_seconds values per (prompt, run_number, model)
    data_by_prompt_run_model: Dict[Tuple[str, int, str], List[float]] = defaultdict(list)
    prompts = set()
    runs = set()

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            # Check if iteration_duration_seconds column exists
            if "iteration_duration_seconds" not in reader.fieldnames:
                print(f"Warning: CSV file {csv_path} does not contain 'iteration_duration_seconds' column.")
                print("Skipping average iteration duration plots.")
                return
            
            for row in reader:
                prompt = row.get("prompt", "").strip()
                model = row.get("model", "").strip()
                
                # Get run_number, handling missing values
                try:
                    run_number = int(row.get("run_number", 0))
                except (ValueError, TypeError):
                    run_number = 0
                
                # Get iteration_duration_seconds, handling None/empty values
                duration_str = row.get("iteration_duration_seconds", "").strip()
                if duration_str and duration_str.lower() not in ("none", "null", ""):
                    try:
                        duration = float(duration_str)
                        if duration is not None and duration >= 0:
                            if prompt and model and run_number > 0:
                                data_by_prompt_run_model[(prompt, run_number, model)].append(duration)
                                prompts.add(prompt)
                                runs.add(run_number)
                    except (ValueError, TypeError):
                        # Skip invalid values
                        continue
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return

    if not prompts:
        print("Warning: No valid iteration duration data found in CSV.")
        return

    # Create one HTML file per prompt
    for prompt in sorted(prompts):
        # Get all runs for this prompt
        prompt_runs = sorted([r for r in runs if any(p == prompt for p, _, _ in data_by_prompt_run_model.keys())])
        
        if not prompt_runs:
            continue
        
        # Create subplots: one row per run
        fig = make_subplots(
            rows=len(prompt_runs),
            cols=1,
            subplot_titles=[f"{prompt} - Run {run}" for run in prompt_runs],
            vertical_spacing=0.15,
        )
        
        # Process each run
        for run_idx, run_number in enumerate(prompt_runs, start=1):
            # Get all models for this (prompt, run) combination and calculate averages
            models_data = {}
            for (p, r, model), durations in data_by_prompt_run_model.items():
                if p == prompt and r == run_number:
                    # Filter out None values and calculate average
                    valid_durations = [d for d in durations if d is not None]
                    avg = (sum(valid_durations) / len(valid_durations)) if valid_durations else 0.0
                    models_data[model] = avg
            
            if models_data:
                # Sort models by name for consistent ordering
                models = sorted(models_data.keys())
                averages = [models_data[model] for model in models]
            else:
                models = []
                averages = []
            
            # Add bar trace to the appropriate subplot
            bar_kw: Dict[str, object] = dict(
                x=models,
                y=averages,
                text=[f"{v:.2f}" for v in averages] if averages else [],
                textposition="outside",
                hovertemplate="Model=%{x}<br>Avg Iteration Duration=%{y:.2f} seconds<extra></extra>",
                showlegend=False,
            )
            if model_to_is_coding and models:
                bar_kw["marker_color"] = bar_colors_for_models(models, model_to_is_coding)
            fig.add_trace(
                go.Bar(**bar_kw),
                row=run_idx,
                col=1,
            )
        
        # Update layout
        if len(prompt_runs) > 0:
            mid_row = (len(prompt_runs) + 1) // 2
            fig.update_xaxes(title_text="Model", tickangle=-45, row=mid_row, col=1)
            fig.update_xaxes(tickangle=-45)  # Rotate all x-axis labels
            fig.update_yaxes(title_text="Average Iteration Duration (seconds)", row=mid_row, col=1)
        
        fig.update_layout(
            title_text=f"Average Iteration Duration per Model: {prompt} (All Runs)",
            template="plotly_white",
            height=500 * len(prompt_runs),
        )
        
        # Write HTML file
        output_path = os.path.join(avg_dir, f"{prompt}.html")
        write_html(fig, output_path)
        print(f"Created average iteration duration plot: {output_path}")


def create_average_iteration_duration_scatter_plots(csv_path: str, output_dir: str) -> None:
    """Create dumbbell plots showing min, max, and average iteration duration per (prompt, run, model) combination.
    
    For each prompt, creates one HTML file with subplots (one per run). Each subplot shows
    models on X-axis and iteration duration ranges on Y-axis. Each dumbbell shows:
    - A line connecting minimum to maximum duration
    - Markers at min and max endpoints
    - A dot at the average duration position
    
    This calculates something like 80 models × 3 runs × 8 prompts = 1,920 individual min/max/avg values.
    
    Args:
        csv_path: Path to command_statistics CSV file
        output_dir: Base output directory (files will be saved in average_iteration_duration subdirectory)
    """
    # Use the same directory as bar charts, files will have _scatter suffix
    avg_dir = os.path.join(output_dir, "average_iteration_duration")
    os.makedirs(avg_dir, exist_ok=True)

    # Collect iteration_duration_seconds values per (prompt, run_number, model)
    data_by_prompt_run_model: Dict[Tuple[str, int, str], List[float]] = defaultdict(list)
    prompts = set()
    runs = set()

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            # Check if iteration_duration_seconds column exists
            if "iteration_duration_seconds" not in reader.fieldnames:
                print(f"Warning: CSV file {csv_path} does not contain 'iteration_duration_seconds' column.")
                print("Skipping average iteration duration scatter plots.")
                return
            
            for row in reader:
                prompt = row.get("prompt", "").strip()
                model = row.get("model", "").strip()
                
                # Get run_number, handling missing values
                try:
                    run_number = int(row.get("run_number", 0))
                except (ValueError, TypeError):
                    run_number = 0
                
                # Get iteration_duration_seconds, handling None/empty values
                duration_str = row.get("iteration_duration_seconds", "").strip()
                if duration_str and duration_str.lower() not in ("none", "null", ""):
                    try:
                        duration = float(duration_str)
                        if duration is not None and duration >= 0:
                            if prompt and model and run_number > 0:
                                data_by_prompt_run_model[(prompt, run_number, model)].append(duration)
                                prompts.add(prompt)
                                runs.add(run_number)
                    except (ValueError, TypeError):
                        # Skip invalid values
                        continue
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return

    if not prompts:
        print("Warning: No valid iteration duration data found in CSV.")
        return

    # Create one HTML file per prompt
    for prompt in sorted(prompts):
        # Get all runs for this prompt
        prompt_runs = sorted([r for r in runs if any(p == prompt for p, _, _ in data_by_prompt_run_model.keys())])
        
        if not prompt_runs:
            continue
        
        # Create subplots: one row per run
        fig = make_subplots(
            rows=len(prompt_runs),
            cols=1,
            subplot_titles=[f"{prompt} - Run {run}" for run in prompt_runs],
            vertical_spacing=0.15,
        )
        
        # Process each run
        for run_idx, run_number in enumerate(prompt_runs, start=1):
            # Get all models for this (prompt, run) combination and calculate min, max, average
            models_data = {}
            for (p, r, model), durations in data_by_prompt_run_model.items():
                if p == prompt and r == run_number:
                    # Filter out None values
                    valid_durations = [d for d in durations if d is not None]
                    if valid_durations:
                        min_dur = min(valid_durations)
                        max_dur = max(valid_durations)
                        avg_dur = sum(valid_durations) / len(valid_durations)
                        models_data[model] = {
                            'min': min_dur,
                            'max': max_dur,
                            'avg': avg_dur
                        }
            
            if models_data:
                # Sort models by name for consistent ordering
                models = sorted(models_data.keys())
                min_durations = [models_data[model]['min'] for model in models]
                max_durations = [models_data[model]['max'] for model in models]
                avg_durations = [models_data[model]['avg'] for model in models]
            else:
                models = []
                min_durations = []
                max_durations = []
                avg_durations = []
            
            if models and min_durations:
                # Create dumbbell plot: lines from min to max
                # For each model, create a line from min to max
                for i, model in enumerate(models):
                    fig.add_trace(
                        go.Scatter(
                            x=[model, model],
                            y=[min_durations[i], max_durations[i]],
                            mode="lines",
                            line=dict(color="rgba(128, 128, 128, 0.5)", width=2),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=run_idx,
                        col=1,
                    )
                
                # Add markers at min endpoints
                fig.add_trace(
                    go.Scatter(
                        x=models,
                        y=min_durations,
                        mode="markers",
                        marker=dict(
                            size=8,
                            color="rgba(255, 0, 0, 0.7)",
                            symbol="circle",
                            line=dict(width=1, color="darkred")
                        ),
                        hovertemplate="Model=%{x}<br>Min Duration=%{y:.2f} seconds<extra></extra>",
                        name="Min",
                        showlegend=(run_idx == 1),  # Only show legend on first subplot
                    ),
                    row=run_idx,
                    col=1,
                )
                
                # Add markers at max endpoints
                fig.add_trace(
                    go.Scatter(
                        x=models,
                        y=max_durations,
                        mode="markers",
                        marker=dict(
                            size=8,
                            color="rgba(0, 128, 0, 0.7)",
                            symbol="circle",
                            line=dict(width=1, color="darkgreen")
                        ),
                        hovertemplate="Model=%{x}<br>Max Duration=%{y:.2f} seconds<extra></extra>",
                        name="Max",
                        showlegend=(run_idx == 1),  # Only show legend on first subplot
                    ),
                    row=run_idx,
                    col=1,
                )
                
                # Add markers at average positions
                fig.add_trace(
                    go.Scatter(
                        x=models,
                        y=avg_durations,
                        mode="markers",
                        marker=dict(
                            size=12,
                            color="rgba(70, 130, 180, 0.9)",
                            symbol="circle",
                            line=dict(width=2, color="steelblue")
                        ),
                        hovertemplate="Model=%{x}<br>Avg Duration=%{y:.2f} seconds<extra></extra>",
                        name="Average",
                        showlegend=(run_idx == 1),  # Only show legend on first subplot
                    ),
                    row=run_idx,
                    col=1,
                )
                
                # Update X-axis to show model names directly (categorical)
                fig.update_xaxes(
                    tickangle=-45,
                    row=run_idx,
                    col=1,
                )
        
        # Update layout
        if len(prompt_runs) > 0:
            mid_row = (len(prompt_runs) + 1) // 2
            fig.update_xaxes(title_text="Model", row=mid_row, col=1)
            fig.update_yaxes(title_text="Iteration Duration (seconds)", row=mid_row, col=1)
        
        fig.update_layout(
            title_text=f"Iteration Duration per Model (Dumbbell Plot): {prompt} (All Runs)",
            template="plotly_white",
            height=500 * len(prompt_runs),
        )
        
        # Write HTML file with _scatter suffix to distinguish from bar chart version
        output_path = os.path.join(avg_dir, f"{prompt}_scatter.html")
        write_html(fig, output_path)
        print(f"Created average iteration duration dumbbell plot: {output_path}")


def create_average_iteration_duration_scatter_plots_with_outliers(csv_path: str, output_dir: str) -> None:
    """Create scatter plots showing average iteration duration per (model, run, prompt) combination with outlier separation.
    
    For each prompt, creates one HTML file showing:
    - X-axis: Model names sorted by parameter size (ascending)
    - Y-axis: Average iteration duration (seconds)
    - One point per (model, run) combination
    - Red dashed horizontal line at outlier threshold (Q3 + 1.5*IQR)
    
    Args:
        csv_path: Path to command_statistics CSV file
        output_dir: Base output directory (files will be saved in average_iteration_duration subdirectory)
    """
    avg_dir = os.path.join(output_dir, "average_iteration_duration")
    os.makedirs(avg_dir, exist_ok=True)

    # Load parameter size mapping
    param_size_map = load_model_size_mapping()

    # Collect iteration_duration_seconds values per (prompt, run_number, model)
    data_by_prompt_run_model: Dict[Tuple[str, int, str], List[float]] = defaultdict(list)
    prompts = set()
    runs = set()

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            # Check if iteration_duration_seconds column exists
            if "iteration_duration_seconds" not in reader.fieldnames:
                print(f"Warning: CSV file {csv_path} does not contain 'iteration_duration_seconds' column.")
                print("Skipping average iteration duration scatter plots with outliers.")
                return
            
            for row in reader:
                prompt = row.get("prompt", "").strip()
                model = row.get("model", "").strip()
                
                # Get run_number, handling missing values
                try:
                    run_number = int(row.get("run_number", 0))
                except (ValueError, TypeError):
                    run_number = 0
                
                # Get iteration_duration_seconds, handling None/empty values
                duration_str = row.get("iteration_duration_seconds", "").strip()
                if duration_str and duration_str.lower() not in ("none", "null", ""):
                    try:
                        duration = float(duration_str)
                        if duration is not None and duration >= 0:
                            if prompt and model and run_number > 0:
                                data_by_prompt_run_model[(prompt, run_number, model)].append(duration)
                                prompts.add(prompt)
                                runs.add(run_number)
                    except (ValueError, TypeError):
                        # Skip invalid values
                        continue
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return

    if not prompts:
        print("Warning: No valid iteration duration data found in CSV.")
        return

    # Calculate averages per (prompt, run, model)
    avg_by_prompt_run_model: Dict[Tuple[str, int, str], float] = {}
    for (prompt, run, model), durations in data_by_prompt_run_model.items():
        valid_durations = [d for d in durations if d is not None]
        if valid_durations:
            avg_by_prompt_run_model[(prompt, run, model)] = sum(valid_durations) / len(valid_durations)

    # Create one HTML file per prompt
    for prompt in sorted(prompts):
        # Get all models for this prompt
        models_set: Set[str] = set()
        for (p, r, m) in avg_by_prompt_run_model.keys():
            if p == prompt:
                models_set.add(m)
        
        if not models_set:
            continue
        
        # Sort models by parameter size (ascending), with missing models sorted last
        models = sorted(models_set, key=lambda m: param_size_map.get(m, float('inf')))
        
        # Calculate outlier threshold using IQR method
        # Collect all average durations for this prompt
        all_durations = [avg_by_prompt_run_model[(prompt, r, m)] 
                         for (p, r, m), avg in avg_by_prompt_run_model.items() 
                         if p == prompt]
        
        if not all_durations:
            continue
        
        # Calculate Q1, Q3, and IQR
        sorted_durations = sorted(all_durations)
        n = len(sorted_durations)
        
        if n == 0:
            continue
        
        # Calculate Q1 (25th percentile) - using linear interpolation method
        q1_pos = (n - 1) * 0.25
        q1_low = int(q1_pos)
        q1_high = min(q1_low + 1, n - 1)
        q1_weight = q1_pos - q1_low
        q1 = sorted_durations[q1_low] * (1 - q1_weight) + sorted_durations[q1_high] * q1_weight
        
        # Calculate Q3 (75th percentile) - using linear interpolation method
        q3_pos = (n - 1) * 0.75
        q3_low = int(q3_pos)
        q3_high = min(q3_low + 1, n - 1)
        q3_weight = q3_pos - q3_low
        q3 = sorted_durations[q3_low] * (1 - q3_weight) + sorted_durations[q3_high] * q3_weight
        
        iqr = q3 - q1
        outlier_threshold = q3 + 1.5 * iqr
        
        # Create scatter plot
        fig = go.Figure()
        
        # Add scatter points for each run
        run_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green
        for run_idx, run_number in enumerate(sorted(runs)):
            x_vals = []
            y_vals = []
            hover_texts = []
            
            for model in models:
                key = (prompt, run_number, model)
                if key in avg_by_prompt_run_model:
                    x_vals.append(model)
                    y_vals.append(avg_by_prompt_run_model[key])
                    hover_texts.append(f"Model: {model}<br>Run: {run_number}<br>Avg Duration: {avg_by_prompt_run_model[key]:.2f} seconds")
            
            if x_vals:  # Only add trace if there's data
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode='markers',
                        name=f'Run {run_number}',
                        marker=dict(
                            size=8,
                            color=run_colors[run_idx % len(run_colors)],
                            opacity=0.7
                        ),
                        text=hover_texts,
                        hovertemplate='%{text}<extra></extra>',
                    )
                )
        
        # Add red dashed horizontal line at outlier threshold
        fig.add_hline(
            y=outlier_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Outlier Threshold: {outlier_threshold:.2f}s",
            annotation_position="right",
            annotation=dict(font_size=10, font_color="red")
        )
        
        # Update layout
        fig.update_xaxes(
            title_text="Model (sorted by parameter size)",
            tickangle=-45,
            type='category'
        )
        fig.update_yaxes(title_text="Average Iteration Duration (seconds)")
        fig.update_layout(
            title_text=f"Average Iteration Duration per Model: {prompt} (Outlier Threshold: {outlier_threshold:.2f}s)",
            template="plotly_white",
            height=700,
            width=max(1200, len(models) * 15),  # Adjust width based on number of models
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(b=150)  # Extra bottom margin for rotated labels
        )
        
        # Write HTML file
        output_path = os.path.join(avg_dir, f"{prompt}_outliers.html")
        write_html(fig, output_path)
        print(f"Created average iteration duration scatter plot with outliers: {output_path}")


def create_norag_vs_rag_duration_comparison_plots(no_rag_csv_path: str, rag_csv_path: str, output_dir: str, model_to_is_coding: Optional[Dict[str, bool]] = None) -> None:
    """Create grouped bar charts comparing average iteration duration between NO_RAG and RAG experiments.
    
    For each prompt, creates one HTML file with 3 subplots (one per run). Each subplot shows:
    - X-axis: Model names sorted by parameter size (ascending)
    - Y-axis: Average iteration duration (seconds)
    - Two bars per model: NO_RAG (blue) and RAG (orange) side-by-side
    
    Args:
        no_rag_csv_path: Path to NO_RAG_KORAD_results.csv file
        rag_csv_path: Path to RAG_KORAD_results.csv file
        output_dir: Base output directory (files will be saved in average_iteration_duration subdirectory)
    """
    avg_dir = os.path.join(output_dir, "average_iteration_duration")
    os.makedirs(avg_dir, exist_ok=True)
    
    # Load parameter size mapping for sorting
    param_size_map = load_model_size_mapping()
    
    # Check if both CSV files exist
    if not os.path.exists(no_rag_csv_path):
        print(f"Warning: NO_RAG CSV file not found at {no_rag_csv_path}")
        print("Skipping NO_RAG vs RAG duration comparison plots.")
        return
    
    if not os.path.exists(rag_csv_path):
        print(f"Warning: RAG CSV file not found at {rag_csv_path}")
        print("Skipping NO_RAG vs RAG duration comparison plots.")
        return
    
    # Collect iteration_duration_seconds values per (prompt, run_number, model) for both files
    no_rag_data: Dict[Tuple[str, int, str], List[float]] = defaultdict(list)
    rag_data: Dict[Tuple[str, int, str], List[float]] = defaultdict(list)
    prompts = set()
    runs = set()
    
    # Read NO_RAG CSV file
    try:
        with open(no_rag_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            # Check if iteration_duration_seconds column exists
            if "iteration_duration_seconds" not in reader.fieldnames:
                print(f"Warning: NO_RAG CSV file {no_rag_csv_path} does not contain 'iteration_duration_seconds' column.")
                print("Skipping NO_RAG vs RAG duration comparison plots.")
                return
            
            for row in reader:
                prompt = row.get("prompt", "").strip()
                model = row.get("model", "").strip()
                
                # Get run_number, handling missing values
                try:
                    run_number = int(row.get("run_number", 0))
                except (ValueError, TypeError):
                    run_number = 0
                
                # Get iteration_duration_seconds, handling None/empty values
                duration_str = row.get("iteration_duration_seconds", "").strip()
                if duration_str and duration_str.lower() not in ("none", "null", ""):
                    try:
                        duration = float(duration_str)
                        if duration is not None and duration >= 0:
                            if prompt and model and run_number > 0:
                                no_rag_data[(prompt, run_number, model)].append(duration)
                                prompts.add(prompt)
                                runs.add(run_number)
                    except (ValueError, TypeError):
                        # Skip invalid values
                        continue
    except Exception as e:
        print(f"Error reading NO_RAG CSV file {no_rag_csv_path}: {e}")
        return
    
    # Read RAG CSV file
    try:
        with open(rag_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            # Check if iteration_duration_seconds column exists
            if "iteration_duration_seconds" not in reader.fieldnames:
                print(f"Warning: RAG CSV file {rag_csv_path} does not contain 'iteration_duration_seconds' column.")
                print("Skipping NO_RAG vs RAG duration comparison plots.")
                return
            
            for row in reader:
                prompt = row.get("prompt", "").strip()
                model = row.get("model", "").strip()
                
                # Get run_number, handling missing values
                try:
                    run_number = int(row.get("run_number", 0))
                except (ValueError, TypeError):
                    run_number = 0
                
                # Get iteration_duration_seconds, handling None/empty values
                duration_str = row.get("iteration_duration_seconds", "").strip()
                if duration_str and duration_str.lower() not in ("none", "null", ""):
                    try:
                        duration = float(duration_str)
                        if duration is not None and duration >= 0:
                            if prompt and model and run_number > 0:
                                rag_data[(prompt, run_number, model)].append(duration)
                                prompts.add(prompt)
                                runs.add(run_number)
                    except (ValueError, TypeError):
                        # Skip invalid values
                        continue
    except Exception as e:
        print(f"Error reading RAG CSV file {rag_csv_path}: {e}")
        return
    
    if not prompts:
        print("Warning: No valid iteration duration data found in CSV files.")
        return
    
    # Calculate averages per (prompt, run, model) for both datasets
    no_rag_avg: Dict[Tuple[str, int, str], float] = {}
    rag_avg: Dict[Tuple[str, int, str], float] = {}
    
    for (prompt, run, model), durations in no_rag_data.items():
        valid_durations = [d for d in durations if d is not None]
        if valid_durations:
            no_rag_avg[(prompt, run, model)] = sum(valid_durations) / len(valid_durations)
    
    for (prompt, run, model), durations in rag_data.items():
        valid_durations = [d for d in durations if d is not None]
        if valid_durations:
            rag_avg[(prompt, run, model)] = sum(valid_durations) / len(valid_durations)
    
    # Create one HTML file per prompt
    for prompt in sorted(prompts):
        # Get all runs for this prompt
        prompt_runs = sorted([r for r in runs if any(p == prompt for p, _, _ in no_rag_data.keys()) or any(p == prompt for p, _, _ in rag_data.keys())])
        
        if not prompt_runs:
            continue
        
        # Get all models for this prompt (from both datasets)
        models_set: Set[str] = set()
        for (p, r, m) in list(no_rag_avg.keys()) + list(rag_avg.keys()):
            if p == prompt:
                models_set.add(m)
        
        if not models_set:
            continue
        
        # Sort models by parameter size (ascending), with missing models sorted last
        models = sorted(models_set, key=lambda m: param_size_map.get(m, float('inf')))
        
        # Split models by parameter size
        models_gt_4b = []
        models_le_4b = []
        
        for model in models:
            param_size = param_size_map.get(model, 0.0)
            if param_size > 4.0:
                models_gt_4b.append(model)
            elif param_size <= 4.0:
                models_le_4b.append(model)
            # Skip models not found in mapping (already handled by sorting)
        
        # Helper function to create plot for a group of models
        def create_plot_for_models(model_list: List[str], size_label: str, filename_suffix: str) -> None:
            if not model_list:
                return
            
            # Create subplots: 3 rows, 1 column (one per run)
            fig = make_subplots(
                rows=len(prompt_runs),
                cols=1,
                subplot_titles=[f"{prompt} - Run {run}" for run in prompt_runs],
                vertical_spacing=0.15,
            )
            
            # Process each run
            for run_idx, run_number in enumerate(prompt_runs, start=1):
                # Collect data for this (prompt, run) combination
                no_rag_values = []
                rag_values = []
                model_names = []
                
                for model in model_list:
                    no_rag_key = (prompt, run_number, model)
                    rag_key = (prompt, run_number, model)
                    
                    # Get averages (use 0.0 if not found)
                    no_rag_duration = no_rag_avg.get(no_rag_key, 0.0)
                    rag_duration = rag_avg.get(rag_key, 0.0)
                    
                    # Only include models that have data in at least one dataset
                    if no_rag_key in no_rag_avg or rag_key in rag_avg:
                        model_names.append(model)
                        no_rag_values.append(no_rag_duration)
                        rag_values.append(rag_duration)
                
                if not model_names:
                    continue
                
                bar_colors = bar_colors_for_models(model_names, model_to_is_coding) if (model_to_is_coding and model_names) else None
                no_rag_kw: Dict[str, object] = dict(
                    name="NO_RAG",
                    x=model_names,
                    y=no_rag_values,
                    text=[f"{v:.2f}" for v in no_rag_values],
                    textposition="outside",
                    hovertemplate="Model=%{x}<br>NO_RAG Avg Duration=%{y:.2f} seconds<extra></extra>",
                    showlegend=(run_idx == 1),
                )
                if bar_colors:
                    no_rag_kw["marker_color"] = bar_colors
                else:
                    no_rag_kw["marker_color"] = "#1f77b4"
                fig.add_trace(go.Bar(**no_rag_kw), row=run_idx, col=1)
                
                rag_kw: Dict[str, object] = dict(
                    name="RAG",
                    x=model_names,
                    y=rag_values,
                    text=[f"{v:.2f}" for v in rag_values],
                    textposition="outside",
                    hovertemplate="Model=%{x}<br>RAG Avg Duration=%{y:.2f} seconds<extra></extra>",
                    showlegend=(run_idx == 1),
                )
                if bar_colors:
                    rag_kw["marker_color"] = bar_colors
                else:
                    rag_kw["marker_color"] = "#ff7f0e"
                fig.add_trace(go.Bar(**rag_kw), row=run_idx, col=1)
                
                # Update x-axis for this subplot
                fig.update_xaxes(
                    title_text="Model (sorted by parameter size)",
                    tickangle=-45,
                    row=run_idx,
                    col=1,
                )
            
            # Update layout
            if len(prompt_runs) > 0:
                mid_row = (len(prompt_runs) + 1) // 2
                fig.update_yaxes(title_text="Average Iteration Duration (seconds)", row=mid_row, col=1)
            
            fig.update_layout(
                title_text=f"NO_RAG vs RAG Average Iteration Duration per Model: {prompt} ({size_label})",
                template="plotly_white",
                barmode="group",
                height=500 * len(prompt_runs),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            
            # Write HTML file
            output_path = os.path.join(avg_dir, f"{prompt}_norag_vs_rag_duration{filename_suffix}.html")
            write_html(fig, output_path)
            print(f"Created NO_RAG vs RAG duration comparison plot: {output_path}")
        
        # Create plots for both groups
        create_plot_for_models(models_gt_4b, ">4B", "_gt4b")
        create_plot_for_models(models_le_4b, "<=4B", "_le4b")


def create_success_failure_iteration_plots(no_rag_csv_path: str, output_dir: str, model_to_is_coding: Optional[Dict[str, bool]] = None) -> None:
    """Create stacked bar charts showing successful vs failed iterations for NO_RAG experiments.
    
    For each prompt, creates one HTML file with subplots (one per run).
    Each subplot shows stacked bar charts with:
    - X-axis: Model names (sorted by parameter size)
    - Y-axis: Number of iterations
    - One stacked bar per model
    - Each bar is stacked: Successful iterations (bottom, green) and Failed iterations (top, red)
    
    Failed iterations = cumulative_failures from iteration 50 (or max iteration)
    Successful iterations = total iterations attempted - failed iterations
    
    Args:
        no_rag_csv_path: Path to NO_RAG_KORAD_results.csv file
        rag_csv_path: Path to RAG_KORAD_results.csv file (not used, kept for compatibility)
        output_dir: Base output directory (files will be saved in success_failure_iterations subdirectory)
    """
    output_subdir = os.path.join(output_dir, "success_failure_iterations")
    os.makedirs(output_subdir, exist_ok=True)
    
    # Load parameter size mapping for sorting
    param_size_map = load_model_size_mapping()
    
    # Check if CSV file exists
    if not os.path.exists(no_rag_csv_path):
        print(f"Warning: NO_RAG CSV file not found at {no_rag_csv_path}")
        print("Skipping success/failure iteration plots.")
        return
    
    # Track cumulative_failures per (prompt, run, model) - prefer iteration 50 or max
    cumulative_failures_by_prompt_run_model: Dict[Tuple[str, int, str], Tuple[int, int]] = {}
    # Track max iteration per (prompt, run, model) to determine total iterations attempted
    max_iteration_by_prompt_run_model: Dict[Tuple[str, int, str], int] = {}
    prompts: Set[str] = set()
    runs: Set[int] = set()
    
    # Read NO_RAG CSV file
    try:
        with open(no_rag_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            # Check if cumulative_failures column exists
            if "cumulative_failures" not in reader.fieldnames:
                print(f"Warning: CSV file {no_rag_csv_path} does not contain 'cumulative_failures' column.")
                print("Skipping success/failure iteration plots.")
                return
            
            for row in reader:
                prompt = row.get("prompt", "").strip()
                model = row.get("model", "").strip()
                
                if not prompt or not model:
                    continue
                
                try:
                    run_number = int(row.get("run_number", 0))
                except (ValueError, TypeError):
                    run_number = 0
                
                try:
                    iteration = int(row.get("iteration", 0))
                except (ValueError, TypeError):
                    iteration = 0
                
                try:
                    cumulative_failures = int(row.get("cumulative_failures", 0))
                except (ValueError, TypeError):
                    cumulative_failures = 0
                
                if run_number > 0 and iteration > 0:
                    key = (prompt, run_number, model)
                    
                    # Track max iteration for this combination
                    if key not in max_iteration_by_prompt_run_model or iteration > max_iteration_by_prompt_run_model[key]:
                        max_iteration_by_prompt_run_model[key] = iteration
                    
                    # Track cumulative_failures (prefer iteration 50 or max)
                    if key not in cumulative_failures_by_prompt_run_model:
                        cumulative_failures_by_prompt_run_model[key] = (iteration, cumulative_failures)
                    else:
                        prev_iter, _ = cumulative_failures_by_prompt_run_model[key]
                        # prefer iteration 50 if seen, otherwise keep the max
                        if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                            cumulative_failures_by_prompt_run_model[key] = (iteration, cumulative_failures)
                    
                    prompts.add(prompt)
                    runs.add(run_number)
    except Exception as e:
        print(f"Error reading NO_RAG CSV file {no_rag_csv_path}: {e}")
        return
    
    if not prompts:
        print("Warning: No valid data found in CSV file.")
        return
    
    # For each prompt, create two HTML files (one for <=4B, one for >4B) with subplots per run
    for prompt in sorted(prompts):
        # Get all models for this prompt
        models_set: Set[str] = set()
        for (p, r, m) in cumulative_failures_by_prompt_run_model.keys():
            if p == prompt:
                models_set.add(m)
        
        if not models_set:
            continue
        
        # Sort models by parameter size (ascending), with missing models sorted last
        models = sorted(models_set, key=lambda m: param_size_map.get(m, float('inf')))
        
        # Split models by parameter size
        models_le_4b = []
        models_gt_4b = []
        
        for model in models:
            param_size = param_size_map.get(model, 0.0)
            if param_size <= 4.0:
                models_le_4b.append(model)
            elif param_size > 4.0:
                models_gt_4b.append(model)
        
        # Get all runs for this prompt
        prompt_runs = sorted([r for r in runs if any(p == prompt for p, _, _ in cumulative_failures_by_prompt_run_model.keys())])
        
        if not prompt_runs:
            continue
        
        # Helper function to create plot for a group of models
        def create_plot_for_models(model_list: List[str], size_label: str, filename_suffix: str) -> None:
            if not model_list:
                return
            
            # Create subplots: one row per run
            fig = make_subplots(
                rows=len(prompt_runs),
                cols=1,
                subplot_titles=[f"{prompt} - Run {run}" for run in prompt_runs],
                vertical_spacing=0.15,
            )
            
            # Process each run
            for run_idx, run_number in enumerate(prompt_runs, start=1):
                # Collect data for this (prompt, run) combination
                successful_values = []
                failed_values = []
                model_names = []
                
                for model in model_list:
                    key = (prompt, run_number, model)
                    
                    if key in cumulative_failures_by_prompt_run_model:
                        final_iter, cumulative_failures = cumulative_failures_by_prompt_run_model[key]
                        max_iter = max_iteration_by_prompt_run_model.get(key, final_iter)
                        
                        # Failed iterations = cumulative_failures from final iteration (prefer iteration 50)
                        failed = cumulative_failures
                        # Successful iterations = total iterations attempted - failed
                        # Use final_iter if it's 50, otherwise use max_iter
                        total_iterations = final_iter if final_iter == 50 else max_iter
                        successful = total_iterations - failed
                        
                        model_names.append(model)
                        successful_values.append(successful)
                        failed_values.append(failed)
                
                if not model_names:
                    continue
                
                bar_colors = bar_colors_for_models(model_names, model_to_is_coding) if (model_to_is_coding and model_names) else None
                succ_kw: Dict[str, object] = dict(name="Successful", x=model_names, y=successful_values, text=[f"{v}" for v in successful_values], textposition="inside", hovertemplate="Model=%{x}<br>Successful=%{y} iterations<extra></extra>", showlegend=(run_idx == 1))
                if bar_colors:
                    succ_kw["marker_color"] = bar_colors
                else:
                    succ_kw["marker_color"] = "#2ca02c"
                fig.add_trace(go.Bar(**succ_kw), row=run_idx, col=1)
                
                fail_kw: Dict[str, object] = dict(name="Failed", x=model_names, y=failed_values, text=[f"{v}" for v in failed_values], textposition="inside", hovertemplate="Model=%{x}<br>Failed=%{y} iterations<extra></extra>", showlegend=(run_idx == 1))
                if bar_colors:
                    fail_kw["marker_color"] = bar_colors
                else:
                    fail_kw["marker_color"] = "#d62728"
                fig.add_trace(go.Bar(**fail_kw), row=run_idx, col=1)
                
                # Update x-axis for this subplot
                fig.update_xaxes(
                    title_text="Model",
                    tickangle=-45,
                    row=run_idx,
                    col=1,
                )
            
            # Update layout
            if len(prompt_runs) > 0:
                mid_row = (len(prompt_runs) + 1) // 2
                fig.update_yaxes(title_text="Number of Iterations", row=mid_row, col=1)
            
            fig.update_layout(
                title_text=f"Successful vs Failed Iterations: {prompt} ({size_label})",
                template="plotly_white",
                barmode="stack",  # Stack successful and failed segments
                height=500 * len(prompt_runs),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(b=150),  # Extra bottom margin for rotated labels
            )
            
            # Write HTML file
            output_path = os.path.join(output_subdir, f"{prompt}{filename_suffix}.html")
            write_html(fig, output_path)
            print(f"Created success/failure iteration plot: {output_path}")
        
        # Create plots for both groups
        create_plot_for_models(models_le_4b, "<=4B", "_le4b")
        create_plot_for_models(models_gt_4b, ">4B", "_gt4b")


def create_norag_vs_rag_blanks_and_formatting_plots(
    prompts: Set[str],
    no_rag_blanks_per_prompt_model: Dict[Tuple[str, str], int],
    rag_blanks_per_prompt_model: Dict[Tuple[str, str], int],
    output_subdir: str,
) -> None:
    """Create bar charts of average blanks and formatting violations per prompt for NO_RAG vs RAG.

    Formatting violations are currently stubbed as zero until a concrete definition is provided.
    """
    
    sorted_prompts = sorted(prompts)
    avg_blanks_no_rag: List[float] = []
    avg_blanks_rag: List[float] = []
    avg_fmt_no_rag: List[float] = []
    avg_fmt_rag: List[float] = []
    #print(no_rag_blanks_per_prompt_model, rag_blanks_per_prompt_model)
    for prompt in sorted_prompts:
        models_no_rag = [m for (p, m) in no_rag_blanks_per_prompt_model if p == prompt]
        models_rag = [m for (p, m) in rag_blanks_per_prompt_model if p == prompt]

        if models_no_rag:
            avg_blanks_no_rag.append(
                sum(no_rag_blanks_per_prompt_model[(prompt, m)] for m in models_no_rag) / len(models_no_rag)
            )
        else:
            avg_blanks_no_rag.append(0.0)

        if models_rag:
            avg_blanks_rag.append(
                sum(rag_blanks_per_prompt_model[(prompt, m)] for m in models_rag) / len(models_rag)
            )
        else:
            avg_blanks_rag.append(0.0)

        # Formatting violations: currently stubbed as 0.0
        avg_fmt_no_rag.append(0.0)
        avg_fmt_rag.append(0.0)

    # Create blanks bar chart (grey = NO_RAG, purple = RAG)
    fig_blanks = go.Figure()
    fig_blanks.add_trace(
        go.Bar(
            name="NO_RAG",
            x=sorted_prompts,
            y=avg_blanks_no_rag,
            marker_color="#808080",
            hovertemplate="Prompt=%{x}<br>NO_RAG Avg Blanks=%{y:.2f}<extra></extra>",
        )
    )
    fig_blanks.add_trace(
        go.Bar(
            name="RAG",
            x=sorted_prompts,
            y=avg_blanks_rag,
            marker_color="#6A0DAD",
            hovertemplate="Prompt=%{x}<br>RAG Avg Blanks=%{y:.2f}<extra></extra>",
        )
    )
    fig_blanks.update_layout(
        title_text="Average Blanks per Prompt (NO_RAG vs RAG)",
        template="plotly_white",
        barmode="group",
        xaxis_title="Prompt",
        yaxis_title="Average blanks across models",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(b=150),
    )
    blanks_path = os.path.join(output_subdir, "blanks_norag_vs_rag_avg.html")
    write_html(fig_blanks, blanks_path)
    print(f"Created blanks NO_RAG vs RAG plot: {blanks_path}")

    # Create formatting violations bar chart (grey = NO_RAG, purple = RAG)
    fig_fmt = go.Figure()
    fig_fmt.add_trace(
        go.Bar(
            name="NO_RAG",
            x=sorted_prompts,
            y=avg_fmt_no_rag,
            marker_color="#808080",
            hovertemplate="Prompt=%{x}<br>NO_RAG Avg Formatting Violations=%{y:.2f}<extra></extra>",
        )
    )
    fig_fmt.add_trace(
        go.Bar(
            name="RAG",
            x=sorted_prompts,
            y=avg_fmt_rag,
            marker_color="#6A0DAD",
            hovertemplate="Prompt=%{x}<br>RAG Avg Formatting Violations=%{y:.2f}<extra></extra>",
        )
    )
    fig_fmt.update_layout(
        title_text="Average Formatting Violations per Prompt (NO_RAG vs RAG)",
        template="plotly_white",
        barmode="group",
        xaxis_title="Prompt",
        yaxis_title="Average formatting violations across models",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(b=150),
    )
    fmt_path = os.path.join(output_subdir, "formatting_violations_norag_vs_rag_avg.html")
    write_html(fig_fmt, fmt_path)
    print(f"Created formatting violations NO_RAG vs RAG plot: {fmt_path}")


def create_norag_vs_rag_success_failure_plots(no_rag_csv_path: str, rag_csv_path: str, output_dir: str, model_to_is_coding: Optional[Dict[str, bool]] = None) -> None:
    """Create grouped stacked bar charts comparing successful vs failed iterations between NO_RAG and RAG experiments.
    
    For each prompt, creates one HTML file with subplots (one per run).
    Each subplot shows grouped stacked bar charts with:
    - X-axis: Model names (sorted by parameter size)
    - Y-axis: Number of iterations
    - Two bars per model side-by-side: NO_RAG (left) and RAG (right)
    - Each bar is stacked: Successful iterations (bottom, green) and Failed iterations (top, red)
    
    Failed iterations = cumulative_failures from iteration 50 (or max iteration)
    Successful iterations = total iterations attempted - failed iterations
    
    Args:
        no_rag_csv_path: Path to NO_RAG_KORAD_results.csv file
        rag_csv_path: Path to RAG_KORAD_results.csv file
        output_dir: Base output directory (files will be saved in norag_vs_rag_success_failure subdirectory)
    """
    output_subdir = os.path.join(output_dir, "norag_vs_rag_success_failure")
    os.makedirs(output_subdir, exist_ok=True)
    
    # Load parameter size mapping for sorting
    param_size_map = load_model_size_mapping()
    
    # Check if CSV files exist
    if not os.path.exists(no_rag_csv_path):
        print(f"Warning: NO_RAG CSV file not found at {no_rag_csv_path}")
        print("Skipping NO_RAG vs RAG success/failure plots.")
        return
    
    if not os.path.exists(rag_csv_path):
        print(f"Warning: RAG CSV file not found at {rag_csv_path}")
        print("Skipping NO_RAG vs RAG success/failure plots.")
        return
    
    # Track cumulative_failures per (prompt, run, model) for both datasets
    no_rag_cumulative_failures: Dict[Tuple[str, int, str], Tuple[int, int]] = {}
    rag_cumulative_failures: Dict[Tuple[str, int, str], Tuple[int, int]] = {}
    # Track max iteration per (prompt, run, model) to determine total iterations attempted
    no_rag_max_iteration: Dict[Tuple[str, int, str], int] = {}
    rag_max_iteration: Dict[Tuple[str, int, str], int] = {}
    # Track blanks per (prompt, model); formatting violations currently stubbed as 0
    no_rag_blanks_per_prompt_model: Dict[Tuple[str, str], int] = {}
    rag_blanks_per_prompt_model: Dict[Tuple[str, str], int] = {}
    prompts: Set[str] = set()
    runs: Set[int] = set()
    
    # Read NO_RAG CSV file
    try:
        with open(no_rag_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            
            # Check if cumulative_failures column exists
            if "cumulative_failures" not in fieldnames:
                print(f"Warning: NO_RAG CSV file {no_rag_csv_path} does not contain 'cumulative_failures' column.")
                print("Skipping NO_RAG vs RAG success/failure plots.")
                return
            
            value_columns = [c for c in fieldnames if c not in _BLANK_METADATA_COLUMNS]
            
            for row in reader:
                prompt = row.get("prompt", "").strip()
                model = row.get("model", "").strip()
                
                if not prompt or not model:
                    continue
                # Track blanks (row where all value columns are 0 or empty)
                if value_columns and _is_blank_row(row, value_columns):
                    pm_key = (prompt, model)
                    no_rag_blanks_per_prompt_model[pm_key] = no_rag_blanks_per_prompt_model.get(pm_key, 0) + 1
                
                # Formatting violations: stub as 0 (definition TBD)
                
                try:
                    run_number = int(row.get("run_number", 0))
                except (ValueError, TypeError):
                    run_number = 0
                
                try:
                    iteration = int(row.get("iteration", 0))
                except (ValueError, TypeError):
                    iteration = 0
                
                try:
                    cumulative_failures = int(row.get("cumulative_failures", 0))
                except (ValueError, TypeError):
                    cumulative_failures = 0
                
                if run_number > 0 and iteration > 0:
                    key = (prompt, run_number, model)
                    
                    # Track max iteration for this combination
                    if key not in no_rag_max_iteration or iteration > no_rag_max_iteration[key]:
                        no_rag_max_iteration[key] = iteration
                    
                    # Track cumulative_failures (prefer iteration 50 or max)
                    if key not in no_rag_cumulative_failures:
                        no_rag_cumulative_failures[key] = (iteration, cumulative_failures)
                    else:
                        prev_iter, _ = no_rag_cumulative_failures[key]
                        # prefer iteration 50 if seen, otherwise keep the max
                        if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                            no_rag_cumulative_failures[key] = (iteration, cumulative_failures)
                    
                    prompts.add(prompt)
                    runs.add(run_number)
    except Exception as e:
        print(f"Error reading NO_RAG CSV file {no_rag_csv_path}: {e}")
        return
    
    # Read RAG CSV file
    try:
        with open(rag_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            
            # Check if cumulative_failures column exists
            if "cumulative_failures" not in fieldnames:
                print(f"Warning: RAG CSV file {rag_csv_path} does not contain 'cumulative_failures' column.")
                print("Skipping NO_RAG vs RAG success/failure plots.")
                return
            
            value_columns = [c for c in fieldnames if c not in _BLANK_METADATA_COLUMNS]
            
            for row in reader:
                prompt = row.get("prompt", "").strip()
                model = row.get("model", "").strip()
                
                if not prompt or not model:
                    continue
                
                # Track blanks (row where all value columns are 0 or empty)
                if value_columns and _is_blank_row(row, value_columns):
                    pm_key = (prompt, model)
                    rag_blanks_per_prompt_model[pm_key] = rag_blanks_per_prompt_model.get(pm_key, 0) + 1
                
                # Formatting violations: stub as 0 (definition TBD)
                
                try:
                    run_number = int(row.get("run_number", 0))
                except (ValueError, TypeError):
                    run_number = 0
                
                try:
                    iteration = int(row.get("iteration", 0))
                except (ValueError, TypeError):
                    iteration = 0
                
                try:
                    cumulative_failures = int(row.get("cumulative_failures", 0))
                except (ValueError, TypeError):
                    cumulative_failures = 0
                
                if run_number > 0 and iteration > 0:
                    key = (prompt, run_number, model)
                    
                    # Track max iteration for this combination
                    if key not in rag_max_iteration or iteration > rag_max_iteration[key]:
                        rag_max_iteration[key] = iteration
                    
                    # Track cumulative_failures (prefer iteration 50 or max)
                    if key not in rag_cumulative_failures:
                        rag_cumulative_failures[key] = (iteration, cumulative_failures)
                    else:
                        prev_iter, _ = rag_cumulative_failures[key]
                        # prefer iteration 50 if seen, otherwise keep the max
                        if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                            rag_cumulative_failures[key] = (iteration, cumulative_failures)
                    
                    prompts.add(prompt)
                    runs.add(run_number)
    except Exception as e:
        print(f"Error reading RAG CSV file {rag_csv_path}: {e}")
        return
    
    if not prompts:
        print("Warning: No valid data found in CSV files.")
        return

    # Create blanks / formatting-violations comparison charts in a separate helper
    create_norag_vs_rag_blanks_and_formatting_plots(
        prompts=prompts,
        no_rag_blanks_per_prompt_model=no_rag_blanks_per_prompt_model,
        rag_blanks_per_prompt_model=rag_blanks_per_prompt_model,
        output_subdir=output_subdir,
    )

    # For each prompt, create two HTML files (one for <=4B, one for >4B) with subplots per run
    for prompt in sorted(prompts):
        # Get all models for this prompt that have failures in at least one run
        models_set: Set[str] = set()
        for (p, r, m) in list(no_rag_cumulative_failures.keys()) + list(rag_cumulative_failures.keys()):
            if p == prompt:
                # Check if this model has failures in either NO_RAG or RAG for this run
                no_rag_key = (prompt, r, m)
                rag_key = (prompt, r, m)
                no_rag_has_failures = no_rag_key in no_rag_cumulative_failures and no_rag_cumulative_failures[no_rag_key][1] > 0
                rag_has_failures = rag_key in rag_cumulative_failures and rag_cumulative_failures[rag_key][1] > 0
                if no_rag_has_failures or rag_has_failures:
                    models_set.add(m)
        
        if not models_set:
            continue
        
        # Sort models by parameter size (ascending), with missing models sorted last
        models = sorted(models_set, key=lambda m: param_size_map.get(m, float('inf')))
        
        # Split models by parameter size
        models_le_4b = []
        models_gt_4b = []
        
        for model in models:
            param_size = param_size_map.get(model, 0.0)
            if param_size <= 4.0:
                models_le_4b.append(model)
            elif param_size > 4.0:
                models_gt_4b.append(model)
        
        # Get all runs for this prompt
        prompt_runs = sorted([r for r in runs if any(p == prompt for p, _, _ in no_rag_cumulative_failures.keys()) or any(p == prompt for p, _, _ in rag_cumulative_failures.keys())])
        
        if not prompt_runs:
            continue
        
        # Helper function to create plot for a group of models
        def create_plot_for_models(model_list: List[str], size_label: str, filename_suffix: str) -> None:
            if not model_list:
                return
            
            # Create subplots: one row per run
            fig = make_subplots(
                rows=len(prompt_runs),
                cols=1,
                subplot_titles=[f"{prompt} - Run {run}" for run in prompt_runs],
                vertical_spacing=0.15,
            )
            
            # Process each run
            for run_idx, run_number in enumerate(prompt_runs, start=1):
                # Collect data for this (prompt, run) combination
                no_rag_successful = []
                no_rag_failed = []
                rag_successful = []
                rag_failed = []
                model_names = []
                
                for model in model_list:
                    no_rag_key = (prompt, run_number, model)
                    rag_key = (prompt, run_number, model)
                    
                    # Calculate NO_RAG values
                    no_rag_failed_val = 0
                    no_rag_successful_val = 0
                    if no_rag_key in no_rag_cumulative_failures:
                        final_iter, cumulative_failures = no_rag_cumulative_failures[no_rag_key]
                        max_iter = no_rag_max_iteration.get(no_rag_key, final_iter)
                        no_rag_failed_val = cumulative_failures
                        total_iterations = final_iter if final_iter == 50 else max_iter
                        no_rag_successful_val = total_iterations - no_rag_failed_val
                    
                    # Calculate RAG values
                    rag_failed_val = 0
                    rag_successful_val = 0
                    if rag_key in rag_cumulative_failures:
                        final_iter, cumulative_failures = rag_cumulative_failures[rag_key]
                        max_iter = rag_max_iteration.get(rag_key, final_iter)
                        rag_failed_val = cumulative_failures
                        total_iterations = final_iter if final_iter == 50 else max_iter
                        rag_successful_val = total_iterations - rag_failed_val
                    
                    # Only include models that have failures in either NO_RAG or RAG dataset
                    if no_rag_failed_val > 0 or rag_failed_val > 0:
                        model_names.append(model)
                        no_rag_successful.append(no_rag_successful_val)
                        no_rag_failed.append(no_rag_failed_val)
                        rag_successful.append(rag_successful_val)
                        rag_failed.append(rag_failed_val)
                
                if not model_names:
                    continue
                
                bar_colors = bar_colors_for_models(model_names, model_to_is_coding) if (model_to_is_coding and model_names) else None
                common_bar = dict(offsetgroup="norag", alignmentgroup="models", showlegend=(run_idx == 1), width=0.6)
                
                norag_succ_kw: Dict[str, object] = dict(name="NO_RAG Successful", x=model_names, y=no_rag_successful, text=None, hovertemplate="Model=%{x}<br>NO_RAG Successful=%{y} iterations<extra></extra>", marker_color="#2ca02c", **common_bar)
                if bar_colors:
                    norag_succ_kw["marker_color"] = bar_colors
                fig.add_trace(go.Bar(**norag_succ_kw), row=run_idx, col=1)
                
                norag_fail_kw: Dict[str, object] = dict(name="NO_RAG Failed", x=model_names, y=no_rag_failed, text=None, hovertemplate="Model=%{x}<br>NO_RAG Failed=%{y} iterations<extra></extra>", marker_color="#d62728", base=no_rag_successful, **common_bar)
                if bar_colors:
                    norag_fail_kw["marker_color"] = bar_colors
                fig.add_trace(go.Bar(**norag_fail_kw), row=run_idx, col=1)
                
                rag_common = dict(offsetgroup="rag", alignmentgroup="models", showlegend=(run_idx == 1), width=0.6)
                rag_succ_kw: Dict[str, object] = dict(name="RAG Successful", x=model_names, y=rag_successful, text=None, hovertemplate="Model=%{x}<br>RAG Successful=%{y} iterations<extra></extra>", marker_color="#90EE90", **rag_common)
                if bar_colors:
                    rag_succ_kw["marker_color"] = bar_colors
                fig.add_trace(go.Bar(**rag_succ_kw), row=run_idx, col=1)
                
                rag_fail_kw: Dict[str, object] = dict(name="RAG Failed", x=model_names, y=rag_failed, text=None, hovertemplate="Model=%{x}<br>RAG Failed=%{y} iterations<extra></extra>", marker_color="#FF6347", base=rag_successful, **rag_common)
                if bar_colors:
                    rag_fail_kw["marker_color"] = bar_colors
                fig.add_trace(go.Bar(**rag_fail_kw), row=run_idx, col=1)
                
                # Update x-axis for this subplot (model names on x-axis for vertical bars)
                fig.update_xaxes(
                    tickangle=-45,  # Rotate labels for readability
                    row=run_idx,
                    col=1,
                )
            
            # Update layout - set axis titles on middle row
            if len(prompt_runs) > 0:
                mid_row = (len(prompt_runs) + 1) // 2
                fig.update_xaxes(title_text="Model", row=mid_row, col=1)
                fig.update_yaxes(title_text="Number of Iterations", row=mid_row, col=1)
            
            # Calculate dynamic dimensions based on number of models
            num_models = len(model_list) if model_list else 1
            # Height: standard height per run
            dynamic_height = 500 * len(prompt_runs)
            # Width: increased significantly to space out models more (more pixels per model)
            dynamic_width = max(1800, num_models * 50)  # More spacing between models
            
            # For grouped stacked bars, we need to use a combination approach
            # Since Plotly doesn't directly support grouped stacked bars with barmode="group",
            # we'll use barmode="group" and handle stacking manually via base parameter
            # This should work: bars with the same offsetgroup will be grouped, and base will stack them
            fig.update_layout(
                title_text=f"NO_RAG vs RAG Success/Failure Comparison: {prompt} ({size_label})",
                template="plotly_white",
                barmode="group",  # Group NO_RAG and RAG bars side-by-side
                bargap=0.4,  # Add space between bars for different models (0.4 = 40% gap)
                height=dynamic_height,
                width=dynamic_width,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(b=200, l=80, r=50, t=100),  # Increased bottom margin for rotated labels
            )
            
            # Write HTML file
            output_path = os.path.join(output_subdir, f"{prompt}{filename_suffix}.html")
            write_html(fig, output_path)
            print(f"Created NO_RAG vs RAG success/failure plot: {output_path}")
        
        # Create plots for both groups
        create_plot_for_models(models_le_4b, "<=4B", "_le4b")
        create_plot_for_models(models_gt_4b, ">4B", "_gt4b")


def create_heatmap(rag_path: str, no_rag_path:str, output_dir: str) -> None:
    output_subdir = os.path.join(output_dir, "aggregate_results_line_graph")
    os.makedirs(output_subdir, exist_ok=True)

    # Load parameter size mapping for sorting
    param_size_map = load_model_size_mapping()

        # Track final base_commands_seen_so_far and unique_valid_commands per (prompt, run, model) - prefer iteration 50 or max
    final_unique_valid_cmds_NORAG: Dict[Tuple[str, int, str], Tuple[int, str]] = {}
    final_base_by_prompt_run_model_NORAG: Dict[Tuple[str, int, str], Tuple[int, str]] = {}
    final_unique_valid_cmds_RAG: Dict[Tuple[str, int, str], Tuple[int, str]] = {}
    final_base_by_prompt_run_model_RAG: Dict[Tuple[str, int, str], Tuple[int, str]] = {}

    prompts: Set[str] = set()
    runs: Set[int] = set()
    aggregation_base_commands = defaultdict(lambda: (int(), 0))
    aggregation_unique_valid_cmd = defaultdict(lambda: (int(), 0))
    avg_base_cmds = defaultdict(lambda: (int()))
    avg_unique_valid_cmds = defaultdict(lambda: (int()))

    try:
        with open(rag_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                prompt = row.get("prompt", "").strip()
                model = row.get("model", "").strip()
                #if not prompt or not model or (prompt != 'prompt1' and prompt != 'prompt2' and prompt != 'prompt3' and prompt != 'prompt4'):
                #if not prompt or not model or (prompt != 'prompt6'):
                if not prompt or not model or prompt == 'security-incremental-1': #or (prompt != 'security-incremental-1' and prompt != 'prompt5' and prompt != 'prompt7'):
                    continue
                
                try:
                    run_number = int(row.get("run_number", 0))
                except (ValueError, TypeError):
                    run_number = 0
                parameter_size = param_size_map.get(model, 0.0)
                try:
                    iteration = int(row.get("iteration", 0))
                except (ValueError, TypeError):
                    iteration = 0
                
                base_set_str = row.get("base_commands_seen_so_far", "")
                unique_valid_cmds = row.get("unique_valid_commands", "")
                if run_number > 0 and iteration > 0:
                    key = (prompt, run_number, model, parameter_size)
                    
                    # Track final base_commands_seen_so_far (prefer iteration 50 or max)
                    if key not in final_base_by_prompt_run_model_RAG:
                        final_base_by_prompt_run_model_RAG[key] = (iteration, base_set_str)
                    else:
                        prev_iter, _ = final_base_by_prompt_run_model_RAG[key]
                        # prefer iteration 50 if seen, otherwise keep the max
                        if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                            final_base_by_prompt_run_model_RAG[key] = (iteration, base_set_str)
                    
                    if key not in final_unique_valid_cmds_RAG:
                        final_unique_valid_cmds_RAG[key] = (iteration, unique_valid_cmds)
                    else:
                        prev_iter, _ = final_unique_valid_cmds_RAG[key]
                        # prefer iteration 50 if seen, otherwise keep the max
                        if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                            final_unique_valid_cmds_RAG[key] = (iteration, unique_valid_cmds)
                    
                    #NOURA
                    prompts.add(prompt)
                    runs.add(run_number)
        print(prompts)
        base_cmds_deltas = [[0 for _ in range(len(prompts))] for _ in range(len(prompts))]
        unique_valid_cmds_deltas = [[0 for _ in range(len(prompts))] for _ in range(len(prompts))]     
        with open(no_rag_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                prompt = row.get("prompt", "").strip()
                model = row.get("model", "").strip()
                #if not prompt or not model or (prompt != 'prompt1' and prompt != 'prompt2' and prompt != 'prompt3' and prompt != 'prompt4'):
                #if not prompt or not model or (prompt != 'prompt6'):
                if not prompt or not model or prompt == 'security-incremental-1': #and prompt != 'prompt5' and prompt != 'prompt7'):
                    continue
                
                try:
                    run_number = int(row.get("run_number", 0))
                except (ValueError, TypeError):
                    run_number = 0
                parameter_size = param_size_map.get(model, 0.0)
                try:
                    iteration = int(row.get("iteration", 0))
                except (ValueError, TypeError):
                    iteration = 0
                
                base_set_str = row.get("base_commands_seen_so_far", "")
                unique_valid_cmds = row.get("unique_valid_commands", "")
                if run_number > 0 and iteration > 0:
                    key = (prompt, run_number, model, parameter_size)
                    
                    # Track final base_commands_seen_so_far (prefer iteration 50 or max)
                    if key not in final_base_by_prompt_run_model_NORAG:
                        final_base_by_prompt_run_model_NORAG[key] = (iteration, base_set_str)
                    else:
                        prev_iter, _ = final_base_by_prompt_run_model_NORAG[key]
                        # prefer iteration 50 if seen, otherwise keep the max
                        if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                            final_base_by_prompt_run_model_NORAG[key] = (iteration, base_set_str)
                    
                    if key not in final_unique_valid_cmds_NORAG:
                        final_unique_valid_cmds_NORAG[key] = (iteration, unique_valid_cmds)
                    else:
                        prev_iter, _ = final_unique_valid_cmds_NORAG[key]
                        # prefer iteration 50 if seen, otherwise keep the max
                        if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                            final_unique_valid_cmds_NORAG[key] = (iteration, unique_valid_cmds)
        sorted_prompts = sorted(list(prompts))

        for prompt_index in range(0, len(sorted_prompts)):
            for second_prompt_index in range(0, len(prompts)):
                for key in list(final_base_by_prompt_run_model_RAG.keys()):
                        _,run_number,model, parameter_size = key
                        rag_base_cmds_set_prompta = ast.literal_eval(final_base_by_prompt_run_model_RAG[(sorted_prompts[prompt_index], run_number, model,parameter_size)][-1])
                        rag_base_cmds_set_promptb = ast.literal_eval(final_base_by_prompt_run_model_RAG[(sorted_prompts[second_prompt_index], run_number, model,parameter_size)][-1])
                        if base_cmds_deltas[prompt_index][second_prompt_index] == 0:
                            base_cmds_deltas[prompt_index][second_prompt_index] = [round(len(rag_base_cmds_set_prompta) - len(rag_base_cmds_set_promptb), 2)]
                        else: 
                             base_cmds_deltas[prompt_index][second_prompt_index].append(round(len(rag_base_cmds_set_prompta) - len(rag_base_cmds_set_promptb), 2))
                        if unique_valid_cmds_deltas[prompt_index][second_prompt_index] == 0:
                            unique_valid_cmds_deltas[prompt_index][second_prompt_index] = [round(int(final_unique_valid_cmds_RAG[(sorted_prompts[prompt_index], run_number, model,parameter_size)][-1]) - int(final_unique_valid_cmds_RAG[(sorted_prompts[second_prompt_index], run_number, model,parameter_size)][-1]), 2)]
                        else:
                             unique_valid_cmds_deltas[prompt_index][second_prompt_index].append(round(int(final_unique_valid_cmds_RAG[(sorted_prompts[prompt_index], run_number, model,parameter_size)][-1]) - int(final_unique_valid_cmds_RAG[(sorted_prompts[second_prompt_index], run_number, model,parameter_size)][-1]), 2))
                        #base_cmds_deltas[prompt_index][second_prompt_index] = round(len(rag_base_cmds_set_prompta) - len(rag_base_cmds_set_promptb), 2)
                        #unique_valid_cmds_deltas[prompt_index][second_prompt_index] = round(int(final_unique_valid_cmds_RAG[(sorted_prompts[prompt_index], run_number, model,parameter_size)][-1]) - int(final_unique_valid_cmds_RAG[(sorted_prompts[second_prompt_index], run_number, model,parameter_size)][-1]), 2)
       
        for row in range(0, len(base_cmds_deltas)):
            for col in range(0, len(base_cmds_deltas[row])):
                if isinstance(base_cmds_deltas[row][col], list):
                    base_cmds_deltas[row][col] = round(sum(base_cmds_deltas[row][col])/len(base_cmds_deltas[row][col]), 2)
        for row in range(0, len(unique_valid_cmds_deltas)):
            for col in range(0, len(unique_valid_cmds_deltas[row])):
                if isinstance(unique_valid_cmds_deltas[row][col], list):
                    unique_valid_cmds_deltas[row][col] = round(sum(unique_valid_cmds_deltas[row][col])/len(unique_valid_cmds_deltas[row][col]), 2)
        #labels = ["1_shot_v1", "1_shot_v2", "1_shot_v3", "1_shot_v4", "0_shot_v1", "2_shot_v1", "0_shot_v2"]
        labels = ["1_shot_v1", "1_shot_v2","0_shot_v1"]
        M = np.array(unique_valid_cmds_deltas)
        print(base_cmds_deltas)
        #dfM = pd.DataFrame(M, index=labels, columns=labels)
        #ax = sns.heatmap(dfM, annot=True, cmap="viridis", center=0)
        absmax = np.nanmax(np.abs(M))
        M_norm = M / absmax if absmax != 0 else M
        
        dfM = pd.DataFrame(M_norm, index=labels, columns=labels)
        ax = sns.heatmap(dfM, annot=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1)
        plt.tight_layout()
        plt.savefig("pairwise_heatmap_rag_unique_valid_cmds.png", dpi=300, bbox_inches="tight")
        plt.close() 


        base_cmds_deltas = [[0 for _ in range(len(prompts))] for _ in range(len(prompts))]
        unique_valid_cmds_deltas = [[0 for _ in range(len(prompts))] for _ in range(len(prompts))]     
        for prompt_index in range(0, len(sorted_prompts)):
            for second_prompt_index in range(0, len(prompts)):
                for key in list(final_base_by_prompt_run_model_NORAG.keys()):
                        _,run_number,model, parameter_size = key
                        rag_base_cmds_set_prompta = ast.literal_eval(final_base_by_prompt_run_model_NORAG[(sorted_prompts[prompt_index], run_number, model,parameter_size)][-1])
                        rag_base_cmds_set_promptb = ast.literal_eval(final_base_by_prompt_run_model_NORAG[(sorted_prompts[second_prompt_index], run_number, model,parameter_size)][-1])
                        if base_cmds_deltas[prompt_index][second_prompt_index] == 0:
                            base_cmds_deltas[prompt_index][second_prompt_index] = [round(len(rag_base_cmds_set_prompta) - len(rag_base_cmds_set_promptb), 2)]
                        else: 
                             base_cmds_deltas[prompt_index][second_prompt_index].append(round(len(rag_base_cmds_set_prompta) - len(rag_base_cmds_set_promptb), 2))
                        if unique_valid_cmds_deltas[prompt_index][second_prompt_index] == 0:
                            unique_valid_cmds_deltas[prompt_index][second_prompt_index] = [round(int(final_unique_valid_cmds_NORAG[(sorted_prompts[prompt_index], run_number, model,parameter_size)][-1]) - int(final_unique_valid_cmds_NORAG[(sorted_prompts[second_prompt_index], run_number, model,parameter_size)][-1]), 2)]
                        else:
                             unique_valid_cmds_deltas[prompt_index][second_prompt_index].append(round(int(final_unique_valid_cmds_NORAG[(sorted_prompts[prompt_index], run_number, model,parameter_size)][-1]) - int(final_unique_valid_cmds_NORAG[(sorted_prompts[second_prompt_index], run_number, model,parameter_size)][-1]), 2))

        for row in range(0, len(base_cmds_deltas)):
            for col in range(0, len(base_cmds_deltas[row])):
                if isinstance(base_cmds_deltas[row][col], list):
                    base_cmds_deltas[row][col] = round(sum(base_cmds_deltas[row][col])/len(base_cmds_deltas[row][col]), 2)
        for row in range(0, len(unique_valid_cmds_deltas)):
            for col in range(0, len(unique_valid_cmds_deltas[row])):
                if isinstance(unique_valid_cmds_deltas[row][col], list):
                    unique_valid_cmds_deltas[row][col] = round(sum(unique_valid_cmds_deltas[row][col])/len(unique_valid_cmds_deltas[row][col]), 2)
      
        #labels = ["1_shot_v1", "1_shot_v2", "1_shot_v3", "1_shot_v4", "0_shot_v1", "2_shot_v1", "0_shot_v2"]
        labels = ["1_shot_v1", "1_shot_v2", "0_shot_v1"]
        M = np.array(unique_valid_cmds_deltas)

        #dfM = pd.DataFrame(M, index=labels, columns=labels)
        #ax = sns.heatmap(dfM, annot=True, cmap="viridis", center=0)
        absmax = np.nanmax(np.abs(M))
        M_norm = M / absmax if absmax != 0 else M
        dfM = pd.DataFrame(M_norm, index=labels, columns=labels)
        ax = sns.heatmap(dfM, annot=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1)
        plt.tight_layout()
        plt.savefig("pairwise_heatmap_norag_unique_valid_cmds.png", dpi=300, bbox_inches="tight")
        plt.close() 
        M = np.array(unique_valid_cmds_deltas)

        dfM = pd.DataFrame(M, index=labels, columns=labels)
        ax = sns.heatmap(dfM, annot=True, cmap="viridis", center=0)
        #absmax = np.nanmax(np.abs(M))
        #M_norm = M / absmax if absmax != 0 else M
        dfM = pd.DataFrame(M_norm, index=labels, columns=labels)
        #ax = sns.heatmap(dfM, annot=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1)
        ax.set_xlabel("Prompts")
        ax.set_ylabel("Prompts")
        ax.set_title("Pairwise Heatmap of Base Command Prompt Deltas (N0 RAG)")
        plt.tight_layout()
        plt.savefig("pairwise_heatmap_norag_unique_valid_cmds2.png", dpi=300, bbox_inches="tight")
        plt.close() 
        '''
        for prompt_index in range(0, len(sorted_prompts)):
            for second_prompt_index in range(0, len(prompts)):
                for key in list(final_base_by_prompt_run_model_NORAG.keys()):
                    _,run_number,model, parameter_size = key
                    rag_base_cmds_set_prompta = ast.literal_eval(final_base_by_prompt_run_model_NORAG[(sorted_prompts[prompt_index], run_number, model,parameter_size)][-1])
                    rag_base_cmds_set_promptb = ast.literal_eval(final_base_by_prompt_run_model_NORAG[(sorted_prompts[second_prompt_index], run_number, model,parameter_size)][-1])
                    base_cmds_deltas[prompt_index][second_prompt_index] = round(len(rag_base_cmds_set_prompta) - len(rag_base_cmds_set_promptb), 2)
                    unique_valid_cmds_deltas[prompt_index][second_prompt_index] = round(int(final_unique_valid_cmds_NORAG[(sorted_prompts[prompt_index], run_number, model,parameter_size)][-1]) - int(final_unique_valid_cmds_NORAG[(sorted_prompts[second_prompt_index], run_number, model,parameter_size)][-1]), 2)
        print(base_cmds_deltas)       
        labels = ["prompt1", "prompt2", "prompt3", "prompt4", "prompt5", "prompt6", "prompt7", "original"]
        M = np.array(base_cmds_deltas)

        dfM = pd.DataFrame(M, index=labels, columns=labels)
        sns.heatmap(dfM, annot=True, cmap="coolwarm", center=0)
        plt.tight_layout()
        plt.savefig("pairwise_heatmap_norag.png", dpi=300, bbox_inches="tight")
        plt.close() 
        '''
        '''
        for key in list(final_base_by_prompt_run_model_RAG.keys()):
            prompt,run_number,model = key[:3]
            for prompt in prompts: 
            rag_base_cmds_set = ast.literal_eval(final_base_by_prompt_run_model_RAG[key][-1])
            no_rag_base_cmds_set = ast.literal_eval(final_base_by_prompt_run_model_NORAG[key][-1])
            base_cmds_deltas[(prompt, run_number, model)] = len(rag_base_cmds_set) - len(no_rag_base_cmds_set)
            #print(type(final_unique_valid_cmds_RAG[key][-1]), final_unique_valid_cmds_NORAG[key][-1])
            unique_valid_cmds_deltas[(prompt, run_number, model)] = int(final_unique_valid_cmds_RAG[key][-1]) - int(final_unique_valid_cmds_NORAG[key][-1])
        
        def group_and_average_by_prompt(supplied_dict):
            by_prompt = defaultdict(list)
            for (p, run, model), v in supplied_dict.items():
                by_prompt[p].append(v)
            for key, val in by_prompt.items():
                by_prompt[key] = round(sum(val)/len(val), 2) if val else 0.0
            return by_prompt
        
        
        base_by_prompt = group_and_average_by_prompt(base_cmds_deltas)
        unique_valid_by_prompt = group_and_average_by_prompt(unique_valid_cmds_deltas)

        overall_deltas_by_prompt = defaultdict(list)
        for key, value in base_by_prompt.items():
            overall_deltas_by_prompt[key].append(value)
            overall_deltas_by_prompt[key].append(unique_valid_by_prompt[key])
        
        heat = pd.DataFrame(
        list(overall_deltas_by_prompt.values()),
        index=list(overall_deltas_by_prompt.keys()),
        columns=["Unique_Base_delta", "Unique_Valid_delta"]
        )
      
        fig = px.imshow(
        heat,
        text_auto=".2f",
        aspect="auto",
        title="Heatmap"
        )
        fig.update_layout(template="plotly_white", xaxis_title="", yaxis_title="")
        
        cols = ["Unique_Base_delta", "Unique_Valid_delta"]
        
# z-score each column
        heat_z = heat.copy()
        std = heat[cols].std(ddof=0).replace(0, np.nan)   # avoid divide-by-zero
        heat_z[cols] = (heat[cols] - heat[cols].mean()) / std
        heat_z[cols] = heat_z[cols].fillna(0)

        # symmetric color scale around 0
        m = np.nanmax(np.abs(heat_z[cols].values))

        fig = px.imshow(
            heat_z[cols],
            text_auto=".2f",
            aspect="auto",
            zmin=-m,
            zmax=m,
            title="Z-score heatmap of deltas (RAG − NO_RAG): Unique_Base_Cmd_delta and Unique_Valid_Cmd_delta"
        )
        fig.update_layout(template="plotly_white", xaxis_title="", yaxis_title="Prompt")
        
        rows = list(heat.index)              # e.g., prompts
        count_vals = heat["UB_delta"].values    # or "UB_delta" etc
        score_vals = heat["UV_delta"].values    # or "UV_delta" etc

        fig = make_subplots(
            rows=1, cols=2,
            shared_yaxes=True,
            horizontal_spacing=0.08,
            subplot_titles=("UB_delta", "UV_delta")
        )

        # Heatmap 1 (Count) with its own colorscale + colorbar
        fig.add_trace(
            go.Heatmap(
                z=count_vals.reshape(-1, 1),
                x=["UB_delta"],
                y=rows,
                coloraxis="coloraxis",
                showscale=True,
                text=count_vals.reshape(-1, 1),
                texttemplate="%{text:.2f}",
                hovertemplate="Row=%{y}<br>Count=%{z}<extra></extra>",
            ),
            row=1, col=1
        )

        # Heatmap 2 (Score) with a DIFFERENT colorscale + colorbar
        fig.add_trace(
            go.Heatmap(
                z=score_vals.reshape(-1, 1),
                x=["UV_delta"],
                y=rows,
                coloraxis="coloraxis2",
                showscale=True,
                text=score_vals.reshape(-1, 1),
                texttemplate="%{text:.2f}",
                hovertemplate="Row=%{y}<br>Score=%{z}<extra></extra>",
            ),
            row=1, col=2
        )

        fig.update_layout(
            template="plotly_white",
            height=650,
            margin=dict(l=220, r=80, t=60, b=60),

            # left colorbar (Count)
            coloraxis=dict(
                colorbar=dict(title="UB_delta", x=1.02)   # position just right of plots
            ),

            # right colorbar (Score)
            coloraxis2=dict(
                colorbar=dict(title="UV_delta", x=1.12)   # push farther right
            )
        )
    
        # Make y top-to-bottom like your screenshot
        fig.update_yaxes(autorange="reversed")
        output_path = os.path.join(output_subdir, "heatmap.html")
        write_html(fig, output_path)
        '''
    except Exception as e:
        print(f"Error reading NO_RAG CSV file {csv_path}: {e}")
        return

def create_regression_scatter_plot(rag_path: str, no_rag_path:str, output_dir: str) -> None:
        # --- Helper: LOWESS + bootstrap confidence band
    def lowess_with_confidence_band(x, y, frac=0.35, n_boot=250, ci=95, grid_size=200, seed=42):
        """
        Bootstrap confidence band for LOWESS.
        Returns x_grid, y_center, y_lo, y_hi.
        """
        rng = np.random.default_rng(seed)

        x = np.asarray(x, float)
        y = np.asarray(y, float)

        x_grid = np.linspace(x.min(), x.max(), grid_size)

        # center line
        fit = lowess(y, x, frac=frac, return_sorted=True)
        y_center = np.interp(x_grid, fit[:, 0], fit[:, 1])

        # bootstrap
        n = len(x)
        boots = np.empty((n_boot, grid_size), float)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            xb, yb = x[idx], y[idx]
            fit_b = lowess(yb, xb, frac=frac, return_sorted=True)
            boots[b] = np.interp(x_grid, fit_b[:, 0], fit_b[:, 1])

        alpha = (100 - ci) / 2
        y_lo = np.percentile(boots, alpha, axis=0)
        y_hi = np.percentile(boots, 100 - alpha, axis=0)

        return x_grid, y_center, y_lo, y_hi
    output_subdir = os.path.join(output_dir, "regression_line_graph")
    os.makedirs(output_subdir, exist_ok=True)
    
        # Load parameter size mapping for sorting
    param_size_map = load_model_size_mapping()
        
    final_unique_valid_cmds_NORAG: Dict[Tuple[str, int, str], Tuple[int, str]] = {}
    final_base_by_prompt_run_model_NORAG: Dict[Tuple[str, int, str], Tuple[int, str]] = {}
    final_unique_valid_cmds_RAG: Dict[Tuple[str, int, str], Tuple[int, str]] = {}
    final_base_by_prompt_run_model_RAG: Dict[Tuple[str, int, str], Tuple[int, str]] = {}

    prompts: Set[str] = set()
    runs: Set[int] = set()
    total_base_commands = defaultdict(lambda: (list()))
    aggregation_unique_valid_cmd = defaultdict(lambda: (int(), 0))
    avg_base_cmds = defaultdict(lambda: (int()))
    avg_unique_valid_cmds = defaultdict(lambda: (int()))
    total_base_commands_rag = defaultdict(lambda: (list()))
    total_unique_valid_cmds = defaultdict(lambda: (list()))
    total_unique_valid_cmds_rag = defaultdict(lambda: (list()))

    try:
        with open(rag_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                prompt = row.get("prompt", "").strip()
                model = row.get("model", "").strip()
                run = row.get("run_number", "").strip()
                #if not prompt or not model or (prompt != 'prompt1' and prompt != 'prompt2' and prompt != 'prompt3' and prompt != 'prompt4'):
                #if not prompt or not model or (prompt != 'prompt6'):
                if not prompt or not model or prompt == 'security-incremental-1': #or (prompt != 'security-incremental-1' and prompt != 'prompt5' and prompt != 'prompt7'):
                    continue
                
                try:
                    run_number = int(row.get("run_number", 0))
                except (ValueError, TypeError):
                    run_number = 0
                parameter_size = param_size_map.get(model, 0.0)
                try:
                    iteration = int(row.get("iteration", 0))
                except (ValueError, TypeError):
                    iteration = 0
                
                base_set_str = row.get("base_commands_seen_so_far", "")
                unique_valid_cmds = row.get("unique_valid_commands", "")
                if run_number > 0 and iteration > 0:
                    key = (prompt, run_number, model, parameter_size)
                    
                    # Track final base_commands_seen_so_far (prefer iteration 50 or max)
                    if key not in final_base_by_prompt_run_model_RAG:
                        final_base_by_prompt_run_model_RAG[key] = (iteration, base_set_str)
                    else:
                        prev_iter, _ = final_base_by_prompt_run_model_RAG[key]
                        # prefer iteration 50 if seen, otherwise keep the max
                        if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                            final_base_by_prompt_run_model_RAG[key] = (iteration, base_set_str)
                    
                    if key not in final_unique_valid_cmds_RAG:
                        final_unique_valid_cmds_RAG[key] = (iteration, unique_valid_cmds)
                    else:
                        prev_iter, _ = final_unique_valid_cmds_RAG[key]
                        # prefer iteration 50 if seen, otherwise keep the max
                        if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                            final_unique_valid_cmds_RAG[key] = (iteration, unique_valid_cmds)
                    
                    #NOURA
                    prompts.add(prompt)
                    runs.add(run_number)

        with open(no_rag_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                prompt = row.get("prompt", "").strip()
                model = row.get("model", "").strip()
                #if not prompt or not model or (prompt != 'prompt1' and prompt != 'prompt2' and prompt != 'prompt3' and prompt != 'prompt4'):
                #if not prompt or not model or (prompt != 'prompt6'):
                if not prompt or not model or prompt == 'security-incremental-1': # or (prompt != 'security-incremental-1' and prompt != 'prompt5' and prompt != 'prompt7'):
                    continue
                
                try:
                    run_number = int(row.get("run_number", 0))
                except (ValueError, TypeError):
                    run_number = 0
                parameter_size = param_size_map.get(model, 0.0)
                try:
                    iteration = int(row.get("iteration", 0))
                except (ValueError, TypeError):
                    iteration = 0
                
                base_set_str = row.get("base_commands_seen_so_far", "")
                unique_valid_cmds = row.get("unique_valid_commands", "")
                if run_number > 0 and iteration > 0:
                    key = (prompt, run_number, model, parameter_size)
                    
                    # Track final base_commands_seen_so_far (prefer iteration 50 or max)
                    if key not in final_base_by_prompt_run_model_NORAG:
                        final_base_by_prompt_run_model_NORAG[key] = (iteration, base_set_str)
                    else:
                        prev_iter, _ = final_base_by_prompt_run_model_NORAG[key]
                        # prefer iteration 50 if seen, otherwise keep the max
                        if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                            final_base_by_prompt_run_model_NORAG[key] = (iteration, base_set_str)
                    
                    if key not in final_unique_valid_cmds_NORAG:
                        final_unique_valid_cmds_NORAG[key] = (iteration, unique_valid_cmds)
                    else:
                        prev_iter, _ = final_unique_valid_cmds_NORAG[key]
                        # prefer iteration 50 if seen, otherwise keep the max
                        if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                            final_unique_valid_cmds_NORAG[key] = (iteration, unique_valid_cmds)
        #print(f"final_base_by_prompt_run_model_NORAG: {final_unique_valid_cmds_NORAG}")
        #print(f"final_base_by_prompt_run_model_RAG: {final_unique_valid_cmds_RAG}")
        for key in list(final_base_by_prompt_run_model_NORAG.keys()):
            prompt,run_number,model,parameter_size = key
            base_cmds_set_norag = ast.literal_eval(final_base_by_prompt_run_model_NORAG[key][-1])
            total_base_commands[(model, parameter_size)].append(len(base_cmds_set_norag))

        for key in list(final_base_by_prompt_run_model_RAG.keys()):
            prompt,run_number,model,parameter_size = key
            base_cmds_set_rag = ast.literal_eval(final_base_by_prompt_run_model_RAG[key][-1])
            total_base_commands_rag[(model, parameter_size)].append(len(base_cmds_set_rag))

        for key in list(final_unique_valid_cmds_NORAG.keys()):
            prompt, run_number, model, parameter_size = key
            unique_valid_cmds = int(final_unique_valid_cmds_NORAG[key][-1])
            total_unique_valid_cmds[(model, parameter_size)].append(unique_valid_cmds)

        for key in list(final_unique_valid_cmds_RAG.keys()):
            prompt, run_number, model, parameter_size = key
            unique_valid_cmds_rag = int(final_unique_valid_cmds_RAG[key][-1])
            total_unique_valid_cmds_rag[(model, parameter_size)].append(unique_valid_cmds_rag)
        
        #labels = ["prompt1","prompt2","prompt3","prompt4","prompt5","prompt6","prompt7"]  # adjust if needed
        labels = ["prompt1", "prompt4", "prompt7"]
        base_cmds_avg_rag = defaultdict(lambda: (int()))
        rows = [] 

        for (model, param_size), ys in total_base_commands.items():
            for i, y in enumerate(ys):
                rows.append({
                    "model": model,
                    "param_size": float(param_size),
                    "prompt": labels[i] if i < len(labels) else f"idx_{i}",
                    "y": y,
                    "prompt_index": i,
                    "condition": "No RAG",
                })

        # --- RAG
        for (model, param_size), ys in total_base_commands_rag.items():
            for i, y in enumerate(ys):
                rows.append({
                    "model": model,
                    "param_size": float(param_size),
                    "prompt": labels[i] if i < len(labels) else f"idx_{i}",
                    "y": y,
                    "prompt_index": i,
                    "condition": "RAG",
                })
        
        df = pd.DataFrame(rows)
                # jitter for display (multiplicative)
        j = 0.10
        df["param_jitter"] = df["param_size"] * (1 + np.random.uniform(-j, j, size=len(df)))
        fig = go.Figure()
        
        # Scatter: both datasets, different colors
        fig = px.scatter(
            df,
            x="param_jitter",
            y="y",
            color="condition",
            hover_data=["model", "prompt", "prompt_index", "param_size"],
            title="Unique Base Commands Generated vs Model Size (No RAG vs RAG)",
        )
        fig.update_traces(marker=dict(opacity=0.25, size=7))
        
        # LOWESS + 95% confidence band per condition
        line_colors = {"No RAG": "darkblue", "RAG": "red"}

        for cond, g in df.groupby("condition"):
            fit_g = g[(g["y"] > 0) & (g["param_size"] > 0)].copy()
            if len(fit_g) < 15:
                continue

            x = fit_g["param_size"].to_numpy(float)
            y = fit_g["y"].to_numpy(float)

            x_grid, y_center, y_lo, y_hi = lowess_with_confidence_band(
                x, y, frac=0.35, n_boot=250, ci=95, grid_size=200, seed=42
            )

            color = line_colors.get(cond, "gray")

            # Confidence band (draw first)
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_grid, x_grid[::-1]]),
                y=np.concatenate([y_hi, y_lo[::-1]]),
                fill="toself",
                fillcolor=color,
                opacity=0.15,
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                name=f"{cond} 95% CI",
                showlegend=False,
            ))

            # Center LOWESS line
            fig.add_trace(go.Scatter(
                x=x_grid,
                y=y_center,
                mode="lines",
                line=dict(color=color, width=4),
                hovertemplate="param_size=%{x:.2f}<br>trend=%{y:.2f}<extra></extra>",
            ))

        fig.update_layout(
            xaxis_title="Parameter size (B)",
            yaxis_title="Unique Base Commands Generated",
                paper_bgcolor="white",
            plot_bgcolor="white",
        )
        fig.update_layout(showlegend=False)
        fig.update_xaxes(showgrid=True, gridcolor="lightgray", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="lightgray", zeroline=False)
        fig.update_xaxes(tickfont=dict(size=24))
        fig.update_yaxes(tickfont=dict(size=24))
        fig.update_layout(
        xaxis_title_font=dict(size=20),
         yaxis_title_font=dict(size=20),
)
        # save
        output_path = os.path.join(output_subdir, "scatter_norag_vs_rag_lowess.html")
        write_html(fig, output_path)
        fig.write_image(
            os.path.join(output_subdir, "scatter_norag_vs_rag_lowess.png"),
            width=1400,
            height=900,
            scale=2,
        )
        fig.write_image("scatter_norag_vs_rag_lowess.png", width=1400, height=900, scale=2)
        rows = []
        for (model, param_size), ys in total_unique_valid_cmds.items():
           
            for i, y in enumerate(ys):
       
                rows.append({
                "model": model,
                "param_size": param_size,
                "prompt": labels[i] if i < len(labels) else f"idx_{i}",
                "y": y,
                "prompt_index": i
            })
        
        df = pd.DataFrame(rows)

        df["param_jitter"] = df["param_size"] * (1 + np.random.uniform(-0.03, 0.03, size=len(df)))
        fig = px.scatter(
            df,
            x="param_jitter",
            y="y",
                  # makes it readable; remove if you truly want all same color
            hover_data=["model"],
            title="All list values vs parameter size"
        )
        
        fit_df = df[df["y"] > 0].copy()
   
        x = fit_df["param_size"].to_numpy(float)
        y = fit_df["y"].to_numpy(float)

        smoothed = lowess(y, x, frac=0.3, return_sorted=True)  # tweak frac 0.2–0.5

        fig.add_trace(go.Scatter(
            x=smoothed[:,0], y=smoothed[:,1],
            mode="lines",
            name="LOWESS trend (y>0)"
        ))
        fig.update_layout(xaxis_title="Parameter size (B)", yaxis_title="Value")
        output_path = os.path.join(output_subdir, "regression_scatter_rag_unique_valid.html")
        write_html(fig, output_path)

        rows = []
        for (model, param_size), ys in total_base_commands_rag.items():
            for i, y in enumerate(ys):
                rows.append({
                    "model": model,
                    "param_size": param_size,
                    "prompt": labels[i] if i < len(labels) else f"idx_{i}",
                    "y": y,
                    "prompt_index": i
                })
        
        df = pd.DataFrame(rows)
        fit_df = df[df["y"] > 0].copy()
        x = fit_df["param_size"].to_numpy(float)
        y = fit_df["y"].to_numpy(float)

        smoothed = lowess(y, x, frac=0.3, return_sorted=True)  # tweak frac 0.2–0.5

        fig.add_trace(go.Scatter(
            x=smoothed[:,0], y=smoothed[:,1],
            mode="lines",
            name="LOWESS trend (y>0)"
        ))
        df["param_jitter"] = df["param_size"] * (1 + np.random.uniform(-0.03, 0.03, size=len(df)))
        fig = px.scatter(
            df,
            x="param_jitter",
            y="y",
                      # makes it readable; remove if you truly want all same color
            hover_data=["model", "prompt", "prompt_index"],
            title="All list values vs parameter size"
        )

        fig.update_layout(xaxis_title="Parameter size (B)", yaxis_title="Unique Valid Commands Generated")
        output_path = os.path.join(output_subdir, "regression_scatter_norag_uniquevalid.html")
        write_html(fig, output_path)
    except Exception as e:
        print(f"Error reading NO_RAG CSV file {rag_path}: {e}")
        return
def create_aggregate_results(rag_path: str, no_rag_path:str, output_dir: str) -> None:
        output_subdir = os.path.join(output_dir, "aggregate_results_line_graph")
        os.makedirs(output_subdir, exist_ok=True)
    
        # Load parameter size mapping for sorting
        param_size_map = load_model_size_mapping()
        
        def run_aggregation(csv_path: str) -> Dict[Tuple[str, int, str], Tuple[int, str]]:
             # Track final base_commands_seen_so_far and unique_valid_commands per (prompt, run, model) - prefer iteration 50 or max
            final_unique_valid_cmds: Dict[Tuple[str, int, str], Tuple[int, str]] = {}
            final_base_by_prompt_run_model: Dict[Tuple[str, int, str], Tuple[int, str]] = {}

            prompts: Set[str] = set()
            runs: Set[int] = set()
            aggregation_base_commands = defaultdict(lambda: (int(), 0))
            aggregation_unique_valid_cmd = defaultdict(lambda: (int(), 0))
            avg_base_cmds = defaultdict(lambda: (int()))
            avg_unique_valid_cmds = defaultdict(lambda: (int()))
        
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    
                    for row in reader:
                        prompt = row.get("prompt", "").strip()
                        model = row.get("model", "").strip()
                        #if not prompt or not model or (prompt != 'prompt1' and prompt != 'prompt2' and prompt != 'prompt3' and prompt != 'prompt4'):
                        #if not prompt or not model or (prompt != 'prompt6'):
                        if not prompt or not model or prompt == 'security-incremental-1': #and prompt != 'prompt5' and prompt != 'prompt7'):
                            continue
                        
                        try:
                            run_number = int(row.get("run_number", 0))
                        except (ValueError, TypeError):
                            run_number = 0
                        parameter_size = param_size_map.get(model, 0.0)
                        try:
                            iteration = int(row.get("iteration", 0))
                        except (ValueError, TypeError):
                            iteration = 0
                        
                        base_set_str = row.get("base_commands_seen_so_far", "")
                        unique_valid_cmds = row.get("unique_valid_commands", "")
                        if run_number > 0 and iteration > 0:
                            key = (prompt, run_number, model, parameter_size)
                            
                            # Track final base_commands_seen_so_far (prefer iteration 50 or max)
                            if key not in final_base_by_prompt_run_model:
                                final_base_by_prompt_run_model[key] = (iteration, base_set_str)
                            else:
                                prev_iter, _ = final_base_by_prompt_run_model[key]
                                # prefer iteration 50 if seen, otherwise keep the max
                                if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                                    final_base_by_prompt_run_model[key] = (iteration, base_set_str)
                            
                            if key not in final_unique_valid_cmds:
                                final_unique_valid_cmds[key] = (iteration, unique_valid_cmds)
                            else:
                                prev_iter, _ = final_unique_valid_cmds[key]
                                # prefer iteration 50 if seen, otherwise keep the max
                                if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                                    final_unique_valid_cmds[key] = (iteration, unique_valid_cmds)
                            
                            #NOURA
                            prompts.add(prompt)
                            runs.add(run_number)
                
                pos = -1  # which element in the key tuple you care about
                counts = Counter(k[pos] for k in final_base_by_prompt_run_model.keys())
                only_once = {v for v, c in counts.items() if c == 8}
                for key in list(final_base_by_prompt_run_model.keys()):
                    parameter_size = key[-1]
                    
                    val = list(ast.literal_eval(final_base_by_prompt_run_model[key][-1]))
                    if parameter_size == 10.7:
                        cmds, count = aggregation_base_commands[10]
                        aggregation_base_commands[10]=  (cmds + len(val), count + 1)
                    elif parameter_size == 7.6 or parameter_size == 7.2 or parameter_size == 6.9: 
                        cmds, count = aggregation_base_commands[7]
                        aggregation_base_commands[7]=  (cmds + len(val), count + 1)
                    elif parameter_size == 29.5:
                        cmds, count = aggregation_base_commands[30]
                        aggregation_base_commands[30]=  (cmds + len(val), count + 1)
                    elif parameter_size == 33:
                        cmds, count = aggregation_base_commands[32]
                        aggregation_base_commands[32]=  (cmds + len(val), count + 1)
                    elif parameter_size == 20:
                        cmds, count = aggregation_base_commands[22]
                        aggregation_base_commands[22]=  (cmds + len(val), count + 1)
                    elif parameter_size == 4.5:
                        cmds, count = aggregation_base_commands[4]
                        aggregation_base_commands[4]=  (cmds + len(val), count + 1)
                    elif parameter_size == 0.6:
                        cmds, count = aggregation_base_commands[0.5]
                        aggregation_base_commands[0.5]=  (cmds + len(val), count + 1)
                    elif parameter_size == 0.27:
                        cmds, count = aggregation_base_commands[0.36]
                        aggregation_base_commands[0.36]=  (cmds + len(val), count + 1)
                    elif parameter_size == 1.29:
                        cmds, count = aggregation_base_commands[1.5]
                        aggregation_base_commands[1.5]=  (cmds + len(val), count + 1)
                    elif parameter_size == 1.6:
                        cmds, count = aggregation_base_commands[1.5]
                        aggregation_base_commands[1.5]=  (cmds + len(val), count + 1)
                    else:
                        cmds, count = aggregation_base_commands[parameter_size]
                        aggregation_base_commands[parameter_size]=  (cmds + len(val), count + 1)

                for key in list(aggregation_base_commands.keys()):
                    cmds, count = aggregation_base_commands[key]
                    aggregation_base_commands[key] = (cmds, count)
                    avg_base_cmds[key] = round(cmds / count, 2)

                
                for key in list(final_unique_valid_cmds.keys()):
                    val = ast.literal_eval(final_unique_valid_cmds[key][-1])
                        #print(val, type(val))
                    parameter_size = key[-1]
                    if parameter_size == 10.7:
                        cmds, count = aggregation_unique_valid_cmd[10]
                        aggregation_unique_valid_cmd[10]=  (cmds + val, count + 1)
                    elif parameter_size == 7.6 or parameter_size == 7.2 or parameter_size == 6.9: 
                        cmds, count = aggregation_unique_valid_cmd[7]
                        aggregation_unique_valid_cmd[7]=  (cmds + val, count + 1)
                    elif parameter_size == 29.5:
                        cmds, count = aggregation_unique_valid_cmd[30]
                        aggregation_unique_valid_cmd[30]=  (cmds + val, count + 1)
                    elif parameter_size == 33:
                        cmds, count = aggregation_unique_valid_cmd[32]
                        aggregation_unique_valid_cmd[32]=  (cmds + val, count + 1)
                    elif parameter_size == 20:
                        cmds, count = aggregation_unique_valid_cmd[22]
                        aggregation_unique_valid_cmd[22]=  (cmds + val, count + 1)
                    elif parameter_size == 4.5:
                        cmds, count = aggregation_unique_valid_cmd[4]
                        aggregation_unique_valid_cmd[4]=  (cmds + val, count + 1)
                    elif parameter_size == 0.6:
                        cmds, count = aggregation_unique_valid_cmd[0.5]
                        aggregation_unique_valid_cmd[0.5]=  (cmds + val, count + 1)
                    elif parameter_size == 0.27:
                        cmds, count = aggregation_unique_valid_cmd[0.36]
                        aggregation_unique_valid_cmd[0.36]=  (cmds +  val, count + 1)
                    elif parameter_size == 1.29:
                        cmds, count = aggregation_unique_valid_cmd[1.5]
                        aggregation_unique_valid_cmd[1.5]=  (cmds + val, count + 1)
                    elif parameter_size == 1.6:
                        cmds, count = aggregation_unique_valid_cmd[1.5]
                        aggregation_unique_valid_cmd[1.5]=  (cmds + val, count + 1)
                    else:
                        cmds, count = aggregation_unique_valid_cmd[parameter_size]
                        aggregation_unique_valid_cmd[parameter_size]=  (cmds + val, count + 1)
            
                for key in list(aggregation_unique_valid_cmd.keys()):
                    cmd_val, count = aggregation_unique_valid_cmd[key]
                    avg_unique_valid_cmds[key] = round(cmd_val / count,2)
                
                return avg_base_cmds, avg_unique_valid_cmds
            except Exception as e:
                print(f"Error reading NO_RAG CSV file {csv_path}: {e}")
                return
            
        avg_base_cmds_rag, avg_unique_valid_cmds_rag = run_aggregation(rag_path)
        avg_base_cmds_non_rag, avg_unique_valid_cmds_non_rag = run_aggregation(no_rag_path)



# union of keys so both series align
        y_labels = sorted(set(avg_base_cmds_rag.keys()) | set(avg_base_cmds_non_rag.keys()))

        x_vals_rag = [avg_base_cmds_rag.get(k, None) for k in y_labels]
        x_vals_non = [avg_base_cmds_non_rag.get(k, None) for k in y_labels]

    # compute xmin/xmax ignoring None
        all_x = [v for v in (x_vals_rag + x_vals_non) if v is not None]
        xmin = min(all_x) if all_x else 0
        xmax = max(all_x) if all_x else 1

        fig = go.Figure()

        # dotted horizontal guides across each category row
        for y in y_labels:
            fig.add_trace(go.Scatter(
                x=[xmin, xmax],
                y=[y, y],
                mode="lines",
                line=dict(dash="dot", width=1, color="rgba(0,0,0,0.45)"),
                hoverinfo="skip",
                showlegend=False
            ))

        # dots: RAG
        fig.add_trace(go.Scatter(
            x=x_vals_rag,
            y=y_labels,
            mode="markers",
            marker=dict(size=9),
            name="RAG"
        ))

        # dots: NO_RAG
        fig.add_trace(go.Scatter(
            x=x_vals_non,
            y=y_labels,
            mode="markers",
            marker=dict(size=9, symbol="diamond"),
            name="NO_RAG"
        ))

        fig.update_layout(
            title="Avg Base Commands by Parameter Size",
            xaxis_title="Avg Base Commands",
            yaxis_title="Parameter Size (B)",
            template="plotly_white",
            margin=dict(l=220, r=30, t=50, b=60),
            yaxis=dict(
                type="category",
                categoryorder="array",
                categoryarray=y_labels,
                autorange="reversed",
            ),
        )

        output_path = os.path.join(output_subdir, "aggregation_base_commands.html")
        write_html(fig, output_path)

        y_labels = sorted(set(avg_unique_valid_cmds_non_rag.keys()) | set(avg_unique_valid_cmds_rag.keys()))

        x_vals_rag = [avg_unique_valid_cmds_rag.get(k, None) for k in y_labels]
        x_vals_non = [avg_unique_valid_cmds_non_rag.get(k, None) for k in y_labels]

    # compute xmin/xmax ignoring None
        all_x = [v for v in (x_vals_rag + x_vals_non) if v is not None]
        xmin = min(all_x) if all_x else 0
        xmax = max(all_x) if all_x else 1

        fig = go.Figure()

        # dotted horizontal guides across each category row
        for y in y_labels:
            fig.add_trace(go.Scatter(
                x=[xmin, xmax],
                y=[y, y],
                mode="lines",
                line=dict(dash="dot", width=1, color="rgba(0,0,0,0.45)"),
                hoverinfo="skip",
                showlegend=False
            ))

        # dots: RAG
        fig.add_trace(go.Scatter(
            x=x_vals_rag,
            y=y_labels,
            mode="markers",
            marker=dict(size=9),
            name="RAG"
        ))

        # dots: NO_RAG
        fig.add_trace(go.Scatter(
            x=x_vals_non,
            y=y_labels,
            mode="markers",
            marker=dict(size=9, symbol="diamond"),
            name="NO_RAG"
        ))

        fig.update_layout(
            title="Avg Unique valid commands by Parameter Size",
            xaxis_title="Avg Unique valid commands",
            yaxis_title="Parameter Size (B)",
            template="plotly_white",
            margin=dict(l=220, r=30, t=50, b=60),
            yaxis=dict(
                type="category",
                categoryorder="array",
                categoryarray=y_labels,
                autorange="reversed",
            ),
        )

        output_path = os.path.join(output_subdir, "aggregation_unique_valid_cmds.html")
        write_html(fig, output_path)
        
        
def create_norag_vs_rag_base_commands_plots(no_rag_csv_path: str, rag_csv_path: str, output_dir: str, model_to_is_coding: Optional[Dict[str, bool]] = None) -> None:
    """Create grouped bar charts comparing average base commands per model between NO_RAG and RAG experiments.
    
    For each prompt, creates one HTML file with subplots (one per run).
    Each subplot shows grouped bar charts with:
    - X-axis: Model names (sorted by parameter size)
    - Y-axis: Average number of base commands
    - Two bars per model side-by-side: NO_RAG (dark green) and RAG (dark purple)
    
    Args:
        no_rag_csv_path: Path to NO_RAG_KORAD_results.csv file
        rag_csv_path: Path to RAG_KORAD_results.csv file
        output_dir: Base output directory (files will be saved in norag_vs_rag_base_commands subdirectory)
    """
    output_subdir = os.path.join(output_dir, "norag_vs_rag_base_commands")
    os.makedirs(output_subdir, exist_ok=True)
    
    # Load parameter size mapping for sorting
    param_size_map = load_model_size_mapping()
    
    # Check if CSV files exist
    if not os.path.exists(no_rag_csv_path):
        print(f"Warning: NO_RAG CSV file not found at {no_rag_csv_path}")
        print("Skipping NO_RAG vs RAG base commands plots.")
        return
    
    if not os.path.exists(rag_csv_path):
        print(f"Warning: RAG CSV file not found at {rag_csv_path}")
        print("Skipping NO_RAG vs RAG base commands plots.")
        return
    
    # Helper function to parse set string
    def parse_base_set(set_str: str) -> Set[str]:
        """Parse base_commands_seen_so_far string into a set of commands."""
        s = (set_str or "").strip()
        if s == "set()" or s == "{}" or s == "":
            return set()
        else:
            # extract quoted items
            items = re.findall(r"'([^']*)'|\"([^\"]*)\"", s)
            return set([a or b for a, b in items])
    
    # Track final base_commands_seen_so_far per (prompt, run, model) for both datasets
    no_rag_final_base: Dict[Tuple[str, int, str], Tuple[int, str]] = {}
    rag_final_base: Dict[Tuple[str, int, str], Tuple[int, str]] = {}
    prompts: Set[str] = set()
    runs: Set[int] = set()
    
    # Read NO_RAG CSV file
    try:
        with open(no_rag_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            # Check if base_commands_seen_so_far column exists
            if "base_commands_seen_so_far" not in reader.fieldnames:
                print(f"Warning: NO_RAG CSV file {no_rag_csv_path} does not contain 'base_commands_seen_so_far' column.")
                print("Skipping NO_RAG vs RAG base commands plots.")
                return
            
            for row in reader:
                prompt = row.get("prompt", "").strip()
                model = row.get("model", "").strip()
                
                if not prompt or not model:
                    continue
                
                try:
                    run_number = int(row.get("run_number", 0))
                except (ValueError, TypeError):
                    run_number = 0
                
                try:
                    iteration = int(row.get("iteration", 0))
                except (ValueError, TypeError):
                    iteration = 0
                
                base_set_str = row.get("base_commands_seen_so_far", "")
                
                if run_number > 0 and iteration > 0:
                    key = (prompt, run_number, model)
                    
                    # Track final base_commands_seen_so_far (prefer iteration 50 or max)
                    if key not in no_rag_final_base:
                        no_rag_final_base[key] = (iteration, base_set_str)
                    else:
                        prev_iter, _ = no_rag_final_base[key]
                        # prefer iteration 50 if seen, otherwise keep the max
                        if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                            no_rag_final_base[key] = (iteration, base_set_str)
                    
                    prompts.add(prompt)
                    runs.add(run_number)
    except Exception as e:
        print(f"Error reading NO_RAG CSV file {no_rag_csv_path}: {e}")
        return
    
    # Read RAG CSV file
    try:
        with open(rag_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            # Check if base_commands_seen_so_far column exists
            if "base_commands_seen_so_far" not in reader.fieldnames:
                print(f"Warning: RAG CSV file {rag_csv_path} does not contain 'base_commands_seen_so_far' column.")
                print("Skipping NO_RAG vs RAG base commands plots.")
                return
            
            for row in reader:
                prompt = row.get("prompt", "").strip()
                model = row.get("model", "").strip()
                
                if not prompt or not model:
                    continue
                
                try:
                    run_number = int(row.get("run_number", 0))
                except (ValueError, TypeError):
                    run_number = 0
                
                try:
                    iteration = int(row.get("iteration", 0))
                except (ValueError, TypeError):
                    iteration = 0
                
                base_set_str = row.get("base_commands_seen_so_far", "")
                
                if run_number > 0 and iteration > 0:
                    key = (prompt, run_number, model)
                    
                    # Track final base_commands_seen_so_far (prefer iteration 50 or max)
                    if key not in rag_final_base:
                        rag_final_base[key] = (iteration, base_set_str)
                    else:
                        prev_iter, _ = rag_final_base[key]
                        # prefer iteration 50 if seen, otherwise keep the max
                        if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                            rag_final_base[key] = (iteration, base_set_str)
                    
                    prompts.add(prompt)
                    runs.add(run_number)
    except Exception as e:
        print(f"Error reading RAG CSV file {rag_csv_path}: {e}")
        return
    
    if not prompts:
        print("Warning: No valid data found in CSV files.")
        return
    
    # For each prompt, create two HTML files (one for <=4B, one for >4B) with subplots per run
    for prompt in sorted(prompts):
        # Get all models for this prompt (from both datasets)
        models_set: Set[str] = set()
        for (p, r, m) in list(no_rag_final_base.keys()) + list(rag_final_base.keys()):
            if p == prompt:
                models_set.add(m)
        
        if not models_set:
            continue
        
        # Sort models by parameter size (ascending), with missing models sorted last
        models = sorted(models_set, key=lambda m: param_size_map.get(m, float('inf')))
        
        # Split models by parameter size
        models_le_4b = []
        models_gt_4b = []
        
        for model in models:
            param_size = param_size_map.get(model, 0.0)
            if param_size <= 4.0:
                models_le_4b.append(model)
            elif param_size > 4.0:
                models_gt_4b.append(model)
        
        # Get all runs for this prompt
        prompt_runs = sorted([r for r in runs if any(p == prompt for p, _, _ in no_rag_final_base.keys()) or any(p == prompt for p, _, _ in rag_final_base.keys())])
        
        if not prompt_runs:
            continue
        
        # Helper function to create plot for a group of models
        def create_plot_for_models(model_list: List[str], size_label: str, filename_suffix: str) -> None:
            if not model_list:
                return
            
            # Create subplots: one row per run
            fig = make_subplots(
                rows=len(prompt_runs),
                cols=1,
                subplot_titles=[f"{prompt} - Run {run}" for run in prompt_runs],
                vertical_spacing=0.15,
            )
            
            # Process each run
            for run_idx, run_number in enumerate(prompt_runs, start=1):
                # Collect data for this (prompt, run) combination
                no_rag_values = []
                rag_values = []
                no_rag_commands = []  # Store actual command sets
                rag_commands = []  # Store actual command sets
                model_names = []
                
                for model in model_list:
                    no_rag_key = (prompt, run_number, model)
                    rag_key = (prompt, run_number, model)
                    
                    # Calculate NO_RAG base commands count and store commands
                    no_rag_count = 0
                    no_rag_cmd_set = set()
                    if no_rag_key in no_rag_final_base:
                        _, set_str = no_rag_final_base[no_rag_key]
                        parsed_set = parse_base_set(set_str)
                        no_rag_count = len(parsed_set)
                        no_rag_cmd_set = parsed_set
                    
                    # Calculate RAG base commands count and store commands
                    rag_count = 0
                    rag_cmd_set = set()
                    if rag_key in rag_final_base:
                        _, set_str = rag_final_base[rag_key]
                        parsed_set = parse_base_set(set_str)
                        rag_count = len(parsed_set)
                        rag_cmd_set = parsed_set
                    
                    # Only include models that have data in at least one dataset
                    if no_rag_key in no_rag_final_base or rag_key in rag_final_base:
                        model_names.append(model)
                        no_rag_values.append(no_rag_count)
                        rag_values.append(rag_count)
                        # Format commands as comma-separated string for hover
                        no_rag_cmd_str = ", ".join(sorted(no_rag_cmd_set)) if no_rag_cmd_set else "None"
                        rag_cmd_str = ", ".join(sorted(rag_cmd_set)) if rag_cmd_set else "None"
                        no_rag_commands.append(no_rag_cmd_str)
                        rag_commands.append(rag_cmd_str)
                
                if not model_names:
                    continue
                
                bar_colors = bar_colors_for_models(model_names, model_to_is_coding) if (model_to_is_coding and model_names) else None
                no_rag_kw: Dict[str, object] = dict(name="NO_RAG", x=model_names, y=no_rag_values, text=None, customdata=[[cmd] for cmd in no_rag_commands], hovertemplate="Model=%{x}<br>NO_RAG Base Commands=%{y}<br>Commands: %{customdata[0]}<extra></extra>", marker_color="#90EE90", offsetgroup="norag", alignmentgroup="models", showlegend=(run_idx == 1), width=0.6)
                if bar_colors:
                    no_rag_kw["marker_color"] = bar_colors
                fig.add_trace(go.Bar(**no_rag_kw), row=run_idx, col=1)
                
                rag_kw: Dict[str, object] = dict(name="RAG", x=model_names, y=rag_values, text=None, customdata=[[cmd] for cmd in rag_commands], hovertemplate="Model=%{x}<br>RAG Base Commands=%{y}<br>Commands: %{customdata[0]}<extra></extra>", marker_color="#BA55D3", offsetgroup="rag", alignmentgroup="models", showlegend=(run_idx == 1), width=0.6)
                if bar_colors:
                    rag_kw["marker_color"] = bar_colors
                fig.add_trace(go.Bar(**rag_kw), row=run_idx, col=1)
                
                # Update x-axis for this subplot
                fig.update_xaxes(
                    tickangle=-45,  # Rotate labels for readability
                    row=run_idx,
                    col=1,
                )
            
            # Update layout - set axis titles on middle row
            if len(prompt_runs) > 0:
                mid_row = (len(prompt_runs) + 1) // 2
                fig.update_xaxes(title_text="Model", row=mid_row, col=1)
                fig.update_yaxes(title_text="Number of Base Commands", row=mid_row, col=1)
            
            # Calculate dynamic dimensions based on number of models
            num_models = len(model_list) if model_list else 1
            # Height: standard height per run
            dynamic_height = 500 * len(prompt_runs)
            # Width: increased to space out models more
            dynamic_width = max(1800, num_models * 50)
            
            fig.update_layout(
                title_text=f"NO_RAG vs RAG Base Commands Comparison: {prompt} ({size_label})",
                template="plotly_white",
                barmode="group",  # Group NO_RAG and RAG bars side-by-side
                bargap=0.4,  # Add space between bars for different models
                height=dynamic_height,
                width=dynamic_width,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(b=200, l=80, r=50, t=100),  # Increased bottom margin for rotated labels
            )
            
            # Write HTML file
            output_path = os.path.join(output_subdir, f"{prompt}{filename_suffix}.html")
            write_html(fig, output_path)
            print(f"Created NO_RAG vs RAG base commands plot: {output_path}")
        
        # Create plots for both groups
        create_plot_for_models(models_le_4b, "<=4B", "_le4b")
        create_plot_for_models(models_gt_4b, ">4B", "_gt4b")


def main(
    csv_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    model_to_is_coding: Optional[Dict[str, bool]] = None,
) -> None:
    """Main function to create plots from existing CSV file.
    
    Reads command_statistics.csv and generates all four types of plots:
    - Bar charts (valid commands per model per run)
    - Unique valid commands bar charts (unique valid commands from last iteration per model per run)
    - Cumulative time series plots (averaged across runs)
    - Per-iteration time series plots (averaged across runs)
    
    When model_to_is_coding is not None (e.g. from code-command-analysis), bar charts
    color bars by coding vs non-coding model.
    """
    

    # Check if CSV file exists
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        print("Please run create_csv_of_results.py first to generate the CSV file.")
        return
    
    print(f"Reading CSV file: {csv_path}")
    print("\nCreating bar charts...")

    create_valid_commands_barchart(csv_path, output_dir, model_to_is_coding=model_to_is_coding)
    print("\nCreating aggregated bar charts...")
    
 
    #create_average_valid_per_model_plots(csv_path, output_dir, model_to_is_coding=model_to_is_coding)
    create_valid_commands_avg_perrun(csv_path, output_dir, model_to_is_coding=model_to_is_coding)
    create_validity_ratio_plots(csv_path, output_dir, model_to_is_coding=model_to_is_coding)
    print("\nCreating unique valid commands bar charts...")
    
    create_unique_valid_barchart(csv_path, output_dir, model_to_is_coding=model_to_is_coding)
    create_unique_valid_avg_perrun(csv_path, output_dir, model_to_is_coding=model_to_is_coding)
    
    #print("\nCreating cumulative time series plots...")
    #create_cumulative_time_series(csv_path, output_dir)

    #print("\nCreating per-iteration time series plots...")
    #create_per_iteration_time_series(csv_path, output_dir)
    
    print("\nCreating base commands per model plots...")
    create_base_commands_bar_plots(csv_path, output_dir, model_to_is_coding=model_to_is_coding)

    print("\nCreating average base commands per model plots...")
    create_average_base_commands_plots(csv_path, output_dir, model_to_is_coding=model_to_is_coding)

    print("\nCreating base commands comparison plot...")
    create_base_commands_comparison_plot(csv_path, output_dir, model_to_is_coding=model_to_is_coding)
   
    print("\nCreating average valid vs cumulative plots...")
    create_average_valid_vs_cumulative_plots(csv_path, output_dir, model_to_is_coding=model_to_is_coding)
    
    print("\nCreating cumulative valid unique plots...")
    create_cumulative_valid_unique_plots(csv_path, output_dir, model_to_is_coding=model_to_is_coding)
    
    # done HERE 
    print("\nCreating model analysis text file...")
    create_model_analysis_text(csv_path, output_dir)

    print("\nCreating average iteration duration plots...")
    create_average_iteration_duration_plots(csv_path, output_dir, model_to_is_coding=model_to_is_coding)

    print("\nCreating average iteration duration scatter plots with jitter...")
    create_average_iteration_duration_scatter_plots(csv_path, output_dir)

    print("\nCreating average iteration duration scatter plots with outliers...")
    create_average_iteration_duration_scatter_plots_with_outliers(csv_path, output_dir)
    create_cumulative_valid_unique_stacked_plots(csv_path, output_dir, model_to_is_coding=model_to_is_coding)
    print("\nCreating success/failure iteration plots...")
    create_success_failure_iteration_plots(csv_path, output_dir, model_to_is_coding=model_to_is_coding)
    
    print("\nCreating NO_RAG vs RAG duration comparison plots...")
    no_rag_csv_path = os.path.join(output_dir, "FTP_NOrag_results.csv")
    rag_csv_path = os.path.join(output_dir, "FTP_rag_results.csv")
    create_norag_vs_rag_duration_comparison_plots(no_rag_csv_path, rag_csv_path, output_dir, model_to_is_coding=model_to_is_coding)

    print("\nCreating success/failure iteration plots...")
    create_success_failure_iteration_plots(csv_path, output_dir, model_to_is_coding=model_to_is_coding)

    print("\nCreating NO_RAG vs RAG success/failure comparison plots...")
    create_norag_vs_rag_success_failure_plots(no_rag_csv_path, rag_csv_path, output_dir, model_to_is_coding=model_to_is_coding)

    print("\nCreating NO_RAG vs RAG base commands comparison plots...")
    create_norag_vs_rag_base_commands_plots(no_rag_csv_path, rag_csv_path, output_dir, model_to_is_coding=model_to_is_coding)

    print("\nCreating shot type comparison plot...")
    create_shot_type_comparison_plot(rag_csv_path, output_dir, model_to_is_coding=model_to_is_coding)
    print("Creating aggregate results plot...")
    create_aggregate_results(rag_csv_path, no_rag_csv_path, output_dir)
    print("\nCreating heatmap of valid commands across prompts and models...")
    create_heatmap(rag_csv_path, no_rag_csv_path, output_dir)
    print("\nCreating regression scatter plot of valid commands vs parameter size...")
    create_regression_scatter_plot(rag_csv_path, no_rag_csv_path, output_dir=output_dir)
    print("\nAll plots created successfully!")
    

    #create_failure_blanks_plots(rag_csv_path, no_rag_csv_path, output_dir, model_to_is_coding=model_to_is_coding)

if __name__ == "__main__":
    ROOT_DIR = "/data2/nkhajehn/watcher-mcp-server/"
    DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "post_processing", "plots")
    #default_csv = os.path.join(DEFAULT_OUTPUT_DIR, "RAG_KORAD_results.csv")

    parser = argparse.ArgumentParser(description="Create plots from RAG KORAD results CSV.")
    subparsers = parser.add_subparsers(dest="command", help="Command (omit or 'all' = all plots; 'code-command-analysis' = bar charts colored by coding vs non-coding model)")
    subparsers.add_parser("all", help="Generate all plots without coding-model bar coloring")
    cca_parser = subparsers.add_parser("code-command-analysis", help="Generate all plots with bar charts colored by coding vs non-coding model")
    #cca_parser.add_argument("--csv", default=default_csv, help="Path to results CSV (default: RAG_KORAD_results.csv in output dir)")
    cca_parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory for plots")

    args = parser.parse_args()
    csv_path = os.path.join(DEFAULT_OUTPUT_DIR, "FTP_rag_results.csv")
    output_dir = DEFAULT_OUTPUT_DIR
    model_to_is_coding: Optional[Dict[str, bool]] = None

    if args.command == "code-command-analysis":
        csv_path = args.csv
        output_dir = args.output
        model_to_is_coding = load_model_to_is_coding(csv_path)
        if model_to_is_coding is None:
            print("Coding-model bar coloring skipped: is_coding_model column missing or error reading CSV.")
        else:
            print("Coding-model bar coloring enabled.")
    elif args.command == "all" or args.command is None:
        pass

    main(csv_path=csv_path, output_dir=output_dir, model_to_is_coding=model_to_is_coding)

