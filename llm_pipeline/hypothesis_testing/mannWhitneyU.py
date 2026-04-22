import csv
import re
import mpld3
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import mannwhitneyu


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#CSV_PATH = "/data2/nkhajehn/watcher-mcp-server/post_processing/plots/RAG_KORAD_results.csv"
OUTPUT_DIR = "/data2/nkhajehn/watcher-mcp-server/llm_pipeline/hypothesis_testing/csv_files"
CSV_PATH = "/data2/nkhajehn/MCP-Command-Generation/post_processing/plots/RAG_KORAD_results.csv"
THRESHOLDS = [14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
WHOLE_MODEL_THRESHOLD = 10

# Key tuple layout for base_data / unique_valid_data:
#   (prompt, model, parameter_count, run_number, is_code_model, is_chinese_model)
IDX_PROMPT         = 0
IDX_MODEL          = 1
IDX_PARAM_COUNT    = 2
IDX_RUN            = 3
IDX_IS_CODING      = 4
IDX_IS_CHINESE     = 5


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_base_set(set_str: str) -> Set[str]:
    s = (set_str or "").strip()
    if s in ("set()", "{}", ""):
        return set()
    items = re.findall(r"'([^']*)'|\"([^\"]*)\"", s)
    return {a or b for a, b in items}


def calculate_avg_base_commands(base_data: Dict) -> Dict:
    averaged = defaultdict(int)
    for c in base_data:
        prompt, model, parameter_count, _, _ = c
        key = (prompt, model, parameter_count)
        averaged[key] += base_data[c]
    for key in averaged:
        averaged[key] = round(averaged[key] / 3, 2)
    return averaged


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def load_base_commands_data(csv_path: str) -> Dict[Tuple, int]:
    """Load base commands data from CSV.

    Returns dict mapping
    (prompt, model, parameter_count, run_number, is_code_model, is_chinese_model)
    -> base_command_count at the preferred iteration (50 if present, else max).
    """
    best: Dict[Tuple, Tuple[int, int]] = {}  # key -> (best_iter, count)

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                prompt = row.get("prompt", "").strip()
                model  = row.get("model",  "").strip()
                if not prompt or not model:
                    continue

                try:
                    run_number = int(row.get("run_number", 0))
                except Exception:
                    run_number = 0
                try:
                    iteration = int(row.get("iteration", 0))
                except Exception:
                    iteration = 0

                parameter_count  = row.get("parameter_count", "").strip()
                is_code_model    = row.get("is_coding_model", "False").strip()
                is_chinese_model = row.get("is_chinese", "False").strip()

                base_count = len(parse_base_set(row.get("base_commands_seen_so_far", "")))
                key = (prompt, model, parameter_count, run_number, is_code_model, is_chinese_model)

                if key not in best:
                    best[key] = (iteration, base_count)
                else:
                    prev_iter, _ = best[key]
                    if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                        best[key] = (iteration, base_count)

    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return {}

    return {key: count for key, (_, count) in best.items()}


def load_unique_commands_data(csv_path: str) -> Dict[Tuple, int]:
    """Load unique valid command counts from a command-statistics CSV.

    Returns dict mapping
    (prompt, model, parameter_count, run_number, is_coding, is_chinese)
    -> unique_valid_commands at the preferred iteration.
    """
    best: Dict[Tuple, Tuple[int, int]] = {}

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                prompt = row.get("prompt", "").strip()
                model  = row.get("model",  "").strip()
                if not prompt or not model:
                    continue

                try:
                    run_number = int(row.get("run_number", 0))
                except Exception:
                    run_number = 0
                try:
                    iteration = int(row.get("iteration", 0))
                except Exception:
                    iteration = 0

                parameter_count  = row.get("parameter_count", "").strip()
                is_code_model    = row.get("is_coding",  "False").strip()
                is_chinese_model = row.get("is_chinese", "False").strip()

                try:
                    unique_valid_count = int(row.get("unique_valid_commands", 0))
                except Exception:
                    unique_valid_count = 0

                key = (prompt, model, parameter_count, run_number, is_code_model, is_chinese_model)

                if key not in best:
                    best[key] = (iteration, unique_valid_count)
                else:
                    prev_iter, _ = best[key]
                    if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                        best[key] = (iteration, unique_valid_count)

    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return {}

    return {key: count for key, (_, count) in best.items()}


# Backward-compatible alias
load_unique_valid_commands_data = load_unique_commands_data


# ---------------------------------------------------------------------------
# Mann-Whitney U: threshold analysis per prompt
# ---------------------------------------------------------------------------

def organize_data_by_prompt(
    base_data: Dict[Tuple, int],
    thresholds: List[int],
) -> Dict[str, Dict]:
    """Split each prompt's models by parameter count threshold and run Mann-Whitney U.

    For each threshold value, models with parameter_count >= threshold go in the
    "gt" group; models with WHOLE_MODEL_THRESHOLD <= parameter_count < threshold
    go in the "lt" group.

    Returns a dict keyed by prompt name:
        {
            "effect_gt": {threshold -> effect_size},
            "effect_lt": {threshold -> 1 - effect_size},
            "len_gt":    [count per threshold],
            "len_lt":    [count per threshold],
        }
    """
    prompts = sorted({c[IDX_PROMPT] for c in base_data})
    results = {
        p: {"effect_gt": {}, "effect_lt": {}, "len_gt": [], "len_lt": []}
        for p in prompts
    }

    for value in thresholds:
        groups_gt = {p: [] for p in prompts}
        groups_lt = {p: [] for p in prompts}
        
        for c, count in base_data.items():
            p = c[IDX_PROMPT]
            param = float(c[IDX_PARAM_COUNT])
            if param >= value:
                groups_gt[p].append(count)
            elif WHOLE_MODEL_THRESHOLD <= param < value:
                groups_lt[p].append(count)

        for p in prompts:
            gt, lt = groups_gt[p], groups_lt[p]
            print(f"[threshold={value}] {p}: {len(gt)} models >= {value}, {len(lt)} models < {value}")

            if gt and lt:
                res = mannwhitneyu(gt, lt, alternative="greater")
                effect = res.statistic / (len(gt) * len(lt))
                print(f"  U={res.statistic:.1f}, p={res.pvalue:.4f}, effect={effect:.4f}")
            else:
                effect = float("nan")

            results[p]["effect_gt"][value] = effect
            results[p]["effect_lt"][value] = 1.0 - effect
            results[p]["len_gt"].append(len(gt))
            results[p]["len_lt"].append(len(lt))

    return results


# ---------------------------------------------------------------------------
# Mann-Whitney U: coding / Chinese model comparisons
# ---------------------------------------------------------------------------

def compare_by_flag(
    base_data: Dict[Tuple, int],
    flag_index: int,
    label: str,
) -> None:
    """Group data by a boolean flag in the key tuple and run Mann-Whitney U per prompt."""
    prompts = sorted({c[IDX_PROMPT] for c in base_data})
    positive = {p: [] for p in prompts}
    negative = {p: [] for p in prompts}

    for c, count in base_data.items():
        p = c[IDX_PROMPT]
        if c[flag_index] == "True":
            positive[p].append(count)
        else:
            negative[p].append(count)

    print(f"\n{'='*20} {label.upper()} vs NON-{label.upper()} {'='*20}")
    for p in prompts:
        pos, neg = positive[p], negative[p]
        if pos and neg:
            stat, pval = mannwhitneyu(pos, neg, alternative="greater")
            print(f"  {p}: stat={stat:.1f}, p-value={pval:.4f}")
        else:
            print(f"  {p}: insufficient data (pos={len(pos)}, neg={len(neg)})")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_hist(data: List, name: str, path: str) -> None:
    plt.figure()
    plt.hist(data, rwidth=0.9)
    plt.title(f"Histogram of {name} Base Commands Seen So Far")
    plt.xlabel("Number of base commands outputted")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_line_graph(
    data_gt: Dict,
    data_lt: Dict,
    name: str,
    path: str,
    model_count_gt: List = None,
    model_count_lt: List = None,
    thresholds: List[int] = None,
) -> go.Figure:
    x  = thresholds or list(data_gt.keys())
    y1 = list(data_gt.values())
    y2 = list(data_lt.values())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y1, mode="lines+markers+text",
        text=[str(v) for v in (model_count_gt or [])],
        textposition="top center", name="P(Threshold >=)",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y2, mode="lines+markers+text",
        text=[str(int(v)) for v in (model_count_lt or [])],
        textposition="top center", name="P(Threshold <)",
    ))
    fig.update_layout(
        title=f"Effect Size vs Threshold for {name}",
        xaxis_title="Threshold",
        yaxis_title="Effect Size",
    )
    fig.update_xaxes(tickmode="array", tickvals=x)

    if path.lower().endswith(".html"):
        fig.write_html(path)
    else:
        fig.write_image(path)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

base_data        = load_base_commands_data(CSV_PATH)
unique_valid_data = load_unique_commands_data(CSV_PATH)

# Threshold analysis: large-param vs small-param models, per prompt
results = organize_data_by_prompt(base_data, THRESHOLDS)

for prompt_name, data in results.items():
    out_path = f"{OUTPUT_DIR}/{prompt_name}_line_graph.png"
    save_line_graph(
        data["effect_gt"], data["effect_lt"],
        prompt_name, out_path,
        data["len_gt"], data["len_lt"],
        THRESHOLDS,
    )

# Coding model comparison
compare_by_flag(base_data, flag_index=IDX_IS_CODING,  label="Coding")

# Chinese model comparison
compare_by_flag(base_data, flag_index=IDX_IS_CHINESE, label="Chinese")
