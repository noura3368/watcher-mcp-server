import pandas as pd
import statsmodels.formula.api as smf
from typing import Dict, Iterator, List, Tuple, Any, Set
import csv
import re, mpld3
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wilcoxon
# calculate mean of variance first 
import plotly.graph_objects as go



def parse_base_set(set_str: str) -> Set[str]:
    """Parse base_commands_seen_so_far string into a set of commands."""
    s = (set_str or "").strip()
    if s == "set()" or s == "{}" or s == "":
        return set()
    else:
        # extract quoted items
        items = re.findall(r"'([^']*)'|\"([^\"]*)\"", s)
        return set([a or b for a, b in items])


def calculate_avg_base_commands(base_data):

    averaged_data = defaultdict(int)
    for i,c in enumerate(base_data):
        prompt, model, parameter_count, _,_ = c
        if (prompt, model, parameter_count) not in averaged_data:
            averaged_data[(prompt, model, parameter_count)] = base_data[c]
        else:
            averaged_data[(prompt, model, parameter_count)] += base_data[c]
    for i in averaged_data:
        averaged_data[i] = round(averaged_data[i] / 3, 2) # average over 3 runs
    return averaged_data
    
def load_base_commands_data(csv_path: str) -> Dict[Tuple[str, int, str], int]:
    """Load base commands data from CSV.
    
    Args:
        csv_path: Path to command_statistics CSV file
        
    Returns:
        Dictionary mapping (prompt, run_number, model) -> base_command_count
        Uses the last iteration (preferring iteration 50 or max)
    """
    final_base_by_prompt_run_model: Dict[Tuple[str, int, str], Tuple[int, int, float]] = {}
    
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
                parameter_count = row.get("parameter_count", "").strip()
                if not model:
                    continue
                is_code_model = row.get("is_coding_model", "False").strip()
                is_chinese_model = row.get("is_chinese", "False").strip()
                # read raw set string
                base_set_str = row.get("base_commands_seen_so_far", "")
                parsed_set = parse_base_set(base_set_str)
                base_count = len(parsed_set)
                
                key = (prompt, model, parameter_count, run_number,is_code_model, is_chinese_model)
            
                # prefer exact iteration 50; otherwise take the max iteration
                if key not in final_base_by_prompt_run_model:
                    final_base_by_prompt_run_model[key] = (iteration, base_count)
                  
                else:
                    prev_iter, _ = final_base_by_prompt_run_model[key]
                  
                    # prefer iteration 50 if seen, otherwise keep the max
                    if iteration == 50 or (iteration > prev_iter and prev_iter != 50):
                        final_base_by_prompt_run_model[key] = (iteration, base_count)
    
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return {}
  
    # Return just the counts
    return {key: count for key, (_, count) in final_base_by_prompt_run_model.items()}

def load_unique_commands_data(csv_path: str) -> Dict[Tuple[str, str, str, int, str], int]:
    """Load unique valid command counts from a command-statistics CSV.

    For each (prompt, model, parameter_count, run_number, is_coding_model), keeps the
    ``unique_valid_commands`` value from the preferred iteration: iteration 50 if present
    in the data for that key, otherwise the maximum iteration (same rule as
    ``load_base_commands_data``).

    Args:
        csv_path: Path to command_statistics CSV file.

    Returns:
        Dictionary mapping
        (prompt, model, parameter_count, run_number, is_coding_model) -> unique_valid_commands
        from the selected row.
    """
    final_unique_valid_by_prompt_run_model: Dict[
        Tuple[str, str, str, int, str], Tuple[int, int]
    ] = {}

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
                parameter_count = row.get("parameter_count", "").strip()
                if not model:
                    continue
                is_code_model = row.get("is_coding", "False").strip()
                is_chinese_model = row.get("is_chinese", "False").strip()
                try:
                    unique_valid_count = int(row.get("unique_valid_commands", 0))
                except Exception:
                    unique_valid_count = 0

                key = (prompt, model, parameter_count, run_number, is_code_model, is_chinese_model)

                if key not in final_unique_valid_by_prompt_run_model:
                    final_unique_valid_by_prompt_run_model[key] = (
                        iteration,
                        unique_valid_count,
                    )
                else:
                    prev_iter, _ = final_unique_valid_by_prompt_run_model[key]
                    if iteration == 50 or (
                        iteration > prev_iter and prev_iter != 50
                    ):
                        final_unique_valid_by_prompt_run_model[key] = (
                            iteration,
                            unique_valid_count,
                        )

    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return {}

    return {
        key: count for key, (_, count) in final_unique_valid_by_prompt_run_model.items()
    }


# Backward-compatible alias: keep the old function name available for imports.
load_unique_valid_commands_data = load_unique_commands_data

unique_valid_data = load_unique_commands_data(
    "/data2/nkhajehn/MCP-Command-Generation/post_processing/plots/FTP_RAG_results.csv"
)

base_data = load_base_commands_data("/data2/nkhajehn/MCP-Command-Generation/post_processing/plots/FTP_RAG_results.csv")
#averaged_data = calculate_avg_base_commands(base_data)
averaged_data = base_data
threshold = []

def organizing_data_by_prompt(base_data, threshold):
    effect_sizes_prompt1_gt, effect_sizes_prompt1_lt = {}, {}
    effect_sizes_prompt2_gt, effect_sizes_prompt2_lt = {}, {}
    effect_sizes_prompt3_gt, effect_sizes_prompt3_lt  = {}, {}
    effect_sizes_prompt4_gt, effect_sizes_prompt4_lt  = {}, {}
    effect_sizes_prompt5_gt, effect_sizes_prompt5_lt  = {},{}
    effect_sizes_prompt6_gt, effect_sizes_prompt6_lt  = {},{}
    effect_sizes_prompt7_gt, effect_sizes_prompt7_lt = {}, {}
    effect_sizes_security_incremental_1_gt, effect_sizes_security_incremental_1_lt = {}, {}

    prompt1_len_of_models_gt = []
    prompt2_len_of_models_gt = []
    prompt3_len_of_models_gt = []
    prompt4_len_of_models_gt = []
    prompt5_len_of_models_gt = []
    prompt6_len_of_models_gt = []
    prompt7_len_of_models_gt = []
    security_incremental_1_len_of_models_gt = []

    prompt1_len_of_models_lt = []
    prompt2_len_of_models_lt = []
    prompt3_len_of_models_lt = []
    prompt4_len_of_models_lt = []
    prompt5_len_of_models_lt = []
    prompt6_len_of_models_lt = []
    prompt7_len_of_models_lt = []
    security_incremental_1_len_of_models_lt = []
    threshold = [4, 6, 8, 10, 12, 14, 16,18, 20,22,24,26,28,30,32]
    #threshold = [20, 24, 26, 28, 30]
    #threshold = [20]
    #threshold = [8,10,12,14,16,18,20,22,24]
    #threshold = [4, 6, 8, 10, 12, 14, 16,18, 20,22,24]
    for value in threshold:
        data_le_prompt1, data_gt_prompt1 = [] , []
        data_le_prompt2, data_gt_prompt2 = [] , []
        data_le_prompt3, data_gt_prompt3 = [] , []
        data_le_prompt4, data_gt_prompt4 = [] , []
        data_le_prompt5, data_gt_prompt5 = [] , []
        data_le_prompt6, data_gt_prompt6 = [] , []
        data_le_prompt7, data_gt_prompt7 = [] , []
        data_le_security_incremental_1, data_gt_security_incremental_1 = [] , []    



        prompt1 = [] 
        prompt2 = []
        prompt3 = [] 
        prompt4 = []
        prompt5 = []
        prompt6 = []
        prompt7 = []
        security_incremental = [] 

        prompt1_dict = {}
        prompt2_dict = {}
        prompt3_dict = {}
        prompt4_dict = {}
        prompt5_dict = {}
        prompt6_dict = {}
        prompt7_dict = {}
        security_incremental_1_dict = {}
        whole_model_threshold = 10
        for i,c in enumerate(base_data):
            if c[0] == "prompt1":
                prompt1.append(base_data[c])
                prompt1_dict[c[1]] = base_data[c]
                if float(c[2]) >= value:
                    data_gt_prompt1.append(base_data[c])
                elif whole_model_threshold <= float(c[2]) < value:
                    data_le_prompt1.append(base_data[c])
            elif c[0] == "prompt2":
                prompt2.append(base_data[c])
                prompt2_dict[c[1]] = base_data[c]
                if float(c[2]) >= value:
                    data_gt_prompt2.append(base_data[c])
                elif whole_model_threshold <= float(c[2]) < value:
                    data_le_prompt2.append(base_data[c])
            elif c[0] == "prompt3":
                prompt3.append(base_data[c])
                prompt3_dict[c[1]] = base_data[c]
                if float(c[2]) >= value:
                    data_gt_prompt3.append(base_data[c])
                elif whole_model_threshold <= float(c[2]) < value:
                    data_le_prompt3.append(base_data[c])
            elif c[0] == "prompt4":
                prompt4.append(base_data[c])
                prompt4_dict[c[1]] = base_data[c]
                if float(c[2]) >= value:
                    data_gt_prompt4.append(base_data[c])
                elif whole_model_threshold <= float(c[2]) < value:
                    data_le_prompt4.append(base_data[c])
            elif c[0] == "prompt5":
                prompt5.append(base_data[c])
                prompt5_dict[c[1]] = base_data[c]
                if float(c[2]) >= value:
                    data_gt_prompt5.append(base_data[c])
                elif whole_model_threshold <= float(c[2]) < value:
                    data_le_prompt5.append(base_data[c])
            elif c[0] == "prompt6":
                prompt6.append(base_data[c])
                prompt6_dict[c[1]] = base_data[c]
                if float(c[2]) >= value:
                    data_gt_prompt6.append(base_data[c])
                elif whole_model_threshold <= float(c[2]) < value:
                    data_le_prompt6.append(base_data[c])
            elif c[0] == "prompt7":
                prompt7.append(base_data[c])
                prompt7_dict[c[1]] = base_data[c]
                if float(c[2]) >= value:
                    data_gt_prompt7.append(base_data[c])
                elif whole_model_threshold <= float(c[2]) < value:
                    data_le_prompt7.append(base_data[c])
            elif c[0] == "security-incremental-1":
                security_incremental.append(base_data[c])
                security_incremental_1_dict[c[1]] = base_data[c]
                if float(c[2]) >= value:
                    data_gt_security_incremental_1.append(base_data[c])
                elif whole_model_threshold <= float(c[2]) < value:
                    data_le_security_incremental_1.append(base_data[c])
        print(f"Prompt1: {len(data_gt_prompt1)} models >= {value}, {len(data_le_prompt1)} models < {value}")
        p = [data_gt_prompt1, data_le_prompt1, data_gt_prompt2, data_le_prompt2, data_gt_prompt3, data_le_prompt3,
        data_gt_prompt4, data_le_prompt4, data_gt_prompt5, data_le_prompt5, data_gt_prompt6, data_le_prompt6,
        data_gt_prompt7, data_le_prompt7, data_gt_security_incremental_1, data_le_security_incremental_1]
        
        
    
        for i in range(0, len(p), 2):
            res = mannwhitneyu(p[i], p[i+1], alternative="greater")  # large > small
            print("THRESHOLD VALUE: ", value)
            print(f"Prompt {(i//2)+1} : U-statistic = {res.statistic}, p-value = {res.pvalue}")

            n_large = len(p[i])
            n_small = len(p[i+1])
            
            effect_size = res.statistic / (n_large * n_small)
            if "Prompt" + str((i//2)+1) == "Prompt1":
                print(f"Prompt {(i//2)+1} : Effect Size (U/n_large*n_small) = {effect_size}")
                effect_sizes_prompt1_gt[value] = effect_size
                effect_sizes_prompt1_lt[value] = 1- effect_size
                prompt1_len_of_models_gt.append(int(n_large))
                prompt1_len_of_models_lt.append(int(n_small))
            elif"Prompt" + str((i//2)+1) == "Prompt2":
                effect_sizes_prompt2_gt[value] = effect_size
                effect_sizes_prompt2_lt[value] = 1- effect_size
                prompt2_len_of_models_gt.append(int(n_large))
                prompt2_len_of_models_lt.append(int(n_small))
                print(f"Prompt {(i//2)+1} : Effect Size (U/n_large*n_small) = {effect_size}")
            elif "Prompt" + str((i//2)+1) == "Prompt3":
                effect_sizes_prompt3_gt[value] = effect_size
                effect_sizes_prompt3_lt[value] = 1- effect_size
                prompt3_len_of_models_gt.append(int(n_large))
                prompt3_len_of_models_lt.append(int(n_small))    
                print(f"Prompt {(i//2)+1} : Effect Size (U/n_large*n_small) = {effect_size}")
            elif "Prompt" + str((i//2)+1) == "Prompt4":
                effect_sizes_prompt4_gt[value] = effect_size  
                effect_sizes_prompt4_lt[value] = 1- effect_size 
                prompt4_len_of_models_gt.append(int(n_large))
                prompt4_len_of_models_lt.append(int(n_small))
                print(f"Prompt {(i//2)+1} : Effect Size (U/n_large*n_small) = {effect_size}")
            elif "Prompt" + str((i//2)+1)== "Prompt5":
                effect_sizes_prompt5_gt[value] = effect_size
                effect_sizes_prompt5_lt[value] = 1- effect_size
                prompt5_len_of_models_gt.append(int(n_large))
                prompt5_len_of_models_lt.append(int(n_small))
                print(f"Prompt {(i//2)+1} : Effect Size (U/n_large*n_small) = {effect_size}")
            elif "Prompt" + str((i//2)+1) == "Prompt6":
                effect_sizes_prompt6_gt[value] = effect_size
                effect_sizes_prompt6_lt[value] = 1- effect_size
                prompt6_len_of_models_gt.append(int(n_large))
                prompt6_len_of_models_lt.append(int(n_small))
                print(f"Prompt {(i//2)+1} : Effect Size (U/n_large*n_small) = {effect_size}")
            elif "Prompt" + str((i//2)+1)== "Prompt7":
                effect_sizes_prompt7_gt[value] = effect_size
                effect_sizes_prompt7_lt[value] = 1- effect_size
                prompt7_len_of_models_gt.append(int(n_large))
                prompt7_len_of_models_lt.append(int(n_small))
                print(f"Prompt {(i//2)+1} : Effect Size (U/n_large*n_small) = {effect_size}")
            elif "Prompt" + str((i//2)+1)== "security-incremental-1":
                effect_sizes_security_incremental_1_gt[value] = effect_size
                effect_sizes_security_incremental_1_lt[value] = 1- effect_size
                security_incremental_1_len_of_models_gt.append(int(n_large))
                security_incremental_1_len_of_models_lt.append(int(n_small))
                print(f"Prompt {(i//2)+1} : Effect Size (U/n_large*n_small) = {effect_size}")
    return [prompt1_len_of_models_gt, prompt1_len_of_models_lt, prompt2_len_of_models_gt, prompt2_len_of_models_lt, prompt3_len_of_models_gt, prompt3_len_of_models_lt, prompt4_len_of_models_gt, prompt4_len_of_models_lt, prompt5_len_of_models_gt, prompt5_len_of_models_lt, prompt6_len_of_models_gt, prompt6_len_of_models_lt, prompt7_len_of_models_gt, prompt7_len_of_models_lt, security_incremental_1_len_of_models_lt, security_incremental_1_len_of_models_gt, effect_sizes_prompt1_gt, effect_sizes_prompt1_lt, effect_sizes_prompt2_gt, effect_sizes_prompt2_lt, effect_sizes_prompt3_gt, effect_sizes_prompt3_lt, effect_sizes_prompt4_gt, effect_sizes_prompt4_lt, effect_sizes_prompt5_gt, effect_sizes_prompt5_lt, effect_sizes_prompt6_gt, effect_sizes_prompt6_lt, effect_sizes_prompt7_gt, effect_sizes_prompt7_lt, effect_sizes_security_incremental_1_gt, effect_sizes_security_incremental_1_lt]
        

result_of_org = organizing_data_by_prompt(averaged_data, threshold)
prompt1_len_of_models_gt = result_of_org[0]
prompt1_len_of_models_lt = result_of_org[1]
prompt2_len_of_models_gt = result_of_org[2]
prompt2_len_of_models_lt = result_of_org[3]
prompt3_len_of_models_gt = result_of_org[4]
prompt3_len_of_models_lt = result_of_org[5]
prompt4_len_of_models_gt = result_of_org[6]
prompt4_len_of_models_lt = result_of_org[7]
prompt5_len_of_models_gt = result_of_org[8]
prompt5_len_of_models_lt = result_of_org[9]
prompt6_len_of_models_gt = result_of_org[10]
prompt6_len_of_models_lt = result_of_org[11]
prompt7_len_of_models_gt = result_of_org[12]
prompt7_len_of_models_lt = result_of_org[13]
security_incremental_1_len_lt = result_of_org[14]
security_incremental_1_len_gt = result_of_org[15]
effect_sizes_prompt1_gt = result_of_org[16]
effect_sizes_prompt1_lt = result_of_org[17]
effect_sizes_prompt2_gt = result_of_org[18]
effect_sizes_prompt2_lt = result_of_org[19]
effect_sizes_prompt3_gt = result_of_org[20]
effect_sizes_prompt3_lt = result_of_org[21]
effect_sizes_prompt4_gt = result_of_org[22]
effect_sizes_prompt4_lt = result_of_org[23]
effect_sizes_prompt5_gt = result_of_org[24]
effect_sizes_prompt5_lt = result_of_org[25]
effect_sizes_prompt6_gt = result_of_org[26]
effect_sizes_prompt6_lt = result_of_org[27]
effect_sizes_prompt7_gt = result_of_org[28]
effect_sizes_prompt7_lt = result_of_org[29]
effect_sizes_security_incremental_1_gt = result_of_org[30]
effect_sizes_security_incremental_1_lt = result_of_org[31]





def save_hist(data, name, path):
    plt.figure()  # NEW FIGURE
    plt.hist(data, rwidth=0.9)  # 0–12-ish counts
    plt.title(f"Histogram of {name} Base Commands Seen So Far")
    plt.xlabel("Number of base commands outputted")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()  # closes the figure so nothing accumulates


def save_line_graph(data1, data2, name, path, model_count_gt=None, model_count_lt=None):
    """
    data1, data2: dict-like (values are y's). If keys are thresholds, we can use them as x.
    path: "something.html" (interactive) OR "something.png"/".svg"/".pdf" (static; needs kaleido).
    threshold: optional explicit x list to use (your old `threshold`).
    """
    # Decide x-axis (thresholds)

    x = list(threshold)

    y1 = list(data1.values())
    y2 = list(data2.values())
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y1, mode="lines+markers+text", text=[f"{v}" for v in model_count_gt], textposition="top center", name="P(Threshold >=)"
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y2, mode="lines+markers+text", text=[f"{int(v)}" for v in model_count_lt], textposition="top center", name="P(Threshold <)"
    ))

    fig.update_layout(
        title=f"Effect Size vs Threshold for {name}",
        xaxis_title="Threshold",
        yaxis_title="Effect Size",
        legend_title_text=""
    )

    # Match old behavior: show ticks at each threshold
    fig.update_xaxes(tickmode="array", tickvals=x)

    # Save
    if path.lower().endswith(".html"):
        fig.write_html(path)
    else:
        # requires: pip install -U kaleido
        fig.write_image(path)

    return fig  # handy if you also want to display it
'''
save_hist(prompt1, "Prompt 1", "/data2/nkhajehn/MCP-Command-Generation/hypothesis_testing/csv_files/prompt1_hist.png")
save_hist(prompt2, "Prompt 2", "/data2/nkhajehn/MCP-Command-Generation/hypothesis_testing/csv_files/prompt2_hist.png")
save_hist(prompt3, "Prompt 3", "/data2/nkhajehn/MCP-Command-Generation/hypothesis_testing/csv_files/prompt3_hist.png")
save_hist(prompt4, "Prompt 4", "/data2/nkhajehn/MCP-Command-Generation/hypothesis_testing/csv_files/prompt4_hist.png")
save_hist(prompt5, "Prompt 5", "/data2/nkhajehn/MCP-Command-Generation/hypothesis_testing/csv_files/prompt5_hist.png")
save_hist(prompt6, "Prompt 6", "/data2/nkhajehn/MCP-Command-Generation/hypothesis_testing/csv_files/prompt6_hist.png")
save_hist(prompt7, "Prompt 7", "/data2/nkhajehn/MCP-Command-Generation/hypothesis_testing/csv_files/prompt7_hist.png")
'''
save_line_graph(effect_sizes_prompt1_gt, effect_sizes_prompt1_lt, "Prompt1", "/data2/nkhajehn/MCP-Command-Generation/hypothesis_testing/csv_files/prompt1_line_graph.png", prompt1_len_of_models_gt, prompt1_len_of_models_lt)
save_line_graph(effect_sizes_prompt2_gt,effect_sizes_prompt2_lt, "Prompt2", "/data2/nkhajehn/MCP-Command-Generation/hypothesis_testing/csv_files/prompt2_line_graph.png", prompt2_len_of_models_gt, prompt2_len_of_models_lt)
save_line_graph(effect_sizes_prompt3_gt, effect_sizes_prompt3_lt, "Prompt3", "/data2/nkhajehn/MCP-Command-Generation/hypothesis_testing/csv_files/prompt3_line_graph.png", prompt3_len_of_models_gt, prompt3_len_of_models_lt)
save_line_graph(effect_sizes_prompt4_gt, effect_sizes_prompt4_lt,"Prompt4", "/data2/nkhajehn/MCP-Command-Generation/hypothesis_testing/csv_files/prompt4_line_graph.png", prompt4_len_of_models_gt, prompt4_len_of_models_lt)
save_line_graph(effect_sizes_prompt5_gt, effect_sizes_prompt5_lt,"Prompt5", "/data2/nkhajehn/MCP-Command-Generation/hypothesis_testing/csv_files/prompt5_line_graph.png", prompt5_len_of_models_gt, prompt5_len_of_models_lt)
save_line_graph( effect_sizes_prompt6_gt,effect_sizes_prompt6_lt, "Prompt6", "/data2/nkhajehn/MCP-Command-Generation/hypothesis_testing/csv_files/prompt6_line_graph.png", prompt6_len_of_models_gt, prompt6_len_of_models_lt)
save_line_graph(effect_sizes_prompt7_gt, effect_sizes_prompt7_lt,"Prompt7", "/data2/nkhajehn/MCP-Command-Generation/hypothesis_testing/csv_files/prompt7_line_graph.png", prompt7_len_of_models_gt, prompt7_len_of_models_lt)
save_line_graph(effect_sizes_security_incremental_1_gt, effect_sizes_security_incremental_1_lt,"Prompt Security", "/data2/nkhajehn/MCP-Command-Generation/hypothesis_testing/csv_files/prompt7_line_graph.png", prompt7_len_of_models_gt, prompt7_len_of_models_lt)


print("=================CODING MODELS VS NON-CODING MODELS=================")

code_models = {"prompt1":[], "prompt2":[], "prompt3":[], "prompt4":[], "prompt5":[], "prompt6":[], "prompt7":[], "security-incremental-1":[]}
non_code_models = {"prompt1":[], "prompt2":[], "prompt3":[], "prompt4":[], "prompt5":[], "prompt6":[], "prompt7":[], "security-incremental-1":[]}


for i,c in enumerate(base_data):
    prompt, run_number, model, parameter_count, is_code_model, is_chinese_model = c
    count = base_data[c]
    if prompt in code_models.keys():
        if is_code_model == 'True':
            code_models[prompt].append(count)
        else:
            non_code_models[prompt].append(count)


def coding_models_vs_non_coding_models():
    for prompt in code_models.keys():
        code_model_counts = code_models.get(prompt, [])
        non_code_model_counts = non_code_models.get(prompt, [])
        if code_model_counts and non_code_model_counts:
            stat, p = mannwhitneyu(code_model_counts, non_code_model_counts, alternative='greater')
            print(f"Mann Whitney U test between coding models and non-coding models for {prompt}: stat={stat}, p-value={p}")

coding_models_vs_non_coding_models()


print("=================CHINESE MODELS VS NON-CHINESE MODELS=================")

chinese_models = {"prompt1":[], "prompt2":[], "prompt3":[], "prompt4":[], "prompt5":[], "prompt6":[], "prompt7":[], "security-incremental-1":[]}
non_chinese_models = {"prompt1":[], "prompt2":[], "prompt3":[], "prompt4":[], "prompt5":[], "prompt6":[], "prompt7":[], "security-incremental-1":[]}


for i,c in enumerate(base_data):
    prompt, run_number, model, parameter_count, is_code_model, is_chinese_model = c
    count = base_data[c]
    if prompt in chinese_models.keys():
        if is_chinese_model == 'True':
            chinese_models[prompt].append(count)
        else:
            non_chinese_models[prompt].append(count)


def chinese_models_vs_non_chinese_models():
    for prompt in chinese_models.keys():
        chinese_model_counts = chinese_models.get(prompt, [])
        non_chinese_model_counts = non_chinese_models.get(prompt, [])
        if chinese_model_counts and non_chinese_model_counts:
            stat, p = mannwhitneyu(chinese_model_counts, non_chinese_model_counts, alternative='greater')
            print(f"Mann Whitney U test between chinese models and non-chinese models for {prompt}: stat={stat}, p-value={p}")

chinese_models_vs_non_chinese_models()

