
# the number of tests 
# zero-shot vs 1-shot: 1x5, 1x7, 2x5, 2x7, 3x5, 3x7, 4x5, 4x7
# zero short vs two shot: 6x5, 6x7
# 1-shot vs two shot: 6x1, 6x2, 6x3, 6x4
from typing import Dict, Iterator, List, Tuple, Any, Set
import csv
from scipy.stats import wilcoxon
import re 

zero_shot_prompts = ["prompt5", "prompt7", "security-incremental-1"]
one_shot_prompts = ["prompt1", "prompt2", "prompt3"]
two_shot_prompts = ["prompt6"]

#excluded_models = ["qwen2.5vl:7b", "qwen2.5vl:32b", "qwen2.5vl:3b", "qwen3:1.7b", "granite3.2-vision:2b"]

def parse_base_set(set_str: str) -> Set[str]:
    """Parse base_commands_seen_so_far string into a set of commands."""
    s = (set_str or "").strip()
    if s == "set()" or s == "{}" or s == "":
        return set()
    else:
        # extract quoted items
        items = re.findall(r"'([^']*)'|\"([^\"]*)\"", s)
        return set([a or b for a, b in items])

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
               

                # read raw set string
                base_set_str = row.get("base_commands_seen_so_far", "")
                parsed_set = parse_base_set(base_set_str)
                base_count = len(parsed_set)
                
                key = (prompt, run_number, model, parameter_count, is_code_model)
            
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



base_data = load_base_commands_data("/home/nkhajehn/MCP-Command-Generation/post_processing/plots/RAG_KORAD_results.csv")
 
prompt1_counts = []
prompt2_counts = []
prompt3_counts = []
prompt4_counts = []
prompt5_counts = []
prompt6_counts = []
prompt7_counts = []
security_incremental_1_counts = []

prompt_counts = {"prompt1":[], "prompt2":[], "prompt3":[], "prompt5":[], "prompt6":[], "prompt7":[], "security-incremental-1": []}


for i,c in enumerate(base_data):
    prompt, run_number, model, parameter_count, is_code_model = c
    count = base_data[c]
    if prompt in prompt_counts.keys():
        prompt_counts[prompt].append(count)
    
# greater, x beats y, x > y
def zero_shot_vs_one_shot():
    for results_1shot in one_shot_prompts:
        for results_0shot in zero_shot_prompts:
            one_shot = prompt_counts[results_1shot]
            zero_shot = prompt_counts[results_0shot]
            # null hypothesis: one shot doesn't provide an increase in successful results compared to zero shot (one_shot <= zero_shot)
            # alternative hypothesis: one shot provides an increase in successful results
            stat, p = wilcoxon(one_shot, zero_shot, alternative='greater')
            print(f"Wilcoxon test between {results_1shot} and {results_0shot}: stat={stat}, p-value={p}")


def zero_shot_vs_two_shot():
    for results_2shot in two_shot_prompts:
        for results_0shot in zero_shot_prompts:
            two_shot = prompt_counts[results_2shot]
            zero_shot = prompt_counts[results_0shot]
            stat, p = wilcoxon(two_shot, zero_shot, alternative='greater')
            print(f"Wilcoxon test between {results_2shot} and {results_0shot}: stat={stat}, p-value={p}")

def one_shot_vs_two_shot():
    for results_2shot in two_shot_prompts:
        for results_1shot in one_shot_prompts:
            two_shot = prompt_counts[results_2shot]
            one_shot = prompt_counts[results_1shot]
            stat, p = wilcoxon(two_shot, one_shot, alternative='greater')
            print(f"Wilcoxon test between {results_2shot} and {results_1shot}: stat={stat}, p-value={p}")

print(len(prompt_counts["prompt1"]), len(prompt_counts["prompt2"]))
print("ONE-SHOT VS ZERO-SHOT")
zero_shot_vs_one_shot()
print()
print("TWO-SHOT VS ZERO-SHOT")
print()
zero_shot_vs_two_shot()
print()
print("TWO-SHOT VS ONE-SHOT")
one_shot_vs_two_shot()


