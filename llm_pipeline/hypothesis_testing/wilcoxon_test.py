from itertools import permutations
from typing import Dict, Set, Tuple
import csv
import re

from scipy.stats import wilcoxon

zero_shot_prompts = ["prompt5", "prompt7", "prompt9", "prompt8"]
one_shot_prompts = ["prompt1", "prompt2", "prompt3", "prompt4", "prompt10"]
two_shot_prompts = ["prompt6", "prompt11"]


def parse_base_set(set_str: str) -> Set[str]:
    s = (set_str or "").strip()
    if s in ("set()", "{}", ""):
        return set()
    items = re.findall(r"'([^']*)'|\"([^\"]*)\"", s)
    return {a or b for a, b in items}


def load_base_commands_data(csv_path: str) -> Dict[Tuple[str, int, str, str, str], int]:
    """Load base commands data from CSV, keeping the last iteration per (prompt, run, model)."""
    best: Dict[Tuple[str, int, str, str, str], Tuple[int, int]] = {}

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                prompt = row.get("prompt", "").strip()
                model = row.get("model", "").strip()
                if not prompt or not model:
                    continue

                try:
                    run_number = int(row.get("run_number", 0))
                except ValueError:
                    run_number = 0
                try:
                    iteration = int(row.get("iteration", 0))
                except ValueError:
                    iteration = 0

                parameter_count = row.get("parameter_count", "").strip()
                is_code_model = row.get("is_coding_model", "False").strip()
                base_count = len(parse_base_set(row.get("base_commands_seen_so_far", "")))

                key = (prompt, run_number, model, parameter_count, is_code_model)
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


base_data = load_base_commands_data(
    "/data2/nkhajehn/watcher-mcp-server/post_processing/plots/RAG_KORAD_results.csv"
)

all_prompts = zero_shot_prompts + one_shot_prompts + two_shot_prompts

# keyed by (model, run_number) so we can match across prompts
prompt_counts: Dict[str, Dict[Tuple[str, int], int]] = {p: {} for p in all_prompts}

for (prompt, run_number, model, *_), count in base_data.items():
    if prompt in prompt_counts:
        prompt_counts[prompt][(model, run_number)] = count


def compare_all_prompts():
    for pa, pb in permutations(all_prompts, 2):
        keys_a = set(prompt_counts[pa])
        keys_b = set(prompt_counts[pb])

        models_a = {m for m, _ in keys_a}
        models_b = {m for m, _ in keys_b}
        only_a = models_a - models_b
        only_b = models_b - models_a

        if only_a:
            print(f"  Omitting models only in {pa}: {only_a}")
        if only_b:
            print(f"  Omitting models only in {pb}: {only_b}")

        common = sorted(keys_a & keys_b)
        counts_a = [prompt_counts[pa][k] for k in common]
        counts_b = [prompt_counts[pb][k] for k in common]

        stat, p = wilcoxon(counts_a, counts_b, alternative='greater')
        print(f"Wilcoxon test between {pa} and {pb}: stat={stat}, p-value={p}")


compare_all_prompts()
