import os
import json
import re
import math
import csv
import sys
import ast
import subprocess
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Any, Set
from collections import defaultdict
from datetime import datetime

# Add parent directory to path to import get_commands
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_pipeline.services.get_commands import extract_commands_from_response

ROOT_DIR = "/data/nkhajehn/watcher-mcp-server"
DEFAULT_OUTPUT_DIR = os.path.join("/data/nkhajehn/watcher-mcp-server", "post_processing", "plots")
KNOWN_COMMANDS_CSV = os.path.join("/data/nkhajehn/watcher-mcp-server/known_commands", "known_commands_ftp.csv")

CHINESE_MODELS_CSV = "/data/nkhajehn/watcher-mcp-server/chinese_model_folder/chinese_models.csv"
NON_CHINESE_MODELS_CSV = "/data/nkhajehn/watcher-mcp-server/chinese_model_folder/non_chinese_models.csv"


def extract_prompt_name(folder_name: str) -> str:
    """Extract prompt name from folder name by removing -runN suffix.

    For dynamic folders (starting with "dynamic-"), returns the full folder name as-is.
    For run folders (containing "-run"), removes the -runN suffix.

    Examples:
        "prompt1-run1" -> "prompt1"
        "security-incremental-1-run2" -> "security-incremental-1"
        "prompt1-run3" -> "prompt1"
        "dynamic-005" -> "dynamic-005"
        "dynamic-006" -> "dynamic-006"
    """
    match = re.match(r"^(.+)-run\d+$", folder_name)
    if match:
        return match.group(1)
    return folder_name


def find_run_folders(root_dir: str) -> List[str]:
    """Return absolute paths of directories under root whose names match *-run* or start with dynamic-.

    We only scan direct children of root_dir, not nested deeper.
    Skips folders starting with 'failed_outputs-'.
    Includes both run folders (containing "-run") and dynamic folders (starting with "dynamic-").
    """
    run_folders: List[str] = []
    try:
        for entry in os.listdir(root_dir):
            if entry.startswith("failed_outputs-"):
                continue
            entry_path = os.path.join(root_dir, entry)
            if os.path.isdir(entry_path):
                if "-run" in entry or entry.startswith("dynamic-"):
                    run_folders.append(entry_path)
    except FileNotFoundError:
        pass
    return sorted(run_folders)


def iter_json_files(folder: str) -> Iterator[str]:
    """Yield JSON file paths under folder, skipping any in failed_outputs subdirs."""
    for current_dir, dirnames, filenames in os.walk(folder):
        dirnames[:] = [d for d in dirnames if d != "failed_outputs"]
        for filename in filenames:
            if filename.lower().endswith(".json"):
                yield os.path.join(current_dir, filename)


def load_records(json_path: str) -> List[Dict[str, Any]]:
    """Load JSON file and return a list of record dicts.

    Handles either a single object or a list of objects. Returns empty list on errors.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def load_grammar() -> List[str]:
    config_path = Path(__file__).resolve().parent.parent / "config.txt"
    load_grammar = []
    try:
        text = config_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("grammar="):
                rhs = line.split("=", 1)[1].strip()
                try:
                    parsed = ast.literal_eval(rhs)
                    if isinstance(parsed, (list, tuple)):
                        load_grammar = list(parsed)
                    else:
                        load_grammar = [str(parsed)]
                except Exception:
                    matches = re.findall(r"'([^']*)'|\"([^\"]*)\"", rhs)
                    load_grammar = [m[0] or m[1] for m in matches]
                break
    except Exception:
        load_grammar = []
    return load_grammar


def test_param_type(param: str, expected_type: str):
    try:
        if expected_type == "float":
            float(param)
            return True
        elif expected_type == "int":
            int(param)
            return True
        elif expected_type == "bool":
            if param.lower() not in ("on", "off", "1", "0", "true", "false"):
                raise ValueError
            return True
    except ValueError:
        return False


def validate_entry(entry, specs, grammar, known_commands):
    cmd = entry["command"].strip()

    param = entry["parameters"]
    for sep in grammar:
        full = ""
        param_counts = [int(v["num_of_parameters"]) for v in known_commands.values()]
        min_params = min(param_counts)
        max_params = max(param_counts)
        valid_commands = []
        for num_of_params in range(0, max_params + 1):
            if len(param) == num_of_params:
                if len(param) == 0:
                    full = f"{cmd}"
                else:
                    full = cmd + param[0]
                    for i in range(1, num_of_params):
                        full += sep + param[i]
                        if cmd in specs and len(specs[cmd]["parameter_type"]) < i and test_param_type(param[i], specs[cmd]["parameter_type"][i]):
                            valid_commands.append(full)
                            continue
        if cmd in specs:
            valid_commands.append(cmd)
        elif len(param) > 0:
            if cmd + param[0] + sep in specs:
                valid_commands.append(cmd + param[0] + sep)
            elif cmd + param[0] in specs:
                valid_commands.append(cmd + param[0])
        elif cmd + sep in specs:
            valid_commands.append(cmd + sep)

    return full, valid_commands


def parameter_type(parameter, type_from_csv):
    if type_from_csv == "int":
        try:
            int(parameter)
            return True
        except ValueError:
            return False
    elif type_from_csv == "str":
        return isinstance(parameter, str)
    elif type_from_csv == "char":
        return isinstance(parameter, str) and len(parameter) == 1


def extract_command_names(record: Dict[str, Any], model: str, prompt_name, iteration, known_commands) -> List[str]:
    """Extract list of command strings from record's response field."""
    response = record.get("response")

    if response is None:
        return []
    else:
        response = json.loads(response)

    commands_iter = extract_commands_from_response(response, return_params=True)

    commands = []
    valid = []
    base_commands = set()

    for command in list(commands_iter):
        dict_command = dict(command)
        full_string = list(dict_command.keys())[0]
        base_command = list(dict_command.keys())[0].strip()

        if len(list(dict_command.values())[0]) == 0:
            commands.append(full_string)

        parameter_count = 0
        for param in dict_command[base_command]:
            full_string += " " + param
            parameter_count += 1
            key = (base_command, str(parameter_count))
            commands.append(full_string)
            if key in known_commands:
                parameters = full_string.split(" ")[1:] if " " in full_string else ""
                for count in range(0, parameter_count):
                    type_of_param = known_commands[key]
                    if parameter_type(parameters[count], type_of_param[count]):
                        base_commands.add(base_command)
                        valid.append(full_string)
        if (base_command, 0) in known_commands and parameter_count == 0:
            base_commands.add(base_command)
            valid.append(full_string)

    return commands, valid, base_commands


def extract_run_number(folder_name: str) -> int:
    """Extract run number from folder name using regex."""
    match = re.search(r"-run(\d+)$", folder_name)
    if match:
        try:
            return int(match.group(1))
        except (ValueError, TypeError):
            return 0
    return 0


def is_coding_model(model_name: str) -> bool:
    """Return True if the model name contains 'code' (case-insensitive), else False."""
    return "code" in (model_name or "").lower()


def load_model_chinese_mapping(
    chinese_csv_path: str = CHINESE_MODELS_CSV,
    non_chinese_csv_path: str = NON_CHINESE_MODELS_CSV,
) -> Dict[str, bool]:
    """
    Load model -> is_chinese mapping from two CSV files.

    Expected column name in both files: "Model Name"

    Rules:
    - models in non_chinese_models.csv -> False
    - models in chinese_models.csv -> True
    - if a model appears in both, Chinese takes precedence
    """
    model_to_is_chinese: Dict[str, bool] = {}

    def _read_csv(path: str, flag: bool) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    model_name = (row.get("Model Name") or "").strip()
                    if model_name:
                        model_to_is_chinese[model_name] = flag
        except Exception as e:
            print(f"Warning: Could not load model Chinese mapping from {path}: {e}")

    _read_csv(non_chinese_csv_path, False)
    _read_csv(chinese_csv_path, True)

    return model_to_is_chinese


def is_chinese_model(model_name: str, model_to_is_chinese: Dict[str, bool]) -> bool:
    """Return True if model is marked Chinese in mapping, else False."""
    return model_to_is_chinese.get((model_name or "").strip(), False)


def load_known_commands(csv_path: str = KNOWN_COMMANDS_CSV) -> Set[str]:
    """Load known commands from CSV file and return a mapping-like dict."""
    specs = {}
    try:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                specs[row["command"].strip(), row["num_of_parameters"]] = row["parameter_type"].split(" ")
    except Exception as e:
        print(f"Error loading known commands from {csv_path}: {e}")
    return specs


def normalize_command(cmd: str, base_cmd=False) -> str:
    """Normalize a command string by stripping whitespace and commas."""
    if base_cmd:
        s = cmd.strip().upper()
        s = s.split(':', 1)[0].split(' ', 1)[0]
        s = re.sub(r'\d+$', '', s)
        return s
    return cmd.strip().rstrip(",").strip()


_ollama_param_cache: Dict[str, float] = {}


def get_parameter_count_from_ollama(model_name: str) -> float:
    """Run `ollama show <model_name>` and return the parameter count in billions."""
    if model_name in _ollama_param_cache:
        return _ollama_param_cache[model_name]

    count = 0.0
    try:
        result = subprocess.run(
            ["ollama", "show", model_name],
            capture_output=True, text=True, timeout=30,
        )
        for line in result.stdout.splitlines():
            m = re.search(r'parameters\s+([\d.]+)\s*([BbMmKkGgTt]?)', line)
            if m:
                number = float(m.group(1))
                suffix = m.group(2).upper()
                if suffix == "B":
                    count = number
                elif suffix == "M":
                    count = number / 1_000
                elif suffix == "K":
                    count = number / 1_000_000
                elif suffix == "T":
                    count = number * 1_000
                else:
                    count = number / 1e9  # assume raw count
                break
    except Exception as e:
        print(f"Warning: ollama show {model_name} failed: {e}")

    _ollama_param_cache[model_name] = count
    return count


def get_parameter_count_from_model_name(model_name: str) -> float:
    """Extract parameter count in billions from the model name string.

    Handles patterns like 'llama3.2:3b', 'mistral:7b', 'phi3:14b', 'deepseek-coder:6.7b'.
    Returns 0.0 if no size pattern is found.
    """
    m = re.search(r'(\d+(?:\.\d+)?)\s*[bB]\b', model_name)
    if m:
        return float(m.group(1))
    return 0.0


def load_model_parameters(root_dir: str = ROOT_DIR) -> Dict[str, float]:
    """Load model parameter sizes from models_combined_with_num_predict.csv (used as fallback)."""
    model_params_map = {}
    models_csv_path = "/data/nkhajehn/watcher-mcp-server/llm_pipeline/services/models_combined_with_num_predict.csv"
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
                    model_params_map[model_name] = param_size
    except Exception as e:
        print(f"Warning: Could not load model parameter mapping from {models_csv_path}: {e}")

    return model_params_map


def generate_csv_data(
    folder: str,
    prompt_name: str,
    run_number: int,
    known_commands: Set[str],
    model_params_map: Dict[str, float],
    model_to_is_chinese: Dict[str, bool],
) -> List[Dict[str, Any]]:
    """Generate CSV data rows for a single run folder."""
    records_by_model_iteration: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    model_list_name = list(model_params_map.keys())

    for json_path in iter_json_files(folder):
        for record in load_records(json_path):
            model = record.get("model")
            if not model or model == 'granite3-guardian:2b' or model == 'moondream:1.8b' or model == 'llama-guard3:8b':
                continue

            params = record.get("params", {})
            if not isinstance(params, dict):
                continue

            iteration = params.get("ITERATION")
            if iteration is None:
                continue

            try:
                iteration = int(iteration)
            except (ValueError, TypeError):
                continue

            model_str = str(model)
            records_by_model_iteration[(model_str, iteration)].append(record)

    max_iteration_by_model: Dict[str, int] = {}
    for (model, iteration) in records_by_model_iteration.keys():
        if model not in max_iteration_by_model or iteration > max_iteration_by_model[model]:
            max_iteration_by_model[model] = iteration

    rows: List[Dict[str, Any]] = []

    all_commands_seen: Dict[str, list[str]] = defaultdict(list)
    all_valid_commands_seen: Dict[str, list[str]] = defaultdict(list)
    base_commands_seen_so_far: Dict[str, set[str]] = defaultdict(set)

    last_seen_iteration: Dict[str, int] = {}
    cumulative_failures: Dict[str, int] = defaultdict(int)

    for model in sorted(max_iteration_by_model.keys()):
        max_iteration = max_iteration_by_model[model]

        for iteration in range(1, 51):
            if (model, iteration) in records_by_model_iteration and iteration <= max_iteration:
                records = records_by_model_iteration[(model, iteration)]

                if model in last_seen_iteration:
                    gap = iteration - last_seen_iteration[model] - 1
                    if gap > 0:
                        cumulative_failures[model] += gap
                else:
                    if iteration > 1:
                        cumulative_failures[model] += iteration - 1

                last_seen_iteration[model] = iteration

                all_commands = []
                validated_commands = []
                base_commands = set()

                commands = extract_command_names(records[0], model, prompt_name, iteration, known_commands)

                for cmd in commands[0]:
                    if cmd:
                        all_commands_seen[model].append(cmd)
                        all_commands.append(cmd)

                for cmd in commands[1]:
                    if cmd:
                        all_valid_commands_seen[model].append(cmd)
                        validated_commands.append(cmd)

                base_commands = commands[2]
                base_commands_seen_so_far[model] = base_commands_seen_so_far[model].union(base_commands)

                iteration_duration_seconds = None
                try:
                    record = records[0]
                    started_at_str = record.get("started_at")
                    ended_at_str = record.get("ended_at")
                    if started_at_str and ended_at_str:
                        started_at = datetime.fromisoformat(started_at_str.replace('Z', '+00:00'))
                        ended_at = datetime.fromisoformat(ended_at_str.replace('Z', '+00:00'))
                        iteration_duration_seconds = (ended_at - started_at).total_seconds()
                except (ValueError, TypeError, AttributeError):
                    iteration_duration_seconds = None

            else:
                if iteration > max_iteration:
                    cumulative_failures[model] += 1

                all_commands = []
                validated_commands = []
                base_commands = set()
                iteration_duration_seconds = None

            parameter_count = get_parameter_count_from_model_name(model) or get_parameter_count_from_ollama(model)

            rows.append({
                "model": model,
                "is_coding_model": is_coding_model(model),
                "is_chinese": is_chinese_model(model, model_to_is_chinese),
                "prompt": prompt_name,
                "run_number": run_number,
                "iteration": iteration,
                "number_of_commands": len(all_commands),
                "cumulative_commands": len(all_commands_seen[model]),
                "unique_commands": len(set(all_commands_seen[model])),
                "valid_commands": len(validated_commands),
                "unique_valid_commands": len(set(all_valid_commands_seen[model])),
                "number_of_base_commands_in_iteration": len(base_commands),
                "base_commands_seen_so_far": base_commands_seen_so_far[model],
                "parameter_count": parameter_count,
                "iteration_duration_seconds": iteration_duration_seconds,
                "cumulative_failures": cumulative_failures[model]
            })

            if iteration == 50 and model in model_list_name:
                model_list_name.remove(model)

    if len(model_list_name) > 0:
        for model in model_list_name:
            rows.append({
                "model": model,
                "is_coding_model": is_coding_model(model),
                "is_chinese": is_chinese_model(model, model_to_is_chinese),
                "prompt": prompt_name,
                "run_number": run_number,
                "iteration": 50,
                "number_of_commands": 0,
                "cumulative_commands": 0,
                "unique_commands": 0,
                "valid_commands": 0,
                "unique_valid_commands": 0,
                "number_of_base_commands_in_iteration": 0,
                "base_commands_seen_so_far": set(),
                "parameter_count": get_parameter_count_from_model_name(model) or get_parameter_count_from_ollama(model),
                "iteration_duration_seconds": 0,
                "cumulative_failures": 50
            })

    print("model_list_name", model_list_name)

    return rows


def main(output_dir: str = DEFAULT_OUTPUT_DIR) -> None:
    run_folders = find_run_folders(ROOT_DIR)
    if not run_folders:
        print("No *-run* folders found under", ROOT_DIR)
        return

    print("Loading known commands...")
    known_commands = load_known_commands()
    if not known_commands:
        print("Warning: No known commands loaded from CSV")
    else:
        print(f"Loaded {len(known_commands)} known commands")

    print("Loading model parameters...")
    model_params_map = load_model_parameters()
    if not model_params_map:
        print("Warning: No model parameters loaded from CSV")
    else:
        print(f"Loaded parameter counts for {len(model_params_map)} models")

    print("Loading Chinese/non-Chinese model mapping...")
    model_to_is_chinese = load_model_chinese_mapping()
    if not model_to_is_chinese:
        print("Warning: No Chinese/non-Chinese mapping loaded")
    else:
        print(f"Loaded nationality labels for {len(model_to_is_chinese)} models")

    all_csv_rows: List[Dict[str, Any]] = []
    print("hello")
    for folder in sorted(run_folders):
        folder_name = os.path.basename(folder.rstrip(os.sep))
        print(folder_name, "hi", "prompt1" not in folder_name)

        prompt_name = extract_prompt_name(folder_name)
        run_number = extract_run_number(folder_name)

        print(f"Processing {folder_name}...")
        rows = generate_csv_data(
            folder,
            prompt_name,
            run_number,
            known_commands,
            model_params_map,
            model_to_is_chinese,
        )
        all_csv_rows.extend(rows)

    csv_path = os.path.join(output_dir, "FTP_rag_results.csv")
    os.makedirs(output_dir, exist_ok=True)

    if all_csv_rows:
        fieldnames = [
            "model",
            "is_coding_model",
            "is_chinese",
            "prompt",
            "run_number",
            "iteration",
            "number_of_commands",
            "cumulative_commands",
            "unique_commands",
            "valid_commands",
            "unique_valid_commands",
            "number_of_base_commands_in_iteration",
            "base_commands_seen_so_far",
            "parameter_count",
            "iteration_duration_seconds",
            "cumulative_failures",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_csv_rows)
        print(f"\nCSV file written: {csv_path}")
        print(f"Total rows: {len(all_csv_rows)}")
        print("\nTo create plots, run: python post_processing/create_plots.py")
    else:
        print("\nNo data found to write to CSV.")


if __name__ == "__main__":
    main()