#!/usr/bin/env python
"""
Run prompt templates against many Ollama models.

    python run.py [--config config.txt] [--mode rag|norag|lc] [--shard K/N] [--host URL] [-v]

Loop order is model -> (iteration, template) chains -> trials, so each model is
loaded once. Trials within a chain are sequential because FOUNDSOFAR carries the
commands found by earlier trials; separate chains can run concurrently
(`parallel_chains`). Outputs go to

    <output_root>/<system>/<mode>/<template>-run<iteration>/
    <output_root>/<system>/<mode>/failed_outputs-<template>-run<iteration>/

Runs are resumable: finished trials found on disk are skipped and FOUNDSOFAR is
rebuilt from them. No Ollama generation options are set; every model uses its
own defaults.
"""

import argparse
import datetime
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from get_commands import extract_commands_from_response, format_foundsofar
from models import ensure_model, is_present, load_models, remove_model
from structured_output import DETECTION_MODES, GenerationError, generate_structured, make_client

log = logging.getLogger("run")
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULTS = {
    "models_csv": None,
    "templates_dir": None,
    "template": [],             # empty = all templates in templates_dir
    "start_iteration": 1,
    "num_iterations": 1,        # a count: runs start_iteration .. start_iteration+num_iterations-1
    "trials_per_model": 10,
    "target": None,
    "interface": "",
    "timeout": 600.0,
    "model_row": 0,             # first CSV data row to use (1-based; 0/1 = from the top)
    "model_end_row": 0,         # last CSV data row to use (0 = to the end)
    "system": None,             # korad / ftp / wifi; derived from templates_dir if empty
    "mode": None,               # rag / norag / lc; falls back to RAG_ENABLED/LC_ENABLED env vars
    "output_root": "results",
    "parallel_chains": 1,
    "pull_models": True,
    "remove_after": False,      # delete models this run pulled once they are finished
    "rag_db": "",
    "rag_top_k": 5,
    "rag_query": "",            # empty = rag_service.DEFAULT_QUERY; may use {target} and {interface}
    "LC_file": "",
    "runaway_detection": "shadow",  # off / shadow (only record) / enforce (stop the trial)
    "runaway_max_whitespace": 300,  # whitespace-only pieces in a row
    "runaway_max_repeats": 10,      # identical command+parameters entries in a row
}
INT_KEYS = {"start_iteration", "num_iterations", "trials_per_model", "model_row",
            "model_end_row", "parallel_chains", "rag_top_k", "runaway_max_whitespace",
            "runaway_max_repeats"}
FLOAT_KEYS = {"timeout"}
BOOL_KEYS = {"pull_models", "remove_after"}
REQUIRED = ("models_csv", "templates_dir", "target")
SYSTEM_BY_DIR = {"templates": "korad", "ftp_prompts": "ftp", "wifi_prompts": "wifi"}
MODES = ("rag", "norag", "lc")


# ---------------------------------------------------------------- config

def load_config(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    config = dict(DEFAULTS)
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                log.warning("Invalid config line %d: %s", line_num, line)
                continue
            key, value = (s.strip() for s in line.split("=", 1))
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            try:
                if key in INT_KEYS:
                    config[key] = int(value)
                elif key in FLOAT_KEYS:
                    config[key] = float(value)
                elif key in BOOL_KEYS:
                    config[key] = value.lower() in ("1", "true", "yes", "on")
                elif key == "template":
                    config[key] = [] if value.lower() in ("all", "") else [t.strip() for t in value.split(",") if t.strip()]
                else:
                    config[key] = value
            except ValueError:
                raise SystemExit(f"Config line {line_num}: bad value for {key}: {value!r}")

    missing = [k for k in REQUIRED if not config.get(k)]
    if missing:
        raise SystemExit(f"Config {path} is missing required keys: {', '.join(missing)}")
    if config["runaway_detection"] not in DETECTION_MODES:
        raise SystemExit(f"runaway_detection must be one of {DETECTION_MODES}")
    if config["num_iterations"] < 1 or config["trials_per_model"] < 1:
        raise SystemExit("num_iterations and trials_per_model must be >= 1")
    if not config["system"]:
        name = Path(config["templates_dir"]).name
        config["system"] = SYSTEM_BY_DIR.get(name, name)
    return config


def resolve_mode(cli_mode: Optional[str], config: dict) -> str:
    mode = cli_mode or config.get("mode")
    if mode:
        if mode not in MODES:
            raise SystemExit(f"mode must be one of {MODES}, got {mode!r}")
        return mode
    # Legacy environment switches (RAG was the default when unset).
    off = ("0", "false", "no")
    if os.getenv("RAG_ENABLED", "1").lower() not in off:
        mode = "rag"
    elif os.getenv("LC_ENABLED", "0").lower() not in off:
        mode = "lc"
    else:
        mode = "norag"
    log.warning("No mode in config or --mode; using %r from RAG_ENABLED/LC_ENABLED env vars", mode)
    return mode


def load_context(mode: str, config: dict) -> Tuple[str, List[dict]]:
    """Return (context text, retrieved pages) for the mode. Exits if RAG/LC was requested but produced nothing."""
    if mode == "norag":
        return "", []
    pages = []
    try:
        if mode == "rag":
            from rag_service import retrieve_context
            text, pages = retrieve_context(config["target"], config["interface"], top_k=config["rag_top_k"],
                                           db_path=config["rag_db"] or None, query=config["rag_query"])
        else:
            from lc_service import load_lc_content
            text = load_lc_content(config["LC_file"])
    except Exception as e:
        raise SystemExit(f"[{mode}] could not load context: {e}")
    if not text.strip():
        raise SystemExit(f"[{mode}] context is empty; refusing to run a {mode} experiment without it")
    return text, pages


# ---------------------------------------------------------------- templates

def select_templates(config: dict) -> List[str]:
    tdir = Path(config["templates_dir"]).expanduser()
    available = sorted(p.name for p in tdir.glob("*.jinja"))
    if not available:
        raise SystemExit(f"No templates found in {tdir}")
    if not config["template"]:
        return available
    chosen = [t for t in config["template"] if t in available]
    for t in config["template"]:
        if t not in available:
            log.warning("Template %r not found in %s", t, tdir)
    if not chosen:
        raise SystemExit("No valid templates specified")
    return chosen


def make_env(config: dict) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(Path(config["templates_dir"]).expanduser())),
        autoescape=False, undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True,
    )


def build_params(config: dict, context: str, foundsofar: str, trial: int) -> dict:
    # Korad templates use {{ CONTEXT }}, FTP/WiFi templates use {{ RAG }}; both get the same text.
    return {
        "TARGET": config["target"],
        "INTERFACE": config["interface"],
        "FOUNDSOFAR": foundsofar,
        "ITERATION": trial,
        "RAG": context,
        "CONTEXT": context,
    }


def check_templates_render(env: Environment, templates: List[str], config: dict, context: str) -> None:
    for name in templates:
        try:
            env.get_template(name).render(**build_params(config, context, "", 1))
        except Exception as e:
            raise SystemExit(f"Template {name} does not render: {e}")


# ---------------------------------------------------------------- results on disk

def safe_name(model: str) -> str:
    return model.replace("/", "_")


@dataclass
class Chain:
    iteration: int
    template: str
    out_dir: Path
    fail_dir: Path
    done: int = 0
    found: Set[str] = None


def scan_chain(chain: Chain, model: str) -> None:
    """Count trials this model already finished in the chain and rebuild its found-command set."""
    safe = safe_name(model)
    done, found = 0, set()
    for path in chain.out_dir.glob(f"{chain.template}_{safe}_*.json"):
        data = _read_json(path)
        if data.get("model") == model:
            done += 1
            found.update(extract_commands_from_response(data.get("response"), False))
    for path in chain.fail_dir.glob(f"failed_{safe}_*.json"):
        if _read_json(path).get("model") == model:
            done += 1
    chain.done, chain.found = done, found


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)  # atomic, so an interrupted write never leaves a half file that counts as a trial


# ---------------------------------------------------------------- generation

class Runner:
    def __init__(self, config, mode, context, env, client, stop: threading.Event):
        self.config, self.mode, self.context = config, mode, context
        self.env, self.client, self.stop = env, client, stop
        self.context_length = None

    def run_chain(self, model: str, chain: Chain) -> None:
        cfg = self.config
        template = self.env.get_template(chain.template)
        for trial in range(chain.done + 1, cfg["trials_per_model"] + 1):
            if self.stop.is_set():
                return
            log.info("%s | %s | iteration %d | trial %d/%d", model, chain.template, chain.iteration,
                     trial, cfg["trials_per_model"])
            params = build_params(cfg, self.context, format_foundsofar(chain.found), trial)
            record = {
                "template": chain.template,
                "data": "command.json",
                "params": params,
                "model": model,
                "system": cfg["system"],
                "mode": self.mode,
                "iteration": chain.iteration,
                "trial": trial,
                "context_length": self.context_length,
            }
            stamp = str(time.time_ns())
            try:
                prompt = template.render(**params).strip()
            except Exception as e:
                self._fail(chain, model, stamp, record, f"Template render error: {e}", None)
                continue
            record["prompt"] = prompt
            log.debug("Prompt:\n%s", prompt)

            try:
                result = generate_structured(
                    self.client, model, prompt, deadline_s=cfg["timeout"],
                    detection=cfg["runaway_detection"],
                    max_whitespace_run=cfg["runaway_max_whitespace"],
                    max_identical_repeats=cfg["runaway_max_repeats"])
            except GenerationError as e:
                self._fail(chain, model, stamp, record, str(e), e)
                continue

            commands = [c.model_dump() for c in result.response.commands]
            record.update({
                "time": result.elapsed_ms,
                "response": json.dumps(commands, indent=2),
                "structured": True,
                "started_at": result.started_at.isoformat(),
                "ended_at": result.ended_at.isoformat(),
                "stats": result.stats,
            })
            self._warn_if_truncated(model, chain, trial, result.stats)
            if result.stats.get("runaway_flag"):
                # Shadow mode: the detector fired but the answer still finished and validated.
                log.warning("%s | %s | iteration %d | trial %d: detector flagged %s but the answer was valid",
                            model, chain.template, chain.iteration, trial, result.stats["runaway_flag"]["reason"])
            write_json(chain.out_dir / f"{chain.template}_{safe_name(model)}_{stamp}.json", record)
            chain.found.update(extract_commands_from_response(commands, False))

    def _fail(self, chain, model, stamp, record, error, exc: Optional[GenerationError]) -> None:
        record.update({
            "time": getattr(exc, "elapsed_ms", None),
            "error": error,
            "failure_reason": getattr(exc, "reason", None) if exc else "render",
            "response": getattr(exc, "raw", None),
            "fail_time": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "structured": False,
            "started_at": exc.started_at.isoformat() if exc and exc.started_at else None,
            "ended_at": exc.ended_at.isoformat() if exc and exc.ended_at else None,
            "stats": getattr(exc, "stats", {}),
        })
        path = chain.fail_dir / f"failed_{safe_name(model)}_{stamp}.json"
        write_json(path, record)
        log.warning("%s | %s | iteration %d | trial %s failed: %s (saved %s)", model, chain.template,
                    chain.iteration, record["trial"], error[:200], path.name)

    def _warn_if_truncated(self, model, chain, trial, stats) -> None:
        # Heuristic only: prompt caching can make prompt_eval_count smaller than the real prompt.
        n, ctx = stats.get("prompt_eval_count"), self.context_length
        if n and ctx and n >= 0.95 * ctx:
            log.warning("%s | %s | trial %d: prompt used %d of %d context tokens; it was probably truncated",
                        model, chain.template, trial, n, ctx)

    def load_model(self, model: str) -> bool:
        """Load the model into memory once and record the context length Ollama gave it."""
        try:
            self.client.generate(model=model, prompt="")
        except Exception as e:
            log.error("Could not load %s: %s", model, e)
            return False
        self.context_length = None
        try:
            for m in self.client.ps().models:
                if m.model == model or m.name == model:
                    self.context_length = getattr(m, "context_length", None)
        except Exception:
            pass
        log.info("Loaded %s (context length %s)", model, self.context_length or "unknown")
        return True


# ---------------------------------------------------------------- main

def parse_shard(value: Optional[str]) -> Tuple[int, int]:
    if not value:
        return 0, 1
    try:
        k, n = (int(x) for x in value.split("/"))
        assert 0 <= k < n
        return k, n
    except Exception:
        raise SystemExit("--shard must look like K/N with 0 <= K < N, e.g. 0/2")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(SCRIPT_DIR / "config.txt"))
    ap.add_argument("--mode", choices=MODES, help="Overrides 'mode' in the config")
    ap.add_argument("--shard", help="Run only models K, K+N, K+2N, ... (0-based), e.g. 1/2")
    ap.add_argument("--host", default=None, help="Ollama host, e.g. http://localhost:11435")
    ap.add_argument("-v", "--verbose", action="store_true", help="Log full prompts")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    config = load_config(Path(args.config))
    mode = resolve_mode(args.mode, config)
    k, n = parse_shard(args.shard)
    models = load_models(config["models_csv"], config["model_row"], config["model_end_row"])[k::n]
    if not models:
        raise SystemExit("No models selected from the CSV")
    templates = select_templates(config)

    context, pages = load_context(mode, config)
    env = make_env(config)
    check_templates_render(env, templates, config, context)

    root = Path(config["output_root"]).expanduser() / config["system"] / mode
    root.mkdir(parents=True, exist_ok=True)
    if context:
        (root / "context.txt").write_text(context, encoding="utf-8")
    if pages:
        write_json(root / "rag_context.json", {"pages": pages})
        for p in pages:
            log.info("RAG page %s p.%d (score %.2f)", Path(p["source_file"]).name, p["page_num"], p["score"])

    iterations = range(config["start_iteration"], config["start_iteration"] + config["num_iterations"])
    log.info("System %s, mode %s, %d models, templates %s, iterations %s-%s, %d trials, output %s",
             config["system"], mode, len(models), templates, iterations.start, iterations.stop - 1,
             config["trials_per_model"], root.resolve())

    client = make_client(args.host, config["timeout"])
    stop = threading.Event()
    runner = Runner(config, mode, context, env, client, stop)
    pull_pool = ThreadPoolExecutor(max_workers=1)
    pulls = {}

    def start_pull(name):
        if config["pull_models"] and name not in pulls:
            pulls[name] = pull_pool.submit(lambda: (not is_present(client, name), ensure_model(client, name)))

    # Work out what is left to do before loading anything, so finished models are never pulled or loaded.
    work = []
    for spec in models:
        chains = []
        for j in iterations:
            for t in templates:
                base = t[:-len(".jinja")] if t.endswith(".jinja") else t
                c = Chain(j, t, root / f"{base}-run{j}", root / f"failed_outputs-{base}-run{j}")
                scan_chain(c, spec.name)
                if c.done < config["trials_per_model"]:
                    chains.append(c)
        if chains:
            work.append((spec, chains))
        else:
            log.info("%s already complete, skipping", spec.name)
    log.info("%d of %d models have work left", len(work), len(models))

    try:
        for idx, (spec, chains) in enumerate(work):
            model = spec.name
            log.info("[%d/%d] %s (CSV row %d)", idx + 1, len(work), model, spec.row)
            pulled_now = False
            if config["pull_models"]:
                start_pull(model)
                pulled_now, ok = pulls.pop(model).result()
                if not ok:
                    log.error("Skipping %s: model not available", model)
                    continue
            if idx + 1 < len(work):
                start_pull(work[idx + 1][0].name)  # download the next model while this one runs

            if not runner.load_model(model):
                if config["remove_after"] and pulled_now:
                    remove_model(client, model)
                continue
            ex = ThreadPoolExecutor(max_workers=max(1, config["parallel_chains"]))
            try:
                for fut in [ex.submit(runner.run_chain, model, c) for c in chains]:
                    fut.result()
            except KeyboardInterrupt:
                stop.set()  # tell running chains to stop after their current request
                raise
            finally:
                ex.shutdown(wait=True, cancel_futures=True)

            if config["remove_after"] and pulled_now:
                remove_model(client, model)
    except KeyboardInterrupt:
        stop.set()
        log.warning("Interrupted. Rerun the same command to resume.")
        raise
    finally:
        pull_pool.shutdown(wait=False, cancel_futures=True)

    log.info("Done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
