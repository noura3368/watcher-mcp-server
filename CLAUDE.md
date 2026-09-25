# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research harness for measuring how well many local LLMs (~150, served by Ollama) generate security-test commands for a target system. Examples are a Korad KA3005P power supply, the FTP protocol, and a WiFi module over AT commands. Each system is run with and without retrieved documentation (RAG). The results are then scored and compared statistically.

There is no build, lint, or test suite. Everything is a standalone Python script.

## Three stages

1. **Document ingestion (`watcher/`).** `watcher.py` watches `raw_docs/` and keeps the ColQwen page index in `colrag/store.py` (LanceDB at `data/colqwen.lancedb`) in sync with it. Every PDF page is one row. The page image is embedded by ColQwen (`vidore/colqwen2.5-v0.2` by default, set by `COLRAG_MODEL`; ColQwen3 checkpoints also work) into one 128-dim vector per patch, and those vectors are used only for search. The page markdown (from pymupdf4llm) is stored next to them, and that is what the models get as context. Pages with no text are skipped. Non-PDF files are ignored. Processed-file hashes are tracked in `data/watcher_state.json`. The encoder is loaded per file and freed afterwards, so it does not hold GPU memory Ollama needs.
2. **Generation (`llm_pipeline/services/`).** See below.
3. **Analysis.** `post_processing/` turns result folders into CSVs and plots, and `llm_pipeline/hypothesis_testing/` runs Wilcoxon, Mann-Whitney U, and Kruskal-Wallis tests. These scripts find result folders by the `<template>-run<N>` naming (and `dynamic-*`). They also read `known_commands/known_commands_<system>.csv` for validation and `chinese_model_folder/*.csv` for model grouping. Renaming output folders breaks them.

## Generation pipeline

```
cd llm_pipeline/services
python run.py --config config.txt --mode rag     # or norag, or lc (whole LC_file PDF as context)
python run.py --mode norag --shard 0/2 --host http://localhost:11434   # split the model list across servers
python models.py --csv models.csv --pull-all     # pre-download every model in the CSV
```

- **Config.** `config.txt` is a `key=value` file, and every key is documented inline. It sets the model CSV (`Model Name` column), `templates_dir`, which picks the system (`templates/` = Korad, `ftp_prompts/`, `wifi_prompts/`), `template`, `target`, and `interface`. It also sets the iteration and trial counts (`num_iterations` is a count), `system`, `mode`, `output_root`, `parallel_chains`, `pull_models`, and `remove_after`. A missing config file or required key is an error, not a fallback to defaults.
- **Loop.** For each model: pull it if it is missing, while the next model downloads in the background. Load it once, then run every (iteration, template) chain, up to `parallel_chains` at a time. Trials inside a chain are sequential, because `FOUNDSOFAR` holds the commands from earlier trials. It is kept in memory and formatted by `get_commands.format_foundsofar`, and that format must stay stable so prompts remain comparable across runs.
- **Outputs.** Successes go to `<output_root>/<system>/<mode>/<template>-run<N>/` and failures to the sibling `failed_outputs-<template>-run<N>/`. Nothing is moved or deleted. A rerun resumes: completed trials, counted per exact `model` field, are skipped, and `FOUNDSOFAR` is rebuilt from them. RAG or LC context is saved as `context.txt` in the mode folder.
- **Generation.** `structured_output.generate_structured` calls Ollama with `format=` set to the `SecurityTestResponse` JSON schema, which is the same request outlines used to make. It validates the result with Pydantic. By design, no generation options are set (temperature, num_ctx, num_predict), so every model runs with its own defaults. The response is streamed, and `timeout` is enforced as a total deadline per trial. `RunawayDetector` checks the answer text as it arrives for two patterns only: `runaway_max_whitespace` whitespace-only pieces in a row, and the same command with the same parameters `runaway_max_repeats` times in a row. Reasoning text is not checked. With `runaway_detection=shadow` it only records `stats.runaway_flag`. With `enforce` it stops the trial. Failed trials keep their partial text and a `failure_reason`: timeout, runaway:<pattern>, schema, error, or render.
- **Templates.** Jinja is rendered with `StrictUndefined`. Every template is test-rendered at startup. The variables are `TARGET`, `INTERFACE`, `FOUNDSOFAR`, `ITERATION`, and the context, which is passed as both `RAG` (FTP/WiFi) and `CONTEXT` (Korad).
- **Modes.** `rag` and `lc` exit if the context is empty or fails to load. When `mode` is empty, the legacy `RAG_ENABLED`/`LC_ENABLED` environment variables decide, and RAG is the default.
- **RAG.** `rag_service.retrieve_context` embeds `rag_query` (default `Commands, syntax and parameters for {target}`), scores every indexed page with exact MaxSim, and joins the text of the top `rag_top_k` pages under `[file p.N]` headers. The retrieved pages and their scores are saved to `rag_context.json` in the mode folder. `rag` results produced before the switch from haiku.rag/mxbai chunk retrieval to ColQwen are not comparable with newer ones.

## Environment notes

- Absolute paths under `/data/nkhajehn/watcher-mcp-server` (and `/data2/...` in some analysis scripts) are still hardcoded in `config.txt`, `watcher.py`, and the post-processing scripts. `rag_service.py` resolves the index from the repo root, or from `COLRAG_DB`/`rag_db`.
- ColQwen needs a CUDA build of torch. On the aarch64 GB10 machine, install it from the PyTorch CUDA wheel index. Attention uses `sdpa`, so flash-attn is not needed.
- Post-processing `ROOT_DIR` must point at `<output_root>/<system>/<mode>`.
- Parallel chains need `OLLAMA_NUM_PARALLEL` set on the Ollama server. Every parallel slot reserves a full context cache.
- Dependencies: `llm_pipeline/requirements.txt` and `watcher/requirements.txt`. Ollama must be running on `localhost:11434`.
- Outputs, results, `data/`, `raw_docs/`, and `docs_processed/` are gitignored.
