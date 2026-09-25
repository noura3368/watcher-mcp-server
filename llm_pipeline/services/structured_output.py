#!/usr/bin/env python3
"""
Structured generation against Ollama.

The request is the same one outlines' Ollama adapter sends: `client.generate`
with `format=<JSON schema of SecurityTestResponse>`. Calling Ollama directly
also gives us token counts and durations. No generation options (temperature,
num_ctx, num_predict, ...) are set, so every model runs with its own defaults.

The response is streamed so that a RunawayDetector can watch it and the
wall-clock deadline can be enforced. Streaming only changes how the text is
delivered; Ollama applies the schema the same way either way.
"""

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import ollama
from pydantic import BaseModel, Field


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


RESPONSE_SCHEMA = SecurityTestResponse.model_json_schema()


@dataclass
class GenerationResult:
    response: SecurityTestResponse
    raw: str
    started_at: datetime
    ended_at: datetime
    elapsed_ms: int
    stats: dict = field(default_factory=dict)


class GenerationError(Exception):
    """Raised when the call fails or the output does not match the schema.

    Carries whatever was known at the time of failure so it can be logged.
    """

    def __init__(self, message, raw=None, started_at=None, ended_at=None, elapsed_ms=None, stats=None,
                 reason="error"):
        super().__init__(message)
        self.reason = reason  # timeout / runaway:whitespace / runaway:repetition / schema / error
        self.raw = raw
        self.started_at = started_at
        self.ended_at = ended_at
        self.elapsed_ms = elapsed_ms
        self.stats = stats or {}


def make_client(host: Optional[str] = None, timeout: float = 600) -> ollama.Client:
    return ollama.Client(host=host, timeout=timeout)


def _stats(resp) -> dict:
    keys = ("prompt_eval_count", "eval_count", "total_duration", "load_duration",
            "prompt_eval_duration", "eval_duration", "done_reason")
    return {k: getattr(resp, k, None) for k in keys}


ENTRY_RE = re.compile(r"\{[^{}]*\}")  # one finished command entry; the outer {"commands": [...]} never matches
DETECTION_MODES = ("off", "shadow", "enforce")


class RunawayDetector:
    """Watches streamed answer text for two patterns a valid answer never contains:

    - whitespace: `max_whitespace_run` whitespace-only pieces in a row
    - repetition: the same command with the same parameters `max_identical_repeats` times in a row
    """

    def __init__(self, max_whitespace_run: int = 300, max_identical_repeats: int = 10):
        self.max_ws = max_whitespace_run
        self.max_same = max_identical_repeats
        self.buf = ""
        self.scan_pos = 0
        self.ws_run = 0
        self.last_key = None
        self.same = 0

    def feed(self, piece: str) -> Optional[str]:
        """Add one streamed piece. Returns 'whitespace' or 'repetition' when the output looks broken."""
        if not piece:
            return None
        self.buf += piece
        if piece.strip() == "":
            self.ws_run += 1
            if self.ws_run >= self.max_ws:
                return "whitespace"
            return None
        self.ws_run = 0
        if "}" in piece:
            return self._scan_entries()
        return None

    def _scan_entries(self) -> Optional[str]:
        # Only text after the last finished entry is searched, and only complete entries match.
        for m in ENTRY_RE.finditer(self.buf, self.scan_pos):
            self.scan_pos = m.end()
            try:
                entry = json.loads(m.group())
            except ValueError:
                continue
            key = (str(entry.get("command", "")).strip(),
                   json.dumps(entry.get("parameters", []), sort_keys=True))
            self.same = self.same + 1 if key == self.last_key else 1
            self.last_key = key
            if self.same >= self.max_same:
                return "repetition"
        return None


def generate_structured(client: ollama.Client, model_name: str, prompt: str, deadline_s: float = 600,
                        detection: str = "shadow", max_whitespace_run: int = 300,
                        max_identical_repeats: int = 10) -> GenerationResult:
    """Generate one schema-constrained response. Raises GenerationError on any failure.

    detection: 'enforce' stops the trial when the detector fires, 'shadow' only records
    where it would have stopped (stats['runaway_flag']), 'off' disables the detector.
    Partial text is kept on every failure, including timeouts.
    """
    started_at, t0 = datetime.now(timezone.utc), time.monotonic()
    detector = RunawayDetector(max_whitespace_run, max_identical_repeats) if detection != "off" else None
    parts: List[str] = []
    stats: dict = {}
    flag = None
    reason, message = None, None
    stream = None
    try:
        stream = client.generate(model=model_name, prompt=prompt, format=RESPONSE_SCHEMA, stream=True)
        for chunk in stream:
            piece = chunk.response or ""  # reasoning text arrives in chunk.thinking and is not checked
            parts.append(piece)
            if chunk.done:
                stats = _stats(chunk)
                break
            elapsed = time.monotonic() - t0
            if elapsed > deadline_s:
                reason, message = "timeout", f"Timed out after {deadline_s:.0f}s"
                break
            if detector and flag is None:
                hit = detector.feed(piece)
                if hit:
                    flag = {"reason": hit, "at_ms": int(elapsed * 1000), "at_chars": len(detector.buf)}
                    if detection == "enforce":
                        reason, message = f"runaway:{hit}", f"Stopped early: {hit} detected after {elapsed:.0f}s"
                        break
    except Exception as e:
        reason, message = "error", f"{type(e).__name__}: {e}"
    finally:
        if stream is not None:
            stream.close()  # closes the HTTP connection, which makes Ollama stop generating

    ended_at = datetime.now(timezone.utc)
    elapsed_ms = int((ended_at - started_at).total_seconds() * 1000)
    raw = "".join(parts)
    stats["runaway_flag"] = flag
    if reason:
        raise GenerationError(message, raw=raw, started_at=started_at, ended_at=ended_at,
                              elapsed_ms=elapsed_ms, stats=stats, reason=reason)

    try:
        parsed = SecurityTestResponse.model_validate_json(raw)
    except Exception as e:
        raise GenerationError(
            f"Response does not match schema: {e}", raw=raw, started_at=started_at,
            ended_at=ended_at, elapsed_ms=elapsed_ms, stats=stats, reason="schema",
        ) from e

    return GenerationResult(parsed, raw, started_at, ended_at, elapsed_ms, stats)
