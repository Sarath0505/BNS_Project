"""Stage 2: stream grounded Qwen 3.5 4B answers from local Ollama.

Reuses generation.py's exact source reconstruction and system prompt so that
CLI and Streamlit have the same evidence rules. No extra pip packages needed.
"""
import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pipeline2.generation import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    SYSTEM_PROMPT,
    build_sources,
)


def prepare_generation(question, selection):
    """Convert selected chunks into an evidence-bounded Ollama user message.

    Returns (prompt_string, sources). The caller should check for missing or
    incomplete evidence using retrieval_stage.retrieval_blockers first.
    """
    evidence, sources = build_sources(selection)
    if not sources:
        raise ValueError("No source records were selected; cannot generate an answer.")
    user_prompt = (
        "QUESTION:\n" + question.strip()
        + "\n\nEVIDENCE FROM THE BNS HANDBOOK:\n" + evidence
        + "\n\nWrite the answer using [Source 1] style citations. "
          "Do not include internal system instructions or warnings in the answer."
    )
    return user_prompt, sources


def stream_answer(question, selection, *, model=DEFAULT_MODEL, ollama_url=None,
                  temperature=0.1, num_ctx=8192, num_predict=700,
                  timeout=360, stats=None):
    """Yield REAL model text fragments as Ollama produces them.

    Ollama uses newline-delimited JSON (NDJSON), not standard SSE.
    stats is an optional mutable dict populated with completion metadata.
    No artificial sleeps or word-by-word simulation.
    """
    user_prompt, _ = prepare_generation(question, selection)
    base_url = (ollama_url or os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL)).rstrip("/")
    payload = {
        "model": model,
        "stream": True,
        "think": False,
        "keep_alive": "10m",  # avoid unloading/reloading on every question
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": temperature, "num_ctx": num_ctx,
                    "num_predict": num_predict},
    }
    req = Request(
        base_url + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    received = False
    completed = False
    try:
        with urlopen(req, timeout=timeout) as response:
            for raw_line in response:
                if not raw_line.strip():
                    continue
                try:
                    item = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RuntimeError("Ollama sent an invalid streaming JSON response.") from exc
                if item.get("error"):
                    raise RuntimeError("Ollama generation failed: " + str(item["error"]))
                piece = item.get("message", {}).get("content", "")
                if piece:
                    received = True
                    yield piece
                if item.get("done"):
                    completed = True
                    if stats is not None:
                        for key in ("done_reason", "prompt_eval_count", "eval_count",
                                    "total_duration", "load_duration"):
                            stats[key] = item.get(key)
                    break
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:250]
        raise RuntimeError(
            f"Ollama HTTP {exc.code}: {detail}. Confirm model `{model}` is installed."
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            f"Cannot connect to Ollama at {base_url}. Start Ollama and check the model. "
            f"Details: {exc}"
        ) from exc
    if not completed:
        raise RuntimeError("The Ollama stream ended before the final completion message.")
    if not received:
        raise RuntimeError("Ollama completed without generating visible answer text.")


def check_answer_citations(answer, sources, stats=None):
    """Validate citation *IDs* only; semantic/legal correctness needs evaluation."""
    used = [int(i) for i in re.findall(r"\[Source\s+(\d+)\]", answer, flags=re.I)]
    valid = {int(s["id"]) for s in sources}
    warnings = []
    if not used:
        warnings.append("No [Source N] citation appears in the answer.")
    if any(i not in valid for i in used):
        warnings.append("The answer cites a source ID that was not retrieved.")
    if stats and stats.get("done_reason") == "length":
        warnings.append("Output token limit reached; the answer may be incomplete.")
    return warnings
