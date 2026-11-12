"""Stage 1: retrieve and select evidence from the existing BNS Qdrant index.

Nothing here embeds document chunks, deletes Qdrant collections, or calls an LLM.
Kept separate so retrieval performance can be measured independently.
"""
from pathlib import Path
from time import perf_counter


def run_retrieval(question, qdrant_dir, *, scope="auto", candidate_k=15,
                  max_records=5, max_chunks=12, max_chars=12000):
    """Return (selection, elapsed_seconds) using the project's existing selector."""
    if not str(question).strip():
        raise ValueError("Question is empty")
    from pipeline2.context_selection import retrieve_context

    t0 = perf_counter()
    selection = retrieve_context(
        question=question,
        qdrant_dir=Path(qdrant_dir),
        candidate_k=candidate_k,
        max_records=max_records,
        max_chunks=max_chunks,
        max_chars=max_chars,
        scope=scope,
    )
    elapsed = perf_counter() - t0
    return selection, elapsed


def retrieval_blockers(selection):
    """Apply the same evidence-completeness gate used by generation.py."""
    missing = list(selection.get("uncovered_topics") or [])
    warnings = list(selection.get("warnings") or [])
    incomplete = [str(w) for w in warnings if "Incomplete record" in str(w)]
    blockers = []
    if not selection.get("chunks"):
        blockers.append("No evidence was selected from the indexed handbook excerpt.")
    if missing:
        blockers.append("Uncovered requested topics: " + "; ".join(map(str, missing)))
    if incomplete:
        blockers.append("Some selected records were truncated by the context limits.")
    return blockers
