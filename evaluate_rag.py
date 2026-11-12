"""Offline evaluation of the existing BNS RAG app.

No LangChain, PyTorch, Ragas, or DeepEval dependency. Reuses your saved
Qdrant index and existing pipeline2 modules; judgments use local Ollama.

Examples, from RAG_project/:
  python evaluate_rag.py collect --limit 3
  python evaluate_rag.py score
  python evaluate_rag.py all

WARNING: Scores from an LLM judge are estimates, not legal verification.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ROOT / "eval_cases.json"
DEFAULT_RUNS = ROOT / "eval_results" / "runs.jsonl"
DEFAULT_DETAILS = ROOT / "eval_results" / "scores.jsonl"
DEFAULT_SUMMARY = ROOT / "eval_results" / "summary.json"
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "chunk_relevant": {"type": "array", "items": {"type": "boolean"}},
        "answer_relevance": {"type": "integer", "minimum": 0, "maximum": 4},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "supported": {"type": "boolean"}
                },
                "required": ["text", "supported"]
            }
        },
        "facts_covered": {"type": "array", "items": {"type": "boolean"}},
        "abstained": {"type": "boolean"},
        "reason": {"type": "string"}
    },
    "required": ["chunk_relevant", "answer_relevance", "claims",
                 "facts_covered", "abstained", "reason"]
}


def load_cases(path: Path, limit: int = 0) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("eval_cases.json must be a JSON array")
    ids = set()
    for case in cases:
        if not case.get("id") or not case.get("question"):
            raise ValueError("Every case needs id and question")
        if case["id"] in ids:
            raise ValueError(f"Duplicate test id: {case['id']}")
        ids.add(case["id"])
        if not isinstance(case.get("expected_sections"), list):
            raise ValueError(f"expected_sections must be a list: {case['id']}")
        if not isinstance(case.get("required_facts"), list):
            raise ValueError(f"required_facts must be a list: {case['id']}")
        if not isinstance(case.get("answerable"), bool):
            raise ValueError(f"answerable must be boolean: {case['id']}")
    return cases[:limit] if limit else cases


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def collect(cases: list[dict], args: argparse.Namespace) -> list[dict]:
    """Call the *existing* retrieval and generation functions. No re-indexing."""
    from pipeline2.context_selection import retrieve_context
    from pipeline2.generation import generate_answer

    store = ROOT / "data" / "qdrant_store"
    if not store.exists():
        raise FileNotFoundError(f"Qdrant index not found at {store}. Do not re-index unless needed.")
    runs = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] Collecting {case['id']}: {case['question']}", flush=True)
        record = {"id": case["id"], "question": case["question"],
                  "expected_sections": case["expected_sections"],
                  "required_facts": case["required_facts"],
                  "answerable": case["answerable"]}
        try:
            start = time.perf_counter()
            selection = retrieve_context(
                case["question"], store, candidate_k=args.candidate_k,
                max_records=args.max_records, max_chunks=args.max_chunks,
                max_chars=args.max_chars)
            record["retrieval_seconds"] = round(time.perf_counter() - start, 3)
            record["selected_sections"] = selection.get("selected_sections", [])
            record["uncovered_topics"] = selection.get("uncovered_topics", [])
            record["warnings"] = selection.get("warnings", [])
            record["chunks"] = [
                {"chunk_id": ch.get("chunk_id"), "record_id": ch.get("record_id"),
                 "section_number": ch.get("section_number"),
                 "section_listing": ch.get("section_in_bns"),
                 "text": ch.get("chunk_text", "")}
                for ch in selection.get("chunks", [])
            ]
            start = time.perf_counter()
            result = generate_answer(case["question"], selection, model=args.model)
            record["generation_seconds"] = round(time.perf_counter() - start, 3)
            record["answer"] = result.get("answer", "")
            record["generated"] = bool(result.get("generated"))
            record["warnings"].extend(result.get("warnings", []))
        except Exception as exc:
            # Never turn a failing test into an invented numerical score.
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  ERROR: {record['error']}")
        runs.append(record)
        write_jsonl(args.runs, runs)  # checkpoint after each test
    return runs


def ollama_judge(question: str, chunks: list[dict], answer: str,
                 required_facts: list[str], args: argparse.Namespace) -> dict:
    """One local model call per test, returning structured judge notes."""
    evidence = [
        {"chunk_id": str(ch.get("chunk_id", "")),
         "section": ch.get("section_number"),
         "text": str(ch.get("text", ""))[:1600]}
        for ch in chunks
    ]
    instructions = (
        "You are a strict evaluator of a legal-document RAG prototype. "
        "Do not use external legal knowledge: ONLY the supplied retrieved chunks. "
        "Answer with the requested JSON fields. "
        "chunk_relevant: one boolean per retrieved chunk IN THE SAME ORDER, "
        "true only when the chunk contributes evidence to answer some part of the question "
        "(including a necessary definition or qualification); do not count superficial "
        "keyword similarity. For split clauses, consider neighboring chunks. "
        "answer_relevance: integer 0-4 (0 unrelated, 1 mostly unrelated, 2 partly on-topic, "
        "3 mostly responsive, 4 directly responsive). This is NOT correctness. "
        "claims: decompose the answer into its material, atomic factual/legal claims; "
        "supported=true ONLY when the retrieved chunks support the exact claim, "
        "including numbers and legal conditions. Mark unsupported, contradicted, or "
        "unverifiable claims false. Do not count citations or disclaimers as claims. "
        "facts_covered: one boolean PER required fact IN THE SAME ORDER; true only if the "
        "answer accurately and explicitly includes that fact (not merely a similar idea). "
        "abstained: true if the answer withholds a substantive answer because evidence is "
        "missing; false if it supplies a substantive answer. "
        "Never claim you verified the actual law. Never infer a legal provision outside evidence."
    )
    payload = {
        "question": question,
        "retrieved_chunks": evidence,
        "answer": answer,
        "required_facts": required_facts,
        "required_output_lengths": {"chunk_relevant": len(evidence),
                                    "facts_covered": len(required_facts)},
    }
    data = {"model": args.judge_model, "stream": False, "think": False,
            "format": JUDGE_SCHEMA, "keep_alive": "10m",
            "options": {"temperature": 0, "num_ctx": args.judge_ctx,
                        "num_predict": args.judge_tokens},
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
            ]}
    req = Request(args.ollama_url.rstrip("/") + "/api/chat",
                  data=json.dumps(data).encode("utf-8"),
                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=args.timeout) as response:
            result = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Ollama judge request failed: {exc}") from exc
    message = result.get("message", {}).get("content", "")
    try:
        evaluation = json.loads(message)
    except (ValueError, TypeError) as exc:
        raise ValueError("Judge did not return valid JSON") from exc
    if (len(evaluation.get("chunk_relevant", [])) != len(chunks) or
            len(evaluation.get("facts_covered", [])) != len(required_facts)):
        raise ValueError("Judge output list lengths are incorrect; no metrics calculated")
    if not all(isinstance(x, bool) for x in evaluation["chunk_relevant"] + evaluation["facts_covered"]):
        raise ValueError("Judge relevance/coverage labels must be boolean")
    if not isinstance(evaluation.get("abstained"), bool):
        raise ValueError("Judge abstained must be boolean")
    if not isinstance(evaluation.get("answer_relevance"), int) or not 0 <= evaluation["answer_relevance"] <= 4:
        raise ValueError("Judge answer_relevance must be an integer from 0 to 4")
    claims = evaluation.get("claims")
    if not isinstance(claims, list) or not all(
        isinstance(c, dict) and isinstance(c.get("supported"), bool) and isinstance(c.get("text"), str)
        for c in claims
    ):
        raise ValueError("Judge claims format is invalid")
    return evaluation


def average_precision(labels: list[bool]) -> float | None:
    """Selected-context average precision, not recall over the entire database."""
    if not labels:
        return None
    total_relevant = sum(labels)
    if total_relevant == 0:
        return 0.0
    hits, total = 0, 0.0
    for rank, relevant in enumerate(labels, 1):
        if relevant:
            hits += 1
            total += hits / rank
    return total / total_relevant


def section_recall(expected_sections: list[int], chunks: list[dict]) -> float | None:
    if not expected_sections:
        return None
    seen = {int(ch["section_number"]) for ch in chunks
            if ch.get("section_number") not in (None, "")}
    expected = {int(s) for s in expected_sections}
    return len(expected & seen) / len(expected)


def score_run(run: dict, judge: dict) -> dict:
    claims = judge["claims"]
    required = run["required_facts"]
    result = {
        **{k: run.get(k) for k in ("id", "question", "answerable", "selected_sections",
                                      "answer", "generated", "retrieval_seconds", "generation_seconds")},
        "retrieved_chunks": len(run.get("chunks", [])),
        "groundedness": (sum(c["supported"] for c in claims) / len(claims)) if claims else None,
        "answer_relevance": judge["answer_relevance"] / 4,
        "context_precision": average_precision(judge["chunk_relevant"]),
        "completeness": (sum(judge["facts_covered"]) / len(required)) if required else None,
        "section_recall": section_recall(run["expected_sections"], run.get("chunks", [])),
        "abstention_correct": judge["abstained"] == (not run["answerable"]),
        "judge": judge,
        "warnings": run.get("warnings", []),
    }
    return result


def summarize(results: list[dict], args: argparse.Namespace) -> dict:
    metrics = ["groundedness", "answer_relevance", "context_precision", "completeness",
               "section_recall"]
    summary = {"judge_model": args.judge_model, "generator_model": args.model,
               "evaluated": sum("judge" in row for row in results),
               "failed": sum("error" in row for row in results), "metrics": {}}
    for metric in metrics:
        values = [r[metric] for r in results if r.get(metric) is not None]
        summary["metrics"][metric] = {"mean": round(statistics.mean(values), 3) if values else None,
                                      "count": len(values)}
    abstentions = [r["abstention_correct"] for r in results if "abstention_correct" in r]
    summary["metrics"]["abstention_accuracy"] = {
        "mean": round(sum(abstentions) / len(abstentions), 3) if abstentions else None,
        "count": len(abstentions)}
    return summary


def score(args: argparse.Namespace) -> list[dict]:
    if not args.runs.is_file():
        raise FileNotFoundError(f"No runs at {args.runs}. Run 'collect' first.")
    runs = read_jsonl(args.runs)
    results = []
    for index, run in enumerate(runs, 1):
        print(f"[{index}/{len(runs)}] Judging {run['id']}", flush=True)
        if run.get("error"):
            results.append({"id": run["id"], "error": run["error"]})
        else:
            try:
                judgement = ollama_judge(run["question"], run.get("chunks", []),
                                         run.get("answer", ""), run["required_facts"], args)
                results.append(score_run(run, judgement))
            except Exception as exc:
                results.append({"id": run["id"], "error": f"{type(exc).__name__}: {exc}"})
                print(f"  ERROR: {results[-1]['error']}")
        write_jsonl(args.details, results)
    summary = summarize(results, args)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nRESULTS (0-1; N/A means not measurable)")
    print(f"{'Test':18} {'Ground':>7} {'Relevant':>9} {'CtxPrec':>8} {'Complete':>9} {'SecRecall':>9} {'Abstain':>8}")
    for r in results:
        if "error" in r:
            print(f"{r['id']:18} ERROR: {r['error']}")
            continue
        def fmt(key):
            v = r.get(key)
            return f"{v:.2f}" if v is not None else "N/A"
        print(f"{r['id']:18} {fmt('groundedness'):>7} {fmt('answer_relevance'):>9} "
              f"{fmt('context_precision'):>8} {fmt('completeness'):>9} "
              f"{fmt('section_recall'):>9} {str(r['abstention_correct']):>8}")
    print("\nAGGREGATE:", json.dumps(summary["metrics"], indent=2))
    print(f"\nSaved detailed results: {args.details}\nSaved summary: {args.summary}")
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Offline BNS RAG metrics (local Ollama)")
    ap.add_argument("mode", choices=["collect", "score", "all"])
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    ap.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--limit", type=int, default=0, help="Only first N questions during collect")
    ap.add_argument("--model", default="qwen3.5:4b", help="Existing generator")
    ap.add_argument("--judge-model", default="qwen3.5:4b", help="Prefer a separate, stronger judge if available")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--candidate-k", type=int, default=15)
    ap.add_argument("--max-records", type=int, default=5)
    ap.add_argument("--max-chunks", type=int, default=12)
    ap.add_argument("--max-chars", type=int, default=12000)
    ap.add_argument("--judge-ctx", type=int, default=8192)
    ap.add_argument("--judge-tokens", type=int, default=1300)
    ap.add_argument("--timeout", type=int, default=360)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    if args.mode in ("collect", "all"):
        cases = load_cases(args.cases, args.limit)
        collect(cases, args)
        print(f"\nCollected {len(cases)} runs to {args.runs}")
    if args.mode in ("score", "all"):
        score(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print("ERROR:", error, file=sys.stderr)
        sys.exit(1)
