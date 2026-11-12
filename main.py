"""BNS RAG: indexing, retrieval, context selection, and Qwen3.5 answer generation.

Commands:
    python main.py index
    python main.py query "What is the punishment for theft?"
    python main.py context "Compare theft and snatching"
    python main.py context "..." --scope broad
    python main.py context-chat            # interactive context selection
    python main.py answer-chat              # interactive Qwen3.5 grounded answers
    python main.py answer "What is the punishment for theft?"
    python main.py                         # interactive Qwen answers (default)
"""

import argparse
from pathlib import Path

from pipeline1.ingestion import parse_pdf
from pipeline1.chunking import chunk_csv
from pipeline1.embedding import embed_chunks
from pipeline2.retrieval import search_chunks
from pipeline2.context_selection import retrieve_context
from pipeline2.generation import generate_answer, DEFAULT_MODEL

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PDF_NAME = "BNS_Handbook_Pages_129-208_RAG_Ingestion.pdf"


def index():
    records = parse_pdf(DATA_DIR / PDF_NAME, DATA_DIR / "bns_parsed.csv")
    chunks = chunk_csv(DATA_DIR / "bns_parsed.csv", DATA_DIR / "bns_chunks.csv", overlap=0)
    stored = embed_chunks(DATA_DIR / "bns_chunks.csv", DATA_DIR / "qdrant_store")
    print(f"Index complete: {len(records)} records, {len(chunks)} chunks, {stored} vectors")


def query(question, top_k=5, section=None, annexure=None):
    hits = search_chunks(
        question, DATA_DIR / "qdrant_store", top_k=top_k,
        section_number=section, annexure=annexure,
    )
    if not hits:
        print("No chunks matched the search filters.")
        return
    for rank, hit in enumerate(hits, 1):
        print(
            f"\nResult {rank} | Section listing {hit.get('section_in_bns', '?')}"
            f" | Annexure {hit.get('annexure', '?')}"
            f" | Pages {hit.get('source_page_start', '?')}-{hit.get('source_page_end', '?')}"
            f" | Cosine {hit.get('score', 0.0):.4f}"
            f" | Hybrid {hit.get('hybrid_score', 0.0):.4f}"
        )
        print(hit.get("chunk_text", ""))


def show_context(question, scope="auto", candidate_k=15,
                 max_records=5, max_chunks=12, section=None, annexure=None):
    selection = retrieve_context(
        question, DATA_DIR / "qdrant_store",
        candidate_k=candidate_k, max_records=max_records,
        max_chunks=max_chunks, scope=scope,
        section_number=section, annexure=annexure,
    )
    print(f"\nDetected scope: {selection['scope']}")
    print(f"Selected sections: {selection['selected_sections']}")
    print(f"Selected chunks: {len(selection['chunks'])}")
    for warning in selection["warnings"]:
        print("WARNING:", warning)
    if not selection["chunks"]:
        print("No relevant context was selected.")
        return
    print("\n--- CONTEXT FOR THE LLM (NOT AN ANSWER) ---")
    print(selection["context_text"])
    print("\nNote: Evidence completeness and legal applicability are not verified.")


def interactive_context(scope="auto", candidate_k=15, max_records=5,
                        max_chunks=12, section=None, annexure=None):
    print("\nBNS Context Selection Test (type 'exit' to quit)")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nProgram closed.")
            break
        if question.lower() in {"quit", "exit"}:
            print("Program closed.")
            break
        if question:
            show_context(
                question, scope=scope, candidate_k=candidate_k,
                max_records=max_records, max_chunks=max_chunks,
                section=section, annexure=annexure,
            )


def interactive_query(top_k=5, section=None, annexure=None):
    print("\nBNS RAG Retrieval System (type 'exit' to quit)")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nProgram closed.")
            break
        if question.lower() in {"quit", "exit"}:
            print("Program closed.")
            break
        if question:
            query(question, top_k=top_k, section=section, annexure=annexure)



def answer_question(question, scope="auto", candidate_k=15, max_records=5,
                    max_chunks=12, max_chars=12000, model=DEFAULT_MODEL,
                    section=None, annexure=None, show_context=False):
    """Retrieve, select, and generate one grounded answer."""
    selection = retrieve_context(
        question, DATA_DIR / "qdrant_store",
        candidate_k=candidate_k, max_records=max_records,
        max_chunks=max_chunks, max_chars=max_chars, scope=scope,
        section_number=section, annexure=annexure,
    )
    print(f"\nScope: {selection['scope']} | Sections: {selection['selected_sections']} "
          f"| Chunks: {len(selection['chunks'])}")
    if show_context:
        print("\n--- SELECTED CONTEXT ---")
        print(selection["context_text"] or "(no context)")

    result = generate_answer(question, selection, model=model)
    print("\n--- ANSWER ---")
    print(result["answer"])
    for warning in result["warnings"]:
        print("WARNING:", warning)
    if result["sources"]:
        print("\nEvidence sources:")
        for src in result["sources"]:
            print(f"  [Source {src['id']}] BNS section {src['section_number']}; "
                  f"Annexure {src['annexure']}; pages {src['page_start']}-{src['page_end']}")
    print("\nNote: Handbook excerpt only; citations and legal applicability require verification.")
    return result


def interactive_answer(scope="auto", candidate_k=15, max_records=5,
                       max_chunks=12, max_chars=12000, model=DEFAULT_MODEL,
                       section=None, annexure=None, show_context=False):
    print(f"\nBNS RAG Answer Generation — {model} (type 'exit' to quit)")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nProgram closed.")
            break
        if question.lower() in {"exit", "quit"}:
            print("Program closed.")
            break
        if not question:
            continue
        try:
            answer_question(
                question, scope=scope, candidate_k=candidate_k,
                max_records=max_records, max_chunks=max_chunks, max_chars=max_chars,
                model=model, section=section, annexure=annexure,
                show_context=show_context,
            )
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            print("ERROR:", exc)


def main():
    parser = argparse.ArgumentParser(description="BNS RAG pipeline")
    parser.add_argument(
        "mode", nargs="?", default="answer-chat",
        choices=("index", "query", "chat", "context", "context-chat", "answer", "answer-chat"),
    )
    parser.add_argument("question", nargs="*")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--section", type=int, default=None)
    parser.add_argument("--annexure", choices=("V", "VI"), default=None)
    parser.add_argument("--scope", choices=("auto", "focused", "broad"), default="auto")
    parser.add_argument("--candidate-k", type=int, default=15)
    parser.add_argument("--max-records", type=int, default=5)
    parser.add_argument("--max-chunks", type=int, default=12)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--show-context", action="store_true")
    args = parser.parse_args()

    if args.mode == "index":
        index()
        return

    question = " ".join(args.question).strip()
    if args.mode == "answer":
        if not question:
            question = input("Question: ").strip()
        if question:
            try:
                answer_question(question, scope=args.scope, candidate_k=args.candidate_k,
                                max_records=args.max_records, max_chunks=args.max_chunks,
                                max_chars=args.max_chars, model=args.model,
                                section=args.section, annexure=args.annexure,
                                show_context=args.show_context)
            except (RuntimeError, FileNotFoundError, ValueError) as exc:
                print("ERROR:", exc)
    elif args.mode == "answer-chat":
        interactive_answer(scope=args.scope, candidate_k=args.candidate_k,
                           max_records=args.max_records, max_chunks=args.max_chunks,
                           max_chars=args.max_chars, model=args.model,
                           section=args.section, annexure=args.annexure,
                           show_context=args.show_context)
    elif args.mode == "context":
        if not question:
            question = input("Question: ").strip()
        if question:
            show_context(
                question, scope=args.scope, candidate_k=args.candidate_k,
                max_records=args.max_records, max_chunks=args.max_chunks,
                section=args.section, annexure=args.annexure,
            )
    elif args.mode == "context-chat":
        interactive_context(
            scope=args.scope, candidate_k=args.candidate_k,
            max_records=args.max_records, max_chunks=args.max_chunks,
            section=args.section, annexure=args.annexure,
        )
    elif args.mode == "query":
        if not question:
            question = input("Question: ").strip()
        if question:
            query(question, args.top_k, args.section, args.annexure)
    else:
        interactive_query(args.top_k, args.section, args.annexure)


if __name__ == "__main__":
    main()
