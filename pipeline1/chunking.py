"""Split the parsed BNS CSV into section-safe RAG chunks.

Standard library only (no PyTorch, LangChain, or pandas).
Input:  data/bns_parsed.csv, created by ingestion.py
Output: data/bns_chunks.csv

The parser has already merged entries that continue onto another PDF page.
This chunker never mixes separate source records or annexures.
"""

import csv
from pathlib import Path


CHUNK_SIZE = 1000  # Maximum characters per chunk, including its section header.
CHUNK_OVERLAP = 0  # No repeated source-text spans between adjacent chunks.

FIELDS = [
    "chunk_id", "record_id", "chunk_index", "total_chunks",
    "annexure", "section_in_bns", "section_number",
    "source_page_start", "source_page_end", "pdf_page_start", "pdf_page_end",
    "char_start", "char_end", "chunk_text",
]


def choose_split(text, start, limit):
    """Prefer punctuation boundaries; fall back to whitespace, then hard limit."""
    if limit == len(text):
        return limit

    # Search only the last 35% of this window to avoid tiny chunks.
    lower_bound = start + int((limit - start) * 0.65)
    for separator in (". ", "; ", ": ", ", ", " "):
        index = text.rfind(separator, lower_bound, limit)
        if index != -1:
            return index + len(separator)
    return limit


def split_text(text, max_chars, overlap):
    """Yield text spans. Zero overlap produces consecutive, non-overlapping spans."""
    if max_chars <= 0 or not 0 <= overlap < max_chars:
        raise ValueError("Require max_chars > 0 and 0 <= overlap < max_chars")
    if not text.strip():
        return

    start = 0
    while start < len(text):
        limit = min(start + max_chars, len(text))
        end = choose_split(text, start, limit)
        content = text[start:end].strip()
        if content:
            yield start, end, content
        if end == len(text):
            break
        if overlap == 0:
            start = end  # Essential: avoid repeating text via word-boundary fallback.
            continue

        # Start the next chunk near 'overlap' characters before the cut.
        # Choose a word boundary rather than cutting a word in two.
        desired_start = max(start + 1, end - overlap)
        next_space = text.find(" ", desired_start, end)
        if next_space != -1:
            start = next_space + 1
        else:
            previous_space = text.rfind(" ", start + 1, desired_start)
            start = previous_space + 1 if previous_space != -1 else desired_start


def chunk_csv(input_csv, output_csv, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Read parsed CSV records and write independent, traceable chunks."""
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    if not input_csv.is_file():
        raise FileNotFoundError(f"Parsed CSV not found: {input_csv}")
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")

    chunks = []
    with input_csv.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"record_id", "annexure", "section_in_bns", "section_number", "rag_text",
                    "source_page_start", "source_page_end", "pdf_page_start", "pdf_page_end"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")

        for record in reader:
            # The input rag_text already contains the section header once.
            # Separate it so that *every* output chunk gets this header.
            header = (
                f"BNS Section: {record['section_in_bns']} | "
                f"Annexure: {record['annexure']} | "
            )
            rag_text = record["rag_text"]
            if not rag_text.startswith(header):
                raise ValueError(f"Unexpected rag_text header for {record['record_id']}")
            body = rag_text[len(header):]
            max_body_chars = chunk_size - len(header)
            if max_body_chars <= overlap:
                raise ValueError(
                    f"Chunk size too small for header/overlap: {record['record_id']}"
                )
            if not body.strip():
                raise ValueError(f"Empty content in {record['record_id']}")

            pieces = list(split_text(body, max_body_chars, overlap))
            for index, (start, end, content) in enumerate(pieces, start=1):
                chunk_text = header + content
                if len(chunk_text) > chunk_size:
                    raise AssertionError("Chunk exceeded chunk_size")
                chunks.append({
                    "chunk_id": f"{record['record_id']}-C{index:03d}",
                    "record_id": record["record_id"],
                    "chunk_index": index,
                    "total_chunks": len(pieces),
                    "annexure": record["annexure"],
                    "section_in_bns": record["section_in_bns"],
                    "section_number": record["section_number"],
                    "source_page_start": record["source_page_start"],
                    "source_page_end": record["source_page_end"],
                    "pdf_page_start": record["pdf_page_start"],
                    "pdf_page_end": record["pdf_page_end"],
                    "char_start": start,
                    "char_end": end,
                    "chunk_text": chunk_text,
                })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(chunks)

    print(f"Created {len(chunks)} chunks from {len(set(r['record_id'] for r in chunks))} records")
    print(f"Saved to {output_csv}")
    return chunks


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent  # file lives in Pipeline1/
    chunk_csv(root / "data" / "bns_parsed.csv", root / "data" / "bns_chunks.csv")
