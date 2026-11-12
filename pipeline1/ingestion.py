"""Parse the BNS Handbook (printed pages 129-208) into RAG-ready CSV records.

Requirements: pip install pdfplumber
No PyTorch, LangChain, OCR, or pandas required.
"""

import csv
import re
from pathlib import Path

import pdfplumber


FIRST_PRINTED_PAGE = 129
FIELDS = [
    "record_id", "annexure", "section_in_bns", "section_number",
    "source_page_start", "source_page_end", "pdf_page_start", "pdf_page_end",
    "heading_and_content", "rag_text",
]


def clean_text(value):
    """Remove layout whitespace without changing the legal wording."""
    text = (value or "").replace("\u00ad", "").replace("\u00a0", " ")
    text = re.sub(r"-\s*\n\s*(?=\w)", "-", text)  # inter-\ngovernmental
    return re.sub(r"\s+", " ", text).strip()


def parse_pdf(pdf_path, csv_path):
    """Extract one record per listed section/entry, joining its continuations."""
    pdf_path = Path(pdf_path)
    csv_path = Path(csv_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    records = []
    annexure = None
    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            marker = re.search(r"Annexure\s*[-–]\s*(VI|V)\b", page_text)
            if marker:
                annexure = marker.group(1)
            if annexure is None:
                raise ValueError(f"Annexure missing before PDF page {page_index + 1}")

            tables = page.extract_tables()
            if not tables:
                raise ValueError(f"No table found on PDF page {page_index + 1}")

            for table in tables:
                for row in table:
                    if len(row) != 2:
                        raise ValueError(f"Unexpected table layout on PDF page {page_index + 1}")

                    section = clean_text(row[0])
                    content = clean_text(row[1])
                    if section.lower() == "section in bns":  # repeated table header
                        continue
                    if not content:
                        continue

                    printed_page = FIRST_PRINTED_PAGE + page_index
                    if section:
                        match = re.match(r"^(\d{1,3})\b", section)
                        if not match:
                            raise ValueError(f"Unexpected section label on PDF page {page_index + 1}: {section}")
                        records.append({
                            "record_id": f"BNS-{len(records) + 1:03d}",
                            "annexure": annexure,
                            "section_in_bns": section,
                            "section_number": match.group(1),
                            "source_page_start": printed_page,
                            "source_page_end": printed_page,
                            "pdf_page_start": page_index + 1,
                            "pdf_page_end": page_index + 1,
                            "heading_and_content": content,
                        })
                    else:
                        # A blank section cell continues the previous section,
                        # sometimes on the next page.
                        if not records or records[-1]["annexure"] != annexure:
                            raise ValueError(
                                f"Continuation has no matching section on PDF page {page_index + 1}"
                            )
                        records[-1]["heading_and_content"] += " " + content
                        records[-1]["source_page_end"] = printed_page
                        records[-1]["pdf_page_end"] = page_index + 1

    for record in records:
        record["rag_text"] = (
            f"BNS Section: {record['section_in_bns']} | "
            f"Annexure: {record['annexure']} | "
            f"{record['heading_and_content']}"
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)

    print(f"Parsed {len(records)} entries from {pdf_path.name}")
    print(f"Saved to {csv_path}")
    return records


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    ROOT = folder.parent if folder.name == "pipeline1" else folder
    parse_pdf(
        ROOT / "data" / "BNS_Handbook_Pages_129-208_RAG_Ingestion.pdf",
        ROOT / "data" / "bns_parsed.csv",
    )
