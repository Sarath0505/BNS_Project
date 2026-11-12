"""Pipeline 2: select relevant BNS evidence, without PyTorch or LangChain.

Compatible with existing main.py and retrieval.py. Uses Qdrant payloads
already indexed. For comparison questions, match each named topic against ALL
section headings, then fetch all chunks of each matching record. Heuristic only:
legal applicability and evidence completeness must still be checked.
"""

import re
from collections import defaultdict
from pathlib import Path

STOP = set("""a an and are as at be between by compare constitutes do does different
    difference explain for from general how in including is it its law laws me of on
    or ordinary basic penalty penalties please punishment punishments section sections
    the their to under versus vs what when where which who with involving""".split())


def words(text):
    """Lightweight normalization (including the common 'grevious' typo)."""
    return re.findall(r"[a-z]+|\d+", str(text).lower().replace("grevious", "grievous"))


def terms(text):
    return set(words(text)) - STOP


def body(chunk):
    return str(chunk.get("chunk_text", "")).split(" | ", 2)[-1]


def section(chunk):
    try:
        return int(chunk.get("section_number", -1))
    except (TypeError, ValueError):
        return -1


def heading(chunks):
    """Heading from the first chunk of an indexed source record."""
    first = min(chunks, key=lambda c: int(c.get("chunk_index", 1)))
    text = body(first)
    match = re.search(rf"(?<!\d){section(first)}\s*[.:]", text)
    return text[:match.start()].strip(" .,;:–—-\n") if match and match.start() <= 220 else ""


def requested_sections(question):
    """Parse 'section 117' or 'sections 117, 120 and 121'."""
    numbers = []
    pattern = (r"\bsections?\s+(\d{1,3}(?:\s*\(\d+\))?"
               r"(?:\s*(?:,|and|&)\s*\d{1,3}(?:\s*\(\d+\))?)*)")
    for match in re.finditer(pattern, question, flags=re.I):
        for number in re.findall(r"\d{1,3}(?=\s*(?:\(|\b))", match.group(1)):
            if int(number) not in numbers:
                numbers.append(int(number))
    return numbers


def question_topics(question):
    """Extract explicit comparison parts; avoids inventing parts for simple Qs."""
    if not re.search(r"\b(compare|contrast|difference|differences|versus|vs)\b", question, re.I):
        return []
    q = question.strip().rstrip("?.!")
    q = re.sub(r"^\s*(?:compare|contrast)\s+(?:the\s+)?"
               r"(?:punishments?\s+|penalties\s+)?(?:for|of|between)?\s*", "", q, flags=re.I)
    q = re.sub(r"^\s*(?:what\s+(?:is|are)\s+)?(?:the\s+)?"
               r"differences?\s+between\s+", "", q, flags=re.I)
    parts = re.split(r"\s*,\s*(?:and\s+)?|\s+(?:and|versus|vs)\s+", q, flags=re.I)
    parts = [re.sub(r"^(?:the\s+)?(?:for|of)\s+", "", p.strip(), flags=re.I)
             for p in parts]
    parts = [p for p in parts if terms(p)]
    return parts if len(parts) >= 2 else []


def intent_for(question):
    if re.search(r"\b(punish\w*|penalt\w*|sentenc\w*|imprison\w*)\b", question, re.I):
        return "punishment"
    if re.search(r"\b(exception\w*|proviso\w*|exempt\w*)\b", question, re.I):
        return "exception"
    if re.search(r"\b(defin\w*|constitut\w*|meaning)\b", question, re.I):
        return "definition"
    return "general"


def evidence_score(chunk, intent):
    """Used only to prioritize chunks inside an already-selected section."""
    text = body(chunk).lower()
    if intent == "punishment":
        return int("shall be punished" in text)
    if intent == "exception":
        return int(bool(re.search(r"\b(exception|provided that|proviso)\b", text)))
    if intent == "definition":
        return int(bool(re.search(r"\b(is said to|means|commits)\b", text)))
    return 0


def match_topic(topic, groups, headings, rank):
    """Find best heading for one comparison topic across ALL stored records."""
    query_terms = terms(topic)
    if not query_terms:
        return None
    choices = []
    for rid, chunks in groups.items():
        title = headings[rid]
        heading_terms = terms(title)
        common = query_terms & heading_terms
        if not common:
            continue
        score = (5 * len(common) / len(query_terms)
                 + 3 * len(common) / len(heading_terms))
        if re.search(r"\b(ordinary|general|basic)\b", topic, re.I):
            if re.search(r"\b(to extort|to deter|by dangerous|on provocation)\b", title, re.I):
                score -= 2
            if "(2)" in str(chunks[0].get("section_in_bns", "")):
                score += 0.2
        if rid in rank:
            score += 0.2 / (1 + rank[rid] / 5)
        choices.append((score, rid))
    if not choices:
        return None
    choices.sort(key=lambda v: (-v[0], rank.get(v[1], 10**6), v[1]))
    return choices[0][1] if choices[0][0] >= 5 else None


def select_context(question, candidates, all_chunks, *, scope="auto",
                   max_records=5, max_chunks=12, max_chars=12000):
    """Select and expand legal records; preserves existing main.py output keys."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty")
    if scope not in {"auto", "focused", "broad"}:
        raise ValueError("scope must be auto, focused or broad")
    if min(max_records, max_chunks, max_chars) < 1:
        raise ValueError("Context limits must be positive")

    groups = defaultdict(list)
    for chunk in all_chunks:
        if chunk.get("record_id") and chunk.get("chunk_text"):
            groups[str(chunk["record_id"])].append(chunk)
    for chunks in groups.values():
        chunks.sort(key=lambda c: int(c.get("chunk_index", 1)))
    headings = {rid: heading(chunks) for rid, chunks in groups.items()}

    ranked = []
    for hit in candidates:
        rid = str(hit.get("record_id", ""))
        if rid in groups and rid not in ranked:
            ranked.append(rid)
    rank = {rid: index for index, rid in enumerate(ranked)}
    explicit = requested_sections(question)
    topics = question_topics(question)
    broad = (len(explicit) > 1 or len(topics) > 1 or bool(re.search(
        r"\b(various|multiple|circumstances|types of|all the punishments)\b",
        question, re.I)))
    scope = ("broad" if broad else "focused") if scope == "auto" else scope

    selected_records, matched_topics, uncovered_topics = [], [], []

    def choose(rid, topic):
        if rid is None or (rid not in selected_records and len(selected_records) >= max_records):
            uncovered_topics.append(topic)
            return
        if rid not in selected_records:
            selected_records.append(rid)
        matched_topics.append({"topic": topic, "section_number": section(groups[rid][0]),
                               "record_id": rid})

    if explicit:
        for num in explicit:
            possible = [rid for rid in groups if section(groups[rid][0]) == num]
            possible.sort(key=lambda rid: (rank.get(rid, 10**6), rid))
            choose(possible[0] if possible else None, f"BNS Section {num}")
    elif topics and scope == "broad":
        for topic in topics:
            choose(match_topic(topic, groups, headings, rank), topic)
    else:
        selected_records = ranked[:1 if scope == "focused" else max_records]

    selected, seen_ids, seen_text = [], set(), set()
    used_chars = 0

    def add(chunk):
        nonlocal used_chars
        cid = str(chunk.get("chunk_id", ""))
        text = str(chunk["chunk_text"])
        key = (section(chunk), " ".join(body(chunk).casefold().split()))
        if cid in seen_ids or key in seen_text:
            return
        if len(selected) >= max_chunks or used_chars + len(text) > max_chars:
            return
        selected.append(chunk)
        seen_ids.add(cid)
        seen_text.add(key)
        used_chars += len(text)

    # Reserve at least one chunk for each required topic before expansion.
    query_intent = intent_for(question)
    for rid in selected_records:
        add(max(groups[rid], key=lambda c: (evidence_score(c, query_intent),
                                           -int(c.get("chunk_index", 1)))))
    for rid in selected_records:
        for chunk in groups[rid]:
            add(chunk)

    warnings = []
    if uncovered_topics:
        warnings.append("Uncovered requested topics: " + "; ".join(uncovered_topics))
    if not selected:
        warnings.append("No context selected. Do not produce a supported legal answer.")
    for rid in selected_records:
        missing = [c for c in groups[rid] if str(c.get("chunk_id", "")) not in seen_ids
                   and (section(c), " ".join(body(c).casefold().split())) not in seen_text]
        if missing:
            warnings.append(f"Incomplete record {rid}: {len(missing)} chunk(s) omitted. "
                            "Increase max_chunks or max_chars.")
            for match in matched_topics:
                if match["record_id"] == rid:
                    uncovered_topics.append(match["topic"] + " (incomplete)")

    # For focused questions, show the directly responsive chunk first.
    # For broad comparisons, keep original chunk order to preserve split clauses.
    position = {rid: idx for idx, rid in enumerate(selected_records)}
    selected.sort(key=lambda c: (position[str(c["record_id"])],
                                 -evidence_score(c, query_intent) if scope == "focused" else 0,
                                 int(c.get("chunk_index", 1))))

    sections = list(dict.fromkeys(str(c.get("section_in_bns", c.get("section_number", "?")))
                                  for c in selected))
    context_parts = []
    for i, c in enumerate(selected, 1):
        context_parts.append(
            f"[Evidence {i} | Section listing: {c.get('section_in_bns', '?')} "
            f"| Annexure {c.get('annexure', '?')} | Pages "
            f"{c.get('source_page_start', '?')}-{c.get('source_page_end', '?')} "
            f"| Chunk {c.get('chunk_id', '?')} "
            f"| Position {c.get('chunk_index', '?')}/{c.get('total_chunks', '?')}]\n"
            f"{c['chunk_text']}")

    return {
        "scope": scope,
        "chunks": selected,
        "context_text": "\n\n".join(context_parts),
        "selected_sections": sections,
        "matched_topics": matched_topics,
        "uncovered_topics": uncovered_topics,
        "warnings": warnings,
        "coverage_verified": False,
    }


def load_qdrant_chunks(qdrant_dir):
    """Read all payloads from the existing local Qdrant collection."""
    from qdrant_client import QdrantClient
    from pipeline1.embedding import COLLECTION_NAME

    qdrant_dir = Path(qdrant_dir)
    if not qdrant_dir.is_dir():
        raise FileNotFoundError(f"Qdrant store not found: {qdrant_dir}")
    client = QdrantClient(path=str(qdrant_dir))
    try:
        if not client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(f"Collection not found: {COLLECTION_NAME}")
        rows, offset = [], None
        while True:
            points, offset = client.scroll(
                collection_name=COLLECTION_NAME, offset=offset, limit=256,
                with_payload=True, with_vectors=False)
            rows.extend(p.payload or {} for p in points)
            if offset is None:
                return rows
    finally:
        client.close()


def split_question_parts(question):
    """Split explicit multi-part questions without splitting legal phrases like
    'hurt and restraint' or comparisons like 'theft, robbery, and dacoity'.

    This is a lightweight heuristic; it is not a legal query planner.
    """
    # Examples: "What is punishment for theft? And is there an exception?"
    pieces = re.split(r"\?\s*(?:and\s+|also\s+)?(?=[A-Za-z])", question.strip(), flags=re.I)
    pieces = [p.strip(" .?!") for p in pieces if p.strip(" .?!")]
    if len(pieces) > 1:
        return pieces

    # Also catch two direct questions separated by "and" but no '?' character.
    pieces = re.split(
        r"\s+and\s+(?=(?:is|are|can|could|does|do|will|would|what|who|how|why|where|when)\b)",
        question.strip(" .?!"), flags=re.I
    )
    pieces = [p.strip(" .?!") for p in pieces if p.strip(" .?!")]
    return pieces or [question.strip()]


def retrieve_context(question, qdrant_dir, cache_dir=None, candidate_k=15,
                     max_records=5, max_chunks=12, max_chars=12000,
                     scope="auto", section_number=None, annexure=None):
    """Retrieve separately for independent question parts, then merge evidence.

    The original single-question behavior is preserved. Multiple parts are
    retrieved independently, preventing a clause about a special case from
    displacing the basic provision requested in another clause.
    """
    from pipeline2.retrieval import search_chunks

    parts = split_question_parts(question)
    all_chunks = load_qdrant_chunks(qdrant_dir)
    if section_number is not None:
        all_chunks = [c for c in all_chunks if section(c) == int(section_number)]
    if annexure is not None:
        all_chunks = [c for c in all_chunks if str(c.get("annexure", "")).upper() == annexure.upper()]

    def fetch(part):
        return search_chunks(
            question=part, qdrant_dir=qdrant_dir, cache_dir=cache_dir,
            top_k=candidate_k, section_number=section_number, annexure=annexure
        )

    if len(parts) == 1:
        return select_context(
            question, fetch(question), all_chunks, scope=scope,
            max_records=max_records, max_chunks=max_chunks, max_chars=max_chars
        )

    selected = []
    seen_chunks, seen_text, seen_records = set(), set(), set()
    warnings, uncovered, matched = [], [], []
    total_chars = 0

    for part in parts:
        # Keep ordinary standalone questions focused. Preserve the existing
        # broad/compare logic for comparison clauses.
        part_scope = "auto" if question_topics(part) or requested_sections(part) else "focused"
        sub = select_context(
            part, fetch(part), all_chunks, scope=part_scope,
            max_records=max_records, max_chunks=max_chunks, max_chars=max_chars
        )
        warnings.extend(sub["warnings"])
        uncovered.extend(sub["uncovered_topics"])
        matched.extend(sub["matched_topics"])
        if not sub["chunks"] and part not in uncovered:
            uncovered.append(part)

        # Reject incomplete source records rather than silently truncating them.
        by_record = defaultdict(list)
        for chunk in sub["chunks"]:
            by_record[str(chunk.get("record_id", ""))].append(chunk)
        for rid, record_chunks in by_record.items():
            additions = []
            for c in record_chunks:
                cid = str(c.get("chunk_id", ""))
                key = (section(c), " ".join(body(c).casefold().split()))
                if cid not in seen_chunks and key not in seen_text:
                    additions.append((c, cid, key))
            if rid not in seen_records and len(seen_records) >= max_records:
                warnings.append(f"Context limit: record {rid} omitted for question part '{part}'.")
                uncovered.append(part)
                continue
            if (len(selected) + len(additions) > max_chunks or
                    total_chars + sum(len(str(c["chunk_text"])) for c, _, _ in additions) > max_chars):
                warnings.append(f"Context limit: record {rid} omitted for question part '{part}'.")
                uncovered.append(part)
                continue
            seen_records.add(rid)
            for c, cid, key in additions:
                seen_chunks.add(cid)
                seen_text.add(key)
                selected.append(c)
                total_chars += len(str(c["chunk_text"]))

    if re.search(r"\b(loopholes?|evad\w*|escape\s+(?:punishment|liability))\b", question, re.I):
        warnings.append(
            "Do not infer that a lawful exemption or defence exists merely because "
            "an excerpt mentions escape. Distinguish an established legal exception "
            "from an aggravated offence, and acknowledge limitations."
        )

    context_parts = []
    for i, c in enumerate(selected, 1):
        context_parts.append(
            f"[Evidence {i} | Section listing: {c.get('section_in_bns', '?')} "
            f"| Annexure {c.get('annexure', '?')} | Pages "
            f"{c.get('source_page_start', '?')}-{c.get('source_page_end', '?')} "
            f"| Chunk {c.get('chunk_id', '?')} "
            f"| Position {c.get('chunk_index', '?')}/{c.get('total_chunks', '?')}]\n"
            f"{c['chunk_text']}"
        )

    return {
        "scope": "broad",  # Independent parts require multiple-topic coverage.
        "question_parts": parts,
        "chunks": selected,
        "context_text": "\n\n".join(context_parts),
        "selected_sections": list(dict.fromkeys(str(c.get("section_in_bns", c.get("section_number", "?"))) for c in selected)),
        "matched_topics": matched,
        "uncovered_topics": list(dict.fromkeys(uncovered)),
        "warnings": warnings,
        "coverage_verified": False,
    }
