
"""Pipeline 2: Hybrid retrieval using FastEmbed and Qdrant.

Combines:
1. Dense vector similarity
2. BM25 keyword matching
3. Exact offence-heading preference
4. Reciprocal Rank Fusion
5. Duplicate removal

No PyTorch, LangChain, or additional dependencies.
"""

import math
import re
from collections import Counter
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from pipeline1.embedding import MODEL_NAME, COLLECTION_NAME


# Ranking configuration
RRF_K = 60
DENSE_WEIGHT = 0.5
KEYWORD_WEIGHT = 0.5
HEADING_BONUS = 0.012
PUNISHMENT_BONUS = 0.001


# Words that generally do not identify a specific offence
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "can", "do", "does", "explain", "for", "from",
    "give", "how", "i", "in", "is", "it", "its",
    "me", "of", "on", "or", "please", "tell",
    "the", "their", "there", "to", "under",
    "what", "when", "where", "which", "who",
    "why", "with", "bns", "section", "sections",
    "law", "legal", "offence", "offences",
    "offender", "offenders", "punishment",
    "punishments", "punishable", "penalty",
    "penalties", "sentence", "sentencing",
    "imprisonment", "years", "year", "fine",
    "fines", "maximum", "minimum", "person",
    "persons", "whoever", "commits", "committed",
    "conviction", "convicted", "term",
    "defined", "definition", "define",
    "meaning", "means"
}


def tokenize(text):
    """Convert text into lowercase searchable tokens."""
    return re.findall(r"[a-z]+|\d+", str(text).casefold())


def get_body(payload):
    """Remove the stored section/annexure prefix."""
    text = str(payload.get("chunk_text", ""))
    return text.split(" | ", 2)[-1]


def get_heading(payload):
    """Extract the offence heading from the first chunk."""

    if int(payload.get("chunk_index", 0)) != 1:
        return ""

    body = get_body(payload)
    section = payload.get("section_number")

    if section is None:
        return ""

    # Example:
    # "Theft. 303. (1) Whoever..." becomes "Theft"
    pattern = rf"(?<!\d){re.escape(str(section))}\s*[.:]"
    match = re.search(pattern, body)

    if not match or match.start() > 220:
        return ""

    return body[:match.start()].strip(" \n\t.,:;–—-")


def heading_score(heading, query_terms):
    """Score how closely the offence heading matches the query."""

    heading_terms = [
        token for token in tokenize(heading)
        if token not in STOPWORDS
    ]

    if not heading_terms or not query_terms:
        return 0.0

    query_phrase = " ".join(query_terms)
    heading_phrase = " ".join(heading_terms)

    if heading_phrase == query_phrase:
        return 1.0

    if heading_phrase.startswith(query_phrase + " "):
        return 0.55

    if query_phrase in heading_phrase:
        return 0.35

    shared = len(set(query_terms) & set(heading_terms))

    return 0.15 * shared / len(set(query_terms))


def bm25_scores(documents, query_terms):
    """Calculate BM25 keyword scores without extra libraries."""

    n = len(documents)

    if not n or not query_terms:
        return [0.0] * n

    lengths = [len(tokens) for tokens in documents]
    avgdl = sum(lengths) / n or 1.0

    document_frequency = Counter()

    for tokens in documents:
        document_frequency.update(set(tokens))

    k1 = 1.5
    b = 0.75

    scores = []

    for tokens, length in zip(documents, lengths):

        counts = Counter(tokens)
        score = 0.0

        for term in set(query_terms):

            frequency = counts.get(term, 0)

            if frequency == 0:
                continue

            df = document_frequency[term]

            idf = math.log(
                1.0 + (n - df + 0.5) / (df + 0.5)
            )

            score += (
                idf * frequency * (k1 + 1.0)
            ) / (
                frequency
                + k1 * (1.0 - b + b * length / avgdl)
            )

        scores.append(score)

    return scores


def fetch_all_points(client, query_filter):
    """Read stored Qdrant payloads using pagination."""

    points = []
    offset = None

    while True:

        page, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=query_filter,
            offset=offset,
            limit=256,
            with_payload=True,
            with_vectors=False,
        )

        points.extend(page)

        if offset is None:
            break

    return points


def search_chunks(
    question,
    qdrant_dir,
    cache_dir=None,
    top_k=5,
    section_number=None,
    annexure=None
):
    """Perform hybrid retrieval and return unique BNS chunks."""

    # ---------------------------
    # 1. Validate inputs
    # ---------------------------

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty")

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if annexure is not None and annexure.upper() not in ("V", "VI"):
        raise ValueError("annexure must be V or VI")

    qdrant_dir = Path(qdrant_dir)

    if not qdrant_dir.is_dir():
        raise FileNotFoundError(
            f"Qdrant database not found: {qdrant_dir}"
        )

    cache_dir = (
        Path(cache_dir)
        if cache_dir
        else qdrant_dir.parent / "model_cache"
    )

    # ---------------------------
    # 2. Embed user query
    # ---------------------------

    embedding_model = TextEmbedding(
        model_name=MODEL_NAME,
        cache_dir=str(cache_dir)
    )

    query_vector = next(
        iter(embedding_model.query_embed(question.strip()))
    ).tolist()

    # ---------------------------
    # 3. Optional metadata filters
    # ---------------------------

    must_conditions = []

    if section_number is not None:
        must_conditions.append(
            models.FieldCondition(
                key="section_number",
                match=models.MatchValue(
                    value=int(section_number)
                )
            )
        )

    if annexure is not None:
        must_conditions.append(
            models.FieldCondition(
                key="annexure",
                match=models.MatchValue(
                    value=annexure.upper()
                )
            )
        )

    query_filter = (
        models.Filter(must=must_conditions)
        if must_conditions
        else None
    )

    client = QdrantClient(path=str(qdrant_dir))

    try:

        if not client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' not found"
            )

        # ---------------------------
        # 4. Load stored chunk metadata
        # ---------------------------

        stored_points = fetch_all_points(
            client, query_filter
        )

        if not stored_points:
            return []

        # ---------------------------
        # 5. Dense vector retrieval
        # ---------------------------

        dense_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=len(stored_points),
            with_payload=True,
        )

        dense_rank = {
            str(point.id): rank
            for rank, point in enumerate(
                dense_result.points, 1
            )
        }

        similarities = {
            str(point.id): float(point.score)
            for point in dense_result.points
        }

        # ---------------------------
        # 6. Extract offence headings
        # ---------------------------

        headings = {}

        for point in stored_points:
            payload = point.payload or {}

            title = get_heading(payload)

            if title:
                headings[payload.get("record_id")] = title

        # ---------------------------
        # 7. Prepare keyword query
        # ---------------------------

        query_terms = [
            token for token in tokenize(question)
            if token not in STOPWORDS
        ]

        if not query_terms:
            query_terms = tokenize(question)

        # ---------------------------
        # 8. BM25 keyword search
        # ---------------------------

        documents = []

        for point in stored_points:

            payload = point.payload or {}

            title = headings.get(
                payload.get("record_id"), ""
            )

            # Repeat heading tokens to emphasize offence names
            text = (title + " ") * 3 + get_body(payload)

            documents.append(tokenize(text))

        keyword_scores = bm25_scores(
            documents, query_terms
        )

        lexical_order = sorted(
            (
                i for i, score in enumerate(keyword_scores)
                if score > 0
            ),
            key=lambda i: (
                -keyword_scores[i],
                str(stored_points[i].id)
            )
        )

        lexical_rank = {
            str(stored_points[i].id): rank
            for rank, i in enumerate(lexical_order, 1)
        }

        # ---------------------------
        # 9. Detect question requirements
        # ---------------------------

        explicit_section = re.search(
            r"\b(?:section|sec)\.?\s*(\d{1,3})\b",
            question,
            re.IGNORECASE
        )

        requested_number = (
            int(explicit_section.group(1))
            if explicit_section
            else None
        )

        wants_punishment = bool(
            re.search(
                r"\b(punish\w*|penalt\w*|sentenc\w*)\b",
                question,
                re.IGNORECASE
            )
        )

        # ---------------------------
        # 10. Hybrid ranking using RRF
        # ---------------------------

        ranked = []

        for point in stored_points:

            payload = point.payload or {}
            point_id = str(point.id)

            dr = dense_rank.get(point_id)
            lr = lexical_rank.get(point_id)

            if dr is None and lr is None:
                continue

            hybrid = (
                DENSE_WEIGHT / (RRF_K + dr)
                if dr else 0.0
            )

            hybrid += (
                KEYWORD_WEIGHT / (RRF_K + lr)
                if lr else 0.0
            )

            title = headings.get(
                payload.get("record_id"), ""
            )

            hybrid += (
                HEADING_BONUS
                * heading_score(title, query_terms)
            )

            if (
                wants_punishment
                and "shall be punished"
                in get_body(payload).casefold()
            ):
                hybrid += PUNISHMENT_BONUS

            if (
                requested_number is not None
                and payload.get("section_number")
                == requested_number
            ):
                hybrid += HEADING_BONUS

            ranked.append((
                hybrid,
                point_id,
                {
                    **payload,
                    "score": similarities.get(point_id, 0.0),
                    "hybrid_score": hybrid
                }
            ))

        ranked.sort(
            key=lambda item: (-item[0], item[1])
        )

        # ---------------------------
        # 11. Remove duplicate chunks
        # ---------------------------

        unique_results = []
        seen = set()

        for _, _, hit in ranked:

            normalized_body = " ".join(
                get_body(hit).casefold().split()
            )

            key = (
                hit.get("section_number"),
                normalized_body
            )

            if key in seen:
                continue

            seen.add(key)
            unique_results.append(hit)

            if len(unique_results) == top_k:
                break

        return unique_results

    finally:
        client.close()
