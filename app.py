"""Local Streamlit chat UI for the existing BNS RAG project.

Run from the RAG_project folder:
    python -m streamlit run app.py

Two explicit stages:
    retrieval_stage.run_retrieval()    -> evidence and section metadata
    stream_generation.stream_answer() -> Ollama streaming tokens

No document re-embedding or Qdrant indexing is performed by this file.
"""
from pathlib import Path
from time import perf_counter

import streamlit as st

from pipeline2.retrieval_stage import retrieval_blockers, run_retrieval
from pipeline2.stream_generation import (
    check_answer_citations,
    prepare_generation,
    stream_answer,
)

ROOT = Path(__file__).resolve().parent
QDRANT_DIR = ROOT / "data" / "qdrant_store"
MODEL = "qwen3.5:4b"

st.set_page_config(page_title="BNS Legal RAG | Local Chat", page_icon="⚖️", layout="centered")
st.markdown("""
<style>
.block-container {max-width: 930px; padding-top: 1.8rem;}
.hero-eyebrow {font-size: 0.78rem; letter-spacing: .14em; font-weight: 700;
  color: #16b8a6; text-transform: uppercase; margin-bottom: .35rem;}
.hero-title {font-size: 2rem; font-weight: 750; line-height: 1.16; margin-bottom: .5rem;}
.hero-sub {color: #8f9aa6; margin-bottom: 1.4rem;}
section[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.12);}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚖️ BNS Legal RAG")
    st.caption("Runs locally • Qwen 3.5 4B • Qdrant")
    st.markdown("**System**")
    if QDRANT_DIR.is_dir():
        st.success("Qdrant directory found", icon="✅")
    else:
        st.error("Qdrant index not found", icon="⛔")
        st.caption("Run the existing `python main.py index` once if needed.")
    st.markdown("**Retrieval settings**")
    scope = st.selectbox("Question scope", ["auto", "focused", "broad"], index=0,
                         help="Auto detects simple vs comparison questions.")
    candidate_k = st.slider("Initial retrieval candidates", 5, 30, 15, 1)
    with st.expander("Advanced evidence limits"):
        max_records = st.slider("Maximum source records", 1, 8, 5)
        max_chunks = st.slider("Maximum chunks", 2, 24, 12)
        max_chars = st.slider("Character budget", 4000, 24000, 12000, 1000)
    show_evidence = st.toggle("Display retrieved evidence", value=True)
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("Chat history is displayed locally for this session; follow-up questions should restate the relevant offence or section. The retriever does not resolve references to earlier turns.")

st.markdown('<div class="hero-eyebrow">Bharatiya Nyaya Sanhita</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Ask the BNS handbook</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Search the indexed legal excerpt and watch Qwen answer in real time, with traceable sources.</div>', unsafe_allow_html=True)
st.info("This assistant covers only the indexed handbook excerpt (pages 129–208). It does not provide verified legal advice or coverage of the complete BNS.", icon="ℹ️")

if "messages" not in st.session_state:
    st.session_state.messages = []


def display_sources(sources, show_full=False):
    """Render the evidence used by the model under the answer."""
    if not sources:
        return
    with st.expander(f"Evidence sources ({len(sources)})", expanded=False):
        for source in sources:
            st.markdown(
                f"**[Source {source['id']}] — BNS Section {source['section_number']}** "
                f"· Annexure {source['annexure']} "
                f"· Handbook pages {source['page_start']}–{source['page_end']}"
            )
            st.caption("Indexed listing: " + str(source["section_listing"])
                       + " · Chunks: " + ", ".join(source["chunk_ids"]))
            if show_full:
                st.text(source["text"])
            st.divider()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            st.caption(message.get("timing", ""))
            if message.get("warnings"):
                for warn in message["warnings"]:
                    st.warning(warn, icon="⚠️")
            display_sources(message.get("sources", []), show_evidence)

if not st.session_state.messages:
    st.markdown("**Try asking**")
    suggestions = [
        "What is the punishment for theft?",
        "Compare ordinary grievous hurt and grievous hurt to extort a confession.",
        "What exceptions apply to stalking under Section 78?",
    ]
    cols = st.columns(len(suggestions))
    chosen = None
    for col, suggestion in zip(cols, suggestions):
        with col:
            if st.button(suggestion, use_container_width=True):
                chosen = suggestion
else:
    chosen = None

question = st.chat_input("Ask about an offence, punishment, or BNS section…") or chosen

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not QDRANT_DIR.is_dir():
            answer = "Qdrant index directory is missing. Index the source documents before opening the chatbot."
            st.error(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.stop()

        # STAGE 1: retrieve and select evidence. No LLM is invoked here.
        try:
            with st.status("Stage 1 of 2 · Retrieving legal evidence…", expanded=True) as stage:
                selection, retrieval_seconds = run_retrieval(
                    question, QDRANT_DIR, scope=scope, candidate_k=candidate_k,
                    max_records=max_records, max_chunks=max_chunks,
                    max_chars=max_chars,
                )
                st.write(f"Sections: {', '.join(selection.get('selected_sections', [])) or 'None'}")
                st.write(f"Chunks selected: {len(selection.get('chunks', []))}")
                st.caption(f"Retrieval completed in {retrieval_seconds:.2f} seconds")
                stage.update(label="Stage 1 complete · Evidence retrieved", state="complete",
                             expanded=False)
        except Exception as exc:
            answer = "Retrieval failed. Verify that the Qdrant index is available and not locked by another process."
            st.error(answer)
            st.exception(exc)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.stop()

        blockers = retrieval_blockers(selection)
        warnings = [str(w) for w in selection.get("warnings", [])]
        if blockers:
            answer = "**Insufficient verified evidence for this request.**\n\n" + "\n\n".join(blockers)
            st.warning(answer, icon="⚠️")
            st.session_state.messages.append({
                "role": "assistant", "content": answer, "warnings": warnings,
                "sources": [], "timing": f"Retrieval: {retrieval_seconds:.2f}s · Generation skipped",
            })
            st.stop()

        # Keep existing citation reconstruction, separate from Streamlit markup.
        _, sources = prepare_generation(question, selection)
        st.caption("Stage 2 of 2 · Qwen 3.5 4B is generating…")
        stats = {}
        timings = {"started": perf_counter(), "first": None}

        def measured_stream():
            for token in stream_answer(question, selection, model=MODEL, stats=stats):
                if timings["first"] is None:
                    timings["first"] = perf_counter()
                yield token

        
        try:
            # Stream directly into a Streamlit placeholder.
            # Avoid st.write_stream's dataframe-type inspection.
            placeholder = st.empty()

            collected_tokens = []
            last_render = 0.0

            for token in measured_stream():
                collected_tokens.append(token)

                # Refresh the UI approximately every 75 milliseconds.
                now = perf_counter()

                if now - last_render >= 0.075:
                    current_answer = "".join(collected_tokens)
                    placeholder.markdown(current_answer + " ▌")
                    last_render = now

            # Final answer without the streaming cursor.
            answer = "".join(collected_tokens)
            placeholder.markdown(answer)

        except Exception as exc:
            st.error(
                "The local Qwen generation stream failed; "
                "this answer was not saved."
            )
            st.exception(exc)
            st.stop()


        total_generation = perf_counter() - timings["started"]
        first_token = (timings["first"] - timings["started"]
                       if timings["first"] is not None else None)
        timing = f"Retrieval: {retrieval_seconds:.2f}s · Generation: {total_generation:.2f}s"
        if first_token is not None:
            timing += f" · First token: {first_token:.2f}s after generation began"
        st.caption(timing)

        warnings.extend(check_answer_citations(answer, sources, stats))
        # Selector heuristics cannot certify complete legal applicability.
        if show_evidence:
            display_sources(sources, show_full=True)
        else:
            display_sources(sources, show_full=False)
        for warning in dict.fromkeys(warnings):
            if warning.startswith("Do not infer"):
                continue  # internal selection instruction, not a factual user warning
            st.warning(warning, icon="⚠️")

        st.session_state.messages.append({
            "role": "assistant", "content": answer, "sources": sources,
            "warnings": [w for w in dict.fromkeys(warnings) if not w.startswith("Do not infer")],
            "timing": timing,
        })
