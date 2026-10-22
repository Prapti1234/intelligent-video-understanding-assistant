"""Streamlit interface for the Intelligent Video Understanding AI Assistant.

This file only handles the UI: collecting input, showing progress and results
and the chat. All processing lives in pipeline.py.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import logging

import streamlit as st

import pipeline as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="Video Understanding Assistant", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Loading the embedding model (first run downloads it)...")
def get_embeddings(model_name: str):
    return pl.load_embeddings(model_name)


def md_escape(text: str) -> str:
    """Stop '$' in transcripts from being rendered as LaTeX by Streamlit's Markdown."""
    return text.replace("$", "\\$")


def reset_results() -> None:
    for key in ("result", "chat", "chunk_count"):
        st.session_state.pop(key, None)


def render_finding(f: pl.Finding, approximate: bool) -> None:
    extras = []
    if f.owner:
        extras.append(f"**Owner:** {md_escape(f.owner)}")
    if f.deadline:
        extras.append(f"**Deadline:** {md_escape(f.deadline)}")
    span = pl.format_span(f.start, f.end, approximate)
    if f.quote_found:
        badge = f"`{span}` · ✓ quote found in transcript"
    else:
        badge = f"`{span}` (section range) · ⚠ quote not found - treat with caution"
    st.markdown(f"- {md_escape(f.text)}  \n  {' · '.join(extras + [badge])}")
    st.caption(f"“{f.quote}”")


def render_answer(answer: pl.Answer, approximate: bool) -> None:
    if answer.not_found:
        st.warning(answer.text)
    else:
        st.markdown(md_escape(answer.text))
    if answer.evidence:
        label = "Closest passages (judge for yourself)" if answer.not_found else "Evidence used"
        with st.expander(f"{label} - {len(answer.evidence)} passages"):
            for e in answer.evidence:
                st.markdown(f"**[{e.number}]** `{pl.format_span(e.start, e.end, approximate)}` "
                            f"· similarity {e.score:.2f}")
                st.markdown(f"> {md_escape(e.text)}")


settings = pl.load_settings()

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🎬 Video Understanding Assistant")
    source_kind = st.radio("Source", ["YouTube URL", "Upload a file"], horizontal=True)
    url, upload = "", None
    if source_kind == "YouTube URL":
        url = st.text_input("YouTube link", placeholder="https://www.youtube.com/watch?v=...")
    else:
        upload = st.file_uploader(
            "Audio or video file",
            type=sorted(ext.lstrip(".") for ext in pl.SUPPORTED_EXTENSIONS),
        )

    engine = st.selectbox("Transcription", list(pl.ENGINE_LABELS), format_func=pl.ENGINE_LABELS.get)
    if engine == pl.ENGINE_SARVAM:
        st.caption("Audio is sent to Sarvam AI in 25-second pieces. Output is English; "
                   "timestamps are approximate (per piece).")
    else:
        st.caption(f"Runs locally with Whisper '{settings.whisper_model}'. "
                   "Audio does not leave your computer.")

    problems = pl.config_problems(settings, engine)
    for problem in problems:
        st.warning(problem)

    analyze_clicked = st.button("Analyze", type="primary", width="stretch",
                                disabled=bool(problems), key="analyze")

    if "result" in st.session_state:
        st.divider()
        rec = st.session_state["result"].transcript
        st.caption(f"Index for this recording: {st.session_state.get('chunk_count', 0)} chunks")
        if st.button("Delete this recording's index", width="stretch", key="delete_index"):
            existed = pl.delete_index(rec.recording_id)
            st.session_state["chunk_count"] = 0
            st.success("Index deleted. Other recordings were not touched."
                       if existed else "There was no index to delete.")

    with st.expander("Privacy & data"):
        st.markdown(
            "- **Local:** Whisper, FFmpeg, embeddings, ChromaDB.\n"
            "- **Sent to Mistral:** transcript text and your questions. On Mistral's free "
            "Experiment plan, requests may be used to train their models.\n"
            "- **Sent to Sarvam (only in Sarvam mode):** your audio.\n"
            "- Downloaded audio is deleted after transcription; the search index stays in "
            "`chroma_db/` until you delete it."
        )

# --------------------------------------------------------------------------- processing
if analyze_clicked:
    source = None
    if source_kind == "YouTube URL":
        source = url.strip() or None
    elif upload is not None:
        source = pl.UploadedMedia(upload.name, upload.getvalue())

    if source is None:
        st.error("Enter a YouTube link or upload a file first.")
    else:
        reset_results()
        with st.status("Processing...", expanded=True) as status:
            def progress(message: str) -> None:
                status.update(label=message)
                st.write(message)

            try:
                pl.identify_source(source)          # reject bad input before loading any model
                result = pl.process_recording(
                    source, engine, settings,
                    llm=pl.make_llm(settings),
                    embeddings=get_embeddings(settings.embedding_model),
                    progress=progress,
                )
            except pl.PipelineError as exc:
                status.update(label="Failed", state="error")
                st.error(str(exc))
            except Exception as exc:  # unexpected bug: show it rather than hide it
                status.update(label="Failed with an unexpected error", state="error")
                st.exception(exc)
            else:
                st.session_state["result"] = result
                st.session_state["chunk_count"] = result.chunk_count
                st.session_state["chat"] = []
                status.update(label="Done", state="complete", expanded=False)

# --------------------------------------------------------------------------- results
if "result" not in st.session_state:
    st.title("Intelligent Video Understanding AI Assistant")
    st.markdown(
        "Paste a YouTube link or upload a recording, choose a transcription engine and press "
        "**Analyze**. You will get a transcript with timestamps, a summary, extracted decisions, "
        "action items and open questions, and a chat that answers from the recording with evidence."
    )
    st.stop()

result: pl.ProcessedRecording = st.session_state["result"]
transcript, report = result.transcript, result.report
approx = transcript.approximate_timestamps

st.title(md_escape(report.title))
st.caption(f"{transcript.source_title} · {pl.format_timestamp(transcript.duration, True)} · "
           f"{pl.ENGINE_LABELS[transcript.engine]}")
if approx:
    st.info(pl.timestamp_note(transcript))

summary_tab, insights_tab, transcript_tab, ask_tab = st.tabs(
    ["Summary", "Decisions & actions", "Transcript", "Ask"])

with summary_tab:
    st.markdown("\n".join(f"- {md_escape(p)}" for p in report.summary))
    st.divider()
    col1, col2 = st.columns(2)
    col1.download_button("Download report (.md)", pl.report_to_markdown(transcript, report),
                         file_name=f"{transcript.recording_id}-report.md", mime="text/markdown",
                         width="stretch")
    col2.download_button("Download transcript (.txt)", pl.transcript_to_text(transcript),
                         file_name=f"{transcript.recording_id}-transcript.txt", mime="text/plain",
                         width="stretch")

with insights_tab:
    st.caption("✓ means the supporting quote appears word-for-word in the transcript. It does not "
               "prove the item is interpreted correctly - check the timestamp in the recording.")
    for heading, items in (("Decisions", report.decisions),
                           ("Action items", report.action_items),
                           ("Open questions", report.open_questions)):
        st.subheader(heading)
        if not items:
            st.write("None found.")
        for f in items:
            render_finding(f, approx)

with transcript_tab:
    st.caption(pl.timestamp_note(transcript))
    with st.container(height=500):
        st.text(pl.transcript_to_text(transcript))

with ask_tab:
    if st.session_state.get("chunk_count", 0) == 0:
        st.warning("This recording's index was deleted. Run Analyze again to chat with it.")
    for turn in st.session_state.get("chat", []):
        with st.chat_message("user"):
            st.markdown(md_escape(turn["question"]))
        with st.chat_message("assistant"):
            render_answer(turn["answer"], approx)

    question = st.chat_input("Ask a question about this recording")
    if question:
        with st.chat_message("user"):
            st.markdown(md_escape(question))
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the transcript..."):
                    answer = pl.answer_question(
                        question, transcript.recording_id, pl.make_llm(settings),
                        get_embeddings(settings.embedding_model), approximate=approx)
            except pl.PipelineError as exc:
                st.error(str(exc))
            else:
                render_answer(answer, approx)
                st.session_state.setdefault("chat", []).append({"question": question, "answer": answer})
