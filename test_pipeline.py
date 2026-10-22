"""Offline tests for pipeline.py and app.py.

No API keys, network access or model downloads are needed:
  * Mistral is replaced by small fake runnables (clearly marked FAKE below).
  * Whisper and Sarvam are replaced by fakes at their boundaries (model.transcribe / HTTP POST).
  * Embeddings are a deterministic keyword-count model (FAKE, but it ranks by real word overlap).
  * ChromaDB is REAL, running in a temporary directory, so isolation and de-duplication
    are tested against the actual database.
  * FFmpeg is REAL when installed (tests that need it are skipped otherwise).

These tests prove the application logic. They do NOT prove the quality of live
transcription, live LLM answers or the real embedding model.

Run:  pytest -q
"""

from __future__ import annotations

import math
import re
import shutil
import struct
import wave
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda

import pipeline as pl

HAS_FFMPEG = shutil.which("ffmpeg") is not None
SMALL = 150   # chunk size for retrieval tests, so each recording has several chunks

# Chroma warns when LangChain (not Chroma) computes the embeddings. It is internal to chromadb and harmless.
pytestmark = pytest.mark.filterwarnings("ignore:legacy embedding function config:DeprecationWarning")


# --------------------------------------------------------------------------- helpers / fakes

def make_wav(path: Path, seconds: float, rate: int = 16_000, channels: int = 1, tone: bool = False) -> Path:
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        if tone:
            samples = (int(8000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(frames))
            w.writeframes(b"".join(struct.pack("<h", s) * channels for s in samples))
        else:
            w.writeframes(b"\x00\x00" * channels * frames)
    return path


class KeywordEmbeddings(Embeddings):
    """FAKE embedding model: counts vocabulary words. Deterministic and offline."""

    VOCAB = ["vendor", "contract", "price", "budget", "launch", "date", "march", "team",
             "gradient", "descent", "learning", "rate", "loss", "function", "model", "lecture"]

    def _vec(self, text: str) -> list[float]:
        words = re.findall(r"[a-z]+", text.lower())
        v = [float(words.count(w)) for w in self.VOCAB] + [0.1]   # bias: never a zero vector
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeStructuredLLM:
    """FAKE Mistral for analyze(): returns prepared objects per schema, records inputs."""

    def __init__(self, section_notes: list, final_summary):
        self.section_notes = list(section_notes)
        self.final_summary = final_summary
        self.calls: list[tuple[str, object]] = []

    def with_structured_output(self, schema):
        def respond(prompt_value):
            self.calls.append((schema.__name__, prompt_value))
            if schema is pl.SectionNotes:
                return self.section_notes.pop(0)
            return self.final_summary
        return RunnableLambda(respond)


def fake_chat_llm(reply: str, calls: list):
    """FAKE Mistral for answer_question(): records the prompt, returns a fixed reply."""
    def respond(prompt_value):
        calls.append(prompt_value.to_string())
        return reply
    return RunnableLambda(respond)


MEETING = [  # recording A: a meeting about a vendor contract
    pl.Segment(0.0, 6.0, "Welcome everyone, today we review the vendor contract and the budget."),
    pl.Segment(6.0, 14.0, "The vendor price went up by ten percent compared to last year."),
    pl.Segment(14.0, 22.0, "We decided to sign the vendor contract only if the price stays under budget."),
    pl.Segment(22.0, 30.0, "Priya will send the revised contract to legal by Friday."),
    pl.Segment(30.0, 38.0, "The launch date moves to March so the team has time for testing."),
    pl.Segment(38.0, 45.0, "We still need to confirm who owns the launch checklist."),
]
LECTURE = [  # recording B: a lecture on gradient descent
    pl.Segment(0.0, 9.0, "In this lecture we study gradient descent for training a model."),
    pl.Segment(9.0, 18.0, "Gradient descent updates the parameters against the gradient of the loss function."),
    pl.Segment(18.0, 27.0, "The learning rate controls the step size of gradient descent."),
    pl.Segment(27.0, 36.0, "If the learning rate is too large, the loss can diverge."),
]


def settings(**overrides) -> pl.Settings:
    base = dict(mistral_api_key="test-key", sarvam_api_key="test-key", whisper_model="small",
                mistral_model="mistral-small-latest", sarvam_model="saaras:v4",
                embedding_model="unused")
    base.update(overrides)
    return pl.Settings(**base)


# --------------------------------------------------------------------------- 1. input validation

@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtube.com/watch?v=dQw4w9WgXcQ&t=223s",
    "https://youtu.be/dQw4w9WgXcQ?si=abc",
    "https://www.youtube.com/shorts/dQw4w9WgXcQ",
    "https://m.youtube.com/live/dQw4w9WgXcQ",
])
def test_valid_youtube_urls_map_to_same_id(url):
    assert pl.validate_youtube_url(url) == "dQw4w9WgXcQ"


@pytest.mark.parametrize("url", [
    "", "not a url", "https://vimeo.com/123456", "https://www.youtube.com/playlist?list=PL123",
    "https://www.youtube.com/watch?v=short", "ftp://youtube.com/watch?v=dQw4w9WgXcQ",
    "https://evil.com/?u=https://youtu.be/dQw4w9WgXcQ",
])
def test_invalid_youtube_urls_are_rejected(url):
    with pytest.raises(pl.PipelineError):
        pl.validate_youtube_url(url)


def test_upload_extension_validation():
    assert pl.validate_upload_name("Meeting.MP4") == ".mp4"
    with pytest.raises(pl.PipelineError, match="Unsupported"):
        pl.validate_upload_name("notes.pdf")


def test_recording_ids_and_collection_names():
    assert pl.recording_id_for_file(b"abc") == pl.recording_id_for_file(b"abc")
    assert pl.recording_id_for_file(b"abc") != pl.recording_id_for_file(b"abd")
    name = pl.collection_name("yt-dQw4w9WgXcQ")
    assert re.fullmatch(r"rec-[0-9a-f]{20}", name)          # valid Chroma name (3-63 chars)
    assert name != pl.collection_name("yt-aaaaaaaaaaa")


# --------------------------------------------------------------------------- configuration

def test_config_problems_report_missing_keys(monkeypatch):
    monkeypatch.setattr(pl.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    s = settings(mistral_api_key=None, sarvam_api_key=None)
    assert any("MISTRAL_API_KEY" in p for p in pl.config_problems(s, pl.ENGINE_WHISPER))
    assert any("SARVAM_API_KEY" in p for p in pl.config_problems(s, pl.ENGINE_SARVAM))
    assert not any("SARVAM" in p for p in pl.config_problems(s, pl.ENGINE_WHISPER))
    with pytest.raises(pl.ConfigError):
        pl.check_requirements(s, pl.ENGINE_WHISPER)


def test_turbo_cannot_translate(monkeypatch):
    monkeypatch.setattr(pl.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    problems = pl.config_problems(settings(whisper_model="turbo"), pl.ENGINE_WHISPER_TRANSLATE)
    assert any("not trained for translation" in p for p in problems)
    assert pl.config_problems(settings(whisper_model="turbo"), pl.ENGINE_WHISPER) == []


def test_missing_ffmpeg_is_reported(monkeypatch):
    monkeypatch.setattr(pl.shutil, "which", lambda name: None)
    assert any("FFmpeg" in p for p in pl.config_problems(settings(), pl.ENGINE_WHISPER))


# --------------------------------------------------------------------------- 2-3. audio

def test_split_wav_offsets_and_durations(tmp_path):
    wav = make_wav(tmp_path / "a.wav", 70)
    pieces = pl.split_wav(wav, 25, tmp_path / "pieces")
    assert [round(o, 3) for _, o in pieces] == [0.0, 25.0, 50.0]
    assert [round(pl.wav_duration(p), 3) for p, _ in pieces] == [25.0, 25.0, 20.0]


@pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg not installed")
def test_normalize_audio_produces_16k_mono(tmp_path):
    src = make_wav(tmp_path / "stereo.wav", 2, rate=44_100, channels=2, tone=True)
    out = pl.normalize_audio(src, tmp_path / "out.wav")
    with wave.open(str(out), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16_000, 1, 2)
    assert abs(pl.wav_duration(out) - 2.0) < 0.05


@pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg not installed")
def test_normalize_audio_rejects_non_media(tmp_path):
    bad = tmp_path / "fake.mp4"
    bad.write_text("this is not a video")
    with pytest.raises(pl.PipelineError, match="FFmpeg"):
        pl.normalize_audio(bad, tmp_path / "out.wav")


# --------------------------------------------------------------------------- 4. transcription (FAKE engines)

def test_whisper_segments_get_absolute_timestamps(tmp_path, monkeypatch):
    class FakeWhisper:  # FAKE: stands in for whisper.load_model(...)
        device = type("D", (), {"type": "cpu"})()
        def __init__(self):
            self.calls = []
        def transcribe(self, path, task, fp16):
            self.calls.append((task, fp16))
            return {"segments": [{"start": 1.0, "end": 4.0, "text": " hello "},
                                 {"start": 5.0, "end": 6.0, "text": "  "}]}

    fake = FakeWhisper()
    monkeypatch.setattr(pl, "load_whisper_model", lambda name: fake)
    wav = make_wav(tmp_path / "audio.wav", 25)
    segs = pl._transcribe_whisper(wav, settings(), translate=True, progress=None, chunk_seconds=10)
    assert [(s.start, s.end, s.text) for s in segs] == [(1.0, 4.0, "hello"), (11.0, 14.0, "hello"),
                                                        (21.0, 24.0, "hello")]
    assert fake.calls == [("translate", False)] * 3      # empty segments dropped, fp32 on CPU


class FakeResponse:  # FAKE: stands in for requests.Response
    def __init__(self, status, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload or {}, text
    def json(self):
        return self._payload
    def raise_for_status(self):
        import requests
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


def test_sarvam_pieces_and_approximate_timestamps(tmp_path, monkeypatch):
    import requests
    sent = []

    def fake_post(url, headers, files, data, timeout):
        sent.append((url, headers["api-subscription-key"], data))
        return FakeResponse(200, {"transcript": f"piece {len(sent)}"})

    monkeypatch.setattr(requests, "post", fake_post)
    wav = make_wav(tmp_path / "audio.wav", 60)
    segs = pl.transcribe(wav, pl.ENGINE_SARVAM, settings(), progress=None)
    assert [(round(s.start), round(s.end), s.text) for s in segs] == [
        (0, 25, "piece 1"), (25, 50, "piece 2"), (50, 60, "piece 3")]
    assert sent[0] == (pl.SARVAM_URL, "test-key", {"model": "saaras:v4", "mode": "translate"})
    t = pl.Transcript("x", "t", pl.ENGINE_SARVAM, segs, 60)
    assert t.approximate_timestamps and "approximate" in pl.timestamp_note(t)


def test_sarvam_bad_key_gives_clear_error(tmp_path, monkeypatch):
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(403, text="invalid key"))
    with pytest.raises(pl.PipelineError, match="rejected the API key"):
        pl.transcribe(make_wav(tmp_path / "a.wav", 5), pl.ENGINE_SARVAM, settings())


def test_empty_transcript_is_an_error(tmp_path, monkeypatch):
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(200, {"transcript": ""}))
    with pytest.raises(pl.PipelineError, match="No speech"):
        pl.transcribe(make_wav(tmp_path / "a.wav", 5), pl.ENGINE_SARVAM, settings())


# --------------------------------------------------------------------------- retries

class StatusError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.response = type("R", (), {"status_code": status})()


def test_retry_on_429_then_success():
    attempts, sleeps = [], []
    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise StatusError(429)
        return "ok"
    assert pl.with_retries(flaky, sleep=sleeps.append) == "ok"
    assert sleeps == [2.0, 4.0]


def test_no_retry_on_401():
    attempts = []
    def unauthorized():
        attempts.append(1)
        raise StatusError(401)
    with pytest.raises(StatusError):
        pl.with_retries(unauthorized, sleep=lambda s: None)
    assert len(attempts) == 1


# --------------------------------------------------------------------------- 5. analysis (FAKE LLM)

def test_quote_location_and_its_limits():
    # Found, even across a segment boundary and with different punctuation/case:
    assert pl.locate_quote("ten percent compared to last year. We DECIDED to sign", MEETING) == (6.0, 22.0)
    # Invented quote:
    assert pl.locate_quote("we agreed to cancel the vendor entirely", MEETING) is None
    # Too short to count as evidence:
    assert pl.locate_quote("the vendor", MEETING) is None


def test_analyze_map_reduce_merges_and_flags_unsupported_items():
    notes_1 = pl.SectionNotes(
        summary_points=["Vendor price rose 10%."],
        decisions=[pl.ExtractedItem(text="Sign only if under budget.",
                                    quote="sign the vendor contract only if the price stays under budget")],
        action_items=[pl.ExtractedAction(text="Send revised contract to legal.", owner="Priya",
                                         deadline="Friday", quote="Priya will send the revised contract")],
    )
    notes_2 = pl.SectionNotes(
        summary_points=["Launch moves to March."],
        decisions=[pl.ExtractedItem(text="sign only if under budget",          # duplicate
                                    quote="sign the vendor contract only if the price stays under budget"),
                   pl.ExtractedItem(text="Cancel the vendor.",                  # hallucinated
                                    quote="we agreed to cancel the vendor entirely")],
        open_questions=[pl.ExtractedItem(text="Who owns the launch checklist?",
                                         quote="confirm who owns the launch checklist")],
    )
    llm = FakeStructuredLLM([notes_1, notes_2],
                            pl.FinalSummary(title="Vendor review", summary=["Price up.", "Launch in March."]))

    report = pl.analyze(MEETING, llm, section_chars=250)   # -> exactly two sections

    assert [c[0] for c in llm.calls] == ["SectionNotes", "SectionNotes", "FinalSummary"]  # N + 1 calls
    assert report.title == "Vendor review"
    assert [d.text for d in report.decisions] == ["Sign only if under budget.", "Cancel the vendor."]
    real, invented = report.decisions
    assert real.quote_found and (real.start, real.end) == (14.0, 22.0)
    assert not invented.quote_found                       # flagged, not silently trusted
    assert report.action_items[0].owner == "Priya" and report.action_items[0].start == 22.0
    assert report.open_questions[0].quote_found


def test_analyze_handles_recordings_without_decisions():
    llm = FakeStructuredLLM([pl.SectionNotes(summary_points=["Gradient descent basics."])],
                            pl.FinalSummary(title="Gradient descent", summary=["Basics."]))
    report = pl.analyze(LECTURE, llm)
    assert report.decisions == report.action_items == report.open_questions == []
    assert len(llm.calls) == 2


# --------------------------------------------------------------------------- 6. chunking + REAL Chroma

def test_chunks_carry_timestamps_metadata_and_overlap():
    docs = pl.build_chunks(MEETING, "yt-A", max_chars=150)
    assert len(docs) > 1
    by_end = {s.end: s for s in MEETING}
    for i, d in enumerate(docs):
        assert d.metadata["recording_id"] == "yt-A" and d.metadata["chunk_index"] == i
        assert d.metadata["start"] < d.metadata["end"]
        segments_in_chunk = [s for s in MEETING if d.metadata["start"] <= s.start < d.metadata["end"]]
        assert d.page_content == " ".join(s.text for s in segments_in_chunk)   # times match the text
        assert len(d.page_content) <= 150 or len(segments_in_chunk) <= 2      # only overlap may exceed
    for prev, nxt in zip(docs, docs[1:], strict=False):                # last segment of a chunk opens the next
        overlap = by_end[prev.metadata["end"]]
        assert nxt.page_content.startswith(overlap.text)
        assert nxt.metadata["start"] == overlap.start
    covered = " ".join(d.page_content for d in docs)
    assert all(s.text in covered for s in MEETING)


def test_oversized_single_segment_becomes_its_own_chunk():
    long = [pl.Segment(0, 60, "word " * 400), pl.Segment(60, 61, "short end.")]
    docs = pl.build_chunks(long, "r", max_chars=100)
    assert [d.metadata["start"] for d in docs] == [0, 60]


def test_retrieval_is_isolated_per_recording(tmp_path):
    emb = KeywordEmbeddings()
    pl.index_recording("yt-A", MEETING, emb, tmp_path, max_chars=SMALL)
    pl.index_recording("yt-B", LECTURE, emb, tmp_path, max_chars=SMALL)

    # Ask recording A about B's topic: every result must still come from A.
    store_a = pl._open_store("yt-A", emb, tmp_path)
    hits = store_a.similarity_search_with_relevance_scores("explain gradient descent learning rate", k=10)
    assert len(hits) == pl.count_chunks("yt-A", tmp_path) > 1        # asked for 10, A only has these
    assert all(d.metadata["recording_id"] == "yt-A" for d, _ in hits)

    # And retrieval within B actually finds the relevant passage.
    store_b = pl._open_store("yt-B", emb, tmp_path)
    top_doc, _ = store_b.similarity_search_with_relevance_scores("what does the learning rate control", k=1)[0]
    assert "learning rate" in top_doc.page_content.lower()


def test_reindexing_does_not_duplicate_or_leave_stale_chunks(tmp_path):
    emb = KeywordEmbeddings()
    expected = len(pl.build_chunks(MEETING, "yt-A", SMALL))
    assert expected > 1
    pl.index_recording("yt-A", MEETING, emb, tmp_path, max_chars=SMALL)
    pl.index_recording("yt-A", MEETING, emb, tmp_path, max_chars=SMALL)
    assert pl.count_chunks("yt-A", tmp_path) == expected

    shorter = MEETING[:1]                                   # e.g. re-processed with a different engine
    pl.index_recording("yt-A", shorter, emb, tmp_path, max_chars=SMALL)
    assert pl.count_chunks("yt-A", tmp_path) == len(pl.build_chunks(shorter, "yt-A", SMALL))
    assert len(pl._chroma_client(tmp_path).list_collections()) == 1


def test_delete_index_only_removes_one_recording(tmp_path):
    emb = KeywordEmbeddings()
    pl.index_recording("yt-A", MEETING, emb, tmp_path, max_chars=SMALL)
    pl.index_recording("yt-B", LECTURE, emb, tmp_path, max_chars=SMALL)
    assert pl.delete_index("yt-A", tmp_path) is True
    assert pl.count_chunks("yt-A", tmp_path) == 0
    assert pl.count_chunks("yt-B", tmp_path) > 0
    assert pl.delete_index("yt-A", tmp_path) is False      # deleting twice is safe
    assert pl.delete_index("yt-A", tmp_path / "missing") is False


# --------------------------------------------------------------------------- 7. answering (FAKE LLM, REAL Chroma)

def test_answer_includes_timestamped_evidence_from_the_right_recording(tmp_path):
    emb = KeywordEmbeddings()
    pl.index_recording("yt-A", MEETING, emb, tmp_path, max_chars=SMALL)
    pl.index_recording("yt-B", LECTURE, emb, tmp_path, max_chars=SMALL)
    calls = []
    answer = pl.answer_question("What happened with the vendor price?", "yt-A",
                                fake_chat_llm("The vendor price rose ten percent [1].", calls), emb, tmp_path, k=2)
    assert not answer.not_found and len(answer.evidence) == 2
    assert "vendor" in answer.evidence[0].text.lower()
    assert "gradient" not in calls[0].lower()                # no contamination reached the prompt
    assert re.search(r"\[1\] \(\d\d:\d\d-\d\d:\d\d\)", calls[0])   # passages are numbered + timestamped
    assert pl.NOT_FOUND_MESSAGE in calls[0]                  # refusal instruction is in the prompt


def test_answer_reports_not_found(tmp_path):
    emb = KeywordEmbeddings()
    pl.index_recording("yt-A", MEETING, emb, tmp_path, max_chars=SMALL)
    answer = pl.answer_question("What is the CEO's salary?", "yt-A",
                                fake_chat_llm(pl.NOT_FOUND_MESSAGE, []), emb, tmp_path)
    assert answer.not_found and answer.evidence             # closest passages still shown


def test_no_index_means_no_llm_call(tmp_path):
    calls = []
    answer = pl.answer_question("anything?", "yt-missing", fake_chat_llm("x", calls),
                                KeywordEmbeddings(), tmp_path)
    assert answer.not_found and calls == []
    with pytest.raises(pl.PipelineError):
        pl.answer_question("   ", "yt-A", fake_chat_llm("x", calls), KeywordEmbeddings(), tmp_path)


# --------------------------------------------------------------------------- 8. export

def test_exports_include_timestamps_and_caveats():
    report = pl.Report("Vendor review", ["Price up."],
                       decisions=[pl.Finding("Sign if under budget.", "only if the price stays under budget",
                                             True, 14.0, 22.0)],
                       action_items=[pl.Finding("Cancel vendor.", "made up", False, 0.0, 45.0)])
    whisper_t = pl.Transcript("yt-A", "Meeting", pl.ENGINE_WHISPER, MEETING, 45.0)
    md = pl.report_to_markdown(whisper_t, report)
    assert "## Decisions" in md and "`00:14-00:22`" in md and "quote found in transcript" in md
    assert "QUOTE NOT FOUND" in md and "does not prove" in md
    assert "## Open questions\n- None found." in md

    txt = pl.transcript_to_text(whisper_t)
    assert "[00:00:14 - 00:00:22] We decided" in txt and "~" not in txt

    sarvam_t = pl.Transcript("yt-A", "Meeting", pl.ENGINE_SARVAM, [pl.Segment(25, 50, "Hello.")], 50)
    assert "[~00:00:25 - 00:00:50] Hello." in pl.transcript_to_text(sarvam_t)
    assert "`~00:14-00:22`" in pl.report_to_markdown(sarvam_t, report)


# --------------------------------------------------------------------------- orchestration (FAKE ASR + LLM)

@pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg not installed")
def test_process_recording_end_to_end_offline(tmp_path, monkeypatch):
    wav_bytes = make_wav(tmp_path / "upload.wav", 3, rate=44_100, channels=2, tone=True).read_bytes()
    seen_wavs = []

    def fake_transcribe(wav, engine, s, progress=None):   # FAKE: replaces Whisper/Sarvam
        with wave.open(str(wav), "rb") as w:
            seen_wavs.append((w.getframerate(), w.getnchannels()))
        return MEETING

    monkeypatch.setattr(pl, "transcribe", fake_transcribe)
    llm = FakeStructuredLLM([pl.SectionNotes(summary_points=["Vendor and launch."])],
                            pl.FinalSummary(title="Vendor review", summary=["Vendor and launch."]))
    messages = []
    result = pl.process_recording(pl.UploadedMedia("meeting.wav", wav_bytes), pl.ENGINE_WHISPER, settings(),
                                  llm, KeywordEmbeddings(), messages.append,
                                  workspace=tmp_path / "ws", persist_dir=tmp_path / "db")

    assert seen_wavs == [(16_000, 1)]                       # audio was normalised before ASR
    assert result.transcript.recording_id == pl.recording_id_for_file(wav_bytes)
    assert result.chunk_count == pl.count_chunks(result.transcript.recording_id, tmp_path / "db") > 0
    assert list((tmp_path / "ws").iterdir()) == []          # temporary audio cleaned up
    assert any("Converting" in m for m in messages)


def test_process_recording_cleans_up_after_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(pl.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    def failing_normalize(src, dst):
        raise pl.PipelineError("boom")
    monkeypatch.setattr(pl, "normalize_audio", failing_normalize)
    with pytest.raises(pl.PipelineError, match="boom"):
        pl.process_recording(pl.UploadedMedia("a.mp3", b"data"), pl.ENGINE_WHISPER, settings(),
                             None, None, workspace=tmp_path / "ws", persist_dir=tmp_path / "db")
    assert list((tmp_path / "ws").iterdir()) == []


# --------------------------------------------------------------------------- app.py starts

def test_app_renders_and_reports_missing_configuration(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("MISTRAL_API_KEY", "")       # empty values are treated as missing
    monkeypatch.setenv("SARVAM_API_KEY", "")
    at = AppTest.from_file(str(Path(__file__).parent / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("MISTRAL_API_KEY" in w.value for w in at.warning)
    assert at.button(key="analyze").disabled
    assert "Intelligent Video Understanding" in at.title[0].value


def test_model_download_failure_is_a_clear_error(monkeypatch):
    import langchain_huggingface

    def offline(**kwargs):            # FAKE: simulates Hugging Face being unreachable
        raise OSError("connection refused")
    monkeypatch.setattr(langchain_huggingface, "HuggingFaceEmbeddings", offline)
    with pytest.raises(pl.PipelineError, match="needs internet access"):
        pl.load_embeddings("some/unique-model-name-for-this-test")


def _app(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("MISTRAL_API_KEY", "dummy")
    monkeypatch.setattr(pl.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    return AppTest.from_file(str(Path(__file__).parent / "app.py"), default_timeout=60)


def test_app_rejects_bad_link_before_loading_models(monkeypatch):
    loaded = []
    monkeypatch.setattr(pl, "load_embeddings", lambda name: loaded.append(name))
    at = _app(monkeypatch).run()
    at.text_input[0].input("https://vimeo.com/123").run()
    at.button(key="analyze").click().run()
    assert not at.exception
    assert any("YouTube link" in e.value for e in at.error)
    assert loaded == []


def test_app_shows_results_exports_and_evidence_backed_chat(monkeypatch):
    transcript = pl.Transcript("yt-dQw4w9WgXcQ", "Vendor meeting", pl.ENGINE_SARVAM, MEETING, 45.0)
    report = pl.Report("Vendor review", ["Price up 10%."],
                       decisions=[pl.Finding("Sign if under budget.", "only if the price stays under budget",
                                             True, 14.0, 22.0)])
    fake_result = pl.ProcessedRecording(transcript, report, chunk_count=3)
    asked = []

    # FAKES: the pipeline itself is tested above; here we test that the UI wires and renders it.
    monkeypatch.setattr(pl, "process_recording", lambda *a, **k: fake_result)
    monkeypatch.setattr(pl, "make_llm", lambda s: "llm")
    monkeypatch.setattr(pl, "load_embeddings", lambda name: "emb")
    def fake_answer(question, recording_id, llm, embeddings, approximate=False):
        asked.append((question, recording_id, approximate))
        return pl.Answer("The price rose [1].", [pl.Evidence(1, "The vendor price went up.", 6.0, 14.0, 0.81)], False)
    monkeypatch.setattr(pl, "answer_question", fake_answer)

    at = _app(monkeypatch).run()
    at.text_input[0].input("https://youtu.be/dQw4w9WgXcQ").run()
    at.button(key="analyze").click().run()
    assert not at.exception
    assert at.title[0].value == "Vendor review"
    assert any("approximate" in i.value for i in at.info)             # Sarvam timestamp caveat shown
    assert len(at.get("download_button")) == 2

    at.chat_input[0].set_value("What happened to the price?").run()
    assert not at.exception
    assert asked == [("What happened to the price?", "yt-dQw4w9WgXcQ", True)]   # current recording only
    assert any("The price rose [1]." in m.value for m in at.markdown)
    assert any("~00:06-00:14" in m.value for m in at.markdown)       # evidence with approximate time


def test_network_failures_are_described_as_connectivity_problems():
    import httpx
    import requests
    for exc in (httpx.ConnectError("refused"), requests.ConnectionError("down"), requests.Timeout("slow")):
        assert "Check your internet connection" in pl._describe_api_error("Mistral", exc)
    assert "rejected the API key" in pl._describe_api_error("Mistral", StatusError(401))
