"""Core processing pipeline for the Intelligent Video Understanding AI Assistant.

Stages (in order):
    1. Input       validate a YouTube URL or an uploaded file, derive a recording ID
    2. Audio       download (yt-dlp) and normalise to 16 kHz mono WAV (FFmpeg)
    3. Split       cut the WAV into pieces with the standard-library `wave` module
    4. Transcribe  local Whisper, or the Sarvam AI API (Hindi/Hinglish -> English)
    5. Analyse     map-reduce over transcript sections with Mistral (structured output)
    6. Index       timestamped chunks -> embeddings -> one Chroma collection per recording
    7. Ask         retrieve evidence from that collection only -> grounded answer
    8. Export      Markdown report and timestamped plain-text transcript

This module has no Streamlit code. Heavy libraries (Whisper, yt-dlp, Chroma,
Hugging Face, Mistral) are imported inside the functions that need them, so the
module imports quickly and the offline tests do not need model downloads.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import time
import wave
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from collections.abc import Callable
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR / "workspace"      # temporary audio, deleted after each run
CHROMA_DIR = PROJECT_DIR / "chroma_db"         # persistent vector store

SAMPLE_RATE = 16_000                  # Whisper and Sarvam both expect 16 kHz mono
WHISPER_CHUNK_SECONDS = 600           # 10-minute pieces keep memory flat for long audio
SARVAM_PIECE_SECONDS = 25             # Sarvam's synchronous API rejects audio > 30 s
SARVAM_URL = "https://api.sarvam.ai/speech-to-text"

SECTION_CHARS = 12_000                # analysis map step: ~3k tokens per section
CHUNK_CHARS = 1_000                   # RAG chunk size (packed from whole segments)
TOP_K = 4                             # passages retrieved per question

ENGINE_WHISPER = "whisper"
ENGINE_WHISPER_TRANSLATE = "whisper-translate"
ENGINE_SARVAM = "sarvam"
ENGINE_LABELS = {
    ENGINE_WHISPER: "Whisper (local) - keep original language",
    ENGINE_WHISPER_TRANSLATE: "Whisper (local) - translate to English",
    ENGINE_SARVAM: "Sarvam AI (cloud) - Hindi/Hinglish to English",
}

SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus",
    ".mp4", ".mkv", ".mov", ".webm",
}

NOT_FOUND_MESSAGE = "I couldn't find this in the recording's transcript."

YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}
YOUTU_BE_HOSTS = {"youtu.be", "www.youtu.be"}
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


# ---------------------------------------------------------------------------
# Errors and configuration
# ---------------------------------------------------------------------------

class PipelineError(Exception):
    """An expected failure with a message that is safe to show to the user."""


class ConfigError(PipelineError):
    """Missing or invalid configuration (API keys, FFmpeg, model choice)."""


@dataclass(frozen=True)
class Settings:
    mistral_api_key: Optional[str]
    sarvam_api_key: Optional[str]
    whisper_model: str
    mistral_model: str
    sarvam_model: str
    embedding_model: str


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name, "").strip()
    return value or default


def load_settings() -> Settings:
    """Read configuration from the environment (and a local .env file, if present).

    Called at run time, never at import time, so a .env file is always loaded
    before any value is read (this was a bug in the reference implementation).
    """
    load_dotenv(PROJECT_DIR / ".env", override=False)
    return Settings(
        mistral_api_key=_env("MISTRAL_API_KEY"),
        sarvam_api_key=_env("SARVAM_API_KEY"),
        whisper_model=_env("WHISPER_MODEL", "small"),
        mistral_model=_env("MISTRAL_MODEL", "ministral-3b-latest"),
        sarvam_model=_env("SARVAM_MODEL", "saaras:v4"),
        embedding_model=_env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    )


def config_problems(settings: Settings, engine: str) -> list[str]:
    """Return human-readable problems that would stop a run with this engine."""
    problems = []
    if engine not in ENGINE_LABELS:
        problems.append(f"Unknown transcription engine: {engine!r}.")
    if shutil.which("ffmpeg") is None:
        problems.append("FFmpeg was not found on PATH. Install it and restart the terminal.")
    if not settings.mistral_api_key:
        problems.append("MISTRAL_API_KEY is not set (needed for analysis and chat). Add it to .env.")
    if engine == ENGINE_SARVAM and not settings.sarvam_api_key:
        problems.append("SARVAM_API_KEY is not set. Add it to .env or choose a Whisper mode.")
    if engine == ENGINE_WHISPER_TRANSLATE and settings.whisper_model.startswith(("turbo", "large-v3-turbo")):
        problems.append(
            f"WHISPER_MODEL={settings.whisper_model} is not trained for translation and would "
            "return the original language. Use small, medium or large for translate mode."
        )
    return problems


def check_requirements(settings: Settings, engine: str) -> None:
    problems = config_problems(settings, engine)
    if problems:
        raise ConfigError(" ".join(problems))


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """One timed piece of speech. Times are seconds from the start of the recording."""
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    recording_id: str
    source_title: str
    engine: str
    segments: list[Segment]
    duration: float

    @property
    def approximate_timestamps(self) -> bool:
        """Sarvam returns no timestamps; we only know which 25-second piece text came from."""
        return self.engine == ENGINE_SARVAM

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)


@dataclass
class Finding:
    """A decision, action item or open question extracted by the LLM."""
    text: str
    quote: str
    quote_found: bool            # the quote exists verbatim in the transcript (NOT proof of correctness)
    start: float
    end: float
    owner: Optional[str] = None
    deadline: Optional[str] = None


@dataclass
class Report:
    title: str
    summary: list[str]
    decisions: list[Finding] = field(default_factory=list)
    action_items: list[Finding] = field(default_factory=list)
    open_questions: list[Finding] = field(default_factory=list)


@dataclass
class Evidence:
    number: int
    text: str
    start: float
    end: float
    score: float


@dataclass
class Answer:
    text: str
    evidence: list[Evidence]
    not_found: bool


@dataclass
class UploadedMedia:
    name: str
    data: bytes


@dataclass
class ProcessedRecording:
    transcript: Transcript
    report: Report
    chunk_count: int


ProgressFn = Callable[[str], None]


def _notify(progress: Optional[ProgressFn], message: str) -> None:
    logger.info(message)
    if progress:
        progress(message)


# ---------------------------------------------------------------------------
# 1. Input validation and recording identity
# ---------------------------------------------------------------------------

def validate_youtube_url(url: str) -> str:
    """Return the 11-character video ID, or raise PipelineError."""
    url = (url or "").strip()
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or host not in YOUTUBE_HOSTS | YOUTU_BE_HOSTS:
        raise PipelineError("Please enter a YouTube link (youtube.com/watch?v=... or youtu.be/...).")

    parts = [p for p in parsed.path.split("/") if p]
    video_id = None
    if host in YOUTU_BE_HOSTS and parts:
        video_id = parts[0]
    elif parts[:1] == ["watch"]:
        video_id = parse_qs(parsed.query).get("v", [None])[0]
    elif len(parts) >= 2 and parts[0] in {"shorts", "live", "embed"}:
        video_id = parts[1]
    elif parts[:1] == ["playlist"]:
        raise PipelineError("Playlists are not supported. Paste the link of a single video.")

    if not video_id or not VIDEO_ID_RE.match(video_id):
        raise PipelineError("That does not look like a valid YouTube video link.")
    return video_id


def validate_upload_name(filename: str) -> str:
    """Return the lowercase extension of a supported media file, or raise PipelineError."""
    ext = Path(filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise PipelineError(f"Unsupported file type '{ext or filename}'. Supported: {allowed}.")
    return ext


def recording_id_for_youtube(video_id: str) -> str:
    return f"yt-{video_id}"


def recording_id_for_file(data: bytes) -> str:
    """Identify uploads by content, so renaming a file does not create a second index."""
    return f"file-{hashlib.sha256(data).hexdigest()[:16]}"


def identify_source(source: str | UploadedMedia) -> tuple[str, str]:
    """Validate the input and return (recording_id, display title). No downloads, no models.

    The UI calls this before loading any model, so bad input fails fast with a clear message.
    """
    if isinstance(source, UploadedMedia):
        validate_upload_name(source.name)
        if not source.data:
            raise PipelineError("The uploaded file is empty.")
        return recording_id_for_file(source.data), source.name
    video_id = validate_youtube_url(source)
    return recording_id_for_youtube(video_id), f"YouTube video {video_id}"


def collection_name(recording_id: str) -> str:
    """Chroma collection for one recording: 'rec-' + 20 hex chars (valid 3-63 char name)."""
    return "rec-" + hashlib.sha256(recording_id.encode("utf-8")).hexdigest()[:20]


# ---------------------------------------------------------------------------
# 2-3. Audio acquisition, normalisation and splitting
# ---------------------------------------------------------------------------

def _run_ffmpeg(args: list[str]) -> None:
    if shutil.which("ffmpeg") is None:
        raise ConfigError("FFmpeg was not found on PATH. Install it and restart the terminal.")
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-y", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-4:])
        raise PipelineError(f"FFmpeg could not process the audio:\n{tail}")


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


def normalize_audio(src: Path, dst: Path) -> Path:
    """Decode any audio/video file into 16 kHz, mono, 16-bit PCM WAV (streamed by FFmpeg)."""
    _run_ffmpeg(["-i", str(src), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
                 "-c:a", "pcm_s16le", str(dst)])
    if wav_duration(dst) <= 0.5:
        raise PipelineError("The file contains no usable audio track.")
    return dst


def download_youtube_audio(url: str, workdir: Path) -> tuple[Path, str]:
    """Download the best audio stream with yt-dlp. Returns (file path, video title)."""
    import yt_dlp

    options = {
        "format": "bestaudio/best",
        "outtmpl": str(workdir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        raise PipelineError(
            f"YouTube download failed: {exc}. The video may be private, age-restricted or "
            "region-locked, or yt-dlp may need updating (pip install -U \"yt-dlp[default,deno]\")."
        ) from exc

    downloads = info.get("requested_downloads") or []
    path = Path(downloads[0]["filepath"]) if downloads else None
    if path is None or not path.exists():
        raise PipelineError("yt-dlp reported success but no audio file was written.")
    return path, info.get("title") or "YouTube video"


def split_wav(path: Path, seconds: float, outdir: Path) -> list[tuple[Path, float]]:
    """Split a WAV into pieces of `seconds`, reading one piece at a time.

    Returns (piece path, offset in seconds) pairs. Memory use is one piece,
    regardless of how long the recording is.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    pieces = []
    with wave.open(str(path), "rb") as src:
        params = src.getparams()
        frames_per_piece = max(1, int(seconds * params.framerate))
        frames_done = 0
        index = 0
        while True:
            frames = src.readframes(frames_per_piece)
            if not frames:
                break
            piece_path = outdir / f"{path.stem}_{index:04d}.wav"
            with wave.open(str(piece_path), "wb") as dst:
                dst.setparams(params)
                dst.writeframes(frames)
            pieces.append((piece_path, frames_done / params.framerate))
            frames_done += len(frames) // (params.sampwidth * params.nchannels)
            index += 1
    return pieces


# ---------------------------------------------------------------------------
# Retry helper for remote APIs (Mistral and Sarvam)
# ---------------------------------------------------------------------------

def _http_status(exc: BaseException) -> Optional[int]:
    """Works for httpx.HTTPStatusError (Mistral) and requests.HTTPError (Sarvam)."""
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def with_retries(fn: Callable[[], Any], attempts: int = 4, base_delay: float = 2.0,
                 sleep: Callable[[float], None] = time.sleep) -> Any:
    """Call fn(), retrying only on HTTP 429 (rate limit) and 5xx, with exponential backoff.

    ChatMistralAI already retries network errors itself, but not HTTP 429, which is
    the usual failure on the free tier. Client errors such as 401 fail immediately.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            status = _http_status(exc)
            retryable = status == 429 or (status is not None and status >= 500)
            if not retryable or attempt == attempts:
                raise
            delay = base_delay * 2 ** (attempt - 1)
            logger.warning("HTTP %s, retrying in %.0f s (attempt %d/%d)", status, delay, attempt, attempts)
            sleep(delay)


def _describe_api_error(service: str, exc: BaseException) -> str:
    status = _http_status(exc)
    if status in (401, 403):
        return f"{service} rejected the API key (HTTP {status}). Check the key in .env."
    if status == 429:
        return f"{service} rate limit reached even after retries. Wait a minute and try again."
    if status is not None:
        return f"{service} returned HTTP {status}: {exc}"
    import httpx
    import requests
    if isinstance(exc, (httpx.TransportError, requests.ConnectionError, requests.Timeout)):
        return f"Could not reach {service} ({type(exc).__name__}: {exc}). Check your internet connection."
    return f"{service} request failed: {exc}"


# ---------------------------------------------------------------------------
# 4. Transcription
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2)
def load_whisper_model(name: str):
    """Load a Whisper model once per process (downloads weights on first use)."""
    import whisper
    try:
        return whisper.load_model(name)
    except RuntimeError as exc:          # unknown model name, or a corrupted download
        raise PipelineError(f"Could not load Whisper model '{name}' (check WHISPER_MODEL): {exc}") from exc
    except OSError as exc:               # download or disk failure (URLError is an OSError)
        raise PipelineError(
            f"Could not load Whisper model '{name}': {exc}. The first run downloads the "
            "weights from OpenAI, so it needs internet access."
        ) from exc


def _transcribe_whisper(wav: Path, settings: Settings, translate: bool,
                        progress: Optional[ProgressFn],
                        chunk_seconds: float = WHISPER_CHUNK_SECONDS) -> list[Segment]:
    model = load_whisper_model(settings.whisper_model)
    use_fp16 = getattr(getattr(model, "device", None), "type", "cpu") == "cuda"
    task = "translate" if translate else "transcribe"
    pieces = split_wav(wav, chunk_seconds, wav.parent / "whisper_pieces")
    segments: list[Segment] = []
    for i, (piece, offset) in enumerate(pieces, start=1):
        _notify(progress, f"Whisper ({settings.whisper_model}, {task}): part {i}/{len(pieces)}")
        result = model.transcribe(str(piece), task=task, fp16=use_fp16)
        for seg in result.get("segments", []):
            text = seg["text"].strip()
            if text:
                segments.append(Segment(offset + seg["start"], offset + seg["end"], text))
        piece.unlink(missing_ok=True)
    return segments


def _sarvam_request(piece: Path, settings: Settings) -> str:
    import requests

    def call() -> str:
        with open(piece, "rb") as fh:
            response = requests.post(
                SARVAM_URL,
                headers={"api-subscription-key": settings.sarvam_api_key or ""},
                files={"file": (piece.name, fh, "audio/wav")},
                data={"model": settings.sarvam_model, "mode": "translate"},
                timeout=120,
            )
        response.raise_for_status()
        return (response.json().get("transcript") or "").strip()

    try:
        return with_retries(call)
    except requests.HTTPError as exc:
        body = exc.response.text[:300] if exc.response is not None else ""
        raise PipelineError(f"{_describe_api_error('Sarvam AI', exc)} {body}".strip()) from exc
    except requests.RequestException as exc:
        raise PipelineError(_describe_api_error("Sarvam AI", exc)) from exc


def _transcribe_sarvam(wav: Path, settings: Settings,
                       progress: Optional[ProgressFn]) -> list[Segment]:
    """Sarvam returns text only, so each segment spans its whole 25-second piece."""
    if not settings.sarvam_api_key:
        raise ConfigError("SARVAM_API_KEY is not set.")
    pieces = split_wav(wav, SARVAM_PIECE_SECONDS, wav.parent / "sarvam_pieces")
    segments: list[Segment] = []
    for i, (piece, offset) in enumerate(pieces, start=1):
        _notify(progress, f"Sarvam AI ({settings.sarvam_model}): piece {i}/{len(pieces)}")
        duration = wav_duration(piece)
        text = _sarvam_request(piece, settings)
        if text:
            segments.append(Segment(offset, offset + duration, text))
        piece.unlink(missing_ok=True)
    return segments


def transcribe(wav: Path, engine: str, settings: Settings,
               progress: Optional[ProgressFn] = None) -> list[Segment]:
    """Speech -> timed segments, using the selected engine."""
    if engine == ENGINE_SARVAM:
        segments = _transcribe_sarvam(wav, settings, progress)
    elif engine in (ENGINE_WHISPER, ENGINE_WHISPER_TRANSLATE):
        segments = _transcribe_whisper(wav, settings, engine == ENGINE_WHISPER_TRANSLATE, progress)
    else:
        raise ConfigError(f"Unknown transcription engine: {engine!r}")
    if not segments:
        raise PipelineError("No speech was recognised in this recording (silence or music only?).")
    return segments


# ---------------------------------------------------------------------------
# 5. Analysis: map-reduce with structured output and quote checking
# ---------------------------------------------------------------------------

class ExtractedItem(BaseModel):
    text: str = Field(description="The decision or open question, in one clear sentence.")
    quote: str = Field(description="5-25 words copied word-for-word from the section that support the item.")


class ExtractedAction(ExtractedItem):
    owner: Optional[str] = Field(default=None, description="Person responsible, only if explicitly named; otherwise null.")
    deadline: Optional[str] = Field(default=None, description="Deadline, only if explicitly stated; otherwise null.")


class SectionNotes(BaseModel):
    summary_points: list[str] = Field(description="2-6 short bullet points covering this section.")
    decisions: list[ExtractedItem] = Field(default_factory=list)
    action_items: list[ExtractedAction] = Field(default_factory=list)
    open_questions: list[ExtractedItem] = Field(default_factory=list)


class FinalSummary(BaseModel):
    title: str = Field(description="A short title for the whole recording, at most 10 words.")
    summary: list[str] = Field(description="5-10 bullet points summarising the whole recording in order.")


MAP_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You analyse one section of a transcript. The recording may be a meeting, lecture, "
     "interview, podcast or video. Use only what is explicitly said in THIS section.\n"
     "- summary_points: the main content, as short bullet points.\n"
     "- decisions: conclusions the speakers explicitly agree on or announce. Opinions and "
     "facts are not decisions.\n"
     "- action_items: tasks someone explicitly commits to or is asked to do. Give owner and "
     "deadline only if they are stated; otherwise null.\n"
     "- open_questions: questions raised and left unresolved, or topics flagged for follow-up.\n"
     "Every decision, action item and open question needs a 'quote' copied word-for-word "
     "from the section (5-25 words). Never paraphrase a quote.\n"
     "Empty lists are normal: lectures and videos often contain no decisions or action items. "
     "Never invent items."),
    ("human", "Section {index} of {total} ({span}):\n\n{text}"),
])

REDUCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are given bullet-point notes from consecutive sections of one recording. Write a "
     "title and a summary of the whole recording, in order. Use only information in the notes."),
    ("human", "{notes}"),
])


def _group_segments(segments: list[Segment], max_chars: int) -> list[list[Segment]]:
    """Pack consecutive segments into groups of at most max_chars (no overlap)."""
    groups: list[list[Segment]] = []
    current: list[Segment] = []
    size = 0
    for seg in segments:
        if current and size + len(seg.text) + 1 > max_chars:
            groups.append(current)
            current, size = [], 0
        current.append(seg)
        size += len(seg.text) + 1
    if current:
        groups.append(current)
    return groups


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace - for quote matching and de-duplication."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(text.split())


def locate_quote(quote: str, segments: list[Segment]) -> Optional[tuple[float, float]]:
    """Find a verbatim quote (ignoring case and punctuation) in the transcript.

    Returns the (start, end) of the segments containing it, or None. This only proves
    the words were said; it does not prove the extracted item interprets them correctly.
    """
    target = _normalize(quote)
    if len(target.split()) < 3:          # too short to be meaningful evidence
        return None
    pieces, starts, position = [], [], 0
    for seg in segments:
        norm = _normalize(seg.text)
        starts.append(position)
        pieces.append(norm)
        position += len(norm) + 1
    haystack = " ".join(pieces)
    hit = haystack.find(target)
    if hit < 0:
        return None
    first = last = 0
    for i, seg_start in enumerate(starts):
        if seg_start <= hit:
            first = i
        if seg_start <= hit + len(target) - 1:
            last = i
    return segments[first].start, segments[last].end


def _to_finding(item: ExtractedItem, segments: list[Segment], section: list[Segment]) -> Finding:
    located = locate_quote(item.quote, segments)
    start, end = located if located else (section[0].start, section[-1].end)
    return Finding(
        text=item.text.strip(),
        quote=item.quote.strip(),
        quote_found=located is not None,
        start=start,
        end=end,
        owner=getattr(item, "owner", None),
        deadline=getattr(item, "deadline", None),
    )


def _dedupe(findings: list[Finding]) -> list[Finding]:
    seen, unique = set(), []
    for f in findings:
        key = _normalize(f.text)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def analyze(segments: list[Segment], llm, approximate: bool = False,
            progress: Optional[ProgressFn] = None,
            section_chars: int = SECTION_CHARS) -> Report:
    """Map: one structured LLM call per section. Reduce: one call for title + summary.

    Findings are merged in Python (not by the LLM) so their quotes and timestamps survive.
    Calls made: (number of sections) + 1.
    """
    sections = _group_segments(segments, section_chars)
    map_chain = MAP_PROMPT | llm.with_structured_output(SectionNotes)
    reduce_chain = REDUCE_PROMPT | llm.with_structured_output(FinalSummary)

    all_notes: list[str] = []
    decisions: list[Finding] = []
    actions: list[Finding] = []
    questions: list[Finding] = []

    for i, section in enumerate(sections, start=1):
        _notify(progress, f"Analysing section {i}/{len(sections)} with Mistral")
        span = format_span(section[0].start, section[-1].end, approximate)
        notes: SectionNotes = with_retries(lambda: map_chain.invoke({
            "index": i, "total": len(sections), "span": span,
            "text": " ".join(s.text for s in section),
        }))
        all_notes.append(f"Section {i} ({span}):\n" + "\n".join(f"- {p}" for p in notes.summary_points))
        decisions += [_to_finding(d, segments, section) for d in notes.decisions]
        actions += [_to_finding(a, segments, section) for a in notes.action_items]
        questions += [_to_finding(q, segments, section) for q in notes.open_questions]

    _notify(progress, "Writing the final summary")
    final: FinalSummary = with_retries(lambda: reduce_chain.invoke({"notes": "\n\n".join(all_notes)}))
    return Report(
        title=final.title.strip(),
        summary=[p.strip() for p in final.summary if p.strip()],
        decisions=_dedupe(decisions),
        action_items=_dedupe(actions),
        open_questions=_dedupe(questions),
    )


def make_llm(settings: Settings):
    """Mistral chat model. Low temperature: we want faithful extraction, not creativity."""
    from langchain_mistralai import ChatMistralAI

    if not settings.mistral_api_key:
        raise ConfigError("MISTRAL_API_KEY is not set.")
    return ChatMistralAI(
        model=settings.mistral_model,
        api_key=settings.mistral_api_key,
        temperature=0.2,
        max_retries=2,
        timeout=120,
    )


# ---------------------------------------------------------------------------
# 6. Indexing: one Chroma collection per recording
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2)
def load_embeddings(model_name: str):
    """Local sentence-transformers model; normalised vectors so cosine similarity is meaningful."""
    from langchain_huggingface import HuggingFaceEmbeddings

    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:   # network, proxy, unknown repo or disk errors from huggingface_hub
        raise PipelineError(
            f"Could not load the embedding model '{model_name}': {exc}. The first run downloads "
            "it from Hugging Face, so it needs internet access; check EMBEDDING_MODEL if you changed it."
        ) from exc


def build_chunks(segments: list[Segment], recording_id: str,
                 max_chars: int = CHUNK_CHARS) -> list[Document]:
    """Pack whole segments into chunks of about max_chars.

    Each new chunk starts with the last segment of the previous one (overlap), so a
    sentence on a boundary appears in both chunks. Building from whole segments keeps
    exact start/end times on every chunk, which a character splitter cannot do.
    """
    groups: list[list[Segment]] = []
    current: list[Segment] = []
    for seg in segments:
        if current and len(" ".join(s.text for s in current + [seg])) > max_chars:
            groups.append(current)
            # Carry the last segment over as overlap, unless the chunk was a single
            # oversized segment (repeating it would only duplicate that one segment).
            current = [current[-1]] if len(current) > 1 else []
        current.append(seg)
    if current:
        groups.append(current)

    return [
        Document(
            page_content=" ".join(s.text for s in group),
            metadata={
                "recording_id": recording_id,
                "chunk_index": i,
                "start": float(group[0].start),
                "end": float(group[-1].end),
            },
        )
        for i, group in enumerate(groups)
    ]


def _open_store(recording_id: str, embeddings, persist_dir: Path):
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=collection_name(recording_id),
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def _chroma_client(persist_dir: Path):
    import chromadb

    return chromadb.PersistentClient(path=str(persist_dir))


def count_chunks(recording_id: str, persist_dir: Path = CHROMA_DIR) -> int:
    """Number of stored chunks for this recording (0 if it has no index)."""
    from chromadb.errors import NotFoundError

    if not Path(persist_dir).exists():
        return 0
    try:
        return _chroma_client(persist_dir).get_collection(collection_name(recording_id)).count()
    except NotFoundError:
        return 0


def index_recording(recording_id: str, segments: list[Segment], embeddings,
                    persist_dir: Path = CHROMA_DIR, max_chars: int = CHUNK_CHARS) -> int:
    """(Re)build this recording's collection. Returns the number of chunks stored.

    Duplicate prevention has two layers:
      1. reset_collection() deletes and recreates the collection, so re-processing
         never leaves stale chunks behind (even if the new transcript is shorter);
      2. chunk IDs are deterministic ("<recording_id>:<index>") and LangChain's
         Chroma wrapper upserts, so adding the same chunk twice overwrites it.
    """
    docs = build_chunks(segments, recording_id, max_chars)
    if not docs:
        raise PipelineError("The transcript is empty, so there is nothing to index.")
    store = _open_store(recording_id, embeddings, persist_dir)
    store.reset_collection()
    store.add_documents(docs, ids=[f"{recording_id}:{d.metadata['chunk_index']}" for d in docs])
    return len(docs)


def delete_index(recording_id: str, persist_dir: Path = CHROMA_DIR) -> bool:
    """Delete only this recording's collection. Returns False if it did not exist."""
    from chromadb.errors import NotFoundError

    if not Path(persist_dir).exists():
        return False
    try:
        _chroma_client(persist_dir).delete_collection(collection_name(recording_id))
        return True
    except NotFoundError:
        return False


# ---------------------------------------------------------------------------
# 7. Question answering (evidence-backed RAG)
# ---------------------------------------------------------------------------

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about ONE recording using ONLY the numbered transcript passages "
     "below.\n"
     "Rules:\n"
     "1. Use only facts stated in the passages. Do not use outside knowledge.\n"
     "2. Cite the passages you used, like [1] or [2][3].\n"
     "3. If the passages do not contain the answer, reply with exactly this sentence and "
     f"nothing else: {NOT_FOUND_MESSAGE}\n"
     "4. Transcripts may contain speech-recognition errors; say so if a passage is unclear.\n"
     "5. Be concise.\n\n"
     "Passages:\n{context}"),
    ("human", "{question}"),
])


def answer_question(question: str, recording_id: str, llm, embeddings,
                    persist_dir: Path = CHROMA_DIR, k: int = TOP_K,
                    approximate: bool = False) -> Answer:
    """Retrieve the top-k passages from THIS recording's collection, then answer from them."""
    question = (question or "").strip()
    if not question:
        raise PipelineError("Please type a question.")
    if count_chunks(recording_id, persist_dir) == 0:
        return Answer(NOT_FOUND_MESSAGE + " (This recording has no index - run Analyze again.)", [], True)

    store = _open_store(recording_id, embeddings, persist_dir)
    hits = store.similarity_search_with_relevance_scores(question, k=k)
    if not hits:
        return Answer(NOT_FOUND_MESSAGE, [], True)

    evidence = [
        Evidence(n, doc.page_content, doc.metadata["start"], doc.metadata["end"], float(score))
        for n, (doc, score) in enumerate(hits, start=1)
    ]
    context = "\n\n".join(
        f"[{e.number}] ({format_span(e.start, e.end, approximate)}) {e.text}" for e in evidence
    )
    chain = ANSWER_PROMPT | llm | StrOutputParser()
    try:
        text = with_retries(lambda: chain.invoke({"context": context, "question": question})).strip()
    except Exception as exc:
        raise PipelineError(_describe_api_error("Mistral", exc)) from exc
    not_found = _normalize(NOT_FOUND_MESSAGE) in _normalize(text)
    return Answer(text, evidence, not_found)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_recording(source: str | UploadedMedia, engine: str, settings: Settings, llm, embeddings,
                      progress: Optional[ProgressFn] = None,
                      workspace: Path = WORKSPACE_DIR, persist_dir: Path = CHROMA_DIR) -> ProcessedRecording:
    """Run stages 1-6 for one recording. Temporary audio is always deleted afterwards."""
    check_requirements(settings, engine)
    recording_id, source_title = identify_source(source)

    workdir = workspace / recording_id
    shutil.rmtree(workdir, ignore_errors=True)       # leftovers from an interrupted run
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        if isinstance(source, UploadedMedia):
            raw = workdir / f"source{Path(source.name).suffix.lower()}"
            raw.write_bytes(source.data)
        else:
            _notify(progress, "Downloading audio from YouTube")
            raw, source_title = download_youtube_audio(source, workdir)

        _notify(progress, "Converting audio to 16 kHz mono WAV")
        wav = normalize_audio(raw, workdir / "audio.wav")
        duration = wav_duration(wav)
        raw.unlink(missing_ok=True)

        segments = transcribe(wav, engine, settings, progress)
    finally:
        try:
            shutil.rmtree(workdir)
        except OSError as exc:
            logger.warning("Could not delete temporary folder %s: %s", workdir, exc)

    transcript = Transcript(recording_id, source_title, engine, segments, duration)
    try:
        report = analyze(segments, llm, transcript.approximate_timestamps, progress)
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(_describe_api_error("Mistral", exc)) from exc

    _notify(progress, "Building the search index")
    chunk_count = index_recording(recording_id, segments, embeddings, persist_dir)
    return ProcessedRecording(transcript, report, chunk_count)


# ---------------------------------------------------------------------------
# 8. Export
# ---------------------------------------------------------------------------

def format_timestamp(seconds: float, always_hours: bool = False) -> str:
    total = int(round(max(seconds, 0)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if (h or always_hours) else f"{m:02d}:{s:02d}"


def format_span(start: float, end: float, approximate: bool = False) -> str:
    span = f"{format_timestamp(start)}-{format_timestamp(end)}"
    return f"~{span}" if approximate else span


def timestamp_note(transcript: Transcript) -> str:
    if transcript.approximate_timestamps:
        return ("Timestamps are approximate: Sarvam AI returns text without timestamps, so each "
                "line is placed at the 25-second audio piece it came from.")
    return "Timestamps are Whisper segment times (typically a few seconds of speech each)."


def transcript_to_text(transcript: Transcript) -> str:
    lines = [
        f"Transcript: {transcript.source_title}",
        f"Engine: {ENGINE_LABELS.get(transcript.engine, transcript.engine)}",
        f"Duration: {format_timestamp(transcript.duration, always_hours=True)}",
        timestamp_note(transcript),
        "",
    ]
    prefix = "~" if transcript.approximate_timestamps else ""
    for seg in transcript.segments:
        lines.append(f"[{prefix}{format_timestamp(seg.start, True)} - "
                     f"{format_timestamp(seg.end, True)}] {seg.text}")
    return "\n".join(lines) + "\n"


def _finding_markdown(f: Finding, approximate: bool) -> str:
    extras = []
    if f.owner:
        extras.append(f"owner: {f.owner}")
    if f.deadline:
        extras.append(f"deadline: {f.deadline}")
    extra = f" ({'; '.join(extras)})" if extras else ""
    if f.quote_found:
        check = "quote found in transcript"
    else:
        check = "QUOTE NOT FOUND in transcript - treat with caution; time is the section range"
    return (f"- {f.text}{extra}\n"
            f"  - `{format_span(f.start, f.end, approximate)}` - {check}\n"
            f"  - > \"{f.quote}\"")


def report_to_markdown(transcript: Transcript, report: Report) -> str:
    approx = transcript.approximate_timestamps
    out = [
        f"# {report.title}",
        "",
        f"- **Source:** {transcript.source_title}",
        f"- **Transcription:** {ENGINE_LABELS.get(transcript.engine, transcript.engine)}",
        f"- **Duration:** {format_timestamp(transcript.duration, always_hours=True)}",
        f"- **Timestamps:** {timestamp_note(transcript)}",
        "",
        "## Summary",
        *[f"- {p}" for p in report.summary],
    ]
    for heading, items in (("Decisions", report.decisions),
                           ("Action items", report.action_items),
                           ("Open questions", report.open_questions)):
        out += ["", f"## {heading}"]
        out += [_finding_markdown(f, approx) for f in items] or ["- None found."]
    out += [
        "",
        "---",
        "*Generated by an AI system from an automatic transcript. \"Quote found\" means the "
        "supporting words appear verbatim in the transcript; it does not prove the item was "
        "interpreted correctly. Check the timestamp in the original recording before acting on it.*",
    ]
    return "\n".join(out) + "\n"
