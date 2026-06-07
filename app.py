"""
app.py — Streamlit web UI for the Speech Transcription System.

Run with:
    streamlit run app.py

Lets users upload an audio file, pick a Whisper model, and download
the transcript in TXT, DOCX, or SRT format — no command line needed.
"""

import tempfile
from pathlib import Path

import streamlit as st

from core import TranscriptionEngine
from exporter import export_transcript

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RSA Speech Transcription",
    page_icon="🎙️",
    layout="centered",
)

st.title("🎙️ Speech Transcription System")
st.caption("Road Safety Agency — Meeting & Speech Transcription Tool")

# ── Sidebar settings ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    model_size = st.selectbox(
        "Whisper model",
        ["tiny", "base", "small", "medium", "large"],
        index=1,
        help="Larger models are more accurate but slower. 'base' is good for most speeches.",
    )
    language = st.selectbox(
        "Language",
        ["en", "fr", "pt", "sw", "ny", "auto"],
        index=0,
        help="Select 'auto' to let Whisper detect the language automatically.",
    )
    denoise = st.checkbox("Noise reduction", value=True, help="Recommended for hall/outdoor recordings.")
    export_fmt = st.selectbox("Export format", ["txt", "docx", "srt", "all"], index=0)

    st.divider()
    st.markdown("**Model sizes guide**")
    st.markdown("- `tiny` — fastest, lower accuracy  \n- `base` — good balance  \n- `small` / `medium` — better accuracy  \n- `large` — best quality, needs GPU")

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload audio or video file",
    type=["mp3", "wav", "m4a", "ogg", "flac", "mp4", "webm"],
    help="Supports MP3, WAV, M4A, OGG, FLAC, MP4, WebM.",
)

if uploaded is None:
    st.info("👆 Upload an audio file to get started.")
    st.stop()

# ── Transcribe ────────────────────────────────────────────────────────────────
st.audio(uploaded)

transcribe_btn = st.button("▶ Transcribe", type="primary", use_container_width=True)

if transcribe_btn:
    # Save upload to a temp file
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner(f"Loading Whisper '{model_size}' model and transcribing…"):
        lang = None if language == "auto" else language
        engine = TranscriptionEngine(
            model_size=model_size,
            language=lang,
            denoise=denoise,
            identify_speakers=False,
        )
        result = engine.transcribe_file(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)

    if result is None:
        st.error("Transcription failed. Please try a different file or model.")
        st.stop()

    # ── Show results ──────────────────────────────────────────────────────────
    st.success(f"Done! Duration: {result['duration_seconds']:.1f}s | Language detected: {result['language'].upper()}")

    st.subheader("📝 Transcript")
    st.text_area("Full text", result["full_text"], height=300)

    with st.expander("View timestamped segments"):
        for seg in result["segments"]:
            mins = int(seg["start"] // 60)
            secs = seg["start"] % 60
            st.markdown(f"`{mins:02d}:{secs:05.2f}` {seg['text']}")

    # ── Export & download ─────────────────────────────────────────────────────
    st.subheader("⬇️ Download transcript")
    base_name = Path(uploaded.name).stem
    formats = ["txt", "docx", "srt"] if export_fmt == "all" else [export_fmt]

    with tempfile.TemporaryDirectory() as tmpdir:
        for fmt in formats:
            out_path = export_transcript(result, f"{tmpdir}/{base_name}", fmt)
            with open(out_path, "rb") as f:
                st.download_button(
                    label=f"Download .{fmt}",
                    data=f.read(),
                    file_name=f"{base_name}_transcript.{fmt}",
                    mime={
                        "txt":  "text/plain",
                        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        "srt":  "text/plain",
                    }.get(fmt, "application/octet-stream"),
                )
