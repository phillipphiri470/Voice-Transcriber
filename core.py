"""
core.py — TranscriptionEngine: loads Whisper, runs pre-processing,
transcribes files or live audio, and optionally identifies speakers.
"""

import logging
import tempfile
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Lazy imports — only loaded when needed so the app starts fast
_whisper     = None
_noisereduce = None
_pyannote    = None


def _get_whisper():
    global _whisper
    if _whisper is None:
        import whisper
        _whisper = whisper
    return _whisper


def _get_noisereduce():
    global _noisereduce
    if _noisereduce is None:
        try:
            import noisereduce as nr
            _noisereduce = nr
        except ImportError:
            log.warning("noisereduce not installed — skipping noise reduction. pip install noisereduce")
    return _noisereduce


class TranscriptionEngine:
    """
    Wraps OpenAI Whisper with noise reduction and optional speaker diarisation.

    Parameters
    ----------
    model_size : str
        Whisper model size: tiny | base | small | medium | large
        'base' is a good starting point — accurate and fast on a laptop CPU.
    language : str
        BCP-47 language code, e.g. 'en'. Pass None to auto-detect.
    denoise : bool
        Whether to apply noisereduce before transcribing.
    identify_speakers : bool
        Whether to run pyannote.audio speaker diarisation.
    """

    def __init__(
        self,
        model_size: str = "base",
        language: str = "en",
        denoise: bool = True,
        identify_speakers: bool = False,
    ):
        self.language = language
        self.denoise  = denoise
        self.identify_speakers = identify_speakers

        log.info(f"Loading Whisper '{model_size}' model (downloads on first use)...")
        whisper = _get_whisper()
        self.model = whisper.load_model(model_size)
        log.info("Whisper model ready.")

        if identify_speakers:
            self._load_diarisation_pipeline()

    # ------------------------------------------------------------------
    # Speaker diarisation (optional)
    # ------------------------------------------------------------------

    def _load_diarisation_pipeline(self):
        """
        Load pyannote.audio for speaker diarisation.
        Requires a free HuggingFace token — see README for setup instructions.
        """
        try:
            from pyannote.audio import Pipeline
            import os
            token = os.getenv("HUGGINGFACE_TOKEN")
            if not token:
                log.warning(
                    "HUGGINGFACE_TOKEN not set. Speaker identification disabled.\n"
                    "Get a free token at https://huggingface.co/settings/tokens\n"
                    "Then: export HUGGINGFACE_TOKEN=hf_your_token_here"
                )
                self.identify_speakers = False
                return
            self.diarisation_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token,
            )
            log.info("Speaker diarisation pipeline loaded.")
        except ImportError:
            log.warning("pyannote.audio not installed. pip install pyannote.audio")
            self.identify_speakers = False

    # ------------------------------------------------------------------
    # Audio pre-processing
    # ------------------------------------------------------------------

    def _load_audio(self, audio_path: str) -> tuple:
        """Load audio using ffmpeg — supports M4A, MP3, WAV, FLAC, OGG, MP4, etc."""
        import subprocess
        import shutil
        import os

        RATE = 16000

        ffmpeg_exe = shutil.which("ffmpeg")
        if ffmpeg_exe is None:
            candidates = [
                r"C:\ffmpeg\ffmpeg.exe",
                r"C:\ffmpeg\bin\ffmpeg.exe",
                r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
                r"C:\tools\ffmpeg\bin\ffmpeg.exe",
                r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
            ]
            for path in candidates:
                if os.path.isfile(path):
                    ffmpeg_exe = path
                    break

        if ffmpeg_exe is None:
            raise RuntimeError(
                "ffmpeg not found. Download from https://www.gyan.dev/ffmpeg/builds/ "
                "and add its bin folder to your system PATH, or place ffmpeg.exe at C:\\ffmpeg\\ffmpeg.exe"
            )

        cmd = [
            ffmpeg_exe,
            "-i", audio_path,
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ac", "1",
            "-ar", str(RATE),
            "-",
            "-loglevel", "error",
            "-nostdin",
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {result.stderr.decode()}")

        samples = np.frombuffer(result.stdout, dtype=np.float32)
        return samples, RATE

    def _preprocess_audio(self, samples: np.ndarray, sr: int) -> np.ndarray:
        """Apply noise reduction if enabled."""
        if not self.denoise:
            return samples
        nr = _get_noisereduce()
        if nr is None:
            return samples
        log.info("Applying noise reduction...")
        cleaned = nr.reduce_noise(y=samples, sr=sr, stationary=False)
        return cleaned

    # ------------------------------------------------------------------
    # Transcription — from file
    # ------------------------------------------------------------------
    
    def transcribe_file(self, audio_path: str) -> dict | None:
        try:
            log.info(f"Step 1: loading audio from {audio_path}")
            samples, sr = self._load_audio(audio_path)
            log.info(f"Step 2: loaded {len(samples)} samples at {sr}Hz")
    
            samples = self._preprocess_audio(samples, sr)
            log.info("Step 3: preprocessing done")
    
            import soundfile as sf
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, samples, sr)
            log.info(f"Step 4: wrote temp WAV to {tmp_path}")
    
            log.info("Step 5: running Whisper...")
            result = self.model.transcribe(
                tmp_path,
                language=self.language,
                verbose=False,
                word_timestamps=True,
                fp16=False,
            )
            log.info("Step 6: Whisper done")
    
            Path(tmp_path).unlink(missing_ok=True)
            output = self._build_output(result, samples, sr)
            return output
    
        except Exception as e:
            import traceback
            log.error(f"Transcription error: {e}")
            log.error(traceback.format_exc())
            return None
    # ------------------------------------------------------------------
    # Transcription — live microphone
    # ------------------------------------------------------------------

    def transcribe_live(self, duration_seconds: int = 0) -> dict | None:
        """
        Record from the microphone and transcribe.
        If duration_seconds=0, records until Ctrl+C is pressed.
        """
        try:
            import pyaudio
        except ImportError:
            log.error("pyaudio not installed. pip install pyaudio")
            return None

        RATE     = 16000
        CHANNELS = 1
        CHUNK    = 1024
        FORMAT   = pyaudio.paInt16

        pa     = pyaudio.PyAudio()
        stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        frames = []
        log.info("Recording… Press Ctrl+C to stop and transcribe.")

        try:
            if duration_seconds > 0:
                total_chunks = int(RATE / CHUNK * duration_seconds)
                for _ in range(total_chunks):
                    frames.append(stream.read(CHUNK, exception_on_overflow=False))
            else:
                while True:
                    frames.append(stream.read(CHUNK, exception_on_overflow=False))
        except KeyboardInterrupt:
            log.info("Recording stopped.")
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            with wave.open(tmp.name, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(pa.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b"".join(frames))
            tmp_path = tmp.name

        log.info("Transcribing recorded audio...")
        result = self.transcribe_file(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)
        return result

    # ------------------------------------------------------------------
    # Output builder
    # ------------------------------------------------------------------

    def _build_output(self, whisper_result: dict, samples: np.ndarray, sr: int) -> dict:
        """Convert raw Whisper output into a clean structured result."""
        segments = []
        for seg in whisper_result.get("segments", []):
            segments.append({
                "start":   round(seg["start"], 2),
                "end":     round(seg["end"],   2),
                "text":    seg["text"].strip(),
                "speaker": None,
            })

        full_text = " ".join(s["text"] for s in segments)

        return {
            "full_text":        full_text,
            "segments":         segments,
            "language":         whisper_result.get("language", "unknown"),
            "duration_seconds": len(samples) / sr if sr > 0 else 0,
        }

    # ------------------------------------------------------------------
    # Speaker diarisation
    # ------------------------------------------------------------------

    def _add_speaker_labels(self, output: dict, audio_path: str) -> dict:
        """Run pyannote diarisation and annotate segments with speaker labels."""
        if not hasattr(self, "diarisation_pipeline"):
            return output
        try:
            log.info("Running speaker diarisation...")
            diarisation = self.diarisation_pipeline(audio_path)
            for seg in output["segments"]:
                seg_start, seg_end = seg["start"], seg["end"]
                best_speaker = None
                best_overlap = 0.0
                for turn, _, speaker in diarisation.itertracks(yield_label=True):
                    overlap = min(turn.end, seg_end) - max(turn.start, seg_start)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_speaker = speaker
                seg["speaker"] = best_speaker
        except Exception as e:
            log.error(f"Speaker diarisation failed: {e}")
        return output