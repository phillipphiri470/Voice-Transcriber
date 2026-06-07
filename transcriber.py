"""
Road Safety Agency — Speech Transcription System
=================================================
Transcribes speeches and presentations from audio files or live microphone input.
Supports export to plain text, Word (.docx), and subtitle (.srt) formats.

Dependencies:
    pip install -r requirements.txt

    For Whisper model download (first run downloads ~140MB for 'base'):
    It downloads automatically — just needs internet access once.

Usage examples:
    python transcriber.py --file speech.mp3
    python transcriber.py --file meeting.wav --model large --speakers
    python transcriber.py --live
    python transcriber.py --file speech.mp3 --export docx
"""

import argparse
import logging
import sys
from pathlib import Path

from core import TranscriptionEngine
from exporter import export_transcript

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="RSA Speech Transcription System")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=str, help="Path to an audio or video file to transcribe")
    source.add_argument("--live", action="store_true", help="Record from microphone and transcribe")

    parser.add_argument(
        "--model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size. Larger = more accurate but slower. Default: base",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code, e.g. 'en' for English, 'fr' for French. Default: en",
    )
    parser.add_argument(
        "--speakers",
        action="store_true",
        help="Identify and label different speakers (requires pyannote.audio)",
    )
    parser.add_argument(
        "--export",
        choices=["txt", "docx", "srt", "all"],
        default="txt",
        help="Output format. Default: txt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (without extension). Default: same as input filename",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        default=True,
        help="Apply noise reduction before transcription (default: on)",
    )

    args = parser.parse_args()

    engine = TranscriptionEngine(
        model_size=args.model,
        language=args.language,
        denoise=args.denoise,
        identify_speakers=args.speakers,
    )

    if args.live:
        log.info("Starting live microphone transcription. Press Ctrl+C to stop.")
        result = engine.transcribe_live()
    else:
        audio_path = Path(args.file)
        if not audio_path.exists():
            log.error(f"File not found: {args.file}")
            sys.exit(1)
        log.info(f"Transcribing: {audio_path.name}")
        result = engine.transcribe_file(str(audio_path))

    if result is None:
        log.error("Transcription failed.")
        sys.exit(1)

    # Determine output base name
    output_base = args.output
    if output_base is None:
        if args.live:
            from datetime import datetime
            output_base = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            output_base = Path(args.file).stem

    # Export
    formats = ["txt", "docx", "srt"] if args.export == "all" else [args.export]
    for fmt in formats:
        out_path = export_transcript(result, output_base, fmt)
        log.info(f"Saved: {out_path}")

    # Print summary to console
    print("\n" + "=" * 60)
    print(f"  Transcription complete")
    print(f"  Duration  : {result['duration_seconds']:.1f} seconds")
    print(f"  Segments  : {len(result['segments'])}")
    print(f"  Language  : {result['language']}")
    print("=" * 60)
    print("\nTranscript preview (first 500 chars):\n")
    print(result["full_text"][:500])
    if len(result["full_text"]) > 500:
        print("  ...")
    print()


if __name__ == "__main__":
    main()
