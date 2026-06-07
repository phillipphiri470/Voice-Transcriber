# RSA AI Projects — Getting Started Guide

Two AI projects for the Road Safety Agency of Zambia.

---

## Project 1 — ANPR Overspeeding Detection & Auto-Email System

### Folder: `project1_anpr/`

### What it does
1. Captures a vehicle image from a camera or video file
2. Detects the number plate region using OpenCV
3. Reads the plate text using EasyOCR
4. Checks if the measured speed exceeds the configured limit
5. Looks up the vehicle owner in a SQLite database
6. Sends an HTML violation email with the capture image attached
7. Provides a live web dashboard for RSA staff

### Quick start

```bash
cd project1_anpr

# 1. Install dependencies
pip install -r requirements.txt

# 2. Edit config.py — set your email credentials and speed limit

# 3. Run on a test image (set a simulated speed)
python main.py path/to/vehicle_photo.jpg 95

# 4. Run on live webcam (speed is simulated — wire up a real sensor later)
python main.py

# 5. Start the web dashboard
uvicorn dashboard:app --reload
# Open http://localhost:8000
```

### File structure
```
project1_anpr/
  main.py        ← Core pipeline: detect plate → check speed → send email
  database.py    ← Vehicle registry + offence logging (SQLite/PostgreSQL)
  emailer.py     ← HTML email composer and sender
  dashboard.py   ← FastAPI web dashboard
  config.py      ← Speed limits, email credentials, camera settings
  requirements.txt
  captures/      ← Auto-created — violation images saved here
  rsa.db         ← Auto-created — SQLite database
```

### Connecting a real speed sensor
In `main.py`, replace `simulated_speed` with a live reading:
```python
import serial
port = serial.Serial('/dev/ttyUSB0', 9600)  # adjust port
speed = float(port.readline().decode().strip())
```

### Improving plate accuracy
The starter code uses OpenCV's built-in Haar cascade which works on clear images.
For better real-world accuracy, switch to YOLOv8:
```bash
pip install ultralytics
# Then replace detect_plate_region() in main.py with a YOLOv8 model
# trained on Zambian plate formats (GRZ, BAA, ABA etc.)
```

---

## Project 2 — Speech Transcription System

### Folder: `project2_transcription/`

### What it does
1. Accepts an audio/video file or live microphone input
2. Applies optional noise reduction
3. Transcribes speech using OpenAI Whisper (runs fully offline)
4. Optionally identifies and labels different speakers
5. Exports the transcript to TXT, Word (.docx), or subtitle (.srt) format
6. Provides a drag-and-drop Streamlit web interface

### Quick start

```bash
cd project2_transcription

# 1. Install dependencies
pip install -r requirements.txt

# 2a. Transcribe a file (command line)
python transcriber.py --file speech.mp3

# 2b. Transcribe with a larger model for better accuracy
python transcriber.py --file meeting.wav --model small

# 2c. Export as Word document
python transcriber.py --file speech.mp3 --export docx

# 2d. Record from microphone
python transcriber.py --live

# 3. Start the web UI (easier for non-technical users)
streamlit run app.py
# Open http://localhost:8501
```

### File structure
```
project2_transcription/
  transcriber.py  ← CLI entry point with all options
  core.py         ← TranscriptionEngine: Whisper + noise reduction + diarisation
  exporter.py     ← TXT, DOCX, SRT export handlers
  app.py          ← Streamlit web UI for file upload and download
  requirements.txt
```

### Model size guide
| Model  | Size    | Speed (CPU) | Accuracy | Use case                    |
|--------|---------|-------------|----------|-----------------------------|
| tiny   | 75 MB   | Very fast   | Basic    | Quick drafts                |
| base   | 145 MB  | Fast        | Good     | Most speeches (recommended) |
| small  | 465 MB  | Moderate    | Better   | Important meetings          |
| medium | 1.5 GB  | Slow        | High     | Multi-speaker recordings    |
| large  | 3 GB    | Very slow   | Best     | Critical transcriptions     |

### Enabling speaker identification
Speaker diarisation (labelling Speaker 1, Speaker 2 etc.) requires a free
HuggingFace token:
1. Sign up at https://huggingface.co
2. Go to Settings → Access Tokens → New token
3. Accept the pyannote model terms at https://hf.co/pyannote/speaker-diarization-3.1
4. Run:
```bash
pip install pyannote.audio torch
export HUGGINGFACE_TOKEN=hf_your_token_here
python transcriber.py --file meeting.mp3 --speakers
```

---

## Recommended development order

**Project 1:**
1. Get OCR working on a saved photo first (`python main.py photo.jpg 95`)
2. Add sample vehicles to the database and confirm email sending works
3. Connect the live camera feed
4. Wire up the real speed sensor
5. Deploy the dashboard on a server

**Project 2:**
1. Run Whisper on a short audio clip to confirm it works
2. Try the Streamlit UI (`streamlit run app.py`)
3. Test different model sizes on real RSA recordings
4. Add speaker identification if needed for meeting transcripts
