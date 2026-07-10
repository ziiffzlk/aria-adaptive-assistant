# ARIA

ARIA is a desktop AI assistant that learns how *you* specifically communicate
and adapts to it over time, rather than treating every user the same. It
fuses voice tone, facial expression, and conversation text to estimate mood
in real time, builds a personal behavioural profile from usage history
(frequent topics, active hours, recurring patterns), and uses that profile to
shape both *what* it offers and *how* it responds — not just a fixed tone
overlay on identical answers. Built as a final-year Computer Science project.

## Key features

- **Continuous voice conversation** — always-listening microphone pipeline
  (webrtcvad + energy gating) with real-time barge-in: talk over ARIA
  mid-sentence and she stops immediately, no wake word required by default.
- **Multimodal mood detection** — fuses voice acoustics (pitch, speaking
  speed, pauses via librosa), facial expression (mediapipe face landmarks),
  and text sentiment into a single mood estimate, refined by a personal KNN
  classifier trained on the user's own labelled history.
- **Behavioural pattern learning** — tracks frequent topics, app usage, and
  timing patterns from real conversation history, and uses them to
  personalize suggestions (e.g. YouTube recommendations tied to actual
  tracked interests, not generic content).
- **Action-oriented, mood-aware responses** — ARIA's replies change what she
  offers to do based on detected mood and known interests, not only her tone
  of voice.
- **Local + cloud hybrid** — conversational responses via the Groq API
  (Llama models); speech-to-text, text-to-speech, and face/mood analysis all
  run locally.
- **Academic evaluation suite** — scripts to measure real accuracy,
  confusion matrices, per-modality comparison, and learning-curve
  improvement against the user's actual database, for defensible reporting
  rather than hypothetical claims.

## Tech stack

- **App shell**: Python, Flask, [pywebview](https://pywebview.flowrl.com/)
  (borderless desktop window over a local web UI)
- **LLM**: [Groq API](https://console.groq.com/) (Llama models)
- **Speech-to-text**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
  Google Speech Recognition fallback
- **Voice activity detection**: webrtcvad, with ambient-noise-calibrated
  energy gating on top
- **Text-to-speech**: [Chatterbox-TTS](https://github.com/resemble-ai/chatterbox)
  (voice cloning) with [Kokoro](https://github.com/thewh1teagle/kokoro-onnx)
  (offline ONNX) as fallback
- **Face / mood analysis**: mediapipe, OpenCV
- **Voice feature analysis**: librosa
- **Personal mood classifier**: scikit-learn (KNN)
- **Storage**: SQLite

## Setup

```powershell
# 1. Clone
git clone <this-repo-url>
cd aria-desktop

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
.venv\Scripts\pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env
# then edit .env and add your own Groq API key
# (free at https://console.groq.com/keys)

# 5. Run
.venv\Scripts\python.exe main_desktop.py
```

On first run, ARIA downloads the TTS/STT model weights it needs (Kokoro,
faster-whisper, and optionally Chatterbox — see `download_chatterbox.py` to
pre-fetch Chatterbox separately). GPU acceleration (CUDA) is used
automatically where available, with CPU fallback otherwise.

## Project context

This is a final-year Computer Science project exploring adaptive,
behaviourally-personalized AI assistants — the focus is on genuine learning
from usage data (measured and evaluated against real interaction history),
not a scripted demo.
