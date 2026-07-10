"""
Pre-download the Chatterbox TTS model weights from HuggingFace.
Run once before starting main_desktop.py:  python download_chatterbox.py
"""
from huggingface_hub import snapshot_download

print("Downloading ResembleAI/chatterbox model (~600 MB)…")
print("This only runs once; files are cached for future launches.\n")

path = snapshot_download(
    repo_id="ResembleAI/chatterbox",
    ignore_patterns=["*.md", "*.txt", "*.py", "*.gitattributes"],
)

print(f"\nModel cached at: {path}")
print("You can now launch:  python main_desktop.py")
