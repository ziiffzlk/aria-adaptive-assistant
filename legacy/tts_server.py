"""
ARIA Desktop - TTS Server (subprocess)
Runs a hidden pywebview window hosting Puter.js Gemini TTS, entirely in its
own process. This exists because pywebview's event loop refuses to run on
anything but the main thread, and CustomTkinter's mainloop already owns the
main thread of the primary ARIA process - the only way to use both is to give
pywebview a process of its own.

Protocol: the parent process writes one line per utterance to this script's
stdin, formatted "mood|text". This script speaks it via Puter's Gemini TTS.
Readiness is tracked client-side via document.title (LOADING -> READY/FAILED)
and reported to the parent over stdout. Closing stdin (parent shutdown)
causes a clean exit.
"""

import json
import sys
import time
import threading

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import webview

print("TTS Server starting...", flush=True)

TTS_HTML = """
<!DOCTYPE html>
<html>
<head>
<script src="https://js.puter.com/v2/"></script>
</head>
<body>
<script>
const moodInstructions = {
    stressed: "Speak slowly and very calmly. Be soothing and grounding.",
    anxious: "Speak gently and slowly. Be reassuring and steady.",
    sad: "Speak softly and gently. Warm and compassionate.",
    happy: "Speak with warmth and lightness. Slightly upbeat.",
    excited: "Speak with energy and enthusiasm. Lively.",
    frustrated: "Speak calmly and patiently. Be understanding.",
    calm: "Speak naturally and warmly like a trusted assistant.",
    tired: "Speak softly and gently. Very calm and soothing.",
    surprised: "Speak with warm curiosity and engagement.",
    default: "Speak naturally and warmly like a helpful assistant."
};

// speakText is intentionally a plain (non-async) function: calling an async
// function via evaluate_js() with no callback just synchronously serializes
// the pending Promise itself (-> "{}"), and the callback-based Promise
// bridge turned out to be unreliable for this hidden window in testing.
// Instead, the result is stashed in window.__speakResult, which Python polls
// with the same plain evaluate_js() pattern already used for readiness checks.
window.speakText = function(text, mood){
    window.__speakResult = null;
    window.__speakError = null;
    (async () => {
        try{
            const instruction = moodInstructions[mood] || moodInstructions.default;
            const audio = await puter.ai.txt2speech(text, {
                provider: "gemini",
                model: "gemini-2.5-flash-preview-tts",
                voice: "Aoede",
                instructions: instruction
            });
            audio.onended = () => { window.__speakResult = "done"; };
            audio.onerror = () => { window.__speakResult = "error"; window.__speakError = "audio playback error"; };
            audio.play();
        }catch(e){
            console.error(e);
            window.__speakResult = "error";
            window.__speakError = (e && (e.message || e.toString())) || "unknown error";
        }
    })();
}

window.addEventListener('load', function(){
    document.title = 'LOADING';

    let puterCheckInterval = setInterval(function(){
        if(typeof puter !== 'undefined' && puter.ai){
            document.title = 'READY';
            console.log('Puter ready');
            clearInterval(puterCheckInterval);
        }
    }, 500);

    setTimeout(function(){
        if(document.title !== 'READY'){
            document.title = 'FAILED';
        }
    }, 15000);
});
</script>
</body>
</html>
"""

_window = None


def _wait_until_ready_then_signal():
    """Poll document.title until Puter.js reports READY/FAILED, signal the parent over stdout."""
    deadline = time.time() + 20
    last_title = None
    while time.time() < deadline:
        try:
            title = _window.evaluate_js("document.title")
        except Exception as e:
            print(f"ERROR: could not read document.title: {e}", file=sys.stderr, flush=True)
            title = None

        if title != last_title:
            last_title = title

        if title == "READY":
            print("Puter.js loaded", flush=True)
            print("READY", flush=True)
            return
        if title == "FAILED":
            print("Puter.js failed to load (client-side timeout)", flush=True)
            print("FAILED", flush=True)
            return
        time.sleep(0.2)

    print("ERROR: Puter.js did not become ready within 20s", file=sys.stderr, flush=True)
    print("FAILED", flush=True)


def _read_stdin_commands():
    """Read 'mood|text' lines from stdin and speak each one via Puter.js."""
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line or "|" not in line:
            continue
        mood, _, text = line.partition("|")
        if not text:
            continue

        print(f"Speaking: {text[:30]}", flush=True)
        try:
            # json.dumps safely escapes quotes/backticks/newlines for JS string literals.
            js = f"speakText({json.dumps(text)}, {json.dumps(mood)})"
            _window.evaluate_js(js)

            # Poll window.__speakResult (plain synchronous evaluate_js - the
            # same pattern already proven reliable for readiness checks)
            # rather than relying on pywebview's async-callback bridge, which
            # testing showed never fires for this hidden window.
            result = None
            deadline = time.time() + 30
            while time.time() < deadline:
                result = _window.evaluate_js("window.__speakResult")
                if result is not None:
                    break
                time.sleep(0.2)

            if result == "done":
                print("Puter.js success", flush=True)
            elif result is None:
                print("Puter.js failed (timed out waiting for playback to resolve)", flush=True)
            else:
                error_detail = None
                try:
                    error_detail = _window.evaluate_js("window.__speakError")
                except Exception:
                    pass
                print(f"Puter.js failed (result={result!r}, error={error_detail!r})", flush=True)
        except Exception as e:
            print("Puter.js failed", flush=True)
            print(f"ERROR: speak failed: {e}", file=sys.stderr, flush=True)

    # stdin closed -> parent is shutting down -> close the window so this process can exit
    try:
        _window.destroy()
    except Exception:
        pass


def main():
    global _window
    _window = webview.create_window(
        "ARIA TTS", html=TTS_HTML, width=1, height=1, x=-200, y=-200, hidden=True
    )
    print("Window created", flush=True)

    def _on_started():
        # Starting these threads here (rather than before webview.start())
        # matters: evaluate_js calls made before the native window actually
        # exists fail with "Main window failed to start" - this callback only
        # fires once pywebview's GUI loop has truly finished starting it.
        threading.Thread(target=_wait_until_ready_then_signal, daemon=True).start()
        threading.Thread(target=_read_stdin_commands, daemon=True).start()

    webview.start(_on_started)


if __name__ == "__main__":
    main()
