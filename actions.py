import os
import subprocess
import webbrowser
import urllib.parse

class UniversalActions:
    @staticmethod
    def media(query: str) -> str:
        """Autonomously launches media via Spotify desktop URI or YouTube browser stream."""
        encoded = urllib.parse.quote(query.strip())
        try:
            os.system(f"start spotify:search:{encoded}")
        except Exception:
            webbrowser.open(f"https://www.youtube.com/results?search_query={encoded}")
        return f"Dispatched media: {query}"

    @staticmethod
    def launch(target: str) -> str:
        """Dynamically launches any application, executable, local file path, or URL."""
        target_clean = target.strip().lower()
        try:
            subprocess.Popen(target_clean, shell=True)
            return f"Launched {target}"
        except Exception as e:
            return f"Launch failed for {target}: {e}"

    @staticmethod
    def search(query: str) -> str:
        """Performs real-time web lookups."""
        encoded = urllib.parse.quote(query.strip())
        webbrowser.open(f"https://www.google.com/search?q={encoded}")
        return f"Searched: {query}"

    @staticmethod
    def execute(command: str) -> str:
        """Executes native OS commands in a detached subprocess."""
        try:
            subprocess.Popen(command, shell=True)
            return f"Executed: {command}"
        except Exception as e:
            return f"Execution error: {e}"

ACTION_REGISTRY = {
    "media": UniversalActions.media,
    "launch": UniversalActions.launch,
    "search": UniversalActions.search,
    "execute": UniversalActions.execute,
}
