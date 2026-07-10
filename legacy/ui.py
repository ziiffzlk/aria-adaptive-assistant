"""
ARIA Desktop - UI Module (PyQt6)

Premium rebuild: cool sophisticated dark palette, fluid 50-layer additive-blending
presence orb, premium serif captions, pyqtgraph mood chart in the memory window.
All PyQt6 safety patterns from prior builds are preserved.
"""

import math
import random
import threading
import time
import webbrowser
from collections import Counter
from datetime import datetime
from io import BytesIO
from urllib.parse import urlparse

from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QEvent, QRectF, QPointF,
    QPropertyAnimation, QEasingCurve,
)
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QRadialGradient, QLinearGradient,
    QImage, QPixmap, QFont, QPainterPath, QFontDatabase,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QFrame, QLineEdit, QPushButton,
    QCheckBox, QHBoxLayout, QVBoxLayout, QScrollArea,
    QGraphicsOpacityEffect, QGraphicsDropShadowEffect, QGraphicsBlurEffect,
)

import database
import brain
import patterns
import voice

try:
    import pyqtgraph as pg
    _HAS_PYQTGRAPH = True
except ImportError:
    _HAS_PYQTGRAPH = False

# ----------------------------------------------------------------------
# Palette  (cool sophisticated darks)
# ----------------------------------------------------------------------

BG_DEEP  = "#06050a"
BG       = "#0a0812"
ACCENT   = "#b09ee0"          # lavender
INK      = "#e8e4f2"          # near-white with slight violet warmth
MUTED    = "#6a6580"
ROSE     = "#c17b7b"
SUCCESS  = "#7eb89a"
WARN     = "#e8a87c"          # warm amber (used sparingly)

SURFACE_QCOLOR = QColor(255, 255, 255, 10)
BORDER_QCOLOR  = QColor(255, 255, 255, 15)

# Mood -> RGB tuple for presence orb and mood indicators
MOOD_RGB = {
    "calm":           (176, 158, 224),   # lavender (= ACCENT)
    "calm confident": (200, 180, 240),   # bright lavender
    "engaged":        (160, 180, 240),   # blue-lavender
    "happy":          (138, 200, 155),   # cool green
    "grateful":       (138, 200, 155),
    "excited":        (230, 195, 110),   # warm gold
    "surprised":      (230, 195, 110),
    "sad":            (90,  120, 210),   # deep blue
    "tired":          (130, 120, 175),   # dim purple
    "distracted":     (130, 120, 175),
    "stressed":       (200, 115, 135),   # rose
    "frustrated":     (200, 115, 135),
    "angry":          (200, 115, 135),
    "anxious":        (190, 105, 175),   # purple-rose
    "fearful":        (190, 105, 175),
    "fear":           (190, 105, 175),
}

MOOD_HEX = {mood: "#{:02x}{:02x}{:02x}".format(*rgb) for mood, rgb in MOOD_RGB.items()}

# Numeric encoding for pyqtgraph mood chart (0-10 scale, higher = more positive)
MOOD_NUMERIC = {
    "happy": 8, "excited": 9, "grateful": 8, "surprised": 7,
    "calm confident": 6, "engaged": 6, "calm": 5,
    "distracted": 4, "tired": 3, "anxious": 3,
    "sad": 2, "stressed": 2, "frustrated": 1, "angry": 0,
    "fearful": 1, "fear": 1,
}

PATTERN_DESCRIPTIONS = {
    "active_hour":         "Most active around {v}:00",
    "frequent_topic":      "Often talks about {v}",
    "dominant_mood":       "Usually feels {v}",
    "language_used":       "Speaks in {v}",
    "face_emotion_pattern":"Often appears {v}",
}

STATUS_DOT_COLORS = {
    "ready": SUCCESS, "thinking": ACCENT, "listening": ROSE, "speaking": ACCENT,
}
STATUS_TO_PRESENCE = {
    "ready": "idle", "listening": "listening", "thinking": "thinking", "speaking": "speaking",
}


def mood_rgb(mood):
    return MOOD_RGB.get(mood, MOOD_RGB["calm"])


# ----------------------------------------------------------------------
# Font loading
# Prefer Newsreader > Spectral > Playfair Display > Georgia (guaranteed on Windows).
# No network download at runtime — relies on system / previously installed fonts.
# ----------------------------------------------------------------------

_SERIF_FAMILY = "Georgia"   # default, resolved once at import time


def _resolve_serif_family():
    global _SERIF_FAMILY
    candidates = ["Newsreader", "Spectral", "Fraunces", "Playfair Display",
                  "EB Garamond", "Georgia", "Times New Roman"]
    available = set(QFontDatabase.families())
    for name in candidates:
        if name in available:
            _SERIF_FAMILY = name
            return


def serif_font(size, italic=True):
    font = QFont(_SERIF_FAMILY, size)
    font.setItalic(italic)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    return font


def ui_font(size, weight=QFont.Weight.Normal):
    font = QFont("Segoe UI", size)
    font.setWeight(weight)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    return font


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def antialiased_painter(widget):
    p = QPainter(widget)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    return p


# ----------------------------------------------------------------------
# PresenceWidget  — 50-layer fluid orb with CompositionMode_Plus additive
# blending on an intermediate buffer. The layers drift at independent
# frequencies; the center accumulates to full brightness while edges stay
# dark, giving a living, depth-rich glow that changes hue with mood.
# ----------------------------------------------------------------------

_N_ORB_LAYERS  = 50
_PRESENCE_RATE = 0.55   # radians per real second (smooth, not rushed)


class PresenceWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(560, 560)
        # No global blur effect — softness comes from gradient falloff + many layers.
        # (A global QGraphicsBlurEffect added on top would also work but isn't needed.)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._state = "idle"
        self._mood  = "calm"
        self._phase = 0.0
        self._last_tick   = time.time()
        self._flash_energy    = 0.0
        self._mic_energy_phase = 0.0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)

    def start(self):
        if not self._timer.isActive():
            self._last_tick = time.time()
            self._timer.start(33)   # ~30 fps

    def stop(self):
        self._timer.stop()

    def set_state(self, state):
        self._state = state if state in ("idle", "listening", "thinking", "speaking") else "idle"

    def set_mood(self, mood):
        self._mood = mood or "calm"

    def trigger_flash(self):
        self._flash_energy = 1.0

    def _animate(self):
        try:
            now = time.time()
            dt  = now - self._last_tick
            self._last_tick = now
            self._phase           += dt * _PRESENCE_RATE
            self._flash_energy    *= 0.91
            self._mic_energy_phase += dt
            self.update()
        except Exception as e:
            print(f"[ui] presence animation error: {e}")

    def _energy(self):
        s = self._state
        t = self._phase
        flash = self._flash_energy
        mic   = 0.35 + 0.28 * abs(math.sin(self._mic_energy_phase * 2.3)) \
                      + 0.15 * abs(math.sin(self._mic_energy_phase * 5.1))
        if s == "listening":
            return 0.32 + mic * 0.7 + math.sin(t * 1.4) * 0.04
        if s == "speaking":
            return 0.38 + flash * 0.6 + abs(math.sin(t * 2.0)) * 0.18
        if s == "thinking":
            return 0.26 + math.sin(t * 1.05) * 0.14
        return 0.12 + math.sin(t * 0.34) * 0.05    # idle

    def paintEvent(self, event):
        try:
            self._paint()
        except Exception as e:
            print(f"[ui] presence paintEvent error: {e}")

    def _paint(self):
        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        t      = self._phase
        energy = self._energy()
        r, g, b = mood_rgb(self._mood)

        # ---- Build additive layer stack on an opaque-black intermediate buffer ----
        # CompositionMode_Plus: dst = src + dst.  Starting from black (all 0 RGB, 255 alpha),
        # each gradient adds to the accumulated brightness. Centre accumulates to near-white
        # tinted by mood; edges remain near-black.
        buf = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        buf.fill(QColor(6, 5, 10))   # BG_DEEP as solid black-ish base

        bp = QPainter(buf)
        bp.setRenderHint(QPainter.RenderHint.Antialiasing)
        bp.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)

        base_r = w * 0.18

        for i in range(_N_ORB_LAYERS):
            f = i / _N_ORB_LAYERS                       # 0 → 1
            # Each layer drifts at a unique frequency/phase
            speed  = 0.18 + f * 0.55
            offset = (2.0 * math.pi * i) / _N_ORB_LAYERS
            drift  = base_r * (0.12 + 0.28 * f)
            dx = math.sin(t * speed       + offset * 1.37) * drift
            dy = math.cos(t * speed * 0.8 + offset * 0.93) * drift

            # Size: core layers small and tight, outer layers large and soft
            if i < 20:
                radius = base_r * (0.25 + 0.45 * f) * (1.0 + energy * 0.9)
            elif i < 38:
                radius = base_r * (0.60 + 0.35 * f) * (1.0 + energy * 0.65)
            else:
                radius = base_r * (0.80 + 0.55 * f) * (1.0 + energy * 0.5)

            if radius <= 0:
                continue

            # Per-layer alpha: more for core, less for outer halo
            layer_alpha = (0.025 - 0.010 * f) + energy * 0.012
            layer_alpha = max(0.005, min(layer_alpha, 0.06))

            # Slight hue shift per layer: rotates mood color slightly for richness
            hue_shift = (i * 4) % 30 - 15   # -15..+15 degrees
            lc = QColor(r, g, b)
            h_val, s_val, v_val, _ = lc.getHsvF()
            lc.setHsvF((h_val + hue_shift / 360.0) % 1.0, max(0.0, s_val - 0.05 * f),
                       min(1.0, v_val + 0.06 * (1 - f)), layer_alpha)

            bx, by = cx + dx, cy + dy
            grad = QRadialGradient(bx, by, radius)
            grad.setColorAt(0.0, lc)
            grad.setColorAt(1.0, QColor(r, g, b, 0))

            bp.setBrush(QBrush(grad))
            bp.setPen(Qt.PenStyle.NoPen)
            bp.drawEllipse(QRectF(bx - radius, by - radius, radius * 2.0, radius * 2.0))

        bp.end()

        # ---- Composite buffer onto widget ----
        painter = antialiased_painter(self)
        painter.drawImage(0, 0, buf)
        painter.end()


# ----------------------------------------------------------------------
# WaveformWidget — synthetic sine line while speaking
# ----------------------------------------------------------------------

_WAVEFORM_RATE = 3.0


class WaveformWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(360, 60)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._speaking = False
        self._phase    = 0.0
        self._last_tick = time.time()
        self._flash_energy = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)

    def start(self):
        if not self._timer.isActive():
            self._last_tick = time.time()
            self._timer.start(33)

    def stop(self):
        self._timer.stop()

    def set_speaking(self, speaking):
        self._speaking = speaking
        self.update()

    def trigger_flash(self):
        self._flash_energy = 1.0

    def _animate(self):
        try:
            now = time.time()
            dt  = now - self._last_tick
            self._last_tick = now
            self._phase += dt * _WAVEFORM_RATE
            self._flash_energy *= 0.92
            if self._speaking:
                self.update()
        except Exception as e:
            print(f"[ui] waveform animation error: {e}")

    def paintEvent(self, event):
        try:
            self._paint()
        except Exception as e:
            print(f"[ui] waveform paintEvent error: {e}")

    def _paint(self):
        painter = antialiased_painter(self)
        w, h = float(self.width()), float(self.height())
        if not self._speaking or w <= 0:
            painter.end()
            return
        energy = 0.28 + self._flash_energy * 0.5 + abs(math.sin(self._phase * 1.3)) * 0.25
        amp    = energy * h * 0.40
        mid    = h / 2.0
        steps  = max(1, int(w // 4))
        path   = QPainterPath()
        for i in range(steps + 1):
            x = (float(i) / steps) * w
            tr = float(i) / steps
            y  = mid + math.sin(tr * math.pi * 4 + self._phase * 3) * amp * math.sin(tr * math.pi)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        r, g, b = mood_rgb("engaged")
        pen = QPen(QColor(r, g, b, 200), 2.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawPath(path)
        painter.end()


# ----------------------------------------------------------------------
# BackgroundWidget — dark radial gradient filling the window
# ----------------------------------------------------------------------

class BackgroundWidget(QWidget):
    def paintEvent(self, event):
        try:
            painter = antialiased_painter(self)
            w, h = float(self.width()), float(self.height())
            cx, cy = w / 2.0, h / 2.0
            grad = QRadialGradient(cx, cy * 0.45, max(w, h) * 0.75)
            grad.setColorAt(0.0, QColor(14, 12, 22))   # slightly lighter near orb
            grad.setColorAt(1.0, QColor(6, 5, 10))     # BG_DEEP at edges
            painter.fillRect(self.rect(), QBrush(grad))
            painter.end()
        except Exception as e:
            print(f"[ui] background paint error: {e}")


# ----------------------------------------------------------------------
# GrainOverlay — static tiled noise texture (no animation; keeps QPoint
# overload for drawTiledPixmap to avoid the type-mismatch crash from a
# prior build that left QPainter unclosed and blackened all rendering).
# ----------------------------------------------------------------------

class GrainOverlay(QWidget):
    TILE_SIZE = 128

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._tile = self._build_tile()

    def _build_tile(self):
        size  = self.TILE_SIZE
        image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        for y in range(size):
            for x in range(size):
                image.setPixelColor(x, y, QColor(255, 255, 255, random.randint(0, 18)))
        return QPixmap.fromImage(image)

    def paintEvent(self, event):
        painter = antialiased_painter(self)
        try:
            painter.setOpacity(0.45)
            painter.drawTiledPixmap(self.rect(), self._tile)
        except Exception as e:
            print(f"[ui] grain paint error: {e}")
        finally:
            painter.end()


# ----------------------------------------------------------------------
# Chrome widgets
# ----------------------------------------------------------------------

class CircleButton(QWidget):
    def __init__(self, color, parent=None, on_click=None, dim_alpha=90):
        super().__init__(parent)
        self.setFixedSize(12, 12)
        self._color = QColor(color)
        self._dim_alpha = dim_alpha
        self._hover = False
        self._on_click = on_click
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def enterEvent(self, event):
        self._hover = True;  self.update()

    def leaveEvent(self, event):
        self._hover = False; self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._on_click:
            try: self._on_click()
            except Exception as e: print(f"[ui] circle button error: {e}")

    def paintEvent(self, event):
        painter = antialiased_painter(self)
        color = QColor(self._color)
        color.setAlpha(255 if self._hover else self._dim_alpha)
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(0.0, 0.0, 12.0, 12.0))
        painter.end()


class StatusDot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(8, 8)
        self._color   = QColor(SUCCESS)
        self._pulsing = False
        self._pulse_phase = 0.0
        self._glow = QGraphicsDropShadowEffect(self)
        self._glow.setBlurRadius(10)
        self._glow.setOffset(0, 0)
        self._glow.setColor(QColor(ACCENT))
        self.setGraphicsEffect(self._glow)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(60)

    def set_color(self, hex_color, pulsing=False):
        self._color   = QColor(hex_color)
        self._glow.setColor(QColor(hex_color))
        self._pulsing = pulsing
        if not pulsing:
            self.update()

    def _tick(self):
        if self._pulsing:
            self._pulse_phase += 0.12
            self.update()

    def paintEvent(self, event):
        painter = antialiased_painter(self)
        color = QColor(self._color)
        if self._pulsing:
            color.setAlphaF(0.5 + 0.5 * abs(math.sin(self._pulse_phase)))
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(0.0, 0.0, float(self.width()), float(self.height())))
        painter.end()


class GearButton(QWidget):
    def __init__(self, parent=None, on_click=None):
        super().__init__(parent)
        self.setFixedSize(32, 32)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._on_click = on_click
        self._opacity  = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(0.3)
        self.setGraphicsEffect(self._opacity)

    def enterEvent(self, event):  self._opacity.setOpacity(1.0)
    def leaveEvent(self, event):  self._opacity.setOpacity(0.3)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._on_click:
            try: self._on_click()
            except Exception as e: print(f"[ui] gear button error: {e}")

    def paintEvent(self, event):
        painter = antialiased_painter(self)
        w, h = float(self.width()), float(self.height())
        cx, cy = w / 2.0, h / 2.0
        painter.setBrush(QBrush(QColor(255, 255, 255, 8)))
        painter.setPen(QPen(QColor(255, 255, 255, 35), 1))
        painter.drawEllipse(QRectF(1.0, 1.0, w - 2.0, h - 2.0))
        outer_r, inner_r, tooth_len, num_teeth = 6.0, 2.6, 2.4, 8
        icon_color = QColor(*[int(c) for c in (224, 218, 242)])
        painter.setPen(QPen(icon_color, 1.4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for i in range(num_teeth):
            angle = (2 * math.pi * i) / num_teeth
            p1 = QPointF(cx + math.cos(angle) * outer_r, cy + math.sin(angle) * outer_r)
            p2 = QPointF(cx + math.cos(angle) * (outer_r + tooth_len), cy + math.sin(angle) * (outer_r + tooth_len))
            painter.drawLine(p1, p2)
        painter.drawEllipse(QRectF(cx - outer_r, cy - outer_r, outer_r * 2.0, outer_r * 2.0))
        painter.drawEllipse(QRectF(cx - inner_r, cy - inner_r, inner_r * 2.0, inner_r * 2.0))
        painter.end()


class MicButton(QWidget):
    def __init__(self, parent=None, on_click=None):
        super().__init__(parent)
        self.setFixedSize(38, 38)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._on_click   = on_click
        self._listening  = False

    def set_listening(self, listening):
        self._listening = listening
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._on_click:
            try: self._on_click()
            except Exception as e: print(f"[ui] mic button error: {e}")

    def paintEvent(self, event):
        painter = antialiased_painter(self)
        w, h = float(self.width()), float(self.height())
        cx, cy = w / 2.0, h / 2.0
        color  = QColor(ROSE) if self._listening else QColor(ACCENT)
        painter.setBrush(QBrush(QColor(255, 255, 255, 10)))
        painter.setPen(QPen(QColor(255, 255, 255, 15), 1))
        painter.drawEllipse(QRectF(1.0, 1.0, w - 2.0, h - 2.0))
        pen = QPen(color, 1.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRectF(cx - 3.5, cy - 8.0, 7.0, 11.0), 3.5, 3.5)
        painter.drawArc(QRectF(cx - 7.0, cy - 3.0, 14.0, 14.0), 180 * 16, 180 * 16)
        painter.drawLine(QPointF(cx, cy + 8.0), QPointF(cx, cy + 11.0))
        painter.end()


class SendButton(QWidget):
    def __init__(self, parent=None, on_click=None):
        super().__init__(parent)
        self.setFixedSize(34, 34)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._on_click = on_click
        self._hover    = False

    def enterEvent(self, event):  self._hover = True;  self.update()
    def leaveEvent(self, event):  self._hover = False; self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._on_click:
            try: self._on_click()
            except Exception as e: print(f"[ui] send button error: {e}")

    def paintEvent(self, event):
        painter = antialiased_painter(self)
        w, h  = float(self.width()), float(self.height())
        cx, cy = w / 2.0, h / 2.0
        color  = QColor(ACCENT) if self._hover else QColor(MUTED)
        pen = QPen(color, 1.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(QPointF(cx - 7.0, cy), QPointF(cx + 7.0, cy))
        painter.drawLine(QPointF(cx + 1.0, cy - 6.0), QPointF(cx + 7.0, cy))
        painter.drawLine(QPointF(cx + 1.0, cy + 6.0), QPointF(cx + 7.0, cy))
        painter.end()


# ----------------------------------------------------------------------
# Background QThreads — all blocking ops here, never on GUI thread.
# Signals auto-queue onto the main thread via Qt's event system.
# ----------------------------------------------------------------------

class ListenThread(QThread):
    result_ready = pyqtSignal(object)

    def run(self):
        try:
            result = voice.listen_once()
        except Exception as e:
            print(f"[ui] listen failed: {e}")
            result = None
        # result.get("text") may be an explicit None (not missing key) —
        # the `or ""` catches that; a plain .get("text", "") would not.
        text      = (result.get("text") or "").strip() if result else ""
        audio_raw = result.get("audio_raw") if result else None
        self.result_ready.emit({"text": text, "audio_raw": audio_raw})


class ResponseThread(QThread):
    """
    Full pipeline: voice/face fusion → brain.run_parallel → mood fusion
    → persistence. brain.run_parallel returns {ai_response, text_mood, prompt_mood};
    NOT {response, mood, confidence} — the extraction below matches the real API.
    """
    response_ready = pyqtSignal(dict)

    def __init__(self, text, audio_raw, user_id, face_analyzer, parent=None):
        super().__init__(parent)
        self.text         = text
        self.audio_raw    = audio_raw
        self.user_id      = user_id
        self.face_analyzer = face_analyzer

    def run(self):
        try:
            language = voice.detect_language(self.text)

            if self.audio_raw:
                features = voice.analyze_voice_features(self.audio_raw, self.text)
                features["mood"] = voice.classify_voice_mood(
                    features["pitch"], features["speaking_speed"], features["pause_ratio"]
                )
            else:
                features = {"pitch": 0.0, "pitch_std": 0.0, "speaking_speed": 0.0,
                            "pause_ratio": 0.0, "mood": "calm"}

            face_signals = (
                self.face_analyzer.get_latest_signals()
                if self.face_analyzer and self.face_analyzer.camera_available
                else {"emotion": "neutral", "confidence": 0.0, "fatigue": 0.0,
                      "engagement": 0.5, "fused_mood": "calm", "face_detected": False}
            )

            result       = brain.run_parallel(self.text, self.user_id, features, face_signals, language)
            ai_response  = result["ai_response"]
            text_mood    = result["text_mood"]

            fused = brain.fuse_moods(
                features["mood"], face_signals["fused_mood"], text_mood["mood"],
                features.get("pitch", 0.0), face_signals.get("fatigue", 0.0),
                face_signals.get("engagement", 0.5),
            )

            confidence = brain.extract_confidence(ai_response)
            clean_text = brain.clean_response(ai_response)
            task       = brain.extract_task(ai_response)
            url        = brain.extract_url(ai_response)

            database.save_conversation(
                self.user_id, self.text, clean_text, fused["mood"], confidence, language,
                features.get("pitch", 0.0), features.get("speaking_speed", 0.0),
                face_signals.get("emotion", "neutral"), face_signals.get("fatigue", 0.0),
                face_signals.get("engagement", 0.5),
            )
            database.save_mood_reading(
                self.user_id, features["mood"], face_signals["fused_mood"],
                text_mood["mood"], fused["mood"], fused["intensity"],
            )
            patterns.update_all_patterns(
                self.user_id, self.text, fused["mood"], datetime.now().hour, language,
                face_signals.get("emotion"),
            )
            if task:
                database.save_task(self.user_id, task["description"], task["due_date"])

            self.response_ready.emit({
                "response": clean_text, "fused": fused, "confidence": confidence,
                "url": url, "language": language, "voice_features": features,
                "face_signals": face_signals,
            })
        except Exception as e:
            print(f"[ui] response generation failed: {e}")
            self.response_ready.emit({
                "response": "I had trouble with that — could you try again?",
                "fused": {"mood": "calm", "intensity": 0.5, "description": ""},
                "confidence": "Low", "url": None, "language": "en",
                "voice_features": {"mood": "calm"},
                "face_signals": {"fused_mood": "calm", "fatigue": 0.0},
            })


class SpeakThread(QThread):
    def __init__(self, text, mood, parent=None):
        super().__init__(parent)
        self.text = text
        self.mood = mood

    def run(self):
        try:
            voice.speak(self.text, self.mood)
        except Exception as e:
            print(f"[ui] speak failed: {e}")


# ----------------------------------------------------------------------
# ARIAWindow
# ----------------------------------------------------------------------

class ARIAWindow(QMainWindow):
    greeting_ready    = pyqtSignal(str)
    memory_data_ready = pyqtSignal(object)

    def __init__(self, user_id, user_name, face_analyzer, on_close=None):
        super().__init__()
        # Resolve best available serif font before building any labels
        _resolve_serif_family()

        self.user_id       = user_id
        self.user_name     = user_name
        self.face_analyzer = face_analyzer
        self.on_close      = on_close

        self.is_listening  = False
        self.is_speaking   = False
        self.status_text   = "ready"
        self._drag_pos     = None
        self._auto_listen_generation = 0
        self._listen_generation      = 0
        self._url_prompt   = None
        self._memory_window = None
        self._wake_enabled  = True

        self.last_voice_features = {"pitch": 0.0, "speaking_speed": 0.0,
                                    "pause_ratio": 0.0, "mood": "calm"}
        self.last_face_signals   = {"emotion": "neutral", "confidence": 0.0,
                                    "fatigue": 0.0, "engagement": 0.5,
                                    "fused_mood": "calm", "face_detected": False}
        self.last_fused_mood = {"mood": "calm", "intensity": 0.5, "description": ""}

        user = database.get_user(user_id) or {}
        self.personality_mode        = user.get("personality_mode", "friendly")
        self._current_language_code  = user.get("language_preference", "en")

        self.setWindowTitle("ARIA")
        self.resize(1400, 900)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setMouseTracking(True)

        central = QWidget()
        central.setStyleSheet(f"background: {BG};")
        central.setMouseTracking(True)
        self.setCentralWidget(central)
        self._central = central

        self.background = BackgroundWidget(central)
        self.background.setGeometry(0, 0, self.width(), self.height())

        self.grain = GrainOverlay(central)
        self.grain.setGeometry(0, 0, self.width(), self.height())

        self.presence = PresenceWidget(central)
        self.waveform = WaveformWidget(central)

        self._build_chrome()
        self._build_identity_labels()
        self._build_captions()
        self._build_settings()
        self._build_camera_feed()
        self._build_control_bar()

        self.greeting_ready.connect(self.show_greeting)
        self.memory_data_ready.connect(self._on_memory_data_ready)

        qt_app = QApplication.instance()
        if qt_app is not None:
            qt_app.installEventFilter(self)

        self._reposition_all()
        self.presence.start()
        self.waveform.start()

        self._signals_timer = QTimer(self)
        self._signals_timer.timeout.connect(self._update_signals)
        self._signals_timer.start(200)

    # ------------------------------------------------------------------
    # Window chrome
    # ------------------------------------------------------------------

    def _build_chrome(self):
        central = self._central
        central.mousePressEvent   = self._chrome_mouse_press
        central.mouseMoveEvent    = self._chrome_mouse_move
        central.mouseReleaseEvent = self._chrome_mouse_release
        self.close_btn    = CircleButton("#ff5f57", central, on_click=self.close)
        self.minimize_btn = CircleButton("#febc2e", central, on_click=self.showMinimized)

    def _chrome_mouse_press(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint()

    def _chrome_mouse_move(self, event):
        if self._drag_pos is not None and event.buttons() == Qt.MouseButton.LeftButton:
            delta = event.globalPosition().toPoint() - self._drag_pos
            self.move(self.pos() + delta)
            self._drag_pos = event.globalPosition().toPoint()

    def _chrome_mouse_release(self, event):
        self._drag_pos = None

    def eventFilter(self, obj, event):
        try:
            if event.type() == QEvent.Type.MouseMove:
                self._on_global_mouse_move(event)
            elif event.type() == QEvent.Type.MouseButtonPress:
                self._on_global_mouse_press(event)
        except Exception as e:
            print(f"[ui] event filter error: {e}")
        return super().eventFilter(obj, event)

    def _on_global_mouse_move(self, event):
        try:
            local = self.mapFromGlobal(event.globalPosition().toPoint())
        except Exception:
            return
        if 0 <= local.x() <= self.width() and 0 <= local.y() <= self.height():
            # Deferred: starting an animation synchronously inside eventFilter
            # nests Qt's event dispatch inside itself — singleShot(0, ...)
            # defers to the next event-loop tick, avoiding that reentrancy crash.
            y = local.y()
            QTimer.singleShot(0, lambda: self._handle_hover(y))

    def _on_global_mouse_press(self, event):
        if not self.settings_panel.isVisible():
            return
        try:
            gpos = event.globalPosition().toPoint()
        except Exception:
            return
        on_panel = self.settings_panel.rect().contains(self.settings_panel.mapFromGlobal(gpos))
        on_gear  = self.settings_btn.rect().contains(self.settings_btn.mapFromGlobal(gpos))
        if not on_panel and not on_gear:
            self.settings_panel.hide()

    def _handle_hover(self, y):
        h = self.height()
        if y > h * 0.82:
            self._show_control_bar()
        elif not self.text_input.text().strip() and not self.text_input.hasFocus():
            self._hide_control_bar()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        try:
            self._reposition_all()
        except Exception as e:
            print(f"[ui] reposition on resize error: {e}")

    def closeEvent(self, event):
        self.presence.stop()
        self.waveform.stop()
        # wait(2000) prevents "QThread: Destroyed while thread still running" warnings
        for attr in ("listen_thread", "response_thread", "speak_thread"):
            thread = getattr(self, attr, None)
            if thread is not None and thread.isRunning():
                thread.wait(2000)
        if self.on_close:
            try:
                self.on_close()
            except Exception as e:
                print(f"[ui] on_close error: {e}")
        event.accept()

    # ------------------------------------------------------------------
    # Identity labels
    # ------------------------------------------------------------------

    def _build_identity_labels(self):
        central = self._central

        self.aria_label = QLabel("A  R  I  A", central)
        self.aria_label.setFont(ui_font(10))
        self.aria_label.setStyleSheet(f"color: {MUTED}; background: transparent; border: none;")

        self.status_label = QLabel("ready", central)
        self.status_label.setFont(ui_font(9))
        self.status_label.setStyleSheet("color: #3a3555; background: transparent; border: none;")

    def set_status(self, status_text):
        self.status_text = status_text
        self.status_label.setText(status_text)
        self.status_label.adjustSize()
        self.presence.set_state(STATUS_TO_PRESENCE.get(status_text, "idle"))
        self.waveform.set_speaking(status_text == "speaking")
        self.status_dot.set_color(
            STATUS_DOT_COLORS.get(status_text, SUCCESS),
            pulsing=(status_text != "ready"),
        )

    # ------------------------------------------------------------------
    # Captions — word-by-word reveal in premium serif
    # ------------------------------------------------------------------

    def _build_captions(self):
        central = self._central

        self.user_echo_label = QLabel("", central)
        self.user_echo_label.setFont(serif_font(12, italic=True))
        self.user_echo_label.setStyleSheet(f"color: {MUTED}; background: transparent; border: none;")
        self.user_echo_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.user_echo_label.setWordWrap(True)

        self.caption_label = QLabel("", central)
        self.caption_label.setFont(serif_font(20, italic=True))
        self.caption_label.setStyleSheet(f"color: {INK}; background: transparent; border: none;")
        self.caption_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.caption_label.setWordWrap(True)

        self._caption_words  = []
        self._caption_index  = 0
        self._caption_reveal_timer = QTimer(self)
        self._caption_reveal_timer.timeout.connect(self._reveal_next_word)

        self._user_echo_clear_timer = QTimer(self)
        self._user_echo_clear_timer.setSingleShot(True)
        self._user_echo_clear_timer.timeout.connect(lambda: self.user_echo_label.setText(""))

    def show_user_echo(self, text):
        self.user_echo_label.setText(text)
        self._reposition_captions()
        self._user_echo_clear_timer.start(4000)

    def start_caption(self, text):
        self.caption_label.setText("")
        self._caption_words = text.split()
        self._caption_index = 0
        self._caption_reveal_timer.start(115)
        self._reposition_captions()

    def _reveal_next_word(self):
        if self._caption_index >= len(self._caption_words):
            self._caption_reveal_timer.stop()
            return
        self.caption_label.setText(" ".join(self._caption_words[:self._caption_index + 1]))
        self._caption_index += 1
        self._reposition_captions()

    def _reposition_captions(self):
        w, h   = self.width(), self.height()
        max_w  = int(w * 0.58)
        cx     = (w - max_w) // 2

        self.user_echo_label.setFixedWidth(max_w)
        self.user_echo_label.adjustSize()
        self.user_echo_label.move(cx, h - 148)

        self.caption_label.setFixedWidth(max_w)
        self.caption_label.adjustSize()
        self.caption_label.move(cx, h - 116)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _build_settings(self):
        central = self._central
        self.settings_btn = GearButton(central, on_click=self.toggle_settings)
        self.status_dot   = StatusDot(central)

        self.settings_panel = QFrame(central)
        self.settings_panel.setStyleSheet(f"""
            QFrame {{
                background: #0e0c18;
                border: 1px solid rgba(176,158,224,0.12);
                border-radius: 16px;
            }}
        """)
        self.settings_panel.setFixedWidth(210)
        self.settings_panel.hide()

        sl = QVBoxLayout(self.settings_panel)
        sl.setContentsMargins(14, 14, 14, 14)
        sl.setSpacing(10)

        # Wake word toggle
        wake_row = self._setting_row("Wake word", self._make_checkbox(True, self._on_wake_toggle))
        self.wake_check = wake_row.findChild(QCheckBox)
        sl.addWidget(wake_row)

        # Camera toggle
        cam_on = bool(self.face_analyzer and self.face_analyzer.camera_available)
        cam_row = self._setting_row("Camera", self._make_checkbox(cam_on, self._on_camera_toggle))
        self.cam_check = cam_row.findChild(QCheckBox)
        sl.addWidget(cam_row)

        sl.addWidget(self._separator())

        mode_lbl = QLabel("Personality")
        mode_lbl.setFont(ui_font(9))
        mode_lbl.setStyleSheet(f"color: {MUTED}; background: transparent; border: none;")
        sl.addWidget(mode_lbl)

        mode_row = QWidget()
        mode_row.setStyleSheet("background: transparent;")
        mrl = QHBoxLayout(mode_row)
        mrl.setContentsMargins(0, 0, 0, 0)
        mrl.setSpacing(4)
        self._mode_buttons = {}
        for mode, label in (("friendly", "Friendly"), ("professional", "Pro"), ("motivational", "Motiv")):
            btn = QPushButton(label)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _checked=False, m=mode: self._on_personality_change(m))
            mrl.addWidget(btn)
            self._mode_buttons[mode] = btn
        sl.addWidget(mode_row)
        self._refresh_mode_buttons()

        sl.addWidget(self._separator())

        mem_link = QLabel("View Memory")
        mem_link.setFont(ui_font(12))
        mem_link.setStyleSheet(f"color: {ACCENT}; background: transparent; border: none;")
        mem_link.setCursor(Qt.CursorShape.PointingHandCursor)
        mem_link.mousePressEvent = self._make_settings_click(self._open_memory_window)
        sl.addWidget(mem_link)

        sl.addWidget(self._separator())

        user_lbl = QLabel(self.user_name)
        user_lbl.setFont(ui_font(11))
        user_lbl.setStyleSheet(f"color: {MUTED}; background: transparent; border: none;")
        sl.addWidget(user_lbl)

        signout = QLabel("Sign out")
        signout.setFont(ui_font(11))
        signout.setStyleSheet(f"color: {MUTED}; background: transparent; border: none;")
        signout.setCursor(Qt.CursorShape.PointingHandCursor)
        signout.mousePressEvent = self._make_settings_click(self.close)
        sl.addWidget(signout)

    def _setting_row(self, label_text, widget):
        row = QWidget()
        row.setStyleSheet("background: transparent;")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label_text)
        lbl.setFont(ui_font(12))
        lbl.setStyleSheet(f"color: {INK}; background: transparent; border: none;")
        rl.addWidget(lbl)
        rl.addStretch()
        rl.addWidget(widget)
        return row

    def _make_checkbox(self, checked, handler):
        cb = QCheckBox()
        cb.setChecked(checked)
        cb.setStyleSheet(self._checkbox_qss())
        cb.stateChanged.connect(handler)
        return cb

    def _checkbox_qss(self):
        return f"""
            QCheckBox::indicator {{
                width: 14px; height: 14px;
                border: 1px solid rgba(176,158,224,0.2);
                border-radius: 4px;
                background: rgba(255,255,255,0.04);
            }}
            QCheckBox::indicator:checked {{
                background: {ACCENT};
                border: 1px solid {ACCENT};
            }}
        """

    def _separator(self):
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: rgba(176,158,224,0.08); border: none;")
        return sep

    def _refresh_mode_buttons(self):
        for mode, btn in self._mode_buttons.items():
            if mode == self.personality_mode:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: {ACCENT}; color: {BG};
                        border: none; border-radius: 10px;
                        padding: 4px 10px; font-size: 10px;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: transparent; color: {MUTED};
                        border: none; border-radius: 10px;
                        padding: 4px 10px; font-size: 10px;
                    }}
                    QPushButton:hover {{ color: {ACCENT}; }}
                """)

    def _make_settings_click(self, action):
        def handler(event):
            self.settings_panel.hide()
            action()
        return handler

    def toggle_settings(self):
        if self.settings_panel.isVisible():
            self.settings_panel.hide()
        else:
            self.settings_panel.adjustSize()
            gp = self.settings_btn.pos()
            self.settings_panel.move(
                gp.x() + self.settings_btn.width() - self.settings_panel.width(),
                gp.y() + self.settings_btn.height() + 10,
            )
            self.settings_panel.show()
            self.settings_panel.raise_()

    def _on_wake_toggle(self, _state):
        self._wake_enabled = self.wake_check.isChecked()
        if self._wake_enabled and not self.is_listening and self.status_text == "ready":
            self.start_auto_listen()

    def _on_camera_toggle(self, _state):
        if not self.face_analyzer:
            return
        if self.cam_check.isChecked():
            if not self.face_analyzer.running:
                threading.Thread(target=self.face_analyzer.start, daemon=True).start()
        else:
            if self.face_analyzer.running:
                threading.Thread(target=self.face_analyzer.stop, daemon=True).start()

    def _on_personality_change(self, mode):
        self.personality_mode = mode
        database.set_personality_mode(self.user_id, mode)
        self._refresh_mode_buttons()

    # ------------------------------------------------------------------
    # Signals polling (face/mood — drives presence color)
    # ------------------------------------------------------------------

    def _update_signals(self):
        try:
            if self.face_analyzer and self.face_analyzer.camera_available:
                self.last_face_signals = self.face_analyzer.get_latest_signals()
        except Exception as e:
            print(f"[ui] signal update error: {e}")

    # ------------------------------------------------------------------
    # Camera feed
    # ------------------------------------------------------------------

    def _build_camera_feed(self):
        central = self._central
        self.camera_label = QLabel(central)
        self.camera_label.setFixedSize(200, 150)
        self.camera_label.setStyleSheet("background: transparent; border: none;")
        self.camera_label.hide()
        self._camera_timer = QTimer(self)
        self._camera_timer.timeout.connect(self._update_camera_feed)
        self._camera_timer.start(50)

    def _update_camera_feed(self):
        if not self.last_face_signals.get("face_detected") or \
                not self.face_analyzer or not self.face_analyzer.camera_available:
            if self.camera_label.isVisible():
                self.camera_label.hide()
            return
        try:
            frame = self.face_analyzer.get_current_frame()
        except Exception as e:
            print(f"[ui] camera frame error: {e}")
            frame = None
        if frame is None:
            self.camera_label.hide()
            return
        try:
            raw = self._pil_to_pixmap(frame).scaled(
                200, 150, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.camera_label.setPixmap(self._round_pixmap(raw, 12.0))
            if not self.camera_label.isVisible():
                self.camera_label.show()
                self.camera_label.raise_()
        except Exception as e:
            print(f"[ui] camera render error: {e}")

    @staticmethod
    def _pil_to_pixmap(pil_image):
        buf = BytesIO()
        pil_image.convert("RGB").save(buf, format="PNG")
        px = QPixmap()
        px.loadFromData(buf.getvalue())
        return px

    @staticmethod
    def _round_pixmap(pixmap, radius):
        if pixmap.isNull():
            return pixmap
        rounded = QPixmap(pixmap.size())
        rounded.fill(Qt.GlobalColor.transparent)
        painter = antialiased_painter(rounded)
        try:
            path = QPainterPath()
            path.addRoundedRect(QRectF(0.0, 0.0, float(pixmap.width()), float(pixmap.height())), radius, radius)
            painter.setClipPath(path)
            painter.drawPixmap(0, 0, pixmap)
        finally:
            painter.end()
        return rounded

    # ------------------------------------------------------------------
    # Control bar
    # ------------------------------------------------------------------

    def _build_control_bar(self):
        central = self._central
        self.control_bar = QFrame(central)
        self.control_bar.setStyleSheet("""
            QFrame {
                background: rgba(176,158,224,0.05);
                border: 1px solid rgba(176,158,224,0.1);
                border-radius: 28px;
            }
        """)
        self.control_bar.setFixedSize(520, 54)
        self.control_bar.setVisible(False)

        bl = QHBoxLayout(self.control_bar)
        bl.setContentsMargins(14, 6, 14, 6)
        bl.setSpacing(10)

        self.mic_btn = MicButton(self.control_bar, on_click=self.toggle_mic)
        bl.addWidget(self.mic_btn)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Say something to ARIA…")
        self.text_input.setFont(serif_font(14, italic=False))
        self.text_input.setStyleSheet(f"QLineEdit {{ background: transparent; border: none; color: {INK}; }}")
        self.text_input.returnPressed.connect(self.send_text_message)
        bl.addWidget(self.text_input, 1)

        self.send_btn = SendButton(self.control_bar, on_click=self.send_text_message)
        bl.addWidget(self.send_btn)

        self._control_bar_opacity = QGraphicsOpacityEffect(self.control_bar)
        self._control_bar_opacity.setOpacity(0.0)
        self.control_bar.setGraphicsEffect(self._control_bar_opacity)
        self._control_bar_anim = None

    def _show_control_bar(self):
        if self.control_bar.isVisible():
            return
        w, h = self.width(), self.height()
        self.control_bar.move((w - self.control_bar.width()) // 2, h - 36 - self.control_bar.height())
        self._control_bar_opacity.setOpacity(0.0)
        self.control_bar.setVisible(True)
        self.control_bar.raise_()
        self._animate_control_bar(1.0)

    def _hide_control_bar(self):
        if not self.control_bar.isVisible():
            return
        self._animate_control_bar(0.0, on_complete=lambda: self.control_bar.setVisible(False))

    def _animate_control_bar(self, target, on_complete=None):
        anim = QPropertyAnimation(self._control_bar_opacity, b"opacity", self)
        anim.setDuration(400)
        anim.setStartValue(self._control_bar_opacity.opacity())
        anim.setEndValue(target)
        anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        if on_complete:
            anim.finished.connect(on_complete)
        anim.start()
        self._control_bar_anim = anim   # keep ref — garbage-collected anim stops mid-flight

    def send_text_message(self):
        text = self.text_input.text().strip()
        if not text:
            return
        self.text_input.clear()
        self._hide_control_bar()
        self._auto_listen_generation += 1
        self.process_input(text, None)

    def toggle_mic(self):
        if self.is_listening:
            self._auto_listen_generation += 1
            self.is_listening = False
            self.mic_btn.set_listening(False)
            self.set_status("ready")
            return
        self._begin_listen_cycle()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _reposition_all(self):
        w, h = self.width(), self.height()
        self.background.setGeometry(0, 0, w, h)
        self.grain.setGeometry(0, 0, w, h)

        pw, ph = self.presence.width(), self.presence.height()
        px, py = (w - pw) // 2, (h - ph) // 2 - 50
        self.presence.move(px, py)
        presence_bottom = py + ph

        wf_w = self.waveform.width()
        self.waveform.move((w - wf_w) // 2, presence_bottom - 38)

        self.aria_label.adjustSize()
        self.aria_label.move(w // 2 - self.aria_label.width() // 2, presence_bottom + 24)
        self.status_label.adjustSize()
        self.status_label.move(w // 2 - self.status_label.width() // 2, presence_bottom + 42)

        self.close_btn.move(16, 16)
        self.minimize_btn.move(36, 16)

        self.settings_btn.move(w - 46, 18)
        self.status_dot.move(w - 74, 23)

        self._reposition_captions()
        self.camera_label.move(24, h - 28 - self.camera_label.height())

        if self.control_bar.isVisible():
            self.control_bar.move((w - self.control_bar.width()) // 2, h - 36 - self.control_bar.height())

    # ------------------------------------------------------------------
    # Continuous listening
    # ------------------------------------------------------------------

    def _begin_listen_cycle(self):
        self._auto_listen_generation += 1
        self._listen_generation       = self._auto_listen_generation
        self.is_listening = True
        self.mic_btn.set_listening(True)
        self.set_status("listening")
        self.listen_thread = ListenThread(self)
        self.listen_thread.result_ready.connect(self._on_listen_result)
        self.listen_thread.start()

    def start_auto_listen(self):
        if self.is_listening:
            return
        if not self._wake_enabled:
            self.set_status("ready")
            return
        self._begin_listen_cycle()

    def _on_listen_result(self, result):
        my_gen = self._listen_generation
        self.is_listening = False
        self.mic_btn.set_listening(False)
        if my_gen != self._auto_listen_generation:
            return
        text = result.get("text", "")
        if text:
            self.process_input(text, result.get("audio_raw"))
        else:
            QTimer.singleShot(300, self.start_auto_listen)

    # ------------------------------------------------------------------
    # Conversation pipeline
    # ------------------------------------------------------------------

    def process_input(self, text, audio_raw=None):
        if not text or not text.strip():
            QTimer.singleShot(300, self.start_auto_listen)
            return
        self.set_status("thinking")
        self.show_user_echo(text)
        self.response_thread = ResponseThread(text, audio_raw, self.user_id, self.face_analyzer, self)
        self.response_thread.response_ready.connect(self._on_response_ready)
        self.response_thread.start()

    def _on_response_ready(self, result):
        response = result["response"]
        fused    = result["fused"]
        url      = result.get("url")
        language = result.get("language", "en")

        self._current_language_code = language
        self.last_voice_features    = result.get("voice_features", {})
        self.last_face_signals      = result.get("face_signals", {})
        self.last_fused_mood        = fused

        self.presence.set_mood(fused["mood"])
        self.set_status("speaking")
        self.presence.trigger_flash()
        self.waveform.trigger_flash()
        self.start_caption(response)

        if url:
            QTimer.singleShot(1500, lambda: self.show_url_prompt(url))

        self.speak_thread = SpeakThread(response, fused["mood"], self)
        self.speak_thread.finished.connect(self._on_speaking_done)
        self.speak_thread.start()

    def _on_speaking_done(self):
        self.set_status("ready")
        QTimer.singleShot(600, self.start_auto_listen)

    # ------------------------------------------------------------------
    # Greeting (main.py emits greeting_ready → this slot via signal)
    # ------------------------------------------------------------------

    def show_greeting(self, text):
        self.set_status("speaking")
        self.presence.trigger_flash()
        self.waveform.trigger_flash()
        self.start_caption(text)
        self.speak_thread = SpeakThread(text, "calm", self)
        self.speak_thread.finished.connect(self._on_speaking_done)
        self.speak_thread.start()

    # ------------------------------------------------------------------
    # URL prompt
    # ------------------------------------------------------------------

    def show_url_prompt(self, url):
        if self._url_prompt is not None:
            try: self._url_prompt.deleteLater()
            except Exception: pass
            self._url_prompt = None

        try:
            hostname = urlparse(url).netloc or url
        except Exception:
            hostname = url

        prompt = QFrame(self._central)
        prompt.setStyleSheet(f"""
            QFrame {{
                background: rgba(10,8,18,0.94);
                border: 1px solid rgba(176,158,224,0.25);
                border-radius: 24px;
            }}
        """)
        layout = QHBoxLayout(prompt)
        layout.setContentsMargins(22, 10, 22, 10)
        layout.setSpacing(16)

        label = QLabel(f"Open {hostname}?")
        label.setStyleSheet(f"color: {INK}; font-size: 13px; border: none; background: transparent;")
        open_lbl = QLabel("Open")
        open_lbl.setStyleSheet(f"color: {ACCENT}; font-size: 12px; border: none; background: transparent; text-decoration: underline;")
        open_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
        dismiss_lbl = QLabel("Dismiss")
        dismiss_lbl.setStyleSheet(f"color: {MUTED}; font-size: 12px; border: none; background: transparent;")
        dismiss_lbl.setCursor(Qt.CursorShape.PointingHandCursor)

        layout.addWidget(label)
        layout.addWidget(open_lbl)
        layout.addWidget(dismiss_lbl)

        prompt.adjustSize()
        w, h = self.width(), self.height()
        prompt.move(w // 2 - prompt.width() // 2, h - 210)
        self._url_prompt = prompt

        open_lbl.mousePressEvent    = self._make_url_handler(prompt, url, True)
        dismiss_lbl.mousePressEvent = self._make_url_handler(prompt, url, False)

        prompt.show()
        QTimer.singleShot(8000, lambda: self._dismiss_url_prompt(prompt))

    def _make_url_handler(self, prompt, url, open_now):
        def handler(event):
            if open_now:
                try: webbrowser.open(url)
                except Exception as e: print(f"[ui] URL open error: {e}")
            self._dismiss_url_prompt(prompt)
        return handler

    def _dismiss_url_prompt(self, prompt):
        if self._url_prompt is prompt:
            self._url_prompt = None
        try: prompt.deleteLater()
        except Exception: pass

    # ------------------------------------------------------------------
    # Memory window  (with pyqtgraph mood area chart)
    # ------------------------------------------------------------------

    def _open_memory_window(self):
        try:
            if self._memory_window is not None and self._memory_window.isVisible():
                self._memory_window.raise_()
                self._memory_window.activateWindow()
                return
        except RuntimeError:
            pass

        win = QWidget()
        win.setWindowTitle("ARIA Memory")
        win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        win.resize(920, 640)
        win.setStyleSheet(f"background: {BG};")
        self._memory_window = win

        root = QVBoxLayout(win)
        root.setContentsMargins(52, 44, 52, 34)
        root.setSpacing(10)

        title = QLabel(f"What ARIA knows about {self.user_name}")
        title.setFont(serif_font(24, italic=True))
        title.setStyleSheet(f"color: {INK}; background: transparent; border: none;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        subtitle = QLabel("Loading…")
        subtitle.setStyleSheet(f"color: {MUTED}; font-size: 11px; background: transparent; border: none;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(subtitle)
        self._mem_subtitle = subtitle

        # Mood chart (pyqtgraph area plot, 7 days)
        if _HAS_PYQTGRAPH:
            pg.setConfigOptions(antialias=True, background=BG_DEEP, foreground=MUTED)
            chart = pg.PlotWidget()
            chart.setFixedHeight(160)
            chart.setYRange(0, 10, padding=0.1)
            chart.showGrid(x=False, y=True, alpha=0.12)
            chart.getAxis("bottom").setStyle(showValues=False)
            chart.getAxis("left").setStyle(tickFont=ui_font(8))
            chart.setTitle("Mood this week", color=MUTED, size="10pt")
            chart.setStyleSheet(f"border: 1px solid rgba(176,158,224,0.08); border-radius: 12px;")
            self._mem_chart = chart
            root.addWidget(chart)
        else:
            self._mem_chart = None

        scroll = QScrollArea()
        scroll.setStyleSheet(f"""
            QScrollArea {{ border: none; background: transparent; }}
            QScrollBar:vertical {{ background: {BG}; width: 4px; }}
            QScrollBar::handle:vertical {{ background: rgba(176,158,224,0.12); border-radius: 2px; }}
        """)
        scroll.setWidgetResizable(True)

        content = QWidget()
        content.setStyleSheet("background: transparent;")
        clayout = QHBoxLayout(content)
        clayout.setSpacing(36)

        left_col  = QWidget(); left_col.setStyleSheet("background: transparent;")
        right_col = QWidget(); right_col.setStyleSheet("background: transparent;")
        left_layout  = QVBoxLayout(left_col);  left_layout.setSpacing(6);  left_layout.setContentsMargins(0, 0, 0, 0)
        right_layout = QVBoxLayout(right_col); right_layout.setSpacing(6); right_layout.setContentsMargins(0, 0, 0, 0)

        clayout.addWidget(left_col, 1)
        clayout.addWidget(right_col, 1)

        self._mem_left_layout  = left_layout
        self._mem_right_layout = right_layout

        scroll.setWidget(content)
        root.addWidget(scroll)

        close_lbl = QLabel("close")
        close_lbl.setStyleSheet(f"color: {MUTED}; font-size: 11px; background: transparent; border: none;")
        close_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        close_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
        close_lbl.mousePressEvent = self._make_close_handler(win)
        root.addWidget(close_lbl)

        win.show()
        threading.Thread(target=self._load_memory_data, daemon=True).start()

    @staticmethod
    def _make_close_handler(win):
        def handler(event):
            win.close()
        return handler

    def _load_memory_data(self):
        try:
            user          = database.get_user(self.user_id) or {}
            total         = database.get_total_conversations(self.user_id)
            pats          = database.get_patterns(self.user_id, limit=10)
            tasks         = database.get_pending_tasks(self.user_id)
            mood_readings = database.get_mood_history(self.user_id, days=7)
            summary       = patterns.build_user_profile_summary(self.user_id)
            self.memory_data_ready.emit({
                "user": user, "total": total, "patterns": pats, "tasks": tasks,
                "mood_readings": mood_readings, "summary": summary,
            })
        except Exception as e:
            print(f"[ui] memory data load error: {e}")

    def _on_memory_data_ready(self, data):
        try:
            if self._memory_window is None or not self._memory_window.isVisible():
                return
        except RuntimeError:
            return

        created_at = (data["user"].get("created_at") or "")[:10]
        self._mem_subtitle.setText(
            f"{data['total']} conversations since {created_at or 'recently'}"
        )

        # ---- Mood chart ----
        if _HAS_PYQTGRAPH and self._mem_chart is not None:
            self._populate_mood_chart(data["mood_readings"])

        # ---- Left column: patterns ----
        self._clear_layout(self._mem_left_layout)
        self._mem_left_layout.addWidget(self._mem_section("PATTERNS"))
        pats = data["patterns"]
        if not pats:
            self._mem_left_layout.addWidget(self._mem_text("Still getting to know you."))
        else:
            for p in pats:
                tmpl    = PATTERN_DESCRIPTIONS.get(p.get("pattern_type"), "{v}")
                readable = tmpl.format(v=p.get("pattern_value"))
                self._mem_left_layout.addWidget(self._mem_text(readable))
        self._mem_left_layout.addStretch()

        # ---- Right column: tasks + summary ----
        self._clear_layout(self._mem_right_layout)
        self._mem_right_layout.addWidget(self._mem_section("PENDING TASKS"))
        tasks = data["tasks"]
        if not tasks:
            self._mem_right_layout.addWidget(self._mem_text("Nothing pending.", italic=False))
        else:
            for t in tasks:
                row = QWidget(); row.setStyleSheet("background: transparent;")
                rl  = QHBoxLayout(row); rl.setContentsMargins(0, 0, 0, 0)
                due = f" — {t['due_date']}" if t.get("due_date") else ""
                desc = QLabel(f"• {t['task_description']}{due}")
                desc.setWordWrap(True)
                desc.setStyleSheet(f"color: {INK}; font-size: 12px; background: transparent; border: none;")
                done = QLabel("done")
                done.setStyleSheet(f"color: {SUCCESS}; font-size: 10px; background: transparent; border: none;")
                done.setCursor(Qt.CursorShape.PointingHandCursor)
                done.mousePressEvent = self._make_task_handler(t["id"], row)
                rl.addWidget(desc, 1)
                rl.addWidget(done)
                self._mem_right_layout.addWidget(row)
        self._mem_right_layout.addSpacing(16)

        self._mem_right_layout.addWidget(self._mem_section("ARIA'S UNDERSTANDING"))
        sum_frame = QFrame()
        sum_frame.setStyleSheet(f"""
            QFrame {{
                background: rgba(176,158,224,0.04);
                border-left: 2px solid {ACCENT};
                border-radius: 0px;
            }}
        """)
        sfl = QVBoxLayout(sum_frame)
        sfl.setContentsMargins(12, 12, 12, 12)
        sum_lbl = QLabel(data["summary"])
        sum_lbl.setWordWrap(True)
        sum_lbl.setFont(serif_font(13, italic=True))
        sum_lbl.setStyleSheet(f"color: {MUTED}; background: transparent; border: none;")
        sfl.addWidget(sum_lbl)
        self._mem_right_layout.addWidget(sum_frame)
        self._mem_right_layout.addStretch()

    def _populate_mood_chart(self, mood_readings):
        """Fill the pyqtgraph area chart with 7-day mood history."""
        import numpy as np
        chart = self._mem_chart
        chart.clear()

        if not mood_readings:
            return

        # Build x (timestamp seconds from epoch) and y (mood numeric)
        xs, ys = [], []
        for r in mood_readings:
            ts = r.get("timestamp") or ""
            mood = r.get("fused_mood") or "calm"
            try:
                dt = datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S")
            except Exception:
                try:
                    dt = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
                except Exception:
                    continue
            xs.append(dt.timestamp())
            ys.append(MOOD_NUMERIC.get(mood, 5))

        if not xs:
            return

        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)

        # Area fill
        r, g, b = MOOD_RGB["calm"]
        fill_color = (r, g, b, 55)
        line_color = (r, g, b, 200)

        curve = pg.PlotDataItem(
            xs, ys,
            fillLevel=0,
            fillBrush=pg.mkBrush(*fill_color),
            pen=pg.mkPen(color=line_color, width=2),
            antialias=True,
        )
        chart.addItem(curve)

        # Scatter dots for individual readings
        scatter = pg.ScatterPlotItem(
            x=xs, y=ys,
            symbol="o", size=5,
            brush=pg.mkBrush(*line_color),
            pen=pg.mkPen(None),
        )
        chart.addItem(scatter)

    def _mem_section(self, text):
        lbl = QLabel(text)
        lbl.setFont(ui_font(9))
        lbl.setStyleSheet(f"color: {MUTED}; background: transparent; border: none;")
        return lbl

    def _mem_text(self, text, italic=True):
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setFont(serif_font(13, italic=italic))
        lbl.setStyleSheet(f"color: {MUTED}; background: transparent; border: none;")
        return lbl

    def _make_task_handler(self, task_id, row_widget):
        def handler(event):
            database.complete_task(task_id)
            row_widget.hide()
        return handler

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()


if __name__ == "__main__":
    import sys

    def _log_uncaught(exc_type, exc_value, exc_traceback):
        import traceback
        print("[ui] UNCAUGHT EXCEPTION:")
        traceback.print_exception(exc_type, exc_value, exc_traceback)

    sys.excepthook = _log_uncaught
    database.init_db()
    test_uid = database.get_or_create_user("UI Test User")

    class _StubFace:
        camera_available = False
        running = False
        def get_current_frame(self): return None
        def get_latest_signals(self):
            return {"emotion": "neutral", "confidence": 0.0, "fatigue": 0.0,
                    "engagement": 0.5, "fused_mood": "calm", "face_detected": False}
        def start(self): pass
        def stop(self):  pass

    qt_app = QApplication(sys.argv)
    window = ARIAWindow(test_uid, "UI Test User", _StubFace())
    window.show()
    QTimer.singleShot(900, lambda: window.show_greeting("Hello! ARIA premium UI — 50-layer orb active."))
    sys.exit(qt_app.exec())
