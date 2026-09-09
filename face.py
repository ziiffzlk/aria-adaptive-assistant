"""
ARIA Desktop - Face Module
Real-time webcam analysis: facial emotion (DeepFace), fatigue (Eye Aspect
Ratio + blink rate via MediaPipe), engagement (head pose via solvePnP), a
fused face mood signal, and periodic face-based identity verification
(DeepFace.verify() against a one-time-enrolled reference image) so ARIA can
tell whether it's actually talking to its registered user. Runs entirely in
a background daemon thread so the UI never blocks on camera I/O or model
inference.
"""

import math
import os
import sys
import time
import threading
import urllib.request
from collections import deque, Counter

# DeepFace logs emoji characters; Windows consoles default to cp1252 which
# crashes on them. Reconfigure stdio to UTF-8 (with replacement) defensively.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ----------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------

EAR_THRESHOLD = 0.21          # below this, eyes are considered closed
EMOTION_MIN_INTERVAL_S = 2.5  # DeepFace (full TF inference) at most every 2.5s —
                              # mood detection doesn't need to be frame-perfect, and
                              # every inference competes with Chatterbox for the GPU/CPU
EMOTION_CONFIDENCE_THRESHOLD = 0.4  # below this, keep the previous emotion rather than flapping

IDENTITY_VERIFY_INTERVAL_S = 45  # how often to re-check speaker identity — DeepFace.verify()
                                  # is a full inference pass, no need to run it every frame
ENROLLED_FACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "enrolled_faces")

# SFace's own stock cosine threshold (~0.593) rejected the real registered
# user repeatedly in live use. A first widen to 0.80 (based on 5 early
# samples) still wasn't enough — a longer 19-check live session showed real
# matches spread as wide as 0.5153-0.7646 and mismatches as low as 0.8171,
# i.e. real variance this webcam/lighting setup produces for the SAME person,
# not just a slightly-wrong cutoff. 0.90 absorbs that observed spread with
# roughly a tenth still held in reserve, while the enrolled reference is
# ALSO being redone with a clean, direct, well-lit shot (a poor original
# enrollment photo would explain why every comparison ran high) — the two
# fixes are meant to compound, not substitute for each other. This is a
# false-reject vs false-accept tradeoff: for this personal, single-user,
# non-security app, repeatedly telling the real user "I don't recognize you"
# is strictly worse than being lenient — matches the fail-open philosophy
# used everywhere else in this feature (see _verify_identity).
IDENTITY_MATCH_THRESHOLD = 0.90

# This mediapipe build only ships the new Tasks API (no legacy mp.solutions),
# which requires a downloadable .task model file for face landmark detection.
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# MediaPipe FaceMesh 6-point eye landmark indices (outer, top-outer, top-inner,
# inner, bottom-inner, bottom-outer) used for the standard EAR formula.
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX = [263, 387, 385, 362, 380, 373]

# Generic 3D face model (mm) for solvePnP based head pose estimation, paired
# with these MediaPipe landmark indices: nose tip(1), chin(152),
# left eye corner(33), right eye corner(263), left mouth corner(61), right mouth corner(291)
HEAD_POSE_LANDMARK_IDX = [1, 152, 33, 263, 61, 291]
HEAD_POSE_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
])

# Warm theme colors, converted from ui.py's hex palette to OpenCV's BGR order.
BG_BGR = (10, 11, 13)          # #0d0b0a
BORDER_BGR = (32, 37, 42)      # #2a2520
CREAM_BGR = (232, 240, 245)    # #f5f0e8
MUTED_BGR = (112, 128, 138)    # #8a8070
ACCENT_BGR = (124, 168, 232)   # #e8a87c
ROSE_BGR = (123, 123, 193)     # #c17b7b
SUCCESS_BGR = (154, 184, 126)  # #7eb89a


def _ensure_face_landmarker_model():
    """Download the FaceLandmarker .task model on first run if not already present."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        return MODEL_PATH
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        print("[face] Downloading MediaPipe face landmark model (first run only)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[face] Model downloaded successfully")
        return MODEL_PATH
    except Exception as e:
        print(f"[face] Could not download face landmark model ({e}) - "
              f"fatigue/engagement detection will be disabled")
        return None


class FaceAnalyzer:
    """Owns the webcam and a background thread that continuously analyses frames."""

    def __init__(self):
        self.face_landmarker = None
        self._model_path = _ensure_face_landmarker_model()
        self._create_landmarker()

        self._stream_start_time = None
        self.cap = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.camera_available = False

        self.frame_count = 0
        self.latest_frame = None      # annotated PIL.Image, set under self.lock
        self._latest_raw_frame = None  # raw BGR ndarray, set under self.lock (needed for enroll())
        self._last_emotion_time = 0.0

        self.latest_emotion = "neutral"
        self.latest_emotion_confidence = 0.0
        self.latest_fatigue = 0.0
        self.latest_engagement = 0.5
        self.face_detected = False
        self.emotion_history = deque(maxlen=5)  # rolling window for smoothed/most-common emotion

        # Blink / fatigue tracking state
        self.eyes_closed_prev = False
        self.closed_start_time = None
        self.blink_timestamps = deque()
        self.fatigue_ema = 0.0

        # Identity verification state. Fail-open by default (True) — this is
        # a personal single-user desktop app, not a security product, so the
        # absence of a check (no enrollment yet, camera off, a verification
        # error) must never itself block normal use.
        self._enrolled_path = None
        self._last_identity_check_time = 0.0
        self.speaker_verified = True
        self.verification_distance = None
        # Hysteresis over the raw per-check reads: a single noisy misread
        # must not flip speaker_verified — that flag gates whether the LLM
        # prompt gets RECENT CONVERSATION HISTORY at all, so one bad frame
        # was observed live to make ARIA appear to "forget" a conversation
        # from moments earlier. Real data showed genuine matches for the
        # SAME person spread from 0.16 to 1.02 distance in one session — the
        # signal itself is noisy, not just mis-thresholded. Only 2
        # CONSECUTIVE raw mismatches flip the effective state to unverified;
        # a single subsequent match restores it immediately (asymmetric on
        # purpose — false-reject is the worse failure mode here).
        self._recent_identity_matches = deque(maxlen=2)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _create_landmarker(self):
        """(Re)create the MediaPipe FaceLandmarker. stop() closes it, so every
        start() needs a fresh instance — reusing a closed one throws on each frame."""
        if not self._model_path:
            return
        try:
            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=self._model_path),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"[face] Failed to initialise FaceLandmarker: {e}")
            self.face_landmarker = None

    def start(self):
        """Open the webcam and start the background capture/analysis thread."""
        if self.face_landmarker is None:
            self._create_landmarker()
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap or not self.cap.isOpened():
                print("[face] Camera not available - ARIA will run without face analysis")
                self.camera_available = False
                return False
        except Exception as e:
            print(f"[face] Failed to open camera: {e}")
            self.camera_available = False
            return False

        self.camera_available = True
        self.running = True
        self._stream_start_time = time.time()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("[face] Camera started, face analysis running in background")
        return True

    def stop(self):
        """Stop the analysis thread and release the camera."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.cap is not None:
            self.cap.release()
        if self.face_landmarker is not None:
            try:
                self.face_landmarker.close()
            except Exception:
                pass
            del self.face_landmarker
            self.face_landmarker = None  # start() recreates it fresh
        print("[face] Camera stopped")

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _capture_loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)  # mirror for a natural "selfie" view
            self.frame_count += 1

            landmarks = None
            if self.face_landmarker is not None:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    timestamp_ms = int((time.time() - self._stream_start_time) * 1000)
                    result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
                    if result.face_landmarks:
                        landmarks = result.face_landmarks[0]
                except Exception as e:
                    print(f"[face] face landmark detection failed: {e}")
                    landmarks = None

            # DeepFace does its own face detection internally, so emotion
            # analysis runs independently of whether mediapipe found landmarks.
            # Time-gated: a full TensorFlow inference per 5 frames (~6/sec at
            # 30fps) was sustained CPU burn for a signal read every few seconds.
            now = time.time()
            emotion = confidence = None
            if now - self._last_emotion_time >= EMOTION_MIN_INTERVAL_S:
                self._last_emotion_time = now
                emotion, confidence = self.analyze_emotion(frame)

            fatigue = engagement = None
            if landmarks:
                fatigue = self.detect_fatigue(landmarks)
                engagement = self.detect_engagement(landmarks, frame.shape[1], frame.shape[0])

            with self.lock:
                if emotion is not None:
                    self.latest_emotion = self._get_smoothed_emotion(emotion)
                    self.latest_emotion_confidence = confidence
                self.face_detected = bool(landmarks)
                if fatigue is not None:
                    self.latest_fatigue = fatigue
                if engagement is not None:
                    self.latest_engagement = engagement
                self._latest_raw_frame = frame

            # Identity verification: only worth attempting when a face is
            # actually present this cycle (skips wasted DeepFace calls on
            # empty frames) and an enrollment reference exists. Time-gated
            # like emotion analysis — a full DeepFace.verify() pass every
            # frame would be wasteful for a signal that only needs checking
            # every 30-60s.
            if landmarks and self._enrolled_path and (now - self._last_identity_check_time >= IDENTITY_VERIFY_INTERVAL_S):
                self._last_identity_check_time = now
                self._verify_identity(frame)

            annotated = self.draw_overlay(frame, self.latest_emotion, self.latest_fatigue, self.latest_engagement)
            pil_image = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

            with self.lock:
                self.latest_frame = pil_image

            time.sleep(0.01)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_current_frame(self):
        """Return the latest annotated frame as a PIL Image, or None if unavailable."""
        with self.lock:
            return self.latest_frame

    def get_latest_signals(self):
        """Return a thread-safe snapshot of the latest face signals plus fused mood."""
        with self.lock:
            emotion = self.latest_emotion
            confidence = self.latest_emotion_confidence
            fatigue = self.latest_fatigue
            engagement = self.latest_engagement
            face_detected = self.face_detected
            speaker_verified = self.speaker_verified
            verification_distance = self.verification_distance
        return {
            "emotion": emotion,
            "confidence": confidence,
            "fatigue": fatigue,
            "engagement": engagement,
            "fused_mood": self.get_fused_face_mood(emotion, fatigue, engagement),
            "face_detected": face_detected,
            "speaker_verified": speaker_verified,
            "verification_distance": verification_distance,
        }

    # ------------------------------------------------------------------
    # Identity verification
    # ------------------------------------------------------------------

    def set_enrollment_path(self, path):
        """Point verification at an existing enrolled reference image, if one
        exists on disk. No-op (stays fail-open/unenrolled) if it doesn't."""
        if path and os.path.exists(path):
            self._enrolled_path = path
            print(f"[identity] Using existing enrollment: {path}")
        else:
            print(f"[identity] No enrolled face on file at {path} — "
                  f"verification stays fail-open until enroll() succeeds")

    def enroll(self, save_path, timeout_s=30):
        """
        Capture and save a reference face image for future identity
        verification. Waits up to timeout_s for a confidently-detected face
        from the live camera feed (requires start() to already be running).
        Returns True on success, False if no face was found in time — in
        either case, verification remains fail-open (speaker_verified stays
        True) until a successful enrollment exists, consistent with this
        being a personal convenience feature, not a security gate.
        """
        print("[identity] Enrolling reference face — please look at the camera...")
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self.lock:
                frame = self._latest_raw_frame
                detected = self.face_detected
            if frame is not None and detected:
                try:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    cv2.imwrite(save_path, frame)
                    self._enrolled_path = save_path
                    print(f"[identity] Enrollment saved: {save_path}")
                    return True
                except Exception as e:
                    print(f"[identity] Failed to save enrollment image: {e}")
                    return False
            time.sleep(0.3)
        print(f"[identity] Enrollment failed — no face detected within {timeout_s}s. "
              f"Verification will stay fail-open (always verified) until enrolled "
              f"(re-run enrollment, or delete/replace the file at the expected path to retry).")
        return False

    def _verify_identity(self, frame):
        """
        Compare the current frame against the enrolled reference via
        DeepFace.verify(). model_name="SFace" (~5MB weights, vs DeepFace's
        default VGG-Face at ~580MB) — chosen after a live run showed VGG-
        Face's download taking 14+ hours on this network; SFace is still a
        real, purpose-built face-recognition model, just far more practical
        to fetch. The match decision uses IDENTITY_MATCH_THRESHOLD against
        the raw `distance`, NOT DeepFace's own `verified` field — the stock
        threshold measurably rejected the real user too often in live use
        (see IDENTITY_MATCH_THRESHOLD's comment for the real numbers).
        Fails open (stays verified) on any error — e.g. a transient decode
        issue must not lock the real user out. The EFFECTIVE speaker_verified
        exposed to the rest of the app applies hysteresis over these raw
        reads (see _recent_identity_matches in __init__) — a single mismatch
        does not flip it.
        """
        try:
            from deepface import DeepFace
            result = DeepFace.verify(
                img1_path=self._enrolled_path, img2_path=frame,
                model_name="SFace", enforce_detection=False,
                detector_backend="opencv", silent=True,
            )
            distance = result.get("distance")
            raw_match = isinstance(distance, (int, float)) and distance <= IDENTITY_MATCH_THRESHOLD
            dist_str = f"{distance:.4f}" if isinstance(distance, (int, float)) else "None"
        except Exception as e:
            print(f"[identity] verification failed ({e}) — defaulting to verified (fail-open)")
            raw_match = True
            distance = None
            dist_str = "None"

        with self.lock:
            self._recent_identity_matches.append(raw_match)
            # Unverified only once we HAVE 2 recent reads AND both missed —
            # any match in the last 2 keeps (or restores) verified status.
            effective = (len(self._recent_identity_matches) < 2
                         or any(self._recent_identity_matches))
            self.speaker_verified = effective
            self.verification_distance = distance
        print(f"[identity] Match={raw_match} (effective={effective}), distance={dist_str} "
              f"(threshold={IDENTITY_MATCH_THRESHOLD})")

    # ------------------------------------------------------------------
    # Emotion
    # ------------------------------------------------------------------

    def analyze_emotion(self, frame):
        """
        Run DeepFace emotion analysis on a frame. The frame is first
        contrast-enhanced via CLAHE (on a grayscale copy, then converted back
        to 3-channel) to compensate for poor/uneven webcam lighting - this
        also better matches the grayscale FER2013 data DeepFace's emotion
        model was trained on. Returns (dominant_emotion, confidence 0-1).
        Below EMOTION_CONFIDENCE_THRESHOLD, keeps the previous reading
        instead of flapping to an unreliable low-confidence guess.
        """
        try:
            from deepface import DeepFace

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            frame_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

            result = DeepFace.analyze(
                frame_enhanced, actions=["emotion"], enforce_detection=False,
                detector_backend="opencv", silent=True,
            )
            if isinstance(result, list):
                result = result[0]
            emotions = result.get("emotion", {})
            dominant = result.get("dominant_emotion", "neutral")
            confidence = emotions.get(dominant, 0.0) / 100.0

            if confidence > EMOTION_CONFIDENCE_THRESHOLD:
                return dominant, round(confidence, 2)
            return self.latest_emotion, self.latest_emotion_confidence
        except Exception as e:
            print(f"[face] analyze_emotion failed: {e}")
            return self.latest_emotion, self.latest_emotion_confidence

    def _get_smoothed_emotion(self, new_emotion):
        """Rolling-window smoothing: returns the most common of the last 5 readings."""
        self.emotion_history.append(new_emotion)
        return Counter(self.emotion_history).most_common(1)[0][0]

    # ------------------------------------------------------------------
    # Fatigue (Eye Aspect Ratio + blink rate)
    # ------------------------------------------------------------------

    @staticmethod
    def _eye_aspect_ratio(landmarks, idx):
        p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in idx]

        def dist(a, b):
            return math.hypot(a.x - b.x, a.y - b.y)

        vertical = dist(p2, p6) + dist(p3, p5)
        horizontal = dist(p1, p4)
        if horizontal == 0:
            return 0.3
        return vertical / (2.0 * horizontal)

    def detect_fatigue(self, landmarks):
        """
        Returns a fatigue score 0.0-1.0 based on blink rate (>25/min = fatigued)
        and sustained eye closure (microsleep / heavy eyelids).
        """
        try:
            left_ear = self._eye_aspect_ratio(landmarks, LEFT_EYE_IDX)
            right_ear = self._eye_aspect_ratio(landmarks, RIGHT_EYE_IDX)
            ear = (left_ear + right_ear) / 2.0

            now = time.time()
            closed = ear < EAR_THRESHOLD

            if closed and not self.eyes_closed_prev:
                self.closed_start_time = now
            if not closed and self.eyes_closed_prev:
                self.blink_timestamps.append(now)
            self.eyes_closed_prev = closed

            while self.blink_timestamps and now - self.blink_timestamps[0] > 60:
                self.blink_timestamps.popleft()

            window = max(now - self.blink_timestamps[0], 5.0) if self.blink_timestamps else 5.0
            blink_rate_per_min = (len(self.blink_timestamps) / window) * 60.0
            rate_score = min(blink_rate_per_min / 25.0, 1.0)

            sustained_score = 0.0
            if closed and self.closed_start_time:
                sustained_score = min((now - self.closed_start_time) / 1.5, 1.0)

            raw_fatigue = max(rate_score, sustained_score)
            self.fatigue_ema = 0.8 * self.fatigue_ema + 0.2 * raw_fatigue
            return round(min(max(self.fatigue_ema, 0.0), 1.0), 2)
        except Exception as e:
            print(f"[face] detect_fatigue failed: {e}")
            return self.latest_fatigue

    # ------------------------------------------------------------------
    # Engagement (head pose)
    # ------------------------------------------------------------------

    @staticmethod
    def _rotation_matrix_to_euler(R):
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            pitch = math.atan2(R[2, 1], R[2, 2])
            yaw = math.atan2(-R[2, 0], sy)
            roll = math.atan2(R[1, 0], R[0, 0])
        else:
            pitch = math.atan2(-R[1, 2], R[1, 1])
            yaw = math.atan2(-R[2, 0], sy)
            roll = 0
        return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)

    def detect_engagement(self, landmarks, width, height):
        """
        Estimates head pose (pitch/yaw) via solvePnP and returns an engagement
        score 0.0-1.0: facing the screen squarely scores high, looking away
        or down scores low.
        """
        try:
            image_points = np.array([
                (landmarks[i].x * width, landmarks[i].y * height)
                for i in HEAD_POSE_LANDMARK_IDX
            ], dtype="double")

            focal_length = width
            center = (width / 2.0, height / 2.0)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ], dtype="double")
            dist_coeffs = np.zeros((4, 1))

            success, rotation_vector, _ = cv2.solvePnP(
                HEAD_POSE_MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                return self.latest_engagement

            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            pitch, yaw, _roll = self._rotation_matrix_to_euler(rotation_matrix)

            yaw_norm = min(abs(yaw) / 45.0, 1.0)
            pitch_norm = min(abs(pitch) / 35.0, 1.0)
            deviation = 0.6 * yaw_norm + 0.4 * pitch_norm
            engagement = max(0.0, 1.0 - deviation)
            return round(engagement, 2)
        except Exception as e:
            print(f"[face] detect_engagement failed: {e}")
            return self.latest_engagement

    # ------------------------------------------------------------------
    # Overlay drawing
    # ------------------------------------------------------------------

    def draw_overlay(self, frame, emotion, fatigue, engagement):
        """Draw emotion label, fatigue bar, and engagement indicator onto a copy of frame."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Emotion label, top-left
        cv2.rectangle(annotated, (10, 10), (230, 48), BG_BGR, -1)
        cv2.rectangle(annotated, (10, 10), (230, 48), BORDER_BGR, 1)
        cv2.putText(annotated, emotion.capitalize(), (20, 37),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, ACCENT_BGR, 2)

        # Fatigue bar, right side, vertical, fills bottom-up
        bar_x, bar_top, bar_bottom = w - 40, 25, h - 25
        bar_height = bar_bottom - bar_top
        cv2.rectangle(annotated, (bar_x, bar_top), (bar_x + 20, bar_bottom), BORDER_BGR, 2)
        fill_height = int(bar_height * min(max(fatigue, 0.0), 1.0))
        fill_color = ROSE_BGR if fatigue > 0.6 else CREAM_BGR
        if fill_height > 0:
            cv2.rectangle(annotated, (bar_x, bar_bottom - fill_height), (bar_x + 20, bar_bottom), fill_color, -1)
        cv2.putText(annotated, "Fatigue", (bar_x - 18, bar_top - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, MUTED_BGR, 1)

        # Engagement indicator, bottom-left
        eng_color = SUCCESS_BGR if engagement > 0.6 else (ROSE_BGR if engagement < 0.4 else ACCENT_BGR)
        cv2.circle(annotated, (28, h - 28), 12, eng_color, -1)
        cv2.putText(annotated, f"Engagement {int(engagement * 100)}%", (48, h - 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CREAM_BGR, 1)

        return annotated

    # ------------------------------------------------------------------
    # Fused face mood
    # ------------------------------------------------------------------

    def get_fused_face_mood(self, emotion, fatigue, engagement):
        """Combine emotion + fatigue + engagement into one unified face mood label."""
        if fatigue > 0.7:
            return "tired"
        if emotion == "happy" and engagement > 0.6:
            return "happy"
        if emotion in ("sad", "fear") and engagement < 0.4:
            return "sad"
        if emotion in ("angry", "disgust"):
            return "frustrated"
        if emotion == "surprise":
            return "surprised"
        if emotion == "neutral" and engagement > 0.6:
            return "calm"
        if emotion == "neutral" and engagement < 0.4:
            return "distracted"

        fallback = {"sad": "sad", "fear": "anxious", "happy": "happy"}
        return fallback.get(emotion, "calm")


if __name__ == "__main__":
    # Quick manual smoke test: python face.py
    # Opens the webcam for ~6 seconds, prints live signals, then shuts down cleanly.
    analyzer = FaceAnalyzer()
    if analyzer.start():
        for _ in range(12):
            time.sleep(0.5)
            print(analyzer.get_latest_signals())
        analyzer.stop()
    else:
        print("No camera detected - skipping live test.")
