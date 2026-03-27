"""
Hand tracking module using MediaPipe Hands.

Tracks both hands simultaneously and extracts:
  - Normalized wrist position (x, y) in [0, 1]
  - Wrist roll angle (rotation around the pointing axis)
  - Pinch ratio (thumb-to-index distance, normalised) for gripper control
  - Handedness label ('Left' / 'Right') from MediaPipe

All output values are smoothed with an exponential moving average (EMA)
to suppress per-frame noise before being used as robot commands.
"""

import math

import mediapipe as mp
import numpy as np

# MediaPipe landmark indices
_WRIST = 0
_THUMB_TIP = 4
_INDEX_TIP = 8
_MIDDLE_TIP = 12
_RING_TIP = 16
_PINKY_TIP = 20
_INDEX_MCP = 5
_MIDDLE_MCP = 9
_RING_MCP = 13
_PINKY_MCP = 17


def _detect_gesture(lm) -> str:
    """
    Classify hand gesture from raw landmarks.

    Returns
    -------
    'pointing' : index extended upward, other fingers curled  → START control
    'fist'     : all fingers curled                           → STOP control
    'other'    : any other pose
    """
    # A finger is "extended" if its tip is ABOVE (smaller y) its MCP in screen coords
    index_up = lm[_INDEX_TIP].y < lm[_INDEX_MCP].y
    middle_up = lm[_MIDDLE_TIP].y < lm[_MIDDLE_MCP].y
    ring_up = lm[_RING_TIP].y < lm[_RING_MCP].y
    pinky_up = lm[_PINKY_TIP].y < lm[_PINKY_MCP].y

    if index_up and not middle_up and not ring_up and not pinky_up:
        return 'pointing'
    if not index_up and not middle_up and not ring_up and not pinky_up:
        return 'fist'
    return 'other'


class SmoothedValue:
    """Exponential moving average smoother for a scalar or array value."""

    def __init__(self, alpha: float = 0.25):
        self._alpha = alpha
        self._value = None

    def update(self, new_value):
        if self._value is None:
            self._value = np.asarray(new_value, dtype=float)
        else:
            self._value = (
                self._alpha * np.asarray(new_value, dtype=float)
                + (1.0 - self._alpha) * self._value
            )
        return self._value.copy()

    def reset(self):
        self._value = None


class HandState:
    """Processed, smoothed state for one hand."""

    def __init__(self, alpha: float = 0.25):
        self._pos_smoother = SmoothedValue(alpha)
        self._angle_smoother = SmoothedValue(alpha)
        self._pinch_smoother = SmoothedValue(alpha)

    def update(self, landmarks, frame_w: int, frame_h: int):
        """
        Compute hand state from MediaPipe NormalizedLandmarkList.

        Returns
        -------
        dict with keys:
          x, y         : smoothed wrist position in [0, 1]
          angle        : smoothed wrist roll in radians [-pi, pi]
          pinch        : smoothed pinch ratio in [0, 1]  (0=closed, 1=open)
          raw_x, raw_y : unsmoothed wrist position
        """
        lm = landmarks.landmark

        # --- wrist position (normalised) ---
        wx, wy = lm[_WRIST].x, lm[_WRIST].y
        pos = self._pos_smoother.update([wx, wy])

        # --- roll angle: angle of index_mcp -> pinky_mcp vector (hand "tilt") ---
        dx = lm[_PINKY_MCP].x - lm[_INDEX_MCP].x
        dy = lm[_PINKY_MCP].y - lm[_INDEX_MCP].y
        raw_angle = math.atan2(dy, dx)
        # Smooth on [cos, sin] to avoid ±π wraparound artifacts in EMA
        vec = self._angle_smoother.update([math.cos(raw_angle), math.sin(raw_angle)])
        angle = math.atan2(vec[1], vec[0])

        # --- pinch: normalised thumb-tip to index-tip distance ---
        # Reference distance: wrist to middle MCP (scale-invariant)
        ref_dx = lm[_MIDDLE_MCP].x - lm[_WRIST].x
        ref_dy = lm[_MIDDLE_MCP].y - lm[_WRIST].y
        ref_dist = math.hypot(ref_dx * frame_w, ref_dy * frame_h) + 1e-6

        pinch_dx = lm[_THUMB_TIP].x - lm[_INDEX_TIP].x
        pinch_dy = lm[_THUMB_TIP].y - lm[_INDEX_TIP].y
        pinch_dist = math.hypot(pinch_dx * frame_w, pinch_dy * frame_h)
        # Clip: 0 = fully pinched (fingers touching), 1 = fully open
        pinch_ratio = float(np.clip(pinch_dist / ref_dist, 0.0, 1.0))
        pinch = float(self._pinch_smoother.update(pinch_ratio))

        return {
            "x": float(pos[0]),
            "y": float(pos[1]),
            "angle": angle,
            "pinch": pinch,
            "raw_x": wx,
            "raw_y": wy,
            "gesture": _detect_gesture(lm),
        }

    def reset(self):
        self._pos_smoother.reset()
        self._angle_smoother.reset()
        self._pinch_smoother.reset()


class HandTracker:
    """
    Wraps MediaPipe Hands for dual-hand detection.

    Usage
    -----
        tracker = HandTracker()
        while True:
            frame = camera.read()
            results = tracker.process(frame)
            left  = results.get('Left')   # HandState dict or None
            right = results.get('Right')  # HandState dict or None
    """

    def __init__(
        self,
        alpha: float = 0.25,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._mp_draw = mp.solutions.drawing_utils
        self._states = {
            "Left": HandState(alpha),
            "Right": HandState(alpha),
        }
        self._last_seen = {"Left": False, "Right": False}

    def process(self, bgr_frame):
        """
        Process a BGR frame and return per-hand state dicts.

        Returns
        -------
        dict: {'Left': state_dict_or_None, 'Right': state_dict_or_None}
        """
        h, w = bgr_frame.shape[:2]
        import cv2

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        results_mp = self._hands.process(rgb)

        output = {"Left": None, "Right": None}

        if results_mp.multi_hand_landmarks and results_mp.multi_handedness:
            for hand_landmarks, handedness in zip(
                results_mp.multi_hand_landmarks, results_mp.multi_handedness
            ):
                label = handedness.classification[0].label  # 'Left' or 'Right'
                # MediaPipe detects actual hand anatomy (not image position),
                # so no label flip is needed — 'Right' == operator's right hand.
                state = self._states[label].update(hand_landmarks, w, h)
                output[label] = state
                self._last_seen[label] = True
        else:
            for side in ("Left", "Right"):
                if self._last_seen[side]:
                    self._states[side].reset()
                    self._last_seen[side] = False

        return output, results_mp

    def draw(self, frame, mp_results):
        """Draw hand landmarks on frame (in-place)."""
        if mp_results.multi_hand_landmarks:
            for hand_landmarks in mp_results.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                )
        return frame

    def close(self):
        self._hands.close()
