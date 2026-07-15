"""
DeepSign AI - Shared utilities
Handles MediaPipe hand landmark extraction, feature vector building,
low-light detection, and invalid-gesture confidence checks.
"""

import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Two hands x 21 landmarks x 3 coords (x, y, z) = 126 features
NUM_LANDMARKS = 21
NUM_COORDS = 3
FEATURES_PER_HAND = NUM_LANDMARKS * NUM_COORDS
TOTAL_FEATURES = FEATURES_PER_HAND * 2  # left + right hand slots

# Confidence + validity thresholds
CONFIDENCE_THRESHOLD = 0.70          # below this -> "Uncertain / Invalid gesture"
LOW_LIGHT_BRIGHTNESS_THRESHOLD = 60  # mean pixel brightness (0-255 scale)


def get_hands_detector(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6):
    """Create a MediaPipe Hands detector instance."""
    return mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=0.5,
    )


def check_low_light(frame_bgr):
    """
    Returns (is_low_light: bool, brightness: float)
    Uses mean brightness of the grayscale frame.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    return brightness < LOW_LIGHT_BRIGHTNESS_THRESHOLD, brightness


def extract_landmark_vector(results):
    """
    Given MediaPipe Hands `results`, build a fixed-length 126-dim feature vector.
    Slot 0 = first detected hand, slot 1 = second detected hand (if present).
    Missing hand slots are zero-padded so the model always gets a consistent shape.
    Coordinates are normalized relative to the wrist (landmark 0) of that hand,
    which makes the model robust to hand position in the frame.
    """
    vector = np.zeros(TOTAL_FEATURES, dtype=np.float32)

    if not results.multi_hand_landmarks:
        return vector, False

    for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32,
        )
        # normalize relative to wrist landmark (index 0) for translation invariance
        wrist = coords[0].copy()
        coords -= wrist

        # scale-normalize using the max distance from wrist (size invariance)
        max_dist = np.max(np.linalg.norm(coords, axis=1))
        if max_dist > 1e-6:
            coords /= max_dist

        start = hand_idx * FEATURES_PER_HAND
        end = start + FEATURES_PER_HAND
        vector[start:end] = coords.flatten()

    return vector, True


def draw_landmarks(frame_bgr, results):
    """Draw hand skeleton overlay on the frame for visual feedback."""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_bgr,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )
    return frame_bgr
