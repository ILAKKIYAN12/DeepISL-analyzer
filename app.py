"""
DeepSign AI - Flask Backend
=============================
Run with: python app.py
Then open http://localhost:5000
"""

import os
import time
import sqlite3
import threading
import pickle
from collections import deque

import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify, send_file

from model_utils import (
    get_hands_detector, extract_landmark_vector, draw_landmarks,
    check_low_light, CONFIDENCE_THRESHOLD
)

# ---------- Optional TTS engines (loaded lazily / defensively) ----------
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = "model/deepsign_model.h5"
ENCODER_PATH = "model/label_encoder.pkl"
DB_PATH = "data/history.db"
AUDIO_DIR = "static/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

# ---------------- Load model (if trained) ----------------
model = None
label_encoder = None
if os.path.isfile(MODEL_PATH) and os.path.isfile(ENCODER_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    print(f"Loaded model with signs: {list(label_encoder.classes_)}")
else:
    print("WARNING: No trained model found yet. Run data_collection.py + train_model.py first.")

detector = get_hands_detector(static_image_mode=False)

# ---------------- Shared live-recognition state ----------------
state_lock = threading.Lock()
live_state = {
    "current_word": "",
    "confidence": 0.0,
    "sentence": "",
    "low_light": False,
    "brightness": 0.0,
    "hand_detected": False,
    "emergency_mode": False,
}

STABILITY_FRAMES = 12          # consecutive frames needed to "lock in" a sign
SENTENCE_WORD_GAP_RESET = 1.5  # seconds of no-hand before allowing a repeated word again

_recent_preds = deque(maxlen=STABILITY_FRAMES)
_sentence_words = []
_last_committed_word = None
_last_hand_seen_time = 0.0


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sentence TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


init_db()


def predict_vector(vector):
    """Run the model on a single 126-dim landmark vector. Returns (label, confidence)."""
    if model is None:
        return None, 0.0
    X = vector.reshape(1, 42, 3)
    probs = model.predict(X, verbose=0)[0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    label = label_encoder.classes_[idx]
    return label, confidence


def process_frame_for_sentence(label, confidence, hand_found):
    """
    Stability + sentence-building logic, shared by live feed and video upload.
    Only commits a word once the same prediction is stable for STABILITY_FRAMES,
    and confidence clears CONFIDENCE_THRESHOLD.
    """
    global _last_committed_word, _last_hand_seen_time

    now = time.time()
    if not hand_found:
        _recent_preds.clear()
        if now - _last_hand_seen_time > SENTENCE_WORD_GAP_RESET:
            _last_committed_word = None
        return

    _last_hand_seen_time = now

    if label is None or confidence < CONFIDENCE_THRESHOLD:
        _recent_preds.clear()
        return

    _recent_preds.append(label)

    if len(_recent_preds) == STABILITY_FRAMES and len(set(_recent_preds)) == 1:
        stable_label = _recent_preds[0]
        if stable_label != _last_committed_word:
            _sentence_words.append(stable_label.replace("_", " "))
            _last_committed_word = stable_label
        _recent_preds.clear()


def gen_frames():
    """MJPEG generator for the live webcam feed."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            is_dark, brightness = check_low_light(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)
            frame = draw_landmarks(frame, results)
            vector, hand_found = extract_landmark_vector(results)

            label, confidence = (None, 0.0)
            if hand_found and not is_dark:
                label, confidence = predict_vector(vector)

            process_frame_for_sentence(label, confidence, hand_found)

            with state_lock:
                live_state["current_word"] = label or ""
                live_state["confidence"] = round(confidence * 100, 1)
                live_state["sentence"] = " ".join(_sentence_words)
                live_state["low_light"] = is_dark
                live_state["brightness"] = round(brightness, 1)
                live_state["hand_detected"] = hand_found

            # overlay text on the streamed frame
            overlay_text = f"{label or '...'} ({confidence*100:.0f}%)" if hand_found else "No hand detected"
            color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
            cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if is_dark:
                cv2.putText(frame, "LOW LIGHT", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    finally:
        cap.release()


# ---------------------- ROUTES ----------------------

@app.route("/")
def index():
    signs = list(label_encoder.classes_) if label_encoder is not None else []
    return render_template("index.html", signs=signs, model_ready=(model is not None))


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/get_state")
def get_state():
    with state_lock:
        return jsonify(dict(live_state))


@app.route("/reset_sentence", methods=["POST"])
def reset_sentence():
    global _sentence_words, _last_committed_word
    _sentence_words = []
    _last_committed_word = None
    _recent_preds.clear()
    return jsonify({"ok": True})


@app.route("/toggle_emergency", methods=["POST"])
def toggle_emergency():
    with state_lock:
        live_state["emergency_mode"] = not live_state["emergency_mode"]
        result = live_state["emergency_mode"]
    return jsonify({"emergency_mode": result})


@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400
    if model is None:
        return jsonify({"error": "Model not trained yet"}), 400

    file = request.files["video"]
    temp_path = os.path.join("data", "uploaded_temp.mp4")
    file.save(temp_path)

    cap = cv2.VideoCapture(temp_path)
    local_words = []
    local_preds = deque(maxlen=STABILITY_FRAMES)
    last_committed = None
    frame_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        is_dark, brightness = check_low_light(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        vector, hand_found = extract_landmark_vector(results)

        label, confidence = (None, 0.0)
        if hand_found and not is_dark:
            label, confidence = predict_vector(vector)
            frame_results.append({"label": label, "confidence": round(confidence * 100, 1)})

        if hand_found and label is not None and confidence >= CONFIDENCE_THRESHOLD:
            local_preds.append(label)
            if len(local_preds) == STABILITY_FRAMES and len(set(local_preds)) == 1:
                stable = local_preds[0]
                if stable != last_committed:
                    local_words.append(stable.replace("_", " "))
                    last_committed = stable
                local_preds.clear()
        else:
            local_preds.clear()

    cap.release()
    os.remove(temp_path)

    sentence = " ".join(local_words)
    return jsonify({
        "sentence": sentence,
        "frame_count": len(frame_results),
        "words": local_words,
    })


@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    lang = data.get("lang", "en")  # 'en' or 'ta'
    if not text:
        return jsonify({"error": "No text provided"}), 400

    filename = f"speech_{int(time.time()*1000)}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)

    if lang == "ta":
        if not GTTS_AVAILABLE:
            return jsonify({"error": "gTTS not installed for Tamil voice output"}), 500
        try:
            tts = gTTS(text=text, lang="ta")
            tts.save(filepath)
        except Exception as e:
            return jsonify({"error": f"Tamil TTS failed (needs internet): {str(e)}"}), 500
    else:
        # English: prefer offline pyttsx3 -> wav; fallback to gTTS if unavailable
        if PYTTSX3_AVAILABLE:
            filepath = filepath.replace(".mp3", ".wav")
            filename = filename.replace(".mp3", ".wav")
            engine = pyttsx3.init()
            engine.save_to_file(text, filepath)
            engine.runAndWait()
        elif GTTS_AVAILABLE:
            tts = gTTS(text=text, lang="en")
            tts.save(filepath)
        else:
            return jsonify({"error": "No TTS engine available"}), 500

    return jsonify({"audio_url": f"/static/audio/{filename}"})


@app.route("/history", methods=["GET", "POST", "DELETE"])
def history():
    conn = sqlite3.connect(DB_PATH)
    if request.method == "GET":
        rows = conn.execute(
            "SELECT id, sentence, timestamp FROM history ORDER BY id DESC"
        ).fetchall()
        conn.close()
        return jsonify([{"id": r[0], "sentence": r[1], "timestamp": r[2]} for r in rows])

    if request.method == "POST":
        data = request.get_json(force=True)
        sentence = data.get("sentence", "").strip()
        if sentence:
            conn.execute(
                "INSERT INTO history (sentence, timestamp) VALUES (?, ?)",
                (sentence, time.strftime("%Y-%m-%d %H:%M:%S")),
            )
            conn.commit()
        conn.close()
        return jsonify({"ok": True})

    if request.method == "DELETE":
        conn.execute("DELETE FROM history")
        conn.commit()
        conn.close()
        return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5000)
