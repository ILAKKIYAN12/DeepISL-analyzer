# DeepSign AI — Smart Indian Sign Language Recognition

## Setup (run once)
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Run order (do this tonight, in this order)

### 1. Collect data (~20-30 min, needs your webcam)
Open `data_collection.py`, edit `SIGN_LABELS` to your final 10-15 signs, then:
```bash
python data_collection.py
```
For each sign: press `c` to auto-capture ~180 samples (vary hand angle slightly),
`n` to skip to next sign, `q` to quit early. Do this in good lighting.

### 2. Train the model (~1-5 min)
```bash
python train_model.py
```
Saves `model/deepsign_model.h5`, `model/label_encoder.pkl`, and
`model/training_report.png` (accuracy/loss curves — put this in your report).

### 3. Run the web app
```bash
python app.py
```
Open **http://localhost:5000**

## Architecture (for your report / viva)

**Why landmarks instead of raw video frames for the CNN?**
Training a CNN directly on raw webcam frames for 10-15 gesture classes needs
thousands of labeled images per class and hours of GPU training — not feasible
on a laptop overnight. Instead:

1. **MediaPipe Hands** detects 21 3D landmarks per hand in real time.
2. Landmarks are **normalized** (translated to the wrist, scaled by max
   distance from wrist) so the model is invariant to hand position/size in
   frame — this is a standard preprocessing step, not a shortcut.
3. The normalized landmark vector (up to 2 hands × 21 points × 3 coords = 126
   features, reshaped to 42×3) is fed into a **1D Convolutional Neural
   Network** (Conv1D → BatchNorm → MaxPool → Conv1D → GlobalAveragePooling →
   Dense layers), trained with sparse categorical cross-entropy.

This is a legitimate, published approach to real-time gesture recognition
(landmark-based classification is what MediaPipe's own gesture recognizer and
many ISL/ASL research papers use) — you can defend this design choice
directly if asked why you didn't train on raw pixels.

**Feature checklist mapping:**
- Live Webcam Recognition → `/video_feed` MJPEG stream + live polling of `/get_state`
- Upload Video Recognition → `/upload` route, frame-by-frame processing
- CNN Deep Learning Model → `train_model.py`, Conv1D architecture
- MediaPipe Hand Tracking → `model_utils.py`
- Sentence Formation → stability-buffer logic in `app.py` (`process_frame_for_sentence`)
- English/Tamil Voice Output → `/speak` route (pyttsx3 offline for English, gTTS for Tamil — **Tamil needs internet**, mention this limitation if asked)
- Confidence Score → shown live, thresholded at 70%
- Invalid Gesture Detection → confidence < 70% flagged as "Invalid/Uncertain"
- Low-Light Detection → mean grayscale brightness check in `model_utils.py`
- Conversation History / Clear History → SQLite (`data/history.db`) via `/history`
- Emergency Mode → quick-access phrase overlay, speaks instantly
- Sign Learning Mode → reference tab listing trained signs
- Responsive Web Interface → CSS grid, mobile breakpoint at 900px

## Known limitations (be upfront about these if asked)
- Model accuracy depends entirely on how much/clean data you collect tonight —
  more samples per sign and consistent lighting = better accuracy.
- Tamil TTS requires an internet connection (gTTS calls Google's API).
- This recognizes **isolated static/short-gesture signs**, not continuous
  fluent ISL sentences with grammar — a reasonable and common scope for a
  3rd-year project.
