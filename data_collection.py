"""
DeepSign AI - Data Collection Script
=====================================
Run this on YOUR machine (needs a webcam).

Usage:
    python data_collection.py

For each sign:
  - Get into position, press 'c' to start capturing (auto-captures SAMPLES_PER_SIGN frames)
  - Press 'n' to move to the next sign early
  - Press 'q' to quit anytime (saves whatever is collected so far)

Edit SIGN_LABELS below to match the signs you want to recognize.
Aim for 150-200 samples per sign; vary hand angle/position slightly between samples
for a more robust model.
"""

import cv2
import os
import csv
import time
from model_utils import get_hands_detector, extract_landmark_vector, check_low_light, draw_landmarks

# ---- EDIT THIS LIST to your chosen 10-15 ISL signs ----
SIGN_LABELS = ["I", "You", "Need", "Want", "Water", "Help", "Yes", "No"]

SAMPLES_PER_SIGN = 30
DATA_DIR = "data"
CAPTURE_DELAY = 0.00015  # seconds between auto-captured samples


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, "landmark_data.csv")

    # header: 126 feature columns + label
    header = [f"f{i}" for i in range(126)] + ["label"]
    file_exists = os.path.isfile(csv_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check camera index / permissions.")
        return

    detector = get_hands_detector(static_image_mode=False)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)

        for sign in SIGN_LABELS:
            print(f"\n=== Sign: {sign} ===")
            print("Press 'c' to start auto-capture, 'n' to skip, 'q' to quit.")
            capturing = False
            collected = 0

            while collected < SAMPLES_PER_SIGN:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                is_dark, brightness = check_low_light(frame)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.process(rgb)
                frame = draw_landmarks(frame, results)

                vector, hand_found = extract_landmark_vector(results)

                status = f"Sign: {sign} | Collected: {collected}/{SAMPLES_PER_SIGN}"
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if is_dark:
                    cv2.putText(frame, "LOW LIGHT - improve lighting", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("DeepSign AI - Data Collection", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('c'):
                    capturing = True
                elif key == ord('n'):
                    break
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Quit. Data saved so far.")
                    return

                if capturing and hand_found and not is_dark:
                    writer.writerow(list(vector) + [sign])
                    collected += 1
                    time.sleep(CAPTURE_DELAY)

            print(f"Done with '{sign}': {collected} samples collected.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nAll data saved to {csv_path}")
    print("Next step: run `python train_model.py`")


if __name__ == "__main__":
    main()
