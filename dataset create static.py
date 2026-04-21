import cv2
import mediapipe as mp
import csv
import os
from collections import defaultdict

# === Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# === Constants ===
SAMPLES_PER_SIDE = 200  # samples per side
output_folder = "gesture_dataset"
os.makedirs(output_folder, exist_ok=True)

header = ['label'] + [f'L_{j}_{a}' for j in range(21) for a in ['x','y','z']] + [f'R_{j}_{a}' for j in range(21) for a in ['x','y','z']]

# === Webcam setup ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not access webcam.")
    exit()

# === State Variables ===
person_id = 1
current_label = None
hand_side = "Left"
collecting = False
waiting_for_right = False
asking_extend = False
waiting_for_next = False
sample_counts = defaultdict(int)
import numpy as np
writer, f = None, None

print("👉 Press A–Z to start data collection. Press '-' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # === Collect samples ===
    if results.multi_hand_landmarks and collecting:
        mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        if len(results.multi_hand_landmarks) > 1:
            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[1], mp_hands.HAND_CONNECTIONS)

        # Initialize both hands as zeros
        lh = np.zeros(63)
        rh = np.zeros(63)

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_type = results.multi_handedness[i].classification[0].label
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            if hand_type == 'Left':
                lh = landmarks
            else:
                rh = landmarks

        # Save combined keypoints
        sample_counts[f"P{person_id}_{current_label}"] += 1
        writer.writerow([current_label] + lh.tolist() + rh.tolist())

        if sample_counts[f"P{person_id}_{current_label}"] >= SAMPLES_PER_SIDE:
            collecting = False
            print(f"✅ Finished collection for '{current_label}'")
            f.close()
            asking_extend = True

    # === On-screen messages ===
    if asking_extend:
        text = f"✅ '{current_label}' done! Add more? (Y/N)"
    elif waiting_for_right:
        text = f"Press ENTER to start RIGHT hand for '{current_label}'"
    elif collecting:
        count = sample_counts.get(f"P{person_id}_{current_label}_{hand_side}", 0)
        text = f"Collecting {hand_side} | {current_label} | {count}/{SAMPLES_PER_SIDE}"
    elif waiting_for_next:
        text = "Press A–Z for next letter or '-' to stop"
    else:
        text = "Press A–Z to start new label"

    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Gesture Collector (Hackathon Mode)", frame)

    # === Key Controls ===
    key = cv2.waitKey(1) & 0xFF

    # Exit
    if key == ord('-'):
        print("👋 Exiting...")
        break

    # Start new label
    elif ((65 <= key <= 90 or 97 <= key <= 122) or (48 <= key <= 57)) and not collecting and not asking_extend:
        current_label = chr(key).upper()
        output_file = os.path.join(output_folder, f"P{person_id}_{current_label}.csv")
        file_exists = os.path.isfile(output_file)
        f = open(output_file, 'a', newline='')
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        collecting = True
        print(f"✋ Starting 2-hand collection for '{current_label}'")

    # Extend dataset for same label
    elif asking_extend:
        if key in [ord('y'), ord('Y')]:
            asking_extend = False
            output_file = os.path.join(output_folder, f"P{person_id}_{current_label}.csv")
            f = open(output_file, 'a', newline='')
            writer = csv.writer(f)
            print(f"🔁 Extending dataset for '{current_label}'...")
            collecting = True
        elif key in [ord('n'), ord('N')]:
            asking_extend = False
            waiting_for_next = True
            current_label = None
            print("➡️ Ready for next label")

cap.release()
cv2.destroyAllWindows()

print("\n📊 Final Summary:")
for label in sorted(sample_counts):
    print(f"{label}: {sample_counts[label]}")
