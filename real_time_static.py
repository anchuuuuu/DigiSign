import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import os

# 1. Setup Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 2. Load Model and Labels
if not os.path.exists('static_gesture_model.h5') or not os.path.exists('label_map.pkl'):
    print("❌ Error: Model or label map not found. Run train_static_model.py first!")
    exit()

model = load_model('static_gesture_model.h5')
with open('label_map.pkl', 'rb') as f:
    unique_labels = pickle.load(f)

# 3. Webcam Loop
cap = cv2.VideoCapture(0)

print("🚀 Real-time recognition started! Press 'Q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        # Initialize both hands as zeros
        lh = np.zeros(63)
        rh = np.zeros(63)

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Identify hand type
            hand_type = results.multi_handedness[i].classification[0].label
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            
            if hand_type == 'Left':
                lh = landmarks
            else:
                rh = landmarks
            
        # Combine landmarks for both hands
        combined_landmarks = np.concatenate([lh, rh])
        
        # Predict
        prediction = model.predict(np.expand_dims(combined_landmarks, axis=0), verbose=0)
        class_id = np.argmax(prediction)
        confidence = prediction[0][class_id]
        label = unique_labels[class_id]

        # Display
        if confidence > 0.7:
            text = f"Digit: {label} ({confidence*100:.1f}%)"
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Real-time Static Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
