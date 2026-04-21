import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model

# 1. Setup Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

def extract_keypoints(results):
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Consistency with data_collection.py
            hand_type = results.multi_handedness[i].classification[0].label
            landmarks = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            if hand_type == 'Left':
                lh = landmarks
            else:
                rh = landmarks
    return np.concatenate([lh, rh])

# 2. Parameters
actions = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
sequence = []
sentence = []
sequence_length = 30
threshold = 0.8

# 3. Load Model
if os.path.exists('sign_language_model.h5'):
    model = load_model('sign_language_model.h5')
else:
    print("Error: 'sign_language_model.h5' not found. Please train the model first using train_model.py.")
    exit()

# 4. Real-time Detection
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Make detections
        image, results = mediapipe_detection(frame, hands)
        draw_landmarks(image, results)
        
        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-sequence_length:] # Keep only last N frames
        
        if len(sequence) == sequence_length:
            # Faster prediction for real-time
            res = model(np.expand_dims(sequence, axis=0), training=False).numpy()[0]
            predicted_action = actions[np.argmax(res)]
            
            # Visualization logic
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if predicted_action != sentence[-1]:
                        sentence.append(predicted_action)
                else:
                    sentence.append(predicted_action)

            if len(sentence) > 5:
                sentence = sentence[-5:]

            cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('Sign Language Recognition', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
