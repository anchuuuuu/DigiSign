import cv2
import numpy as np
import os
import time
import mediapipe as mp

# 1. Setup Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB 2 BGR
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
            # Check handedness label to correctly assign landmarks
            hand_type = results.multi_handedness[i].classification[0].label
            landmarks = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            if hand_type == 'Left':
                lh = landmarks
            else:
                rh = landmarks
    return np.concatenate([lh, rh])

# 2. Setup Folders for Collection
DATA_PATH = os.path.join('MP_Data') 
# Actions we want to recognize
actions = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
# Number of videos (sequences) per action
no_sequences = 30
# Frames per video
sequence_length = 30

def setup_folders():
    for action in actions: 
        for sequence in range(no_sequences):
            try: 
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except FileExistsError:
                pass

if __name__ == "__main__":
    setup_folders()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
        
    # Set mediapipe model 
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:
        
        # Loop through actions
        for action in actions:
            # Loop through sequences (videos)
            for sequence in range(no_sequences):
                # Loop through frames
                for frame_num in range(sequence_length):

                    # Read feed
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Make detections
                    image, results = mediapipe_detection(frame, hands)

                    # Draw landmarks
                    draw_landmarks(image, results)
                    
                    # Apply wait logic at the start of each sequence
                    if frame_num == 0: 
                        cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'Collecting frames for {action} | Video {sequence}', (15,20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000)
                    else: 
                        cv2.putText(image, f'Collecting frames for {action} | Video {sequence}', (15,20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                    
                    # Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()
                        
        cap.release()
        cv2.destroyAllWindows()