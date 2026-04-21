import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# 1. Load Data
DATA_PATH = os.path.join('MP_Data') 
actions = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
no_sequences = 30
sequence_length = 30

label_map = {label:num for num, label in enumerate(actions)}

def load_data():
    sequences, labels = [], []
    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path):
            print(f"Warning: Directory for {action} not found. Skipping.")
            continue
            
        for sequence in range(no_sequences):
            window = []
            seq_path = os.path.join(action_path, str(sequence))
            if not os.path.exists(seq_path):
                print(f"Warning: Sequence {sequence} for {action} not found. Skipping.")
                continue
                
            for frame_num in range(sequence_length):
                file_path = os.path.join(seq_path, f"{frame_num}.npy")
                if os.path.exists(file_path):
                    res = np.load(file_path)
                    window.append(res)
                else:
                    # Pad with zeros if frame is missing to maintain sequence length
                    window.append(np.zeros(126))
            
            sequences.append(window)
            labels.append(label_map[action])
            
    if not sequences:
        return None, None
        
    return np.array(sequences), to_categorical(labels).astype(int)

if __name__ == "__main__":
    print("Loading data...")
    X, y = load_data()
    
    if X is None or len(X) == 0:
        print("Error: No data found in MP_Data. Please run data_collection.py first.")
        exit()
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    # 2. Build LSTM Model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,126)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # 3. Train
    print("Starting training...")
    model.fit(X_train, y_train, epochs=200)

    model.summary()

    # 4. Save Model
    model.save('sign_language_model.h5')
    print("Model saved as sign_language_model.h5")
