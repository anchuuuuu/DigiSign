import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pickle

# 1. Load Data
DATA_DIR = "gesture_dataset"
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

data = []
for file in csv_files:
    df = pd.read_csv(os.path.join(DATA_DIR, file))
    data.append(df)

full_df = pd.concat(data, ignore_index=True)

# 2. Preprocess
# Labels are strings ('0', '1', etc.), convert to integers
labels = full_df['label'].astype(str).tolist()
unique_labels = sorted(list(set(labels)))
label_map = {label: i for i, label in enumerate(unique_labels)}

# Save label map for real-time script
with open('label_map.pkl', 'wb') as f:
    pickle.dump(unique_labels, f)

X = full_df.iloc[:, 1:].values # Skip only 'label'
y = np.array([label_map[l] for l in labels])
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(126,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(unique_labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Train
print(f"Training on gestures: {unique_labels}")
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# 5. Save
model.save('static_gesture_model.h5')
print("✅ Model saved as static_gesture_model.h5")
print("✅ Label map saved as label_map.pkl")
