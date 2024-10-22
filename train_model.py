import os
import sys
import io
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from feature_extraction import extract_mfcc  # Ensure this file exists

# Set standard output encoding to UTF-8 to avoid encoding errors in Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Paths to the dataset folders
human_folder = r'C:\Users\kaush\OneDrive\Desktop\voice reduce\data\human'  # Ensure this path is correct
non_human_folder = r'C:\Users\kaush\OneDrive\Desktop\voice reduce\data\non_human'  # Ensure this path is correct

X = []  # List to store MFCC features
y = []  # List to store labels (1 for human, 0 for non-human)

# Process human voice files
for file_name in os.listdir(human_folder):
    file_path = os.path.join(human_folder, file_name)
    try:
        features = extract_mfcc(file_path)  # Extract MFCC features
        if features is not None:  # Check if features were extracted successfully
            X.append(features)  # Append features to the list
            y.append(1)  # Label for human voice
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Process non-human sound files
for file_name in os.listdir(non_human_folder):
    file_path = os.path.join(non_human_folder, file_name)
    try:
        features = extract_mfcc(file_path)  # Extract MFCC features
        if features is not None:  # Check if features were extracted successfully
            X.append(features)  # Append features to the list
            y.append(0)  # Label for non-human sound
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Check if any features were extracted
if len(X) == 0 or len(y) == 0:
    raise ValueError("No features extracted. Please check your audio files and extraction method.")

print(f"Features shape: {X.shape}")  # Print the shape of features
print(f"Labels shape: {y.shape}")  # Print the shape of labels

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))  # Input layer
model.add(Dense(32, activation='relu'))  # Hidden layer
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save('human_voice_classifier.h5')
