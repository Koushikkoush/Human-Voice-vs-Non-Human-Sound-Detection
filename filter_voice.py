import sys
import numpy as np
from tensorflow.keras.models import load_model
from feature_extraction import extract_mfcc

# Reconfigure stdout to use UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Load the trained model
model = load_model('human_voice_classifier.h5')

# Function to predict if a given audio file contains human voice
def is_human_voice(audio_path):
    # Extract MFCC features from the audio file
    features = extract_mfcc(audio_path)
    
    # Reshape the features to match the input shape of the model
    features = np.reshape(features, (1, -1))
    
    # Make a prediction using the trained model
    prediction = model.predict(features)
    
    return prediction >= 0.5  # Return True if human voice, False otherwise

# Test the function with a new audio file
audio_file = 'LJ001-0004.wav'
if is_human_voice(audio_file):
    print("This is a human voice.")
else:
    print("This is not a human voice.")

