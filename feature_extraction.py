# import librosa
import librosa  # {{ edit_1 }}
import numpy as np  # {{ edit_2 }}

def extract_mfcc(file_path):
    # Load the audio file
    audio, sample_rate = librosa.load(file_path, sr=None)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    
    # Return the mean of the MFCCs across time
    return np.mean(mfccs.T, axis=0)
