import librosa
import numpy as np

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path)
        pitch = librosa.yin(y, fmin=75, fmax=300)
        pitch = pitch[~np.isnan(pitch)]  # Remove NaN values

        if len(pitch) == 0:
            return None  # No valid pitch detected

        avg_pitch = np.mean(pitch)

        # If pitch is unrealistically low or high, mark as unclear
        if avg_pitch < 75 or avg_pitch > 300:
            return None

        return avg_pitch
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def predict_gender(file_path):
    pitch = extract_features(file_path)

    if pitch is None:
        return "Voice not clear"
    
    return "female" if pitch > 165 else "male"
