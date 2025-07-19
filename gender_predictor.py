import librosa
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path)
    pitch = librosa.yin(y, fmin=75, fmax=300)
    avg_pitch = np.mean(pitch)
    return avg_pitch

def predict_gender(file_path):
    pitch = extract_features(file_path)
    return "female" if pitch > 165 else "male"
