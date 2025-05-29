# ser.py

import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# RAVDESS emotion map
emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def load_data(data_dir):
    features, labels = [], []
    for file in os.listdir(data_dir):
        if file.endswith(".wav"):
            parts = file.split("-")
            if len(parts) > 2:
                emotion_code = parts[2]
                emotion = emotion_map.get(emotion_code)
            else:
                emotion = None
            if emotion:
                path = os.path.join(data_dir, file)
                features.append(extract_features(path))
                labels.append(emotion)
    return np.array(features), np.array(labels)
