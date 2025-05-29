# live_detect.py

import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
import threading
from ser import extract_features, load_data
from sklearn.ensemble import RandomForestClassifier

def record_live_audio(fs=22050):
    print("🎙️ Recording... Press Enter to stop recording.")

    duration = 60  # fallback max duration in seconds
    audio_chunks = []
    stop_flag = threading.Event()

    # Listen for Enter key in a separate thread
    def wait_for_enter():
        input()
        stop_flag.set()

    threading.Thread(target=wait_for_enter, daemon=True).start()

    def callback(indata, frames, time, status):
        if stop_flag.is_set():
            raise sd.CallbackStop()
        audio_chunks.append(indata.copy())

    try:
        with sd.InputStream(samplerate=fs, channels=1, callback=callback):
            sd.sleep(duration * 1000)
    except sd.CallbackStop:
        pass

    audio = np.concatenate(audio_chunks, axis=0)
    print("✅ Recording stopped.")
    return audio, fs

def train_model():
    print("🧠 Training model on audio files...")
    X, y = load_data("audio")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, sorted(set(y))  # also return label order

def detect_emotion_with_confidence(audio, fs, model, label_order):
    temp_path = "audio/temp_live.wav"
    write(temp_path, fs, audio)
    features = extract_features(temp_path).reshape(1, -1)
    os.remove(temp_path)

    probabilities = model.predict_proba(features)[0]
    predicted_index = np.argmax(probabilities)
    predicted_label = model.classes_[predicted_index]

    print("\n🎯 Emotion Probabilities:")
    for label, prob in zip(model.classes_, probabilities):
        print(f"  {label:<10}: {prob * 100:.1f}%")

    return predicted_label

if __name__ == "__main__":
    if not os.path.exists("audio"):
        os.makedirs("audio")

    model, label_order = train_model()
    audio, fs = record_live_audio()
    emotion = detect_emotion_with_confidence(audio, fs, model, label_order)

    print(f"\n🔊 Predicted Emotion: {emotion}")
