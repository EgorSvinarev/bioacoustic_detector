import librosa
import numpy as np

TARGET_SR = 16000

def load_audio(path):
    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    return audio.astype(np.float32)
