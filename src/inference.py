import tensorflow_hub as hub
import tensorflow as tf
import streamlit as st

@st.cache_resource
def load_model():
    """Загружает модель YAMNet с кэшированием"""
    return hub.load("https://tfhub.dev/google/yamnet/1")

@st.cache_data
def load_class_names():
    """Загружает названия классов с кэшированием"""
    yamnet = load_model()
    class_map = yamnet.class_map_path().numpy().decode("utf-8")
    classes = []
    with open(class_map) as f:
        next(f)
        for line in f:
            classes.append(line.split(",")[2])
    return classes

def run_inference(audio):
    """Выполняет инференс на аудио"""
    yamnet = load_model()
    scores, _, _ = yamnet(audio)
    return tf.reduce_mean(scores, axis=0).numpy()
