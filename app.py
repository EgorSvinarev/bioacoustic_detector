import streamlit as st
import tempfile

from src.audio_utils import load_audio
from src.inference import run_inference, load_class_names
from src.analytics import extract_ecological_events
from src.visualization import plot_activity

st.title("🌍 Environmental Monitoring with ML")

# Загружаем классы с кэшированием
CLASS_NAMES = load_class_names()

uploaded_file = st.file_uploader("Загрузите аудио файл (WAV или MP3)", type=["wav", "mp3"])

if uploaded_file:
    # Определяем расширение файла
    file_extension = uploaded_file.name.split('.')[-1].lower()
    suffix = f".{file_extension}"
    
    with st.spinner("Пожалуйста, подождите, идёт анализ аудио с помощью ML‑модели..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            path = tmp.name

        audio = load_audio(path)
        scores = run_inference(audio)
        df = extract_ecological_events(scores, CLASS_NAMES)

    if df.empty:
        st.warning("Значимые звуки не обнаружены")
    else:
        with st.spinner("Подготавливаем и отображаем графики, пожалуйста, подождите..."):
            st.dataframe(df)
            st.pyplot(plot_activity(df))
