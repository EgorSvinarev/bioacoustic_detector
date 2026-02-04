import streamlit as st
import tempfile

from src.audio_utils import load_audio
from src.inference import run_inference, CLASS_NAMES
from src.analytics import extract_ecological_events
from src.visualization import plot_activity

st.title("🌍 Environmental Monitoring with ML")

uploaded_file = st.file_uploader("Загрузите WAV файл", type=["wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    audio = load_audio(path)
    scores = run_inference(audio)
    df = extract_ecological_events(scores, CLASS_NAMES)

    if df.empty:
        st.warning("Значимые звуки не обнаружены")
    else:
        st.dataframe(df)
        st.pyplot(plot_activity(df))
