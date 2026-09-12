from pathlib import Path
import io
import numpy as np
import torch
import streamlit as st
import librosa
import plotly.graph_objects as go
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

DATASET_PATH = Path(__file__).parent / "data"
REAL_PATH = DATASET_PATH / "real"
FAKE_PATH = DATASET_PATH / "fake"

MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"
LOCAL_MODEL = Path(__file__).parent / "voice_model"
TARGET_SR = 16000

st.set_page_config(
    page_title="VoiceShield AI",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_detector():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else MODEL_NAME
    extractor = AutoFeatureExtractor.from_pretrained(model_path)
    model = AutoModelForAudioClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return extractor, model, device

def predict_audio(audio_file):
    audio_bytes = audio_file.getvalue()
    if not audio_bytes:
        raise ValueError("The audio file is empty.")

    audio, sample_rate = librosa.load(
        io.BytesIO(audio_bytes),
        sr=TARGET_SR,
        mono=True
    )

    if audio is None or len(audio) == 0:
        raise ValueError("Could not decode the audio.")

    audio, _ = librosa.effects.trim(audio, top_db=30)

    if len(audio) < int(0.5 * TARGET_SR):
        raise ValueError("Please provide at least 0.5 seconds of speech.")

    audio = librosa.util.normalize(audio)
    extractor, model, device = load_detector()

    chunk_size = 10 * TARGET_SR
    chunks = [
        audio[i:i + chunk_size]
        for i in range(0, len(audio), chunk_size)
    ]

    fake_probabilities = []

    for chunk in chunks:
        if len(chunk) < int(0.5 * TARGET_SR):
            continue

        inputs = extractor(
            chunk,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        fake_probabilities.append(float(probabilities[1].item()))

    if not fake_probabilities:
        raise ValueError("No usable speech segment was found.")

    fake_probability = float(np.mean(fake_probabilities))
    real_probability = 1.0 - fake_probability

    return real_probability, fake_probability, audio, sample_rate

st.markdown("""
<style>
.main-title {
    font-size: 32px;
    font-weight: 800;
    color: #1E3A8A;
}
.sub-title {
    font-size: 15px;
    color: #475569;
    margin-bottom: 25px;
}
.metric-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    color: #F8FAFC;
}
.risk-box {
    padding: 16px;
    border-radius: 8px;
    text-align: center;
    font-weight: 700;
    font-size: 20px;
    margin: 15px 0;
}
.risk-low {
    background: #DCFCE7;
    color: #166534;
    border: 2px solid #86EFAC;
}
.risk-medium {
    background: #FEF3C7;
    color: #92400E;
    border: 2px solid #FCD34D;
}
.risk-high {
    background: #FEE2E2;
    color: #991B1B;
    border: 2px solid #FCA5A5;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🛡️ VoiceShield AI — Deepfake Voice Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Wav2Vec2-based AI voice anti-spoofing analysis'
    '</div>',
    unsafe_allow_html=True
)

left_col, right_col = st.columns([1.1, 1])

with left_col:
    st.subheader("🎙️ Audio Input")

    tab_mic, tab_upload = st.tabs(["🎙️ Record via Microphone", "📁 Upload Voice Sample"])

    with tab_mic:
        mic_audio = st.audio_input("Record voice for 3–10 seconds:", key="mic_input")

    with tab_upload:
        uploaded_audio = st.file_uploader(
            "Upload audio:",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            key="file_input"
        )

    audio_file = uploaded_audio if uploaded_audio is not None else mic_audio
    result = None

    if audio_file is not None:
        st.audio(audio_file)

        try:
            with st.spinner("Running AI anti-spoofing model..."):
                result = predict_audio(audio_file)

            real_probability, fake_probability, audio, sr = result

            st.markdown("**Waveform:**")

            time_axis = np.arange(len(audio)) / sr
            step = max(1, len(audio) // 1200)

            waveform = go.Figure()
            waveform.add_trace(
                go.Scatter(
                    x=time_axis[::step],
                    y=audio[::step],
                    mode="lines",
                    line=dict(color="#2563EB", width=1.4)
                )
            )

            waveform.update_layout(
                height=180,
                margin=dict(l=10, r=10, t=10, b=20),
                xaxis_title="Time (seconds)",
                yaxis_title="Amplitude"
            )

            st.plotly_chart(waveform, use_container_width=True)

        except Exception as error:
            st.error(f"Detection failed: {error}")

with right_col:
    st.subheader("📊 Detection Result")

    if result is None:
        st.info("Upload or record audio to start detection.")
    else:
        real_probability, fake_probability, _, _ = result

        fake_percent = fake_probability * 100
        real_percent = real_probability * 100

        gauge_color = (
            "#DC2626"
            if fake_percent >= 65
            else "#F59E0B"
            if fake_percent >= 40
            else "#16A34A"
        )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=fake_percent,
                number={"suffix": "%", "font": {"size": 32}},
                title={"text": "AI-Generated Probability", "font": {"size": 18}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": gauge_color},
                    "steps": [
                        {"range": [0, 40], "color": "#DCFCE7"},
                        {"range": [40, 65], "color": "#FEF3C7"},
                        {"range": [65, 100], "color": "#FEE2E2"}
                    ],
                    "threshold": {
                        "line": {"color": "#991B1B", "width": 4},
                        "thickness": 0.75,
                        "value": 65
                    }
                }
            )
        )

        gauge.update_layout(
            height=260,
            margin=dict(l=20, r=20, t=30, b=10)
        )

        st.plotly_chart(gauge, use_container_width=True)

        metric1, metric2 = st.columns(2)

        with metric1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <span style="color: #94A3B8; font-size: 13px; font-weight: 600;">Human Probability</span><br>
                    <span style="color: #38BDF8; font-size: 22px; font-weight: bold;">{real_percent:.2f}%</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        with metric2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <span style="color: #94A3B8; font-size: 13px; font-weight: 600;">AI Probability</span><br>
                    <span style="color: #F87171; font-size: 22px; font-weight: bold;">{fake_percent:.2f}%</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        if fake_percent >= 65:
            st.markdown(
                '<div class="risk-box risk-high">🚨 AI-GENERATED VOICE LIKELY</div>',
                unsafe_allow_html=True
            )
            st.error("Sensitive transaction blocked.")
            st.button("🔒 Approve Wire Transfer", disabled=True)

        elif fake_percent >= 40:
            st.markdown(
                '<div class="risk-box risk-medium">⚠️ UNCERTAIN — VERIFY IDENTITY</div>',
                unsafe_allow_html=True
            )
            st.warning("Request another verification method.")
            st.button("🔒 Approve Wire Transfer", disabled=True)

        else:
            st.markdown(
                '<div class="risk-box risk-low">✅ HUMAN VOICE MORE LIKELY</div>',
                unsafe_allow_html=True
            )
            st.success("No strong AI-voice signal was detected.")
            st.button("✅ Approve Wire Transfer", disabled=False)

        st.caption(
            "This is a research model, not a guaranteed production detector. "
            "Performance can decrease with new voice-cloning systems, noise, "
            "telephone codecs, or very short recordings."
        )