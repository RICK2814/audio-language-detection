import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from inference import AudioClassifier

# Set page config
st.set_page_config(
    page_title="Audio Language Detection System",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.2rem;
        border-radius: 0.75rem;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .pred-badge {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1E40AF;
    }
    .conf-badge {
        font-size: 1.2rem;
        font-weight: 600;
        color: #059669;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_classifier():
    try:
        classifier = AudioClassifier()
        return classifier, None
    except Exception as e:
        return None, str(e)

classifier, load_error = get_classifier()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/microphone.png", width=120)
    st.title("About the System")
    st.markdown("""
    **Multilingual Speech Identifier**
    
    This application identifies spoken Indian languages from raw audio input using acoustic signal processing and machine learning.
    """)
    
    st.divider()
    st.subheader("Supported Languages (9)")
    langs = ["Bengali", "Gujarati", "Kannada", "Konkani", "Malayalam", "Marathi", "Odia", "Tamil", "Telugu"]
    st.write(", ".join([f"`{l}`" for l in langs]))
    
    st.divider()
    st.subheader("Feature Pipeline")
    st.markdown(r"""
    - **Preprocessing**: 16 kHz resample, spectral noise reduction, DC offset removal, energy VAD silence trimming.
    - **Feature Vector (78-D)**:
      - 13 MFCCs + 13 $\Delta$ + 13 $\Delta\Delta$
      - 12 Standard Chroma
      - 7 Spectral Contrast
      - 19 Extended Chroma (Chroma + Tonnetz + Centroid)
      - 1 Zero-Crossing Rate (ZCR)
    - **Classifier**: Random Forest Ensemble
    """)

# Main Content
st.markdown('<div class="main-header">🎙️ Audio Language Classification & Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload an audio recording, speak into your microphone, or choose a sample to detect the spoken language in real time.</div>', unsafe_allow_html=True)

if load_error:
    st.error(f"⚠️ Model bundle not found or could not be loaded: {load_error}")
    st.info("💡 Please wait for the training process to finish generating `model_bundle.joblib`.")
    st.stop()

# Input Options via Tabs
tab_upload, tab_mic, tab_sample = st.tabs(["📁 Upload Audio File", "🎤 Record from Microphone", "🎧 Test with Sample Audio"])

audio_bytes = None
source_name = ""

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload an audio file (WAV, MP3, M4A, OGG, FLAC)",
        type=["wav", "mp3", "m4a", "ogg", "flac"]
    )
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        source_name = uploaded_file.name

with tab_mic:
    st.caption("Record your voice directly using your browser's microphone:")
    recorded_audio = st.audio_input("Click to record audio")
    if recorded_audio is not None:
        audio_bytes = recorded_audio.read()
        source_name = "Live Microphone Recording"

with tab_sample:
    samples_dir = os.path.join(os.path.dirname(__file__), "sample_audios")
    if os.path.exists(samples_dir):
        sample_files = [f for f in os.listdir(samples_dir) if f.endswith((".wav", ".mp3"))]
        if sample_files:
            selected_sample = st.selectbox("Select a benchmark sample clip:", sample_files)
            if st.button("Load Selected Sample"):
                with open(os.path.join(samples_dir, selected_sample), "rb") as f:
                    audio_bytes = f.read()
                source_name = f"Sample: {selected_sample}"
                st.session_state["sample_bytes"] = audio_bytes
                st.session_state["sample_name"] = source_name
            elif "sample_bytes" in st.session_state:
                audio_bytes = st.session_state["sample_bytes"]
                source_name = st.session_state["sample_name"]
        else:
            st.info("Sample audio files will appear here once training generates them.")
    else:
        st.info("No samples directory found yet.")

# Process and Display Results
if audio_bytes is not None:
    st.divider()
    col_play, col_info = st.columns([1, 1])
    
    with col_play:
        st.subheader("🔊 Audio Playback")
        st.audio(audio_bytes)
        st.caption(f"Source: **{source_name}**")
        
    with col_info:
        st.subheader("⚙️ Processing")
        with st.spinner("Analyzing acoustic features..."):
            try:
                results = classifier.predict(audio_bytes)
            except Exception as e:
                st.error(f"Error during audio processing: {e}")
                st.stop()
        st.success("Analysis complete!")
        st.write(f"⏱️ **Original Duration:** {results['orig_duration']:.2f} s")
        st.write(f"✂️ **Trimmed Duration (VAD):** {results['final_duration']:.2f} s")
        st.write(f"🎚️ **Target Sample Rate:** {results['sr']} Hz")

    # Prediction Showcase Card
    st.markdown("---")
    st.subheader("🎯 Detection Results")
    
    top_class = results["top_class"]
    top_prob = results["top_prob"] * 100
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.95rem; color: #6B7280; text-transform: uppercase; font-weight: 600;">Predicted Language</div>
            <div class="pred-badge">{top_class}</div>
            <div class="conf-badge">{top_prob:.1f}% Confidence</div>
        </div>
        """, unsafe_allow_html=True)
        
    with res_col2:
        st.write("**Language Confidence Distribution:**")
        df_probs = pd.DataFrame(
            list(results["probabilities"].items()),
            columns=["Language", "Confidence"]
        )
        df_probs["Confidence (%)"] = df_probs["Confidence"] * 100
        st.bar_chart(df_probs.set_index("Language")["Confidence (%)"], color="#3B82F6")

    # Waveform and Spectrogram Visualizer
    st.markdown("---")
    st.subheader("📊 Audio Visualizations")
    
    viz_col1, viz_col2 = st.columns(2)
    y = results["waveform"]
    sr = results["sr"]
    
    with viz_col1:
        st.write("**Waveform (Time Domain)**")
        fig, ax = plt.subplots(figsize=(6, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax, color="#2563EB")
        ax.set_title("Preprocessed Audio Waveform", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Amplitude", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    with viz_col2:
        st.write("**Mel-Spectrogram (Frequency Domain)**")
        fig, ax = plt.subplots(figsize=(6, 3))
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title("Mel-Spectrogram", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
