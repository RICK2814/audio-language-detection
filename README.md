# 🎙️ Multilingual Audio Language Detection Interface

An interactive, offline web application running on `localhost` for detecting spoken Indian languages (*Bengali, Gujarati, Kannada, Konkani, Malayalam, Marathi, Odia, Tamil, Telugu*) from audio recordings using acoustic feature extraction (78-D) and Random Forest classification.

---

## 📂 Project Structure

```
├── page.py                   # Streamlit web interface (Upload, Record, Visualizations)
├── inference.py              # Audio preprocessing & 78-D feature extraction engine
├── train_and_save_model.py   # Model training & export script
├── model_bundle.joblib       # Pre-trained model bundle (Model, Scaler, LabelEncoder)
├── sample_audios/            # Benchmark audio clips for testing
├── colab_notebook.ipynb      # Research notebook with experimental benchmarks
├── requirements.txt          # Python dependencies
└── work/                     # Preprocessing & evaluation benchmark logs
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Local Web App
```bash
python -m streamlit run page.py
```
Open your browser and navigate to: **`http://localhost:8501`**

---

## 🎯 Features
- **File Upload**: Supports `.wav`, `.mp3`, `.m4a`, `.ogg`, and `.flac`.
- **Microphone Recording**: Directly capture voice in the browser.
- **Audio Preprocessing**: Automatic resampling (16 kHz), spectral noise reduction (`noisereduce`), DC offset correction, and silence trimming (VAD).
- **78-D Acoustic Features**: MFCCs + Deltas + Chroma + Spectral Contrast + Tonnetz + Centroid + ZCR.
- **Visualizations**: Interactive probability distribution bar charts, audio waveforms, and Mel-Spectrograms.
