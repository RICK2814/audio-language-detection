# 🎙️ Audio Language Detection

> **AI-powered spoken-language classification for Indian languages using acoustic features and Random Forest.**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Librosa](https://img.shields.io/badge/Librosa-Audio%20Processing-5C3EE8)](https://librosa.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=2800&pause=900&color=36BCF7&center=true&vCenter=true&width=760&lines=Detect+Indian+Languages+from+Speech;Upload+Audio+%7C+Record+Live+%7C+Analyze+Confidence;78-D+Acoustic+Features+%2B+Random+Forest" alt="Animated project tagline" />
</p>

**Live Demo:** https://audio-language-detection-bhwha53elusztx56o8yb2v.streamlit.app/

---

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=2500&pause=800&color=8BE9FD&width=520&lines=%E2%9C%A8+Project+Overview" alt="Animated Overview header" />
</p>

## ✨ Overview

**Audio Language Detection** is a Streamlit application that analyzes spoken audio and predicts which supported Indian language is being spoken.

The pipeline combines signal preprocessing, handcrafted acoustic feature extraction, feature scaling, and a trained Random Forest classifier. The application supports both uploaded audio files and direct microphone recording, then presents the prediction together with confidence-oriented visualizations.

### Supported Languages

| Language | Supported |
|---|:---:|
| 🇮🇳 Bengali | ✅ |
| 🇮🇳 Gujarati | ✅ |
| 🇮🇳 Kannada | ✅ |
| 🇮🇳 Konkani | ✅ |
| 🇮🇳 Malayalam | ✅ |
| 🇮🇳 Marathi | ✅ |
| 🇮🇳 Odia | ✅ |
| 🇮🇳 Tamil | ✅ |
| 🇮🇳 Telugu | ✅ |

---

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=2500&pause=800&color=50FA7B&width=560&lines=%F0%9F%9A%80+Live+Application" alt="Animated Live Application header" />
</p>

## 🚀 Live Application

Open the deployed application:

**https://audio-language-detection-bhwha53elusztx56o8yb2v.streamlit.app/**

You can upload a supported audio file or record speech directly from the browser and inspect the model output and visualizations.

---

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=2500&pause=800&color=F1FA8C&width=520&lines=%F0%9F%8E%AF+Key+Features" alt="Animated Key Features header" />
</p>

## 🎯 Key Features

### 🎧 Flexible Audio Input
- Upload `.wav`, `.mp3`, `.m4a`, `.ogg`, or `.flac` audio files.
- Record speech directly from the browser microphone.

### 🧹 Audio Preprocessing
The inference pipeline prepares incoming audio through:
- Resampling to **16 kHz**
- Spectral noise reduction with `noisereduce`
- DC-offset correction
- Silence trimming / VAD-style cleanup

### 🧠 78-D Acoustic Representation
The classifier uses a 78-dimensional acoustic feature representation built from:
- MFCCs
- MFCC deltas
- Chroma features
- Spectral contrast
- Tonnetz
- Spectral centroid
- Zero-crossing rate

### 📊 Model Output & Visualization
- Predicted language
- Probability distribution across supported classes
- Audio waveform
- Mel-spectrogram
- Interactive exploratory visuals

---

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=2500&pause=800&color=FF79C6&width=440&lines=%F0%9F%8F%97%EF%B8%8F+ML+Pipeline" alt="Animated Pipeline header" />
</p>

## 🏗️ Pipeline

```text
Audio File / Microphone
          │
          ▼
   Audio Loading
          │
          ▼
 Resampling + Cleanup
          │
          ▼
Noise Reduction + Trimming
          │
          ▼
  78-D Feature Extraction
          │
          ▼
       Scaling
          │
          ▼
 Random Forest Classifier
          │
          ▼
Language Prediction
          │
          ├──────────────► Confidence Distribution
          ├──────────────► Waveform
          └──────────────► Mel-Spectrogram
```

---

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=2500&pause=800&color=BD93F9&width=500&lines=%F0%9F%93%81+Project+Structure" alt="Animated Project Structure header" />
</p>

## 📁 Project Structure

```text
.
├── page.py
├── inference.py
├── train_and_save_model.py
├── model_bundle.joblib
├── requirements.txt
├── sample_audios/
├── colab_notebook.ipynb
└── work/
```

### Core Files

| File | Purpose |
|---|---|
| `page.py` | Streamlit interface for upload, recording, prediction, and visualization |
| `inference.py` | Audio preprocessing and 78-D feature extraction used during inference |
| `train_and_save_model.py` | Model training and bundle generation |
| `model_bundle.joblib` | Serialized model artifacts used for inference |
| `requirements.txt` | Python dependency list |
| `sample_audios/` | Sample/benchmark audio clips |
| `colab_notebook.ipynb` | Experimental and research workflow |
| `work/` | Benchmark/evaluation artifacts |

---

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=2500&pause=800&color=8BE9FD&width=430&lines=%E2%9A%A1+Quick+Start" alt="Animated Quick Start header" />
</p>

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/RICK2814/audio-language-detection.git
cd audio-language-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
python -m streamlit run page.py
```

Then open:

```text
http://localhost:8501
```

---

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=2500&pause=800&color=FFB86C&width=500&lines=%F0%9F%A7%AA+Inference+Flow" alt="Animated Inference Flow header" />
</p>

## 🧪 Inference Flow

The application follows the same general path for uploaded audio and microphone recordings:

1. Load the audio signal.
2. Normalize the sampling rate to 16 kHz.
3. Reduce unwanted spectral noise.
4. Correct DC offset and trim silence.
5. Compute the 78-D acoustic representation.
6. Apply the trained preprocessing/scaling pipeline.
7. Run the Random Forest classifier.
8. Decode the predicted class label.
9. Visualize prediction confidence and audio characteristics.

---

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=2500&pause=800&color=FF5555&width=560&lines=%F0%9F%A4%96+Machine+Learning+Architecture" alt="Animated ML Architecture header" />
</p>

## 🤖 Machine Learning Architecture

The repository stores a pre-trained inference bundle containing the artifacts needed to transform extracted features into the final language prediction.

Conceptually:

```text
Raw Audio
   ↓
Acoustic Feature Engineering
   ↓
Feature Scaling
   ↓
Random Forest
   ↓
Label Decoding
   ↓
Predicted Indian Language
```

This design keeps feature extraction and model inference separated from the Streamlit presentation layer, making the system easier to test and extend.

---

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=2500&pause=800&color=6272A4&width=470&lines=%F0%9F%93%88+Visual+Analysis" alt="Animated Visual Analysis header" />
</p>

## 📈 Visual Analysis

The application is designed not only to return a class label but also to expose interpretable audio-side information through visualizations such as:

- **Waveform** — amplitude variation over time
- **Mel-spectrogram** — time-frequency representation on a mel scale
- **Class probability chart** — relative model confidence across supported languages

---

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=2500&pause=800&color=36BCF7&width=500&lines=%F0%9F%94%AC+Research+%26+Experimentation" alt="Animated Research header" />
</p>

## 🔬 Research / Experimentation

The repository includes a Colab notebook and working artifacts for experimentation and benchmark analysis. These materials can be used to inspect the feature pipeline, training workflow, and evaluation process before deploying changes to the Streamlit interface.

---

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=2500&pause=800&color=50FA7B&width=420&lines=%F0%9F%8C%90+Deployment" alt="Animated Deployment header" />
</p>

## 🌐 Deployment

The application is deployed on **Streamlit Community Cloud**.

For a local deployment:

```bash
python -m streamlit run page.py
```

For Streamlit Community Cloud, connect the GitHub repository and configure the application entry point as:

```text
page.py
```

Install dependencies from:

```text
requirements.txt
```

---

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=2500&pause=800&color=F1FA8C&width=500&lines=%F0%9F%9B%A0%EF%B8%8F+Development+Commands" alt="Animated Development header" />
</p>

## 🛠️ Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python -m streamlit run page.py

# Train / rebuild the model bundle
python train_and_save_model.py
```

> The training command should only be run when you intentionally want to regenerate the model artifacts used by the application.

---

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=2500&pause=800&color=8BE9FD&width=520&lines=%E2%9C%85+Recommended+Test+Checklist" alt="Animated Test Checklist header" />
</p>

## ✅ Recommended Test Checklist

Before publishing a new version, verify:

- [ ] App launches locally
- [ ] `.wav` upload works
- [ ] `.mp3` upload works
- [ ] Microphone recording works
- [ ] Preprocessing completes without errors
- [ ] All supported language labels decode correctly
- [ ] Probability visualization renders
- [ ] Waveform renders
- [ ] Mel-spectrogram renders
- [ ] `model_bundle.joblib` is available in the deployment environment
- [ ] Streamlit deployment starts successfully

---

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=2500&pause=800&color=FF5555&width=500&lines=%E2%9A%A0%EF%B8%8F+Notes+%26+Limitations" alt="Animated Limitations header" />
</p>

## ⚠️ Notes & Limitations

Prediction quality depends on recording conditions, speaker characteristics, pronunciation, background noise, sample duration, and how representative the training data is of real-world speech.

A model confidence score should be interpreted as the classifier's estimated class probability distribution, not as a guarantee of correctness.

---

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=2500&pause=800&color=BD93F9&width=520&lines=%F0%9F%94%90+Model+%26+Data+Handling" alt="Animated Model and Data header" />
</p>

## 🔐 Model & Data Handling

The application performs local preprocessing of the supplied audio before inference. The repository includes the serialized model bundle used for prediction. Avoid committing private or sensitive recordings to the repository.

---

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=2500&pause=800&color=36BCF7&width=520&lines=%F0%9F%93%A6+Project+Links" alt="Animated Project Links header" />
</p>

## 📦 GitHub Repository

**Source:** https://github.com/RICK2814/audio-language-detection

**Live Demo:** https://audio-language-detection-bhwha53elusztx56o8yb2v.streamlit.app/

---

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=2500&pause=800&color=50FA7B&width=440&lines=%F0%9F%97%BA%EF%B8%8F+Roadmap" alt="Animated Roadmap header" />
</p>

## 🗺️ Roadmap

Potential next improvements:

- Add more Indian languages
- Add stronger calibration and evaluation reporting
- Improve robustness to noisy and short recordings
- Add richer model benchmarking
- Add confusion-matrix and per-class evaluation views
- Introduce automated testing for preprocessing and inference
- Add model version metadata to the UI

---

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=2500&pause=800&color=FFB86C&width=400&lines=%F0%9F%93%84+License" alt="Animated License header" />
</p>

## 📄 License

This project is released under the **MIT License**.

---

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=2500&pause=800&color=F1FA8C&width=520&lines=%E2%AD%90+Support+the+Project" alt="Animated Support header" />
</p>

## ⭐ Support the Project

If this project is useful for speech-processing, machine-learning, or Indian-language research, consider starring the repository and sharing feedback.
