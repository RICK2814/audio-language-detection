import os
import shutil
import random
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

TARGET_SR = 16000
DATASET_ROOT = r"c:\Users\sania\OneDrive\Desktop\south_Indian_language_audio"
OUTPUT_BUNDLE = "model_bundle.joblib"
SAMPLES_DIR = "sample_audios"

LANGUAGES = [
    "Bengali", "Gujarati", "Kannada", "konkani",
    "Malayalam", "Marathi", "odia", "tamil", "Telugu"
]

def preprocess_audio(file_path, target_sr=TARGET_SR):
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    if y is None or len(y) == 0:
        raise ValueError("Audio is empty")
    
    # Noise reduction
    try:
        y = nr.reduce_noise(y=y, sr=target_sr)
    except Exception:
        pass
    
    # DC offset removal
    dc_offset = float(np.mean(y))
    y = y - dc_offset
    
    # Silence trimming (VAD)
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trimmed) >= target_sr * 0.1:
        y = y_trimmed
        
    return y.astype(np.float32)

def extract_base_blocks(y, sr=TARGET_SR):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    chroma_std = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    def m(x):
        return np.mean(x, axis=1)

    blocks = {
        "mfcc": m(mfcc),
        "delta": m(delta),
        "delta2": m(delta2),
        "chroma12": m(chroma_std),
        "tonnetz6": m(tonnetz),
        "centroid1": m(centroid),
        "spectral_contrast7": m(spectral_contrast),
        "zcr1": m(zcr),
    }
    return blocks

def extended_chroma_19(blocks):
    return np.concatenate([blocks["chroma12"], blocks["tonnetz6"], blocks["centroid1"]])

def extract_78d(y, sr=TARGET_SR):
    b = extract_base_blocks(y, sr)
    vec = np.concatenate([
        b["mfcc"], b["delta"], b["delta2"],
        b["chroma12"], b["spectral_contrast7"],
        extended_chroma_19(b),
        b["zcr1"],
    ])
    return vec

def main():
    print("=" * 60)
    print("Starting Model Training & Export Pipeline")
    print(f"Dataset root: {DATASET_ROOT}")
    print("=" * 60)
    
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    
    X_list, y_list = [], []
    SAMPLES_PER_LANG = 50  # Balanced fast training set
    
    for lang in LANGUAGES:
        lang_dir = os.path.join(DATASET_ROOT, lang)
        if not os.path.exists(lang_dir) or not os.path.isdir(lang_dir):
            print(f"Warning: {lang_dir} not found!")
            continue
            
        audio_files = [
            f for f in os.listdir(lang_dir)
            if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a"))
        ]
        
        # Take a randomized subset for training
        random.seed(42)
        selected_files = random.sample(audio_files, min(SAMPLES_PER_LANG, len(audio_files)))
        print(f"Processing {lang}: {len(selected_files)} files...")
        
        # Copy 1 sample clip to samples directory for quick UI testing
        if selected_files:
            sample_src = os.path.join(lang_dir, selected_files[0])
            sample_dest = os.path.join(SAMPLES_DIR, f"{lang}_sample{Path(selected_files[0]).suffix}")
            shutil.copyfile(sample_src, sample_dest)
        
        for fname in selected_files:
            fpath = os.path.join(lang_dir, fname)
            try:
                y = preprocess_audio(fpath, TARGET_SR)
                feat = extract_78d(y, TARGET_SR)
                X_list.append(feat)
                y_list.append(lang.capitalize())
            except Exception as e:
                print(f"  Skipped {fname}: {e}")
                
    X = np.array(X_list)
    y_raw = np.array(y_list)
    print(f"\nTotal extracted feature matrix shape: {X.shape}")
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    print("\nTraining RandomForest Classifier...")
    clf = RandomForestClassifier(n_estimators=150, max_depth=None, random_state=42, n_jobs=-1)
    clf.fit(X_train_s, y_train)
    
    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Test Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    bundle = {
        "model": clf,
        "scaler": scaler,
        "label_encoder": le,
        "classes": list(le.classes_),
        "accuracy": acc,
        "feature_dim": 78
    }
    
    joblib.dump(bundle, OUTPUT_BUNDLE)
    print(f"Successfully exported model bundle to {OUTPUT_BUNDLE}!")

if __name__ == "__main__":
    main()
