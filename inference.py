import os
import io
import tempfile
import subprocess
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
import joblib

TARGET_SR = 16000
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_bundle.joblib")

class AudioClassifier:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.bundle = None
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Please run train_and_save_model.py first.")
        self.bundle = joblib.load(self.model_path)
        self.model = self.bundle["model"]
        self.scaler = self.bundle["scaler"]
        self.le = self.bundle["label_encoder"]
        self.classes = self.bundle["classes"]
        self.accuracy = self.bundle.get("accuracy", 0.0)

    def preprocess_audio(self, audio_source, source_name=None):
        """
        Loads and preprocesses audio from a file path, file-like object, or raw bytes.
        Returns: (preprocessed_waveform, sample_rate, duration_seconds)
        """
        # librosa (as of 1.0.0) only decodes formats that `soundfile`/libsndfile
        # natively understand, with no ffmpeg fallback. libsndfile cannot decode
        # AAC/M4A (and some other compressed formats) at all. So instead of
        # relying on librosa/soundfile to understand the original file, we
        # always convert the input to a clean mono WAV via ffmpeg first --
        # ffmpeg reliably handles m4a, mp3, ogg, webm, etc.
        tmp_in_path = None
        tmp_out_path = None
        try:
            if isinstance(audio_source, bytes):
                audio_bytes = audio_source
            elif hasattr(audio_source, "read"):
                audio_bytes = audio_source.read()
            else:
                audio_bytes = None

            if audio_bytes is not None:
                suffix = ".dat"
                if source_name and "." in source_name:
                    suffix = "." + source_name.rsplit(".", 1)[-1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio_bytes)
                    tmp_in_path = tmp.name

                tmp_out_path = tmp_in_path + "_converted.wav"
                result = subprocess.run(
                    ["ffmpeg", "-y", "-i", tmp_in_path, "-ar", str(TARGET_SR), "-ac", "1", tmp_out_path],
                    capture_output=True,
                )
                if result.returncode != 0 or not os.path.exists(tmp_out_path):
                    err_tail = result.stderr.decode(errors="ignore")[-300:]
                    raise RuntimeError(f"Could not decode this audio file: {err_tail}")

                y, sr = librosa.load(tmp_out_path, sr=TARGET_SR, mono=True)
            else:
                y, sr = librosa.load(audio_source, sr=TARGET_SR, mono=True)
        finally:
            for p in (tmp_in_path, tmp_out_path):
                if p and os.path.exists(p):
                    os.remove(p)

        if y is None or len(y) == 0:
            raise ValueError("Input audio is empty or could not be decoded.")

        orig_len = len(y) / TARGET_SR

        # 1. Noise Reduction
        try:
            y = nr.reduce_noise(y=y, sr=TARGET_SR)
        except Exception:
            pass

        # 2. DC Offset Removal
        dc_offset = float(np.mean(y))
        y = y - dc_offset

        # 3. Energy-based Voice Activity Detection (VAD) / silence trimming
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        if len(y_trimmed) >= TARGET_SR * 0.1:
            y = y_trimmed

        final_len = len(y) / TARGET_SR
        return y.astype(np.float32), TARGET_SR, orig_len, final_len

    def extract_features_78d(self, y, sr=TARGET_SR):
        """
        Extracts the exact 78-D acoustic feature vector:
        MFCC (13) + Delta (13) + Delta2 (13) + Standard Chroma (12) +
        Spectral Contrast (7) + Extended Chroma (19) + ZCR (1)
        """
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

        b = {
            "mfcc": m(mfcc),
            "delta": m(delta),
            "delta2": m(delta2),
            "chroma12": m(chroma_std),
            "tonnetz6": m(tonnetz),
            "centroid1": m(centroid),
            "spectral_contrast7": m(spectral_contrast),
            "zcr1": m(zcr),
        }

        ext_chroma19 = np.concatenate([b["chroma12"], b["tonnetz6"], b["centroid1"]])
        
        vec = np.concatenate([
            b["mfcc"], b["delta"], b["delta2"],
            b["chroma12"], b["spectral_contrast7"],
            ext_chroma19,
            b["zcr1"],
        ])
        return vec

    def predict(self, audio_source, source_name=None):
        """
        Given an audio source, returns:
        {
            'top_class': str,
            'top_prob': float,
            'probabilities': dict (class -> float),
            'waveform': np.ndarray,
            'sr': int,
            'orig_duration': float,
            'final_duration': float
        }
        """
        y, sr, orig_len, final_len = self.preprocess_audio(audio_source, source_name=source_name)
        feat_vec = self.extract_features_78d(y, sr)
        feat_vec_scaled = self.scaler.transform(feat_vec.reshape(1, -1))

        probs = self.model.predict_proba(feat_vec_scaled)[0]
        top_idx = int(np.argmax(probs))
        top_class = self.classes[top_idx]
        top_prob = float(probs[top_idx])

        prob_dict = {cls: float(p) for cls, p in zip(self.classes, probs)}
        # Sort descending
        prob_dict = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

        return {
            "top_class": top_class,
            "top_prob": top_prob,
            "probabilities": prob_dict,
            "waveform": y,
            "sr": sr,
            "orig_duration": orig_len,
            "final_duration": final_len,
        }
