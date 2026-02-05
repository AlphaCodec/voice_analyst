from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import io
import numpy as np
import librosa

# ================= CONFIG =================
API_KEY = "sk_test_123456789"
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
# ==========================================

app = FastAPI(title="AI Generated Voice Detection API")

# ---------- Request schema ----------
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


# ---------- Feature helpers ----------

def compute_speech_likelihood(y, sr):
    rms = librosa.feature.rms(y=y)[0]
    energy_ratio = np.mean(rms > np.percentile(rms, 50))

    try:
        f0 = librosa.yin(y, fmin=70, fmax=400, sr=sr)
        pitch_ratio = np.mean(~np.isnan(f0))
    except:
        pitch_ratio = 0.0

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = np.mean(zcr)

    score = 0
    if energy_ratio > 0.10:
        score += 1
    if pitch_ratio > 0.15:
        score += 1
    if zcr_mean < 0.15:
        score += 1

    return score / 3.0


def extract_voice_features(y, sr):
    f0 = librosa.yin(y, fmin=70, fmax=400, sr=sr)
    f0_valid = f0[~np.isnan(f0)]

    pitch_mean = np.mean(f0_valid) if len(f0_valid) > 10 else 0.0
    pitch_variance = np.var(f0_valid) if len(f0_valid) > 10 else 0.0

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = np.mean(centroid)

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = np.mean(zcr)

    duration = librosa.get_duration(y=y, sr=sr)

    return {
        "duration": duration,
        "pitch_variance": pitch_variance,
        "spectral_centroid": centroid_mean,
        "zcr": zcr_mean,
    }


def compute_ai_likelihood(features):
    score = 0.0

    if features["pitch_variance"] < 12:
        score += 0.4

    if features["spectral_centroid"] < 1500:
        score += 0.3

    if features["zcr"] < 0.06:
        score += 0.2

    if features["duration"] > 10:
        score += 0.1

    return min(score, 1.0)


# ---------- Core analysis ----------

def analyze_voice(base64_audio: str):
    audio_bytes = base64.b64decode(base64_audio)

    if len(audio_bytes) < 1000:
        raise ValueError("Audio too small")

    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)

    duration = librosa.get_duration(y=y, sr=sr)
    if duration < 0.5:
        return {
            "classification": "HUMAN",
            "confidence": 0.55,
            "explanation": "Audio too short; insufficient evidence of synthetic speech"
        }

    speech_score = compute_speech_likelihood(y, sr)
    features = extract_voice_features(y, sr)
    ai_score = compute_ai_likelihood(features)

    if speech_score < 0.35:
        classification = "HUMAN"
        confidence = 0.55
        explanation = "Insufficient evidence of synthetic speech patterns"

    elif ai_score > 0.65:
        classification = "AI_GENERATED"
        confidence = round(0.6 + ai_score * 0.35, 2)
        explanation = "Unnaturally stable pitch and spectral patterns detected"

    else:
        classification = "HUMAN"
        confidence = round(0.6 + (1 - ai_score) * 0.3, 2)
        explanation = "Natural pitch variation and human-like speech dynamics"

    return {
        "classification": classification,
        "confidence": min(confidence, 0.95),
        "explanation": explanation
    }


# ---------- API endpoint ----------

@app.post("/api/voice-detection")
def voice_detection(
    data: VoiceRequest,
    x_api_key: str = Header(None)
):
    # API key validation
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Input validation
    if data.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    if data.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 format supported")

    try:
        result = analyze_voice(data.audioBase64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "status": "success",
        "language": data.language,
        "classification": result["classification"],
        "confidenceScore": result["confidence"],
        "explanation": result["explanation"]
    }
