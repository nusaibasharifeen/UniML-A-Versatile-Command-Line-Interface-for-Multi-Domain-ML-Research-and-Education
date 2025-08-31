import librosa
import numpy as np
import soundfile as sf
from so_vits_svc_fork.inference.core import Svc

# === File Paths ===
model_path = "G_200.pth"
config_path = "config.json"
input_path = "target_fixed.wav"
output_path = "output.wav"

# === Load audio ===
audio, sr = librosa.load(input_path, sr=44100, mono=True)
print(f"✅ Loaded input audio: {input_path}, SR: {sr}, Duration: {len(audio)/sr:.2f} seconds")

# === Initialize Model ===
svc = Svc(net_g_path=model_path, config_path=config_path, device="cpu")
print("✅ Model initialized on CPU")

# === Voice Conversion ===
converted, sr_out = svc.infer(
    audio=audio,
    speaker='Biden',  # Single speaker
    transpose=0,
    cluster_infer_ratio=0,
    noise_scale=0.4,
    auto_predict_f0=False,
    f0_method="crepe"
)
print("✅ Inference completed")

# === Save Output ===
sf.write(output_path, converted, sr_out)
print(f"✅ Output saved to: {output_path}")