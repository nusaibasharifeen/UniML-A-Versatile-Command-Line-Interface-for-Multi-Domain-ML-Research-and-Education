import librosa
import numpy as np
import soundfile as sf
from so_vits_svc_fork.inference.core import Svc

# === File Paths ===
# IMPORTANT: Replace these paths with the actual paths to your files.
# Path to your pretrained model (e.g., G_0.pth). This file contains the trained model weights.
# This model inherently defines the "target voice" for the conversion.
model_path = "G_0.pth"
# Path to your model's configuration file (e.g., config.json). This file defines model architecture.
config_path = "config.json"
# Path to your source audio file (input for conversion). This should be a WAV file (1 minute or less).
# This file provides the *speech content* that will be converted.
input_path = input("Enter path to the input audio")
# Path where the converted audio will be saved. The output will be a WAV file.
# This file will contain the speech content from input_path, but in the voice of the G_0.pth model.
output_path = "E:\python\\results\change.wav" # Changed to 'output.wav' as per your request


# === Load Input Audio ===
try:
    # Load the audio file using librosa.
    # sr=44100 ensures a consistent sample rate, which is common for voice models.
    # mono=True converts the audio to a single channel if it's stereo, which is usually required.
    #audio, sr = librosa.load(input_path, sr=44100, mono=True)
    audio, sr = sf.read(input_path)
    audio = np.asarray(audio, dtype=np.float32)
    print(f"✅ Successfully loaded audio from {input_path} with sample rate {sr}.")
except Exception as e:
    # If there's an error loading the audio (e.g., file not found, corrupted), print an error and exit.
    print(f"❌ Failed to load audio from {input_path}: {e}")
    exit() # Exit the script as it cannot proceed without input audio.

# === Initialize Voice Model ===
try:
    # Initialize the Svc (Singing Voice Conversion) model.
    # net_g_path: Path to the model weights file.
    # config_path: Path to the model configuration file.
    # device: Specify "cuda" for GPU acceleration (if you have an NVIDIA GPU and CUDA installed),
    #         otherwise use "cpu" for CPU-only inference. GPU is significantly faster.
    svc = Svc(net_g_path=model_path, config_path=config_path, device="cpu")
    print(f"✅ Successfully initialized voice model from {model_path} and {config_path}.")
except Exception as e:
    # If model initialization fails (e.g., incorrect paths, corrupted model files), print an error and exit.
    print(f"❌ Failed to initialize model: {e}")
    exit() # Exit the script as it cannot proceed without a loaded model.

# === Run Inference ===
try:
    # Perform voice conversion using the initialized model.
    # All arguments are now passed positionally, following the most common full signature
    # for so-vits-svc-fork's infer method when auto_predict_f0 is True.
    result = svc.infer(
        audio=audio,
        speaker=0,  # Always needed even for single-speaker models
        transpose=0,
        cluster_infer_ratio=0,
        noise_scale=0.4,
        auto_predict_f0=True,
        f0_method="crepe"
    )
    
    print("✅ Voice conversion inference completed.")
except Exception as e:
    # Catch any errors that occur during the inference process (e.g., model issues, data type mismatches).
    print(f"❌ Inference failed: {e}")
    exit() # Exit the script if inference fails.

# === Save Output ===
try:
    # The `svc.infer()` method returns a tuple: (converted_audio_data, sample_rate_of_output).
    audio_out, sr_out = result
    # Save the converted audio to the specified output path as a WAV file.
    sf.write(output_path, audio_out, sr_out)
    print(f"✅ Voice conversion complete! Saved to {output_path}")
except Exception as e:
    # If there's an error saving the output audio, print an error.
    print(f"❌ Failed to save output to {output_path}: {e}")