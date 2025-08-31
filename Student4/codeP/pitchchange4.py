import librosa
import numpy as np
import soundfile as sf

# === File Paths ===
# IMPORTANT: Replace these paths with the actual paths to your files.
# Path to your input audio file. This should be a WAV file (1 minute or less).
input_path = input("Enter input file path")
# Path where the pitch-shifted audio will be saved. The output will be a WAV file.
output_path = "E:\python\\results\pitch.wav"

# === Voice Changer Parameters ===

# Positive values increase pitch (e.g., 5 semitones for a higher voice).
# Negative values decrease pitch (e.g., -5 semitones for a deeper voice).
# A semitone is the smallest musical interval. 12 semitones is one octave.
semitones_to_shift = 10 

# === Load Input Audio ===
try:
    # Load the audio file using librosa.
    # sr=None tells librosa to use the original sample rate of the audio file.
    # mono=True converts the audio to a single channel if it's stereo, which is usually required for processing.
    audio, sr = librosa.load(input_path, sr=None, mono=True)
    print(f"✅ Successfully loaded audio from {input_path} with sample rate {sr}.")
except Exception as e:
    # If there's an error loading the audio (e.g., file not found, corrupted), print an error and exit.
    print(f"❌ Failed to load audio from {input_path}: {e}")
    exit() # Exit the script as it cannot proceed without input audio.

# === Perform Pitch Shifting ===
try:
    # Apply pitch shifting using librosa's effects module.
    # `y`: The input audio time series (the 'audio' variable we loaded).
    # `sr`: The sample rate of the audio.
    # `n_steps`: The number of semitones to shift the pitch.
    pitch_shifted_audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=semitones_to_shift)
    print(f"✅ Pitch shifting completed by {semitones_to_shift} semitones.")
except Exception as e:
    # Catch any errors that occur during the pitch shifting process.
    print(f"❌ Pitch shifting failed: {e}")
    exit() # Exit the script if pitch shifting fails.

# === Save Output ===
try:
    # Save the pitch-shifted audio to the specified output path as a WAV file.
    # Ensure the output audio is saved with the original sample rate (`sr`).
    sf.write(output_path, pitch_shifted_audio, sr)
    print(f"✅ Voice changing complete! Saved to {output_path}")
except Exception as e:
    # If there's an error saving the output audio, print an error.
    print(f"❌ Failed to save output to {output_path}: {e}")
