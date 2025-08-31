from pydub import AudioSegment

# Load stereo 32-bit WAV
audio = AudioSegment.from_wav("target.wav")

# Convert to mono + 16-bit + same sample rate (44100)
audio = audio.set_channels(1).set_sample_width(2).set_frame_rate(44100)

# Export fixed file
audio.export("target_fixed.wav", format="wav")
print("✅ source_fixed.wav created successfully.")
