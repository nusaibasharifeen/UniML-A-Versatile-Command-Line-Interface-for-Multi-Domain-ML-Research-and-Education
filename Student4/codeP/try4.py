import os 
from pydub import AudioSegment 
from pydub.playback import play  
import noisereduce as nr  
import numpy as np  
from scipy.io import wavfile  
import tempfile 

tempfile.tempdir = "E:/python/temp"
os.makedirs(tempfile.tempdir, exist_ok=True)

# 1. Class for loading audio files
class AudioLoader:
    def __init__(self, file_path): 
        self.file_path = file_path  

    def load(self):  # Method to load the audio
        audio = AudioSegment.from_file(self.file_path).set_channels(1).set_frame_rate(16000)  
        # Load audio, convert to mono (1 channel), and set frame rate to 16kHz
        print(f"Original Volume: {audio.dBFS:.2f} dBFS") 
        return audio  

# 2. Class for reducing background noise
class NoiseReducer:
    def reduce(self, audio_segment):  # Accepts an audio segment to reduce noise
        print("Reducing noise...")  
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            # Create a temporary WAV file
            audio_segment.export(temp_file.name, format="wav")  
            rate, data = wavfile.read(temp_file.name)  
            reduced = nr.reduce_noise(y=data, sr=rate)  # Apply noise reduction
        return AudioSegment(  
            reduced.tobytes(),  
            frame_rate=rate,  
            sample_width=2,  
            channels=1  
        )

# 3. Class for volume normalization
class VolumeNormalizer:
    def __init__(self, target_dBFS=-20.0):  # Constructor to set target volume level
        self.target_dBFS = target_dBFS  

    def normalize(self, audio_segment):  # Method to normalize audio
        change = self.target_dBFS - audio_segment.dBFS  
        normalized = audio_segment.apply_gain(change)  
        print(f"Normalized Volume: {normalized.dBFS:.2f} dBFS") 
        return normalized  

# 4. Class for playing audio
class AudioPlayer:
    def play(self, audio_segment):  # Accepts audio segment to play
        print("Playing audio...") 
        play(audio_segment)  # Play the audio

# 5. Controller class to process audio
class AudioProcessor:
    def __init__(self, file_path):  # Constructor accepts audio file path
        self.set_ffmpeg()  # Set FFmpeg path for pydub
        self.loader = AudioLoader(file_path)  
        self.reducer = NoiseReducer()  
        self.normalizer = VolumeNormalizer() 
        self.player = AudioPlayer()  

    def set_ffmpeg(self):  # Method to set FFmpeg path (required by pydub)
        ffmpeg_path = "C:\\ffmpeg\\ffmpeg-2024-latest\\bin\\ffmpeg.exe"  # Path to FFmpeg
        os.environ["PATH"] += os.pathsep + ffmpeg_path  

    def process(self):  # Main method to run the whole process
        audio = self.loader.load() 
        clean_audio = self.reducer.reduce(audio)  # Reduce background noise
        normalized_audio = self.normalizer.normalize(clean_audio)  # Normalize the volume
        self.player.play(normalized_audio)  # Play the processed audio

        output_path = "output.wav"  # Define output file name
        normalized_audio.export(output_path, format="wav")  
        print(f"Processed audio saved to {output_path}") 

# Main block to run the project
if __name__ == "__main__": 
    path = input("Enter audio file path: ")  # Ask user to enter path to audio file
    processor = AudioProcessor(path) 
    processor.process()