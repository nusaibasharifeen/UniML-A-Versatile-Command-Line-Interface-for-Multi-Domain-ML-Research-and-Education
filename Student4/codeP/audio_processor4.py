import os
import tempfile
from pydub import AudioSegment
from pydub.playback import play
import noisereduce as nr
from scipy.io import wavfile

# Load the audio file
class AudioLoader:
    def _init_(self, file_path):
        self.file_path = file_path

    def load(self):
        audio = AudioSegment.from_file(self.file_path).set_channels(1).set_frame_rate(16000)
        print(f"Original Volume: {audio.dBFS:.2f} dBFS")
        return audio

# Reduce background noise
class NoiseReducer:
    def reduce(self, audio_segment):
        print("Reducing noise...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            audio_segment.export(temp_file.name, format="wav")
            rate, data = wavfile.read(temp_file.name)
            reduced = nr.reduce_noise(y=data, sr=rate)
        return AudioSegment(
            reduced.tobytes(),
            frame_rate=rate,
            sample_width=2,
            channels=1
        )

# Normalize volume
class VolumeNormalizer:
    def _init_(self, target_dBFS=-20.0):
        self.target_dBFS = target_dBFS

    def normalize(self, audio_segment):
        change = self.target_dBFS - audio_segment.dBFS
        normalized = audio_segment.apply_gain(change)
        print(f"Normalized Volume: {normalized.dBFS:.2f} dBFS")
        return normalized

# Play the audio
class AudioPlayer:
    def play(self, audio_segment):
        print("Playing audio...")
        play(audio_segment)

# Controller that runs all the above steps
class AudioProcessor:
    def _init_(self, file_path):
        self.set_ffmpeg()
        self.loader = AudioLoader(file_path)
        self.reducer = NoiseReducer()
        self.normalizer = VolumeNormalizer()
        self.player = AudioPlayer()

    def set_ffmpeg(self):
        ffmpeg_path = "C:\\ffmpeg\\ffmpeg-2024-latest\\bin\\ffmpeg.exe"
        os.environ["PATH"] += os.pathsep + ffmpeg_path

    def process(self):
        audio = self.loader.load()
        clean = self.reducer.reduce(audio)
        normalized = self.normalizer.normalize(clean)
        self.player.play(normalized)
        normalized.export("output_processed.wav", format="wav")
        print("Processed audio saved as 'output_processed.wav'")