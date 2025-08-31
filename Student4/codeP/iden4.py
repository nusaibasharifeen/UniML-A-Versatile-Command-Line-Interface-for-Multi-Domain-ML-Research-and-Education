from pydub import AudioSegment
from pydub.playback import play

class AudioProcessor:
    vol=input("how much volume you want to increase?")
    def __init__(self):
        # Correct: Set path to ffmpeg converter
        AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"


    def process_and_play(self, input_path, volume_increase=vol):
        # Load audio from user input 
        audio = AudioSegment.from_file(input_path)

        # Print volume and duration
        print("Volume (dBFS):", audio.dBFS)
        print("Duration (seconds):", audio.duration_seconds)

        # Increase volume
        audio = audio + volume_increase

        # Play the processed audio
        play(audio)

if __name__ == "__main__":
    input_path = input("Enter the full path to your audio file: ")
    processor = AudioProcessor()
    processor.process_and_play(input_path)
