from pyannote.audio import Pipeline
from pydub import AudioSegment
import whisper
import numpy as np
import gc
import os


HF_TOKEN = "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"  
path=input("Enter the path of the audio")
# Initialize diarization pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=HF_TOKEN
)

# Run diarization
diarization = pipeline(path)

# Load audio using pydub
audio = AudioSegment.from_mp3(path)
audio = audio.set_frame_rate(16000)

# Load Whisper
model = whisper.load_model("small.en")


def to_millisec(timestamp):
    return int(float(timestamp) * 1000)


# Helper: convert pydub segment to numpy float32 array
def to_float_array(segment):
    arr = np.array(segment.get_array_of_samples())
    return arr.astype(np.float32) / 32768

# Process segments
with open("../results/spd.txt", "w", encoding="utf-8") as f:
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = to_millisec(turn.start)
        end = to_millisec(turn.end)
        segment = audio[start:end]
        audio_array = to_float_array(segment)
        result = model.transcribe(audio_array, fp16=False)
        f.write(f"\n[ {turn.start:.2f} -- {turn.end:.2f} ] {speaker} : {result['text']}")
        del result, segment, audio_array
        gc.collect()

