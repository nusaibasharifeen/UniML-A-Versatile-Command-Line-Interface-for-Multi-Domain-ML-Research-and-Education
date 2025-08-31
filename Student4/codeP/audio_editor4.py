from pydub import AudioSegment

class AudioEditor:
    def _init_(self, file_path):
        self.audio = AudioSegment.from_file(file_path)

    def trim(self, start_ms, end_ms):
        return self.audio[start_ms:end_ms]

    def concat(self, audio_paths):
        combined = AudioSegment.empty()
        for path in audio_paths:
            segment = AudioSegment.from_file(path)
            combined += segment
        return combined