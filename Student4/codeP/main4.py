from audio_processor import AudioProcessor
from audio_editor import AudioEditor

if __name__ == "_main_":
    audio_path = input("Enter audio file path: ")
    processor = AudioProcessor(audio_path)
    processor.process()

    choice = input("Do you want to trim the audio? (yes/no): ")
    if choice.lower() == "yes":
        start = int(input("Enter start time in milliseconds: "))
        end = int(input("Enter end time in milliseconds: "))
        editor = AudioEditor("E:\python\\results\output_processed.wav")
        trimmed = editor.trim(start, end)
        trimmed.export("output_trimmed.wav", format="wav")
        print("Trimmed audio saved as 'output_trimmed.wav'")

    concat_choice = input("Do you want to concatenate multiple audio files? (yes/no): ")
    if concat_choice.lower() == "yes":
        n = int(input("How many audio files do you want to join? "))
        files = []
        for i in range(n):
            file_path = input(f"Enter path for audio file {i+1}: ")
            files.append(file_path)
        editor = AudioEditor(files[0])
        combined_audio = editor.concat(files)
        combined_audio.export("E:\python\\results\output_concatenated.wav", format="wav")
        print("Concatenated audio saved as 'output_concatenated.wav'")