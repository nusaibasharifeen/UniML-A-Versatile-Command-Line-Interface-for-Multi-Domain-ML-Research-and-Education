import speech_recognition

recog=speech_recognition.Recognizer() #function for speech recognition system
filename=input("enter the path\n")
# filename="../data/training/stttest.wav"
while True:
    try:
 
        with speech_recognition.AudioFile(filename) as mic:
            
            audio=recog.record(mic)
            text=recog.recognize_google((audio)) #using google() function to convert the audio to text
            text=text.lower()
            print(f"{text}")
            break
    except speech_recognition.UnknownValueError():
        speech_recognition.Recognizer()
        continue

with open("../results/stt.txt","w") as file:
    file.write(text) #saving the data to the result file