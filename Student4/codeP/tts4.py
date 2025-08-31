from gtts import gTTS
language="en"
textin=input('enter the path')
with open(textin, 'r', encoding='utf-8') as file: #opening the input file
    text = file.read() #reading the file

speech=gTTS(text=text,lang=language,slow=False,tld='com.au') #converting the text to speech and selecting accent
speech.save('../results/texttospeech.mp3') #saving result in the result file