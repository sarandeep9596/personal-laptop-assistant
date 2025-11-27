import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from groq import Groq
from dotenv import dotenv_values
import os
import uuid
import mtranslate as mt

# Load environment variables
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # root folder
env_vars = dotenv_values(os.path.join(BASE_DIR, ".env"))

InputLanguage = env_vars.get("InputLanguage")
GroqAPIKey = env_vars.get("GroqAPIKey")

client = Groq(api_key=GroqAPIKey)

# Directory for temp audio
os.makedirs("Data", exist_ok=True)

def record_audio():
    duration = 5  # seconds
    samplerate = 44100
    print("🎤 Listening... Speak now.")
    
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    
    file_path = f"Data/audio_{uuid.uuid4().hex}.wav"
    write(file_path, samplerate, audio)
    print("✔ Audio captured.")
    
    return file_path

def whisper_transcribe(file_path):
    audio_file = open(file_path, "rb")

    transcript = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3-turbo"   # Groq Whisper model
    )

    text = transcript.text
    audio_file.close()
    os.remove(file_path)
    return text

def SetAssistantStatus(Status):
    TempDirPath = os.path.join(os.getcwd(), "Frontend", "Files")
    os.makedirs(TempDirPath, exist_ok=True)
    with open(os.path.join(TempDirPath, "Status.data"), "w", encoding='utf-8') as file:
        file.write(Status)

def QueryModifier(Query):
    Query = Query.strip().lower()
    words = Query.split()

    question_words = ["how", "what", "who", "where", "when", "why", "which", "whose", "whom", "can you"]

    if any(Query.startswith(w) for w in question_words):
        if not Query.endswith("?"):
            Query += "?"
    else:
        if not Query.endswith(".") and not Query.endswith("?"):
            Query += "."

    return Query.capitalize()

def UniversalTranslator(Text):
    english_translation = mt.translate(Text, "en", "auto")
    return english_translation

def SpeechRecognition():
    path = record_audio()
    text = whisper_transcribe(path)

    print("📝 Raw transcript:", text)

    if InputLanguage.lower() in ["en", "english"]:
        return QueryModifier(text)
    else:
        SetAssistantStatus("Translating...")
        translated = UniversalTranslator(text)
        return QueryModifier(translated)

if __name__ == "__main__":
    try:
        while True:
            result = SpeechRecognition()
            print("Final:", result)
    except KeyboardInterrupt:
        print("\n🛑 Speech Recognition stopped by user.")
