import pyttsx3
import threading

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty("rate", 180)
engine.setProperty("voice", voices[23].id)

# Lock to prevent multiple threads from speaking at the same time
tts_lock = threading.Lock()

def speak(text):
  with tts_lock:
    engine.say(text)
    engine.runAndWait()

def stop_speaking():
  with tts_lock:
    engine.stop()

if __name__ == "__main__":
  speak("Hello world!")