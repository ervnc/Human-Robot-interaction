import pyttsx3
import threading

class TTS:
  def __init__(self):
    self.engine = pyttsx3.init()
    self.voices = self.engine.getProperty('voices')

    self.engine.setProperty("rate", 130)
    self.engine.setProperty("voice", self.voices[23].id)

    self.tts_lock = threading.Lock()

  def speak(self, text: str):
    with self.tts_lock:
      self.engine.say(text)
      self.engine.runAndWait()

  def stop_speaking(self):
    with self.tts_lock:
      self.engine.stop()

if __name__ == "__main__":
  tts = TTS()
  tts.speak("Antares")