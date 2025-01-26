import speech_recognition as sr

def speechToText(timeout=None, phrase_time_limit=None):
  r = sr.Recognizer()
  with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source)
    try:
      audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
      text = r.recognize_google(audio, language="en-US")
      print("You said: ", text)
      return text
    except sr.UnknownValueError:
      print("Could not understand audio")
      return None    
    except sr.RequestError as e:
      print(f"Could not request results from service; {e}")
      return None
    except sr.WaitTimeoutError as e:
      print(f"Timeout; {e}")
      return None
    
if __name__ == "__main__":
    text = speechToText(timeout=5, phrase_time_limit=10)