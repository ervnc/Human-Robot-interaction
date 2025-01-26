import speech_recognition as sr
import threading
import queue

class STT(threading.Thread):
  def __init__(self, recognized_queue, stop_event=None, pause_listening_event=None):
    super().__init__()
    self.recognized_queue = recognized_queue
    self.stop_event = stop_event if stop_event else threading.Event()
    self.pause_listening_event = pause_listening_event if pause_listening_event else threading.Event()
    self.r = sr.Recognizer()
    self.daemon = True

  def run(self):
    with sr.Microphone() as source:
      self.r.adjust_for_ambient_noise(source, duration=1)
      print("Listening...")
      
      while not self.stop_event.is_set():
        try:
          audio = self.r.listen(source, timeout=1)

          try:
            text = self.r.recognize_google(audio, language="en-US")
            text = text.strip()
            lower_text = text.lower()

            if text:
              if self.pause_listening_event.is_set():
                if lower_text == "hello":
                  print("(Paused) Recognized hotword 'hello'")
                  self.recognized_queue.put(lower_text)
                else:
                  print("(Paused) Ignoring:", text)
              else:
                print(f"Recognized: {text}")
                self.recognized_queue.put(lower_text)
          except sr.UnknownValueError:
            pass
          except sr.RequestError as e:
            print(f"Could not request results from service; {e}")
        except sr.WaitTimeoutError as e:
          pass
        except KeyboardInterrupt:
          break
      
if __name__ == "__main__":
  stt = STT(queue.Queue())
  stt.run()