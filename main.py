from speechText import speechToText, textToSpeech
import threading
import time
import sys
import queue

def main():
  print("Initializing...")

  recognized_queue = queue.Queue()
  stop_program_event = threading.Event()
  pause_listening_event = threading.Event()

  speech_thread = speechToText.STT(
    recognized_queue, 
    stop_event=stop_program_event,
    pause_listening_event=pause_listening_event
  )
  speech_thread.start()

  tts_engine = textToSpeech.TTS()

  tts_thread = None
  conversation_active = False

  def speak_in_thread(text):
    pause_listening_event.set()
    tts_engine.speak(text)
    time.sleep(1)
    pause_listening_event.clear()
  
  def interrupt_speaking():
    nonlocal tts_thread
    if tts_thread and tts_thread.is_alive():
      tts_engine.stop_speaking()
      tts_thread.join()  
  
  print("Say 'hello' to start the conversation")

  try:
    while not stop_program_event.is_set():
      try:
        text = recognized_queue.get(timeout=1)
      except queue.Empty:
        text = None

      if text:
        lower_text = text.lower()

        if not conversation_active:
          if lower_text == "hello":
            conversation_active = True
            interrupt_speaking()
            print("How can I help you?")
            tts_thread = threading.Thread(target=speak_in_thread, args=("Hello! How can I help you?",))
            tts_thread.start()
          else:
            pass
        else:
          if lower_text == "hello":
              print("Interrupting conversation")
              interrupt_speaking()
              tts_thread = threading.Thread(target=speak_in_thread, args=("Say your question again",))
              tts_thread.start()
          elif lower_text in ["bye", "exit"]:
            print("Goodbye!")
            interrupt_speaking()
            tts_engine.speak("Goodbye!")
            stop_program_event.set()
            break
          else:
            print("You said:", text)
            interrupt_speaking()
            tts_thread = threading.Thread(target=speak_in_thread, args=(f"You said: {text}",))
            tts_thread.start()

  except KeyboardInterrupt:
    print("Exiting")
  finally:
    stop_program_event.set()
    interrupt_speaking()   
    print("Exiting thread of speech recognition")
    speech_thread.join(timeout=2)
    print("Exit program")
    sys.exit(0)

if __name__ == "__main__":
  main()
