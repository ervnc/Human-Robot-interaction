from speechText import speechToText, textToSpeech
import threading
import time

interrupt_event = threading.Event()

def monitor_commands():
  while not interrupt_event.is_set():
    command = speechToText.speechToText(timeout=2, phrase_time_limit=2)
    if command:
      if command.lower() == "stop":
        print("Stopping...")
        interrupt_event.set()
        break
      elif command.lower() in ["bye", "exit"]:
        print("Goodbye!")
        textToSpeech.speak("Goodbye!")
        exit()
    time.sleep(0.1)

def main():
  print("Initializing...")
  print("Say 'hello' to start the conversation")

  # Wait for the user to say "hello"
  while True:
    text = speechToText.speechToText(timeout=5, phrase_time_limit=5)
    if text and text.lower() == "hello":
      textToSpeech.speak("Hello, how can I help you?")
      break
    else:
      print("Waiting for 'hello'...")
      textToSpeech.speak("Waiting for hello")

  # Start monitoring for commands
  while True:
    text = speechToText.speechToText(timeout=5, phrase_time_limit=10)
    if text:
      command = text.lower()
      if command in ["bye", "exit"]:
        textToSpeech.speak("Goodbye")
        print("Finishing...")
        break
      else:
        response = f"You said: {text}"

        interrupt_event.clear()
        thread_monitor = threading.Thread(target=monitor_commands)
        thread_monitor.start()

        thread_speak = threading.Thread(target=textToSpeech.speak, args=(response,))
        thread_speak.start()


        thread_speak.join()

        if interrupt_event.is_set():
          textToSpeech.speak("Could you please repeat your question?")

        thread_monitor.join()
    else:
      textToSpeech.speak("Please try again")

if __name__ == "__main__":
  main()
