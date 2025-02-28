import os
import time
import numpy as np
import tempfile

from faster_whisper import WhisperModel
import sounddevice as sd
from scipy.io.wavfile import write
import webrtcvad
import collections

import piper

from langdetect import detect

from ollamaLLM import LLM
import re

class WhisperTranscriber:
  def __init__(self, model_size="small", sample_rate=16000):
    self.model_size = model_size
    self.sample_rate = sample_rate
    self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    self.vad = webrtcvad.Vad(2)
    
    self.piper_en = piper.PiperVoice.load("./piperModels/en_US-bryce-medium.onnx")
    self.piper_pt = piper.PiperVoice.load("./piperModels/pt_BR-faber-medium.onnx")

    self.speaking = False

  def detected_language(self, text):
    try:
      lang = detect(text)
      return lang
    
    except:
      return "en"

  def speak(self, text):
    lang = self.detected_language(text)

    tts = self.piper_en if lang == "en" else self.piper_pt

    self.speaking = True

    stream = sd.OutputStream(
      samplerate=tts.config.sample_rate, 
      channels=1, 
      dtype="int16", 
      blocksize=2048,
      latency=0.3)
    
    stream.start()

    for audio_bytes in tts.synthesize_stream_raw(text):
      if not self.speaking:
        break

      int_data = np.frombuffer(audio_bytes, dtype=np.int16)
      stream.write(int_data)
    
    stream.stop()
    stream.close()

    self.speaking = False

  def stop_speech(self):
    self.stop_audio = True
    sd.stop()

  def record_audio(self, frame_duration_ms=30, padding_duration_ms=300, max_record_seconds=10):
    while self.speaking:
      time.sleep(0.1)

    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    frames_per_buffer = int(self.sample_rate * frame_duration_ms / 1000)

    stream = sd.RawInputStream(samplerate=self.sample_rate,
                              blocksize=frames_per_buffer,
                              dtype='int16',
                              channels=1)
                              
    start_time = time.time()
    print("Recording...")

    with stream:
      while True:
        if time.time() - start_time > max_record_seconds:
          print("Max recording time reached")
          break

        frame, _ = stream.read(frames_per_buffer)
        if len(frame) != frames_per_buffer * 2:
          continue

        is_speech = self.vad.is_speech(frame, self.sample_rate)

        if not triggered:
          ring_buffer.append((frame, is_speech))
          num_voiced = len([f for f, speech in ring_buffer if speech])
          if num_voiced > 0.9 * ring_buffer.maxlen:
            triggered = True
            print("Triggered")
            voiced_frames.extend(f for f, _ in ring_buffer)
            ring_buffer.clear()
        else:
          voiced_frames.append(frame)
          ring_buffer.append((frame, is_speech))
          num_unvoiced = len([f for f, speech in ring_buffer if not speech])
          if num_unvoiced > 0.9 * ring_buffer.maxlen:
            print("Untriggered")
            break
    
    audio_data_bytes = b"".join(voiced_frames)
    audio_data = np.frombuffer(audio_data_bytes, dtype=np.int16)
    return audio_data

  def save_temp_audio(self, recording):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_file.name, self.sample_rate, recording)
    return temp_file.name

  def transcribe_audio(self, file_path):
    segments, info = self.model.transcribe(file_path, beam_size=5)
    print("Detected language '%s' with probability %f" % (info.language,
    info.language_probability))
    os.remove(file_path)

    return " ".join(seg.text for seg in segments)

  def run(self):
    self.speak("Hello")
    while True:
      print("Waiting for audio...")
      recording = self.record_audio()
      if recording is None or recording.size == 0:
        print("No audio detected")
        continue

      file_path = self.save_temp_audio(recording)
      transcript = self.transcribe_audio(file_path)

      response_LLM = LLM(transcript)
      print("Transcript:", transcript)

      print(transcript.strip().lower())

      if transcript.strip().lower() in ["bye.", "exit."]:
        self.speak("Goodbye")
        break
      elif transcript.strip().lower() in ["stop.", "shut up."]:
        self.speaking = False
      else:
        response_text = response_LLM.generate().response
        clean_response = re.sub(r'<think>[\s\S]*?</think>', '', response_text).strip()
        self.speak(clean_response)

if __name__ == "__main__":
  transcriber = WhisperTranscriber(model_size="small", sample_rate=16000)
  transcriber.run()
