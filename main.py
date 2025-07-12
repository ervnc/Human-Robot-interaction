import os
import time
import numpy as np
import tempfile
import threading
import queue

from faster_whisper import WhisperModel
import sounddevice as sd
from scipy.io.wavfile import write

import webrtcvad
import re

import pvporcupine
from speexdsp import EchoCanceller

import piper

from langdetect import detect

from ollamaLLM import LLM

class WhisperTranscriber:
  def __init__(self, model_size="small", sample_rate=16000):
    PICOVOICE_ACCESS_KEY = "fNMi2EfTQkvH34LcNQkMvOkajA5+D6hp6xkGyqbiQ6CGwC/coLLnfQ=="
    HOTWORD_PATH = "./Antares_pt_linux_v3_0_0.ppn"

    self.model_size = model_size
    self.sample_rate = sample_rate

    try:
      self.porcupine = pvporcupine.create(
      access_key=PICOVOICE_ACCESS_KEY,
          keyword_paths=[HOTWORD_PATH],
          model_path="./porcupine_params_pt.pv"
      )
      self.frame_length = self.porcupine.frame_length
      print("Porcupine hotword engine loaded successfully.")
    except pvporcupine.PorcupineError as e:
      print(f"FATAL: Could not initialize Porcupine: {e}")
      print("Please check your Access Key and the path to the .ppn file.")
      return

    self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
    self.vad = webrtcvad.Vad(2)
    
    self.piper_en = piper.PiperVoice.load("./piperModels/en_US-bryce-medium.onnx")
    self.piper_pt = piper.PiperVoice.load("./piperModels/pt_BR-faber-medium.onnx")

    self.aec = EchoCanceller.create(frame_size=self.porcupine.frame_length, filter_length=100,sample_rate=self.porcupine.sample_rate)

    self.audio_queue = queue.Queue()
    self.stop_capture = False
    self.tts_interrupt_event = threading.Event()
    self.capture_thread = threading.Thread(target=self.capture_audio)
    self.capture_thread.start()



  def capture_audio(self, channels=1, dtype="int16"):
    try:
      with sd.RawInputStream(samplerate=self.sample_rate,
                              blocksize=self.porcupine.frame_length,
                              dtype='int16',
                              channels=channels) as stream:
        while not self.stop_capture:
          try:
            data, overflowed = stream.read(self.porcupine.frame_length)
            if overflowed:
              print("Buffer overflow")
            self.audio_queue.put(data)
          except Exception as e:
            print(f"Error capturing audio: {e}")
            break
    except Exception as e:
      print(f"Failed to open audio stream: {e}")




  def wait_for_wakeword(self):
    print(f"\nListening for 'Antares'...")
    while True:
      try:
        pcm_bytes = self.audio_queue.get()
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
          
        # Alimenta o Porcupine com o áudio
        result = self.porcupine.process(pcm)
          
        if result >= 0: # Se o resultado for >= 0, a hotword foi detectada
          print(f"Hotword 'Antares' detected!")
          with self.audio_queue.mutex:
              self.audio_queue.queue.clear() # Limpa a fila para não pegar o som da hotword
          return
      except queue.Empty:
          continue
      



  def record_audio(self, timeout=10, silence_duration_ms=1500):
    print("Listening for command...")
    
    # Parâmetros do VAD. Ele funciona melhor com frames de 30ms.
    vad_frame_duration_ms = 30
    vad_frame_bytes = int(self.sample_rate * vad_frame_duration_ms / 1000) * 2
    
    # Quantos frames de silêncio indicam o fim da fala (1.5 segundos)
    silence_frames_needed = silence_duration_ms // vad_frame_duration_ms
    
    is_recording = False
    voiced_frames = []
    silence_counter = 0
    start_time = time.time()
    audio_buffer = b''

    while time.time() - start_time < timeout:
      try:
        audio_buffer += self.audio_queue.get(timeout=0.1)
        
        # Processa o áudio em pedaços que o VAD entende
        while len(audio_buffer) >= vad_frame_bytes:
          vad_frame = audio_buffer[:vad_frame_bytes]
          audio_buffer = audio_buffer[vad_frame_bytes:]
          
          is_speech = self.vad.is_speech(vad_frame, self.sample_rate)
          
          if not is_recording and is_speech:
            print("Speech detected, recording...")
            is_recording = True
          
          if is_recording:
            voiced_frames.append(vad_frame)
            if not is_speech:
              silence_counter += 1
              if silence_counter > silence_frames_needed:
                print("Silence detected, stopping recording.")
                return b"".join(voiced_frames)
            else:
                silence_counter = 0 # Reseta o contador se houver fala
                      
      except queue.Empty:
        if is_recording:
          print("Finished listening (queue empty).")
          break
        continue

    return b"".join(voiced_frames) if voiced_frames else None


  def transcribe_audio(self, audio_data):
    if not audio_data:
        return ""
    
    # Converte os bytes para um array de floats, formato ideal para o Whisper
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    segments, info = self.model.transcribe(audio_np, beam_size=5)
    
    print("Detected language '%s' with probability %f" %
      (info.language, info.language_probability))

    return " ".join(seg.text for seg in segments).strip()



  def detected_language(self, text):
    try:
      lang = detect(text)
      return lang
    
    except:
      return "en"




  # Substitua sua função speak por esta versão final e polida.
  def speak(self, text):
    print("SPEAKING: ", text)
    self.tts_interrupt_event.clear()

    lang = self.detected_language(text)
    tts = self.piper_pt if lang == 'pt' else self.piper_en

    # --- Buffers e Stream ---
    playback_buffer = b''
    mic_buffer = b''
    playback_stream = sd.RawOutputStream(
        samplerate=tts.config.sample_rate,
        blocksize=self.frame_length,
        channels=1,
        dtype="int16"
    )

    # --- Thread Geradora de Voz ---
    def tts_generator_task():
        for chunk in tts.synthesize_stream_raw(text):
            if self.tts_interrupt_event.is_set():
                break
            playback_buffer_queue.put(chunk)
        playback_buffer_queue.put(None)

    playback_buffer_queue = queue.Queue()
    tts_thread = threading.Thread(target=tts_generator_task)
    
    try:
        playback_stream.start()
        tts_thread.start()
        
        # <<< A SOLUÇÃO: Adicionamos um tempo de início e um período de aquecimento
        start_time = time.time()
        warm_up_period_seconds = 0.5  # Não escuta por interrupções no primeiro 0.5s

        # --- Loop Principal de Processamento ---
        while tts_thread.is_alive() or not playback_buffer_queue.empty():
            if self.tts_interrupt_event.is_set():
                break

            # Pega áudio do TTS e do Microfone para os buffers
            try:
                tts_chunk = playback_buffer_queue.get_nowait()
                if tts_chunk is None: tts_thread.join()
                else: playback_buffer += tts_chunk
            except queue.Empty: pass
            
            try:
                mic_buffer += self.audio_queue.get_nowait()
            except queue.Empty: pass
            
            # Processa os buffers em frames sincronizados
            frame_size_bytes = self.frame_length * 2
            while len(playback_buffer) >= frame_size_bytes:
                playback_frame = playback_buffer[:frame_size_bytes]
                playback_buffer = playback_buffer[frame_size_bytes:]

                # Ação 1: Toca a voz do robô sempre
                playback_stream.write(playback_frame)
                
                # <<< A LÓGICA CORRIGIDA >>>
                # Ação 2: Só tenta detectar interrupção se tiver áudio do microfone E
                # se o período de aquecimento já tiver passado.
                if len(mic_buffer) >= frame_size_bytes and (time.time() - start_time > warm_up_period_seconds):
                    mic_frame = mic_buffer[:frame_size_bytes]
                    mic_buffer = mic_buffer[frame_size_bytes:]

                    # Aplica o cancelamento de eco
                    clean_bytes = self.aec.process(mic_frame, playback_frame)
                    clean_frame_np = np.frombuffer(clean_bytes, dtype=np.int16)
                    
                    # Alimenta o Porcupine com o áudio limpo
                    result = self.porcupine.process(clean_frame_np)
                    if result >= 0:
                        print(f"\n--> INTERRUPTION 'Antares' DETECTED! Stopping audio...")
                        self.tts_interrupt_event.set()
                        break
            
            if self.tts_interrupt_event.is_set():
                break
                
    finally:
        # Limpeza final
        if tts_thread.is_alive():
            self.tts_interrupt_event.set()
            tts_thread.join()
        
        time.sleep(playback_stream.latency)
        
        if not playback_stream.stopped:
            playback_stream.stop()
        playback_stream.close()
        print("Playback stream closed.")



  def run(self):
    self.speak("Hello, say Antares to talk to me!")

    while True:
      self.wait_for_wakeword()
      self.speak("Yes?")

      command_audio = self.record_audio()
      if command_audio is None:
        print("No audio detected")
        continue

      transcript = self.transcribe_audio(command_audio)
      if not transcript:
        print("No transcription available")
        continue
      print("Question (ME): ", transcript)
        
      if any(word in transcript.lower() for word in ["exit", "quit", "stop", "bye"]):
        self.speak("Goodbye!")
        break

      response_LLM = LLM(transcript)
      response_text = response_LLM.generate().response
      clean_response = re.sub(r'<think>[\s\S]*?</think>', '', response_text).strip()
      print("Answer (LLM): ", clean_response)
      self.speak(clean_response)

      if self.tts_interrupt_event.is_set():
        print("Speak interrupted, stopping capture")
  

  def shutdown(self):
    self.stop_capture = True
    if hasattr(self, 'porcupine'):
      self.porcupine.delete()
    self.capture_thread.join()
    print("Program finished.")

"""
      response_LLM = LLM(transcript)
      response_text = response_LLM.generate().response
      clean_response = re.sub(r'<think>[\s\S]*?</think>', '', response_text).strip()
      print("Answer (LLM): ", clean_response)
      self.speak(clean_response)
"""

if __name__ == "__main__":
  try:
    transcriber = WhisperTranscriber(model_size="small", sample_rate=16000)
    if hasattr(transcriber, 'porcupine'):
      transcriber.run()
  except Exception as e:
      print(f"An unexpected error occurred: {e}")
  finally:
      # Garante que os recursos sejam liberados mesmo se houver um erro
    if 'transcriber' in locals() and hasattr(transcriber, 'shutdown'):
      transcriber.shutdown()