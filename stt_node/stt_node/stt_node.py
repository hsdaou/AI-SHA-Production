#!/usr/bin/env python3
"""
STT Node - Optimized Faster-Whisper (GPU Accelerated)
Uses small model for maximum accuracy and speed
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import sounddevice as sd
import numpy as np
import threading
import queue
import time as system_time
from faster_whisper import WhisperModel
import subprocess
import time as time_module

class STTNode(Node):
    def __init__(self):
        super().__init__('stt_node')

        # Kill PulseAudio to allow direct ReSpeaker access
        try:
            subprocess.run(['systemctl', '--user', 'stop', 'pulseaudio.socket', 'pulseaudio.service'], 
                         stderr=subprocess.DEVNULL, check=False)
            subprocess.run(['pulseaudio', '--kill'], stderr=subprocess.DEVNULL, check=False)
            time_module.sleep(1)
            self.get_logger().info('Stopped PulseAudio for direct ReSpeaker access')
        except Exception as e:
            self.get_logger().warn(f'Could not stop PulseAudio: {e}')

        # Optimized parameters
        self.sample_rate = 16000
        self.device_channels = 1  
        self.channels = 1  
        self.chunk_duration = 0.5  
        self.silence_threshold = 0.02  
        self.device_index = None
        self.respeaker_found = False

        # Publisher
        self.text_pub = self.create_publisher(String, '/speech/text', 10)

        # Subscribe to /speaker/playing to mute mic during TTS playback
        self.robot_is_speaking = False
        self.speaking_sub = self.create_subscription(
            Bool,
            '/speaker/playing',
            self.speaking_callback,
            10
        )

        # FAILSAFE: auto-mute the moment the brain dispatches speech, in case
        # the TTS node's authoritative /speaker/playing signal never arrives
        # (e.g. TTS runs on the RPi5 and isn't publishing it). tts_callback
        # mutes immediately and sets a duration timer estimated from the text
        # length, so the mic stays off for the whole spoken response.
        #   /robot_speech : AI-SHA brain (brain_node) — the current deployment
        #   /tts_text     : legacy llm_node brain — kept for rollback
        self.tts_mute_timer = None
        self.post_stt_cooldown_timer = None
        self.auto_mute_start_time = 0.0
        self.robot_speech_sub = self.create_subscription(
            String, '/robot_speech', self.tts_callback, 10)
        self.tts_sub = self.create_subscription(
            String, '/tts_text', self.tts_callback, 10)

        # Audio queue
        self.audio_queue = queue.Queue()

        # Model loading
        self.model = None
        self.model_loaded = threading.Event()

        # Find ReSpeaker
        self._find_respeaker()

        # Load optimized model
        self.get_logger().info('Loading Faster-Whisper SMALL model (excellent accuracy)...')
        threading.Thread(target=self._load_model, daemon=True).start()

        # Start audio processing
        threading.Thread(target=self._process_audio, daemon=True).start()

        # Start audio stream
        self._start_stream()

    def speaking_callback(self, msg):
        if self.tts_mute_timer is not None:
            self.tts_mute_timer.cancel()
            self.tts_mute_timer = None

        if msg.data:
            self.robot_is_speaking = True
            self.get_logger().warn('🔇 MIC MUTED - Speakers playing')
        else:
            self.get_logger().warn('🔊 Speakers stopped - unmuting in 1 second...')
            self.tts_mute_timer = threading.Timer(1.0, self._unmute_after_speaker_stop)
            self.tts_mute_timer.start()

    def _unmute_after_speaker_stop(self):
        self.robot_is_speaking = False
        self.get_logger().warn('🎤 MIC UNMUTED - 1s after speakers stopped')

    def tts_callback(self, msg):
        self.robot_is_speaking = True
        self.get_logger().warn('🔇 IMMEDIATE MUTE - LLM response received, waiting for speaker signals...')

        if self.tts_mute_timer is not None:
            self.tts_mute_timer.cancel()
            self.tts_mute_timer = None

        if self.post_stt_cooldown_timer is not None:
            self.post_stt_cooldown_timer.cancel()
            self.post_stt_cooldown_timer = None

        text_length = len(msg.data)
        estimated_speech = text_length * 0.08
        failsafe_duration = max(10.0, min(estimated_speech + 5.0, 30.0))

        self.get_logger().info(f'⏱️ Failsafe timer: {failsafe_duration:.1f}s')
        self.tts_mute_timer = threading.Timer(failsafe_duration, self._unmute_after_tts)
        self.tts_mute_timer.start()

    def _unmute_after_tts(self):
        self.robot_is_speaking = False
        self.get_logger().warn('🎤 FAILSAFE UNMUTE - No /speaker/playing signal received')

    def _unmute_post_stt_cooldown(self):
        self.robot_is_speaking = False
        self.post_stt_cooldown_timer = None
        self.get_logger().warn('🎤 POST-STT COOLDOWN EXPIRED - Mic unmuted')

    def _find_respeaker(self):
        devices = sd.query_devices()
        for idx, device in enumerate(devices):
            name = str(device.get('name', '')).lower()
            if 'respeaker' in name or 'seeed' in name:
                max_input = device.get('max_input_channels', 0)
                self.respeaker_found = True
                self.device_index = idx 
                self.device_channels = 1 
                self.get_logger().info(f'Found ReSpeaker: {device.get("name")} (device {idx})')
                return

        self.get_logger().warn('ReSpeaker not found, using system default mic')
        self.device_index = None
        self.device_channels = 1

    def _load_model(self):
        try:
            import ctranslate2
            has_cuda = ctranslate2.get_cuda_device_count() > 0

            if has_cuda:
                self.get_logger().info('Loading on GPU with optimized settings...')
                device = "cuda"
                compute_type = "float16"
            else:
                self.get_logger().warn('CUDA not available, using CPU with int8 quantization...')
                device = "cpu"
                compute_type = "int8"

            self.model = WhisperModel(
                "small",  
                device=device,
                compute_type=compute_type,
                num_workers=4  
            )

            dummy_audio = np.zeros(self.sample_rate * 2, dtype=np.float32)
            _ = list(self.model.transcribe(
                dummy_audio,
                language="en",
                beam_size=1,
                vad_filter=False
            ))

            self.model_loaded.set()
            if has_cuda:
                self.get_logger().info('✓ Whisper SMALL ready on GPU')
            else:
                self.get_logger().info('✓ Whisper base ready on CPU')

        except Exception as e:
            self.get_logger().error(f'Model load failed: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            self.get_logger().warn(f'Audio status: {status}', throttle_duration_sec=5.0)

        if len(indata.shape) > 1 and indata.shape[1] > 1:
            mono_audio = np.mean(indata, axis=1, keepdims=True)
        else:
            mono_audio = indata.reshape(-1, 1) if len(indata.shape) == 1 else indata

        self.audio_queue.put(mono_audio.copy())

    def _start_stream(self):
        chunk_samples = int(self.sample_rate * self.chunk_duration)
        try:
            self.stream = sd.InputStream(
                device=self.device_index,
                channels=self.device_channels,
                samplerate=self.sample_rate,
                blocksize=chunk_samples,
                callback=self._audio_callback
            )
            self.stream.start()
            self.get_logger().info(f'Recording: {self.sample_rate}Hz, {self.device_channels}ch device -> 1ch mono')

        except Exception as e:
            self.get_logger().error(f'Stream start failed: {e}')
            try:
                self.get_logger().warn('Trying default audio device...')
                self.device_channels = 1
                self.stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    blocksize=chunk_samples,
                    callback=self._audio_callback
                )
                self.stream.start()
            except Exception as e2:
                self.get_logger().error(f'All audio devices failed: {e2}')

    def _is_speech(self, audio_chunk):
        rms = np.sqrt(np.mean(audio_chunk**2))
        return rms > self.silence_threshold

    def _process_audio(self):
        buffer = []
        silence_count = 0
        max_buffer_chunks = 30  
        min_speech_chunks = 2   
        required_silence_chunks = 2 

        while rclpy.ok():
            try:
                chunk = self.audio_queue.get(timeout=0.5)

                if self.robot_is_speaking:
                    buffer = [] 
                    silence_count = 0
                    while not self.audio_queue.empty():
                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            break
                    continue 

                if self._is_speech(chunk):
                    buffer.append(chunk)
                    silence_count = 0
                    if len(buffer) >= max_buffer_chunks:
                        self._transcribe_buffer(buffer)
                        buffer = []
                else:
                    if buffer:
                        silence_count += 1
                        if silence_count >= required_silence_chunks and len(buffer) >= min_speech_chunks:
                            self._transcribe_buffer(buffer)
                            buffer = []
                            silence_count = 0

            except queue.Empty:
                if buffer and len(buffer) >= min_speech_chunks:
                    self._transcribe_buffer(buffer)
                    buffer = []
                continue
            except Exception as e:
                self.get_logger().error(f'Process error: {e}', throttle_duration_sec=5.0)

    def _transcribe_buffer(self, buffer):
        if self.robot_is_speaking:
            return

        if not self.model_loaded.is_set():
            return

        try:
            audio = np.concatenate(buffer).flatten()
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            start_time = system_time.time()
            segments, info = self.model.transcribe(
                audio,
                language="en",
                beam_size=8,  
                best_of=5,    
                temperature=0.0, 
                vad_filter=True, 
                vad_parameters=dict(
                    min_silence_duration_ms=300, 
                    threshold=0.35, 
                    min_speech_duration_ms=150 
                ),
                condition_on_previous_text=False, 
                compression_ratio_threshold=2.2, 
                log_prob_threshold=-1.0, 
                no_speech_threshold=0.5, 
                initial_prompt="Clear speech transcription.", 
                word_timestamps=False 
            )

            text_segments = [segment.text for segment in segments]
            elapsed = system_time.time() - start_time
            text = ' '.join(text_segments).strip()

            if text:
                msg = String()
                msg.data = text
                self.text_pub.publish(msg)
                self.get_logger().info(f'📤 "{text}" ({elapsed:.2f}s)')

                if self.post_stt_cooldown_timer is not None:
                    self.post_stt_cooldown_timer.cancel()
                    self.post_stt_cooldown_timer = None

                self.robot_is_speaking = True
                self.post_stt_cooldown_timer = threading.Timer(10.0, self._unmute_post_stt_cooldown)
                self.post_stt_cooldown_timer.start()

        except Exception as e:
            self.get_logger().error(f'Transcription failed: {e}', throttle_duration_sec=5.0)

    def destroy_node(self):
        if hasattr(self, 'tts_mute_timer') and self.tts_mute_timer is not None:
            self.tts_mute_timer.cancel()
        if hasattr(self, 'post_stt_cooldown_timer') and self.post_stt_cooldown_timer is not None:
            self.post_stt_cooldown_timer.cancel()
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = STTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
