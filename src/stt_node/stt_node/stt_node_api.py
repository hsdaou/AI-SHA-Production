#!/usr/bin/env python3
"""
STT Node API - Ultra-Fast Speech-to-Text using ElevenLabs API
20x faster than local Whisper model
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Empty
import sounddevice as sd
import numpy as np
import threading
import queue
import time
import os
import requests
import json
import subprocess
import time as time_module


class STTNodeAPI(Node):
    def __init__(self):
        super().__init__('stt_node')

        # Note: PulseAudio runs fine with sounddevice - no need to kill it
        # Direct ALSA access works through PortAudio's PulseAudio plugin

        # ElevenLabs API configuration
        self.declare_parameter('api_key', '')
        api_key = self.get_parameter('api_key').value

        if not api_key:
            api_key = os.environ.get('ELEVENLABS_API_KEY', '')

        if not api_key:
            self.get_logger().error('No ElevenLabs API key provided!')
            raise ValueError('Missing ElevenLabs API key')

        self.api_key = api_key
        self.api_url = "https://api.elevenlabs.io/v1/speech-to-text"

        # Audio parameters
        self.sample_rate = 16000
        self.device_channels = 1
        self.channels = 1
        self.chunk_duration = 0.5
        self.silence_threshold = 0.02
        self.device_index = None
        self.respeaker_found = False

        # Publishers
        self.text_pub = self.create_publisher(String, '/speech/text', 10)
        self.interrupt_pub = self.create_publisher(Bool, '/tts/interrupt', 10)
        self.session_reset_pub = self.create_publisher(Bool, '/conversation/reset', 10)
        self.session_deactivated_pub = self.create_publisher(Empty, '/session/deactivated', 10)

        # Wake word configuration
        self.wake_words = [
            'hey robot', 'robot'
        ]

        # Session state - starts inactive, only activates after first wake word
        self.session_active = False
        self.listening_pub = self.create_publisher(String, '/tts_text', 10)  # Publish "I'm listening" directly

        # Conversation timeout - 25 seconds of silence deactivates session
        self.session_timeout = 25.0  # seconds
        self.session_timer = None

        # Conversation memory - temporary chat history for active session
        self.conversation_history = []

        # Subscribe to /audio/playing to mute mic during TTS playback
        self.robot_is_speaking = False
        self.speaking_sub = self.create_subscription(
            Bool,
            '/audio/playing',
            self.speaking_callback,
            10
        )

        # Monitor /tts_text to auto-mute when LLM responds
        self.tts_mute_timer = None
        self.post_stt_cooldown_timer = None
        self.tts_sub = self.create_subscription(
            String,
            '/tts_text',
            self.tts_callback,
            10
        )

        # Audio queue
        self.audio_queue = queue.Queue()

        # Find ReSpeaker
        self._find_respeaker()

        self.get_logger().info('🌐 ElevenLabs STT API ready - ultra-fast cloud transcription')

        # Start audio processing
        threading.Thread(target=self._process_audio, daemon=True).start()

        # Start audio stream
        self._start_stream()

    def _reset_session_timer(self):
        """Reset the 25-second session timeout - ONLY when unmuted"""
        # Cancel existing timer
        if self.session_timer is not None:
            self.session_timer.cancel()
            self.session_timer = None

        # IMPORTANT: Only start timer if mic is unmuted
        if self.robot_is_speaking:
            self.get_logger().info('⏱️  Session timer NOT started - mic still muted (will start after unmute)')
            return

        # Start timer only when unmuted
        self.session_timer = threading.Timer(self.session_timeout, self._deactivate_session)
        self.session_timer.start()
        self.get_logger().warn(f'⏱️  Session timer STARTED - {self.session_timeout}s until auto-deactivate (mic is unmuted)')

    def _deactivate_session(self):
        """Deactivate session after timeout"""
        self.session_active = False
        self.session_timer = None
        self.conversation_history = []  # Clear chat memory

        # Notify LLM to reset conversation history
        reset_msg = Bool()
        reset_msg.data = True
        self.session_reset_pub.publish(reset_msg)

        # Publish session deactivated signal
        self.session_deactivated_pub.publish(Empty())

        self.get_logger().warn('⏰ SESSION DEACTIVATED - 25 seconds of silence → /session/deactivated published')

    def speaking_callback(self, msg):
        """Handle /audio/playing messages to mute mic during TTS"""
        if self.tts_mute_timer is not None:
            self.tts_mute_timer.cancel()
            self.tts_mute_timer = None

        if msg.data:
            # Mute only if not already muted
            if not self.robot_is_speaking:
                self.robot_is_speaking = True
                self.get_logger().warn('🔇 MIC MUTED - Speakers playing')

                # Cancel session timer when robot starts speaking
                if self.session_timer is not None:
                    self.session_timer.cancel()
                    self.session_timer = None
                    self.get_logger().info('⏱️  Session timer CANCELLED - robot speaking')
        else:
            # Unmute only if currently muted (avoid duplicate unmute logs)
            if self.robot_is_speaking:
                self.robot_is_speaking = False
                self.get_logger().warn('🎤 MIC UNMUTED - Speakers stopped (instant)')

                # Start session timer NOW (after robot finishes speaking)
                if self.session_active:
                    self._reset_session_timer()

    def _unmute_after_speaker_stop(self):
        """Unmute 4 seconds after speakers stop"""
        self.robot_is_speaking = False
        self.get_logger().warn('🎤 MIC UNMUTED - 4s after speakers stopped')

        # Clear audio buffer one more time before unmuting
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        # Start session timer NOW (after robot finishes speaking)
        if self.session_active:
            self._reset_session_timer()

    def tts_callback(self, msg):
        """Immediately mute when LLM generates response"""
        self.robot_is_speaking = True
        self.get_logger().warn('🔇 IMMEDIATE MUTE - LLM response received')

        if self.tts_mute_timer is not None:
            self.tts_mute_timer.cancel()
            self.tts_mute_timer = None

        if self.post_stt_cooldown_timer is not None:
            self.post_stt_cooldown_timer.cancel()
            self.post_stt_cooldown_timer = None

        # Cancel session timer when robot starts speaking
        if self.session_timer is not None:
            self.session_timer.cancel()
            self.session_timer = None
            self.get_logger().info('⏱️  Session timer CANCELLED - LLM responding')

        # Failsafe timer
        text_length = len(msg.data)
        estimated_speech = text_length * 0.08
        failsafe_duration = max(10.0, min(estimated_speech + 5.0, 30.0))

        self.tts_mute_timer = threading.Timer(failsafe_duration, self._unmute_after_tts)
        self.tts_mute_timer.start()

    def _unmute_after_tts(self):
        """Failsafe unmute if /audio/playing signal never arrives"""
        self.robot_is_speaking = False
        self.get_logger().warn('🎤 FAILSAFE UNMUTE')

        # Start session timer NOW (after robot finishes speaking)
        if self.session_active:
            self._reset_session_timer()

    def _unmute_post_stt_cooldown(self):
        """Unmute after post-STT cooldown expires"""
        self.robot_is_speaking = False
        self.post_stt_cooldown_timer = None
        self.get_logger().warn('🎤 POST-STT COOLDOWN EXPIRED')

    def _unmute_after_acknowledgment(self):
        """Unmute mic 1.5 seconds after acknowledgment to prevent feedback"""
        self.robot_is_speaking = False
        self.get_logger().warn('🎤 MIC UNMUTED - 1.5s after acknowledgment')

    def _find_respeaker(self):
        """Auto-detect ReSpeaker mic array"""
        devices = sd.query_devices()

        for idx, device in enumerate(devices):
            name = str(device.get('name', '')).lower()
            if 'respeaker' in name or 'seeed' in name:
                self.respeaker_found = True
                self.device_index = idx
                # ReSpeaker has 6 channels, but we'll use channel 0 (mono)
                self.device_channels = 1
                max_channels = device.get('max_input_channels', 1)
                self.get_logger().info(f'✅ Found ReSpeaker: {device.get("name")} (device {idx}, {max_channels} channels available, using 1)')
                return

        self.get_logger().warn('⚠️  ReSpeaker not found, using system default mic')
        self.device_index = None
        self.device_channels = 1

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            self.get_logger().warn(f'Audio status: {status}', throttle_duration_sec=5.0)

        # Convert to mono if multi-channel
        if len(indata.shape) > 1 and indata.shape[1] > 1:
            mono_audio = np.mean(indata, axis=1, keepdims=True)
        else:
            mono_audio = indata.reshape(-1, 1) if len(indata.shape) == 1 else indata

        self.audio_queue.put(mono_audio.copy())

    def _start_stream(self):
        """Start continuous audio capture"""
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
            self.get_logger().info(f'Recording: {self.sample_rate}Hz, {self.device_channels}ch')

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
                self.get_logger().info(f'Using default device: {self.sample_rate}Hz, 1ch')
            except Exception as e2:
                self.get_logger().error(f'All audio devices failed: {e2}')

    def _is_speech(self, audio_chunk):
        """Voice activity detection"""
        rms = np.sqrt(np.mean(audio_chunk**2))
        return rms > self.silence_threshold

    def _is_arabic_or_english(self, text):
        """Check if text is Arabic or English only (bilingual filter)"""
        if not text:
            return False

        # Count character types
        arabic_chars = 0
        english_chars = 0
        other_chars = 0

        for char in text:
            if '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F':  # Arabic Unicode ranges
                arabic_chars += 1
            elif char.isalpha() and ord(char) < 128:  # ASCII English letters
                english_chars += 1
            elif not char.isspace() and char not in '.,!?;:\'"()[]{}0123456789':
                other_chars += 1

        # Accept if majority is Arabic or English
        total_chars = arabic_chars + english_chars + other_chars
        if total_chars == 0:
            return False

        bilingual_ratio = (arabic_chars + english_chars) / total_chars
        return bilingual_ratio > 0.7  # At least 70% Arabic or English

    def _process_audio(self):
        """Process audio chunks from queue"""
        buffer = []
        silence_count = 0
        max_buffer_chunks = 30  # ~15 seconds
        min_speech_chunks = 2   # 1 second minimum
        required_silence_chunks = 2  # 1 second silence

        while rclpy.ok():
            try:
                chunk = self.audio_queue.get(timeout=0.5)

                # If muted, drain queue and clear buffer
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
                        self._transcribe_buffer_api(buffer)
                        buffer = []
                else:
                    if buffer:
                        silence_count += 1
                        if silence_count >= required_silence_chunks and len(buffer) >= min_speech_chunks:
                            self._transcribe_buffer_api(buffer)
                            buffer = []
                            silence_count = 0

            except queue.Empty:
                if buffer and len(buffer) >= min_speech_chunks:
                    self._transcribe_buffer_api(buffer)
                    buffer = []
                continue
            except Exception as e:
                self.get_logger().error(f'Process error: {e}', throttle_duration_sec=5.0)

    def _transcribe_buffer_api(self, buffer):
        """Transcribe using ElevenLabs API - Ultra Fast!"""
        if self.robot_is_speaking:
            return

        try:
            # Concatenate buffer
            audio = np.concatenate(buffer).flatten()

            # Convert to int16 PCM for API
            audio_int16 = (audio * 32767).astype(np.int16)

            # Convert to WAV bytes
            import io
            import wave
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())

            wav_bytes = wav_buffer.getvalue()

            # Call ElevenLabs API
            start_time = time.time()

            headers = {
                'xi-api-key': self.api_key
            }

            files = {
                'file': ('audio.wav', wav_bytes, 'audio/wav')
            }

            data = {
                'model_id': 'scribe_v2',
                'language': 'auto'  # Auto-detect language (supports Arabic, English, etc.)
            }

            response = requests.post(
                self.api_url,
                headers=headers,
                files=files,
                data=data,
                timeout=10
            )

            elapsed = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                text = result.get('text', '').strip()

                if text:
                    # Bilingual filter: Only accept Arabic or English
                    if not self._is_arabic_or_english(text):
                        self.get_logger().info(f'🚫 REJECTED (not Arabic/English): "{text}"')
                        return

                    # Cancel session timer when speech is detected (conversation is active)
                    if self.session_timer is not None:
                        self.session_timer.cancel()
                        self.session_timer = None
                        self.get_logger().info('⏱️  Session timer CANCELLED - speech detected')

                    text_lower = text.lower()

                    # Check for wake word
                    wake_word_found = any(wake_word in text_lower for wake_word in self.wake_words)

                    if wake_word_found:
                        self.get_logger().warn(f'🎯 WAKE WORD DETECTED: "{text}"')

                        # If robot is speaking, interrupt it!
                        if self.robot_is_speaking:
                            interrupt_msg = Bool()
                            interrupt_msg.data = True
                            self.interrupt_pub.publish(interrupt_msg)
                            self.get_logger().warn('⚠️  INTERRUPTING TTS - Wake word detected!')
                            self.robot_is_speaking = False

                        # Activate session if not already active
                        if not self.session_active:
                            self.session_active = True
                            self.conversation_history = []  # Start fresh conversation
                            self.get_logger().warn('✅ SESSION ACTIVATED')

                        # Note: Session timer starts AFTER robot finishes speaking (in unmute callbacks)

                        # Remove wake word from text before sending to LLM
                        cleaned_text = text
                        for wake_word in self.wake_words:
                            cleaned_text = cleaned_text.lower().replace(wake_word, '').strip()

                        # Check if there's a query after wake word
                        if len(cleaned_text) > 3:  # At least a few characters
                            # Send query to LLM
                            msg = String()
                            msg.data = cleaned_text
                            self.text_pub.publish(msg)

                            self.get_logger().warn(f'📤 PUBLISHING TO /speech/text → "{cleaned_text}" ({elapsed:.2f}s)')

                            # No post-STT cooldown - removed for speed
                        else:
                            # Wake word only - acknowledge and wait for command
                            self.get_logger().warn(f'🎤 Wake word only - saying "I\'m listening"')
                            ack_msg = String()
                            # Detect if Arabic wake word was used
                            is_arabic = any(ord(c) >= 0x0600 and ord(c) <= 0x06FF for c in text)
                            ack_msg.data = "أنا أستمع" if is_arabic else "I'm listening"
                            self.listening_pub.publish(ack_msg)

                            # No cooldown after acknowledgment - removed for speed

                    elif self.session_active:
                        # Session is active, process speech without wake word
                        self.get_logger().warn(f'💬 SESSION ACTIVE - Processing: "{text}"')

                        # Note: Session timer resets AFTER robot finishes speaking (in unmute callbacks)

                        msg = String()
                        msg.data = text
                        self.text_pub.publish(msg)

                        # No post-STT cooldown - removed for speed
                    else:
                        # Session not active and no wake word - ignore
                        self.get_logger().info(f'🔒 SESSION INACTIVE: "{text}" (say wake word to activate)')

            else:
                self.get_logger().error(f'ElevenLabs API error {response.status_code}: {response.text}')

        except Exception as e:
            self.get_logger().error(f'Transcription API failed: {e}')

    def destroy_node(self):
        """Cleanup on shutdown"""
        if hasattr(self, 'tts_mute_timer') and self.tts_mute_timer is not None:
            self.tts_mute_timer.cancel()

        if hasattr(self, 'post_stt_cooldown_timer') and self.post_stt_cooldown_timer is not None:
            self.post_stt_cooldown_timer.cancel()

        if hasattr(self, 'session_timer') and self.session_timer is not None:
            self.session_timer.cancel()

        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = STTNodeAPI()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
