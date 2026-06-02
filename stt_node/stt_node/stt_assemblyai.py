#!/usr/bin/env python3
"""
STT Node - AssemblyAI Universal Streaming (v3)
Cloud-based real-time transcription with ~300ms latency
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import sounddevice as sd
import numpy as np
import threading
import queue
import time
import os
import subprocess
import time as time_module
from typing import Type

import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    TerminationEvent,
    TurnEvent,
)


class STTAssemblyAINode(Node):
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

        # Get API key from environment
        api_key = os.environ.get('ASSEMBLYAI_API_KEY')
        if not api_key:
            self.get_logger().error('ASSEMBLYAI_API_KEY not set! Export it first.')
            raise ValueError('Missing ASSEMBLYAI_API_KEY environment variable')

        self.api_key = api_key
        self.get_logger().info(f'AssemblyAI API key configured ({api_key[:8]}...)')

        # Audio parameters (AssemblyAI requires 16kHz)
        self.sample_rate = 16000
        self.device_channels = 1
        self.channels = 1
        self.chunk_duration = 0.1  # 100ms chunks for streaming
        self.device_index = None
        self.respeaker_found = False

        # Publisher
        self.text_pub = self.create_publisher(String, '/speech/text', 10)

        # Muting control
        self.robot_is_speaking = False
        self.tts_mute_timer = None
        self.auto_mute_start_time = 0.0

        # Subscribe to /speaker/playing
        self.speaking_sub = self.create_subscription(
            Bool,
            '/speaker/playing',
            self.speaking_callback,
            10
        )

        # Subscribe to /tts_text for auto-mute
        self.tts_sub = self.create_subscription(
            String,
            '/tts_text',
            self.tts_callback,
            10
        )

        # AssemblyAI streaming client
        self.streaming_client = None
        self.stream_active = False
        self.session_id = None

        # Transcript accumulation (AssemblyAI sends cumulative updates)
        self.current_transcript = ""
        self.transcript_start_time = 0.0
        self.last_update_time = 0.0
        self.silence_timeout = 4.0  # Send after 4s of silence
        self.max_duration = 30.0    # Force send after 30s
        self.send_timer = None

        # Audio queue for streaming
        self.audio_queue = queue.Queue()

        # Find ReSpeaker
        self._find_respeaker()

        # Start AssemblyAI client
        self._start_streaming_client()

        # Start audio stream
        self._start_stream()

        # Start audio streaming thread
        threading.Thread(target=self._stream_audio_worker, daemon=True).start()

    def speaking_callback(self, msg):
        """Handle /speaker/playing messages"""
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
        """Unmute mic 1 second after speakers stop"""
        self.robot_is_speaking = False
        self.get_logger().warn('🎤 MIC UNMUTED - 1s after speakers stopped')

    def tts_callback(self, msg):
        """FAILSAFE: Auto-mute when LLM responds (before RPi5 signals)"""
        self.get_logger().warn('🔇 IMMEDIATE MUTE - LLM response received, waiting for speaker signals...')
        self.robot_is_speaking = True
        self.auto_mute_start_time = time.time()

        # Safety timeout: Force unmute after 30 seconds if RPi5 never signals
        if self.tts_mute_timer is not None:
            self.tts_mute_timer.cancel()
        self.tts_mute_timer = threading.Timer(30.0, self._force_unmute_timeout)
        self.tts_mute_timer.start()

    def _force_unmute_timeout(self):
        """Force unmute if RPi5 never signals completion"""
        elapsed = time.time() - self.auto_mute_start_time
        self.get_logger().warn(f'⚠️  FORCE UNMUTE after {elapsed:.1f}s - RPi5 never signaled completion')
        self.robot_is_speaking = False

    def _find_respeaker(self):
        """Find ReSpeaker Mic Array v3.0"""
        devices = sd.query_devices()
        self.get_logger().info('Available audio devices:')
        for i, device in enumerate(devices):
            self.get_logger().info(f'  [{i}] {device["name"]} (in:{device["max_input_channels"]}, out:{device["max_output_channels"]})')
            if 'ReSpeaker' in device['name'] or 'respeaker' in device['name'].lower():
                self.device_index = i
                self.respeaker_found = True
                self.get_logger().info(f'✅ Found ReSpeaker at device {i}')
                break

        if not self.respeaker_found:
            self.get_logger().warn('⚠️  ReSpeaker not found, using default device')

    def _on_begin(self, client: Type[StreamingClient], event: BeginEvent):
        """Called when session starts"""
        self.session_id = event.id
        self.get_logger().info(f'✅ AssemblyAI session started: {event.id}')

    def _on_turn(self, client: Type[StreamingClient], event: TurnEvent):
        """Called when a turn (transcription) is received

        AssemblyAI sends CUMULATIVE transcripts - each update contains
        the full text so far, not just new words. We keep the latest
        and send after silence or explicit end_of_turn.
        """
        # Skip if muted
        if self.robot_is_speaking:
            return

        current_time = time.time()

        # Cancel existing send timer
        if self.send_timer is not None:
            self.send_timer.cancel()
            self.send_timer = None

        # Update current transcript (AssemblyAI sends cumulative, so just replace)
        if event.transcript:
            text = event.transcript.strip()
            if text:
                # Start timing if this is first update
                if not self.current_transcript:
                    self.transcript_start_time = current_time

                # Replace with latest (AssemblyAI sends cumulative updates)
                self.current_transcript = text
                self.last_update_time = current_time

                # Log updates for debugging
                duration = current_time - self.transcript_start_time
                self.get_logger().debug(
                    f'🔄 {duration:.1f}s: "{text[:50]}..."',
                    throttle_duration_sec=0.5
                )

        # Send conditions
        should_send_now = False

        # 1. Explicit end of turn from AssemblyAI
        if event.end_of_turn:
            should_send_now = True

        # 2. Force send if transcript is getting too long
        elif self.current_transcript:
            duration = current_time - self.transcript_start_time
            if duration >= self.max_duration:
                self.get_logger().warn(f'⏱️ Forcing send after {duration:.1f}s')
                should_send_now = True

        if should_send_now:
            self._send_transcript()
        elif self.current_transcript:
            # Set timer to send after silence
            self.send_timer = threading.Timer(self.silence_timeout, self._send_transcript)
            self.send_timer.start()

    def _send_transcript(self):
        """Send the current accumulated transcript"""
        if self.send_timer is not None:
            self.send_timer.cancel()
            self.send_timer = None

        if not self.current_transcript:
            return

        # Calculate duration
        duration = time.time() - self.transcript_start_time

        # Send transcript
        self.get_logger().warn(f'📤 PUBLISHING TO /speech/text → "{self.current_transcript}" ({duration:.2f}s)')
        self.get_logger().warn(f'   LLM should receive this and respond...')
        msg = String()
        msg.data = self.current_transcript
        self.text_pub.publish(msg)

        # Clear for next utterance
        self.current_transcript = ""
        self.transcript_start_time = 0.0
        self.last_update_time = 0.0

        # Shorter post-transcription cooldown (2s instead of 5s)
        if not self.robot_is_speaking:
            self.get_logger().warn('🔇 Post-transcription cooldown (2s)')
            self.robot_is_speaking = True
            self.tts_mute_timer = threading.Timer(2.0, self._unmute_post_transcription)
            self.tts_mute_timer.start()

    def _unmute_post_transcription(self):
        """Unmute after post-transcription cooldown"""
        self.robot_is_speaking = False
        self.get_logger().warn('🎤 Post-transcription cooldown expired - mic ready')

    def _on_terminated(self, client: Type[StreamingClient], event: TerminationEvent):
        """Called when session terminates"""
        self.get_logger().warn(f'AssemblyAI session terminated: {event.audio_duration_seconds:.1f}s processed')

    def _on_error(self, client: Type[StreamingClient], error: StreamingError):
        """Called on error"""
        self.get_logger().error(f'AssemblyAI error: {error}')

    def _start_streaming_client(self):
        """Initialize AssemblyAI streaming client"""
        self.get_logger().info('🌐 Connecting to AssemblyAI universal streaming (v3)...')

        # Create client with callbacks
        self.streaming_client = StreamingClient(
            StreamingClientOptions(
                api_key=self.api_key,
                api_host="streaming.assemblyai.com",
            )
        )

        # Register event handlers
        self.streaming_client.on(StreamingEvents.Begin, self._on_begin)
        self.streaming_client.on(StreamingEvents.Turn, self._on_turn)
        self.streaming_client.on(StreamingEvents.Termination, self._on_terminated)
        self.streaming_client.on(StreamingEvents.Error, self._on_error)

        # Connect with streaming parameters
        self.streaming_client.connect(
            StreamingParameters(
                sample_rate=self.sample_rate,
                format_turns=True,  # Get formatted turns (complete sentences)
            )
        )

        self.stream_active = True
        self.get_logger().info('✅ Connected to AssemblyAI streaming API')

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            self.get_logger().warn(f'Audio status: {status}', throttle_duration_sec=5.0)

        # Convert to mono if multi-channel
        if len(indata.shape) > 1 and indata.shape[1] > 1:
            mono_audio = np.mean(indata, axis=1, keepdims=True)
        else:
            mono_audio = indata.reshape(-1, 1) if len(indata.shape) == 1 else indata

        # Queue audio for streaming (don't queue if muted)
        if not self.robot_is_speaking:
            self.audio_queue.put(mono_audio.copy())
        else:
            # Drain queue when muted to prevent buffering
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

    def _stream_audio_worker(self):
        """Worker thread to stream audio to AssemblyAI"""
        self.get_logger().info('🎙️ Audio streaming worker started')

        while rclpy.ok() and self.stream_active:
            try:
                # Get audio chunk
                chunk = self.audio_queue.get(timeout=0.5)

                # Convert to int16 for AssemblyAI
                audio_int16 = (chunk.flatten() * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()

                # Send to AssemblyAI
                self.streaming_client.stream(audio_bytes)

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Streaming error: {e}', throttle_duration_sec=5.0)

    def _start_stream(self):
        """Start continuous audio capture"""
        chunk_samples = int(self.sample_rate * self.chunk_duration)

        try:
            self.stream = sd.InputStream(
                device=self.device_index,
                channels=self.device_channels,
                samplerate=self.sample_rate,
                blocksize=chunk_samples,
                callback=self._audio_callback,
                dtype=np.float32
            )
            self.stream.start()
            self.get_logger().info(f'🎤 Audio stream started (device: {self.device_index}, {self.sample_rate}Hz, {chunk_samples} samples/chunk)')
        except Exception as e:
            self.get_logger().error(f'Failed to start audio stream: {e}')
            raise

    def destroy_node(self):
        """Cleanup on shutdown"""
        self.stream_active = False
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()
        if self.streaming_client:
            self.streaming_client.disconnect(terminate=True)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = STTAssemblyAINode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
