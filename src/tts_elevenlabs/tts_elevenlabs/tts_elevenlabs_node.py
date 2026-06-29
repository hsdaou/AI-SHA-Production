#!/usr/bin/env python3
"""
ROS2 TTS Speaker Node using Eleven Labs API
Supports pause command - immediately stops and says "Go ahead"
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from std_msgs.msg import String, Bool, Empty
import subprocess
import os
import signal
import time
import tempfile
import threading
from elevenlabs import ElevenLabs, VoiceSettings


class TTSElevenLabsNode(Node):
    def __init__(self):
        super().__init__('tts_elevenlabs_node')

        # Eleven Labs configuration
        self.api_key = os.environ.get('ELEVENLABS_API_KEY', '')
        if not self.api_key:
            self.get_logger().error(
                'ELEVENLABS_API_KEY not set! '
                'Run: export ELEVENLABS_API_KEY="your-key-here"')
        self.client = ElevenLabs(api_key=self.api_key)
        self.voice_id = "iP95p4xoKVk53GoZ742B"

        # Audio device configuration
        # Using PulseAudio/PipeWire output (RPi5 uses PipeWire)
        # self.audio_device = 'plughw:CARD=MAX98357A,DEV=0'  # Not needed with pulse output

        # Playback control
        self.current_process = None
        self.is_playing = False
        self.pause_requested = threading.Event()
        self.playback_lock = threading.Lock()

        # Track speaking state to prevent premature unmuting
        self.is_currently_speaking = False
        self.pending_messages = 0
        self.speaking_state_lock = threading.Lock()

        # Track last published state to avoid duplicate signals
        self.last_published_state = None

        # QoS for reliable message delivery
        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # QoS for pause - use BEST_EFFORT for faster discovery
        pause_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribe to TTS text
        self.subscription = self.create_subscription(
            String, '/tts_text', self.speech_callback, reliable_qos)

        # Subscribe to PAUSE - multiple QoS profiles
        self.pause_sub1 = self.create_subscription(
            Empty, '/pause', self.pause_callback, reliable_qos)
        self.pause_sub2 = self.create_subscription(
            Empty, '/pause', self.pause_callback, pause_qos)
        
        # Also listen to a String-based pause for compatibility
        self.pause_sub3 = self.create_subscription(
            String, '/pause_cmd', self.pause_string_callback, reliable_qos)

        # Publishers - Status (Bool)
        self.speaking_publisher = self.create_publisher(Bool, '/robot/speaking', 10)
        self.speaker_playing_publisher = self.create_publisher(Bool, '/speaker/playing', 10)

        # Publishers - Events (Empty)
        self.tts_started_publisher = self.create_publisher(Empty, '/tts/started', 10)
        self.tts_finished_publisher = self.create_publisher(Empty, '/tts/finished', 10)

        # Pre-generate "Go ahead" audio
        self.go_ahead_audio = None
        self.prepare_go_ahead_audio()

        # Start a timer to check for pause file (fallback mechanism)
        self.pause_file = '/tmp/tts_pause_signal'
        self.create_timer(0.1, self.check_pause_file)

        self.get_logger().info('TTS Node ready - listening on /tts_text and /pause')
        self.get_logger().info('Publishing: /robot/speaking, /speaker/playing (Bool), /tts/started, /tts/finished (Empty)')

    def publish_speaker_state(self, state):
        """Publish speaker state only if it has changed"""
        if self.last_published_state != state:
            msg = Bool()
            msg.data = state
            self.speaking_publisher.publish(msg)
            self.speaker_playing_publisher.publish(msg)
            self.last_published_state = state
            self.get_logger().warn(f'🔊 /speaker/playing = {state} - {"MUTED 🔇" if state else "UNMUTED 🎤"}')
        else:
            self.get_logger().debug(f'Skipping duplicate state: {state}')

    def prepare_go_ahead_audio(self):
        """Pre-generate the go ahead prompt"""
        try:
            audio_generator = self.client.text_to_speech.convert(
                text="Go ahead, I'm listening.",
                voice_id=self.voice_id,
                model_id="eleven_multilingual_v2",
                voice_settings=VoiceSettings(
                    stability=0.5, similarity_boost=0.75, style=0.0, use_speaker_boost=True
                )
            )
            self.go_ahead_audio = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            for chunk in audio_generator:
                if chunk:
                    self.go_ahead_audio.write(chunk)
            self.go_ahead_audio.close()
            self.get_logger().info('"Go ahead" audio ready')
        except Exception as e:
            self.get_logger().error(f'Failed to prepare go ahead audio: {e}')

    def check_pause_file(self):
        """Check for pause file (fallback for network issues)"""
        if os.path.exists(self.pause_file):
            try:
                os.remove(self.pause_file)
                self.get_logger().warn('PAUSE via file signal')
                self.do_pause()
            except:
                pass

    def pause_string_callback(self, msg):
        """Handle string-based pause command"""
        if 'pause' in msg.data.lower():
            self.pause_callback(Empty())

    def pause_callback(self, msg):
        """Handle pause command"""
        self.get_logger().warn('⏸ PAUSE RECEIVED - Stopping TTS!')
        self.do_pause()

    def do_pause(self):
        """Execute the pause action"""
        self.pause_requested.set()
        self.kill_playback()
        threading.Thread(target=self.play_go_ahead, daemon=True).start()

    def kill_playback(self):
        """Forcefully kill audio playback"""
        # Kill all audio processes immediately
        for cmd in ['pw-play', 'mpg123', 'ffplay', 'aplay']:
            try:
                subprocess.run(['pkill', '-9', cmd], capture_output=True, timeout=1)
            except:
                pass
        
        with self.playback_lock:
            if self.current_process:
                try:
                    os.killpg(os.getpgid(self.current_process.pid), signal.SIGKILL)
                except:
                    pass
                try:
                    self.current_process.kill()
                except:
                    pass
                self.current_process = None
        
        self.is_playing = False

        # Unmute mic
        self.publish_speaker_state(False)

        # Publish TTS finished event when session deactivates via pause
        self.tts_finished_publisher.publish(Empty())

    def play_go_ahead(self):
        """Play go ahead prompt"""

        if self.go_ahead_audio and os.path.exists(self.go_ahead_audio.name):
            try:
                # Brief mute
                self.publish_speaker_state(True)

                proc = subprocess.Popen(
                    ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', self.go_ahead_audio.name],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    preexec_fn=os.setsid
                )
                proc.wait()
            except:
                pass
            finally:
                self.publish_speaker_state(False)

        self.pause_requested.clear()

    def speech_callback(self, msg):
        """Handle TTS request"""
        if self.pause_requested.is_set():
            return

        text = msg.data.strip()
        if not text:
            return

        # Track this message
        with self.speaking_state_lock:
            self.pending_messages += 1

        self.get_logger().info(f'Speaking: "{text[:50]}..." (pending: {self.pending_messages})')
        threading.Thread(target=self.do_playback, args=(text,), daemon=True).start()

    def do_playback(self, text):
        """Perform TTS playback"""
        temp_audio_path = None
        
        try:
            if self.pause_requested.is_set():
                with self.speaking_state_lock:
                    self.pending_messages -= 1
                return

            audio_generator = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id="eleven_multilingual_v2",
                voice_settings=VoiceSettings(
                    stability=0.5, similarity_boost=0.75, style=0.0, use_speaker_boost=True
                )
            )

            if self.pause_requested.is_set():
                with self.speaking_state_lock:
                    self.pending_messages -= 1
                return

            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                temp_audio_path = f.name
                for chunk in audio_generator:
                    if chunk:
                        f.write(chunk)
                    if self.pause_requested.is_set():
                        with self.speaking_state_lock:
                            self.pending_messages -= 1
                        return

            if self.pause_requested.is_set():
                with self.speaking_state_lock:
                    self.pending_messages -= 1
                return

            # Send True signal now (right before starting to speak)
            with self.speaking_state_lock:
                if not self.is_currently_speaking:
                    self.is_playing = True
                    self.publish_speaker_state(True)
                    self.is_currently_speaking = True
                    # Publish TTS started event
                    self.tts_started_publisher.publish(Empty())

            # No delay - start playback immediately
            self.get_logger().info('Starting playback immediately')

            if self.pause_requested.is_set():
                with self.speaking_state_lock:
                    self.pending_messages -= 1
                return

            with self.playback_lock:
                self.get_logger().warn(f'▶️  STARTING PLAYBACK')
                self.current_process = subprocess.Popen(
                    ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', temp_audio_path],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    preexec_fn=os.setsid
                )

            if self.current_process:
                returncode = self.current_process.wait()
                self.get_logger().warn(f'⏹️  PLAYBACK FINISHED (return code: {returncode})')
                if returncode != 0:
                    stderr = self.current_process.stderr.read().decode() if self.current_process.stderr else ''
                    self.get_logger().error(f'mpg123 failed with code {returncode}: {stderr}')

        except Exception as e:
            self.get_logger().error(f'TTS Error: {e}')
        finally:
            with self.playback_lock:
                self.current_process = None

            # Mark this message as processed
            with self.speaking_state_lock:
                self.pending_messages -= 1
                current_pending = self.pending_messages

            if not self.pause_requested.is_set():
                # Only unmute if no more pending messages
                if current_pending == 0:
                    # Unmute immediately when playback finishes
                    with self.speaking_state_lock:
                        if self.is_currently_speaking:
                            self.publish_speaker_state(False)
                            self.is_playing = False
                            self.is_currently_speaking = False

                            # Publish TTS finished event when session deactivates
                            self.tts_finished_publisher.publish(Empty())
                else:
                    self.get_logger().info(f'Playback finished, but {current_pending} message(s) pending - keeping mic muted')

            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass


def main(args=None):
    rclpy.init(args=args)
    try:
        node = TTSElevenLabsNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
