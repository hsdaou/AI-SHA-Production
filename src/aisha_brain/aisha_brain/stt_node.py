import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import threading
import queue
import subprocess
import os
import re
import time
import io
import tempfile
import wave
import numpy as np


# ── Wake word variants ────────────────────────────────────────────────────────
# Whisper may transcribe "AISHA" / "Hey AISHA" in various ways depending on
# accent, noise, and model size.  We match against these known variants.
# Order matters: longer prefixes are checked first so "hey aisha" is stripped
# before "aisha" alone.
_WAKE_PREFIXES = [
    # AI-SHA / Aisha variants (primary robot name)
    'hey aisha', 'hey i sha', 'hey eye sha', 'hey asha', 'hey isha',
    'hi aisha',  'hi i sha',  'hi isha',
    'aisha',     'i sha',     'eye sha',     'asha',     'isha',
    # ARIA variants (legacy / alternate transcriptions)
    'hey aria',  'hey arya',  'hey area',    'hey ariya',
    'hi aria',   'hi arya',   'hi area',
    'aria',      'arya',      'area',        'ariya',
]

# Compiled regex: match any wake prefix at the start of the string,
# optionally followed by a comma / period / colon / space.
_WAKE_RE = re.compile(
    r'^(?:' + '|'.join(re.escape(p) for p in _WAKE_PREFIXES) + r')'
    r'[\s,.:;!?\-]*',
    re.IGNORECASE,
)


class STTNode(Node):
    """Speech-to-Text node using Faster-Whisper (CTranslate2) for local transcription.

    Audio capture modes (in priority order):
      1. Continuous ring buffer via sounddevice (preferred) — captures audio
         continuously, uses RMS energy detection to find speech boundaries,
         and sends complete utterances to Faster-Whisper. No mid-sentence splits.
      2. Fallback: discrete arecord chunks — for systems without sounddevice/
         PortAudio. Captures fixed-duration WAV files in a loop.

    Wake word support:
      When enabled (default), only speech preceded by "AISHA" or "Hey AISHA"
      is published. After a wake word trigger, a listening window stays
      open (default 15 s) so follow-up sentences don't need the wake word
      again. The window is also extended when the robot responds (TTS).

    Feedback prevention: subscribes to /speaker/playing (Bool).
    When True, microphone capture is paused so the robot does not transcribe
    its own TTS output.

    Target architecture topics:
      Publishes:  /speech/text      (std_msgs/String)
      Subscribes: /speaker/playing  (std_msgs/Bool)
                  /robot_speech     (std_msgs/String)  — extends wake window

    Requires:
      - faster-whisper Python package  (pip install faster-whisper)
      - sounddevice + PortAudio (preferred) or arecord (ALSA fallback)
    """

    def __init__(self):
        super().__init__('ai_sha_stt')

        # Publish to architecture-standard topic
        self.publisher_ = self.create_publisher(String, '/speech/text', 10)

        # Feedback prevention: mute mic while TTS is playing
        self._is_speaker_playing = False
        self.create_subscription(Bool, '/speaker/playing', self._on_speaker_playing, 10)

        # Parameters
        self.declare_parameter('whisper_model', 'base')
        self.declare_parameter('whisper_device', 'cpu')
        self.declare_parameter('whisper_compute_type', 'int8')
        self.declare_parameter('language', 'en')
        self.declare_parameter('silence_threshold', 0.06)   # fraction of max int16 (0.06 ≈ 1966 RMS; raised for noisy school environments)
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('chunk_duration', 5.0)       # max seconds per utterance (arecord fallback) / max recording length (continuous mode)
        self.declare_parameter('audio_device', 'plughw:1,0')  # ALSA capture device

        # Continuous capture parameters
        # speech_pad_ms: silence padding before/after speech. Lower values
        # reduce latency (user→transcription) but risk clipping the end of
        # utterances.  200ms is aggressive — increase if words are cut off.
        self.declare_parameter('speech_pad_ms', 200)
        # min_speech_ms: ignore audio shorter than this.  Filters coughs and
        # bumps.  200ms still catches short commands like "stop" or "yes".
        self.declare_parameter('min_speech_ms', 200)
        self.declare_parameter('max_speech_s', 15.0)          # maximum single utterance length

        # Wake word parameters
        self.declare_parameter('wake_word_enabled', True)
        self.declare_parameter('wake_word_timeout', 15.0)    # seconds of continued listening

        self.whisper_model_size = self.get_parameter('whisper_model').get_parameter_value().string_value
        self.whisper_device = self.get_parameter('whisper_device').get_parameter_value().string_value
        self.compute_type = self.get_parameter('whisper_compute_type').get_parameter_value().string_value
        self.language = self.get_parameter('language').get_parameter_value().string_value
        self.silence_threshold = self.get_parameter('silence_threshold').get_parameter_value().double_value
        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        self.chunk_duration = self.get_parameter('chunk_duration').get_parameter_value().double_value
        self.audio_device = self.get_parameter('audio_device').get_parameter_value().string_value

        self.speech_pad_ms = self.get_parameter('speech_pad_ms').get_parameter_value().integer_value
        self.min_speech_ms = self.get_parameter('min_speech_ms').get_parameter_value().integer_value
        self.max_speech_s = self.get_parameter('max_speech_s').get_parameter_value().double_value

        self.wake_word_enabled = self.get_parameter('wake_word_enabled').get_parameter_value().bool_value
        self.wake_word_timeout = self.get_parameter('wake_word_timeout').get_parameter_value().double_value

        # Wake word state: timestamp until which we accept speech without wake word
        self._wake_active_until: float = 0.0

        # Extend listening window when the robot responds (conversation continuation)
        if self.wake_word_enabled:
            self.create_subscription(String, '/robot_speech', self._on_robot_speech, 10)

        self._msg_queue = queue.SimpleQueue()
        self._publish_timer = self.create_timer(0.1, self._publish_pending)
        self._model = None

        if not self._load_model():
            self.get_logger().error(
                'faster-whisper not available. '
                'Install with: pip install faster-whisper --break-system-packages'
            )
            return

        # Detect capture mode: prefer continuous sounddevice, fall back to arecord
        self._use_continuous = False
        try:
            import sounddevice  # noqa: F401
            self._use_continuous = True
            self.get_logger().info('Audio capture: continuous ring buffer (sounddevice)')
        except ImportError:
            if self._check_arecord():
                self.get_logger().warning(
                    'sounddevice not available — falling back to arecord chunked capture. '
                    'Install sounddevice for better accuracy: pip install sounddevice'
                )
            else:
                self.get_logger().error(
                    'Neither sounddevice nor arecord available. '
                    'Install sounddevice (pip install sounddevice) or alsa-utils (apt install alsa-utils).'
                )
                return

        wake_status = (
            f'wake_word=AISHA (timeout={self.wake_word_timeout}s)'
            if self.wake_word_enabled else 'wake_word=disabled (open mic)'
        )
        capture_mode = 'continuous' if self._use_continuous else f'arecord (chunk={self.chunk_duration}s)'
        self.get_logger().info(
            f'AI-SHA STT Active — faster-whisper/{self.whisper_model_size} '
            f'on {self.whisper_device} ({self.compute_type}), '
            f'mic={self.audio_device}, capture={capture_mode}, '
            f'{wake_status}'
        )

        if self._use_continuous:
            threading.Thread(target=self._continuous_listen_loop, daemon=True).start()
        else:
            threading.Thread(target=self._arecord_listen_loop, daemon=True).start()

    # ── Startup checks ────────────────────────────────────────────────────────

    def _check_arecord(self) -> bool:
        try:
            subprocess.run(['which', 'arecord'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _load_model(self) -> bool:
        try:
            from faster_whisper import WhisperModel
            self.get_logger().info(f'Loading faster-whisper "{self.whisper_model_size}"...')
            self._model = WhisperModel(
                self.whisper_model_size,
                device=self.whisper_device,
                compute_type=self.compute_type,
            )
            self.get_logger().info('faster-whisper model ready.')
            return True
        except ImportError:
            return False
        except Exception as e:
            self.get_logger().error(f'Failed to load faster-whisper model: {e}')
            return False

    # ── Speaker playing feedback prevention ───────────────────────────────────

    def _on_speaker_playing(self, msg: Bool):
        self._is_speaker_playing = msg.data
        state = 'muted (TTS playing)' if msg.data else 'listening'
        self.get_logger().debug(f'STT mic: {state}')

    # ── Robot speech callback — extend listening window ───────────────────────

    def _on_robot_speech(self, msg: String):
        """When the robot speaks, extend the wake-word listening window."""
        if msg.data.strip():
            self._wake_active_until = time.monotonic() + self.wake_word_timeout
            self.get_logger().debug(
                f'Wake window extended (robot spoke) — '
                f'listening for {self.wake_word_timeout}s'
            )

    # ── Wake word detection ──────────────────────────────────────────────────

    def _check_wake_word(self, text: str) -> tuple:
        """Check if text contains the wake word and strip it.

        Returns:
            (wake_triggered, cleaned_text)
        """
        match = _WAKE_RE.match(text)
        if match:
            cleaned = text[match.end():].strip()
            return True, cleaned
        return False, text

    def _should_publish(self, text: str) -> tuple:
        """Decide whether to publish this transcription based on wake word state.

        Returns:
            (should_publish, text_to_publish)
        """
        if not self.wake_word_enabled:
            return True, text

        now = time.monotonic()
        triggered, cleaned = self._check_wake_word(text)

        if triggered:
            self._wake_active_until = now + self.wake_word_timeout
            self.get_logger().info(
                f'Wake word detected! Listening for {self.wake_word_timeout}s'
            )
            # If the user said just the wake word (e.g. "Hey AISHA") with no
            # follow-up text, send a standardized trigger so brain_node can
            # greet the user ("Yes, how can I help you?")
            return True, cleaned if cleaned else 'wake_word_triggered'

        if now < self._wake_active_until:
            remaining = self._wake_active_until - now
            self.get_logger().debug(
                f'Wake window active ({remaining:.1f}s left) — passing through'
            )
            self._wake_active_until = now + self.wake_word_timeout
            return True, text

        self.get_logger().debug(
            f'Discarded (no wake word): "{text[:60]}..."'
            if len(text) > 60 else f'Discarded (no wake word): "{text}"'
        )
        return False, ''

    # ── ROS publish (called from timer in executor thread) ────────────────────

    def _publish_pending(self):
        while not self._msg_queue.empty():
            try:
                text = self._msg_queue.get_nowait()
                msg = String()
                msg.data = text
                self.publisher_.publish(msg)
                self.get_logger().info(f'STT → /speech/text: "{text}"')
            except queue.Empty:
                break

    # ── Common: transcribe numpy audio and publish ────────────────────────────

    def _transcribe_and_publish(self, audio_np: np.ndarray):
        """Transcribe a numpy float32 audio array and publish if valid.

        Args:
            audio_np: float32 numpy array, mono, at self.sample_rate
        """
        noise_phrases = {
            '', '.', '...', 'you', 'thank you', 'thanks for watching',
            'thanks for watching.', 'thank you for watching.',
            'thank you for watching', 'bye', 'bye.',
        }

        # Convert float32 [-1, 1] to int16 WAV in memory for faster-whisper
        audio_int16 = (audio_np * 32767).astype(np.int16)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())
        wav_buffer.seek(0)

        segments, _info = self._model.transcribe(
            wav_buffer,
            language=self.language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_speech_duration_ms=300),
        )

        text = ' '.join(seg.text.strip() for seg in segments).strip()

        if text and text.lower() not in noise_phrases and len(text) > 2:
            should_pub, pub_text = self._should_publish(text)
            if should_pub and pub_text:
                self._msg_queue.put(pub_text)

    # ══════════════════════════════════════════════════════════════════════════
    # MODE 1: Continuous ring buffer with VAD-triggered segmentation
    # ══════════════════════════════════════════════════════════════════════════

    def _continuous_listen_loop(self):
        """Capture audio continuously with sounddevice, detect speech boundaries
        using WebRTC VAD (preferred) or RMS energy (fallback), and send complete
        utterances to Faster-Whisper.

        This avoids the mid-sentence splitting problem of fixed-duration arecord
        chunks. WebRTC VAD detects actual human vocal frequencies (not just
        volume), so lockers slamming or bells ringing won't trigger Whisper.
        """
        import sounddevice as sd

        self.get_logger().info('STT: continuous microphone capture started')

        # Parse ALSA device to sounddevice device index
        device = self._resolve_audio_device()

        # WebRTC VAD requires 10/20/30ms frames at 8/16/32/48 kHz
        # We use 30ms blocks at 16kHz = 480 samples per block
        block_duration_ms = 30
        block_size = int(self.sample_rate * block_duration_ms / 1000)
        silence_blocks = int(self.speech_pad_ms / block_duration_ms)
        min_speech_blocks = int(self.min_speech_ms / block_duration_ms)
        max_blocks = int(self.max_speech_s * 1000 / block_duration_ms)

        # Try to use WebRTC VAD (speech-frequency detection, CPU-negligible)
        # Falls back to RMS energy detection if webrtcvad is not installed
        vad = None
        try:
            import webrtcvad
            vad = webrtcvad.Vad()
            vad.set_mode(2)  # 0=least aggressive, 3=most aggressive; 2 is good for noisy schools
            self.get_logger().info('VAD: using WebRTC VAD (speech-frequency detection)')
        except ImportError:
            self.get_logger().warning(
                'webrtcvad not installed — falling back to RMS energy detection. '
                'Install for better noise rejection: pip install webrtcvad'
            )

        # Audio block queue — sounddevice callback pushes here
        audio_q = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            if status:
                pass  # ignore overflow warnings in noisy environments
            audio_q.put(indata[:, 0].copy())  # mono channel

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=block_size,
                device=device,
                callback=audio_callback,
            ):
                self.get_logger().info(f'Sounddevice stream opened (device={device})')
                self._vad_process_loop(audio_q, block_size, silence_blocks,
                                       min_speech_blocks, max_blocks, vad)
        except Exception as e:
            self.get_logger().error(f'Sounddevice stream failed: {e}')
            self.get_logger().warning('Falling back to arecord chunked capture')
            self._use_continuous = False
            self._arecord_listen_loop()

    def _resolve_audio_device(self):
        """Try to resolve ALSA device string to sounddevice device index.
        Returns None (default device) if resolution fails."""
        try:
            import sounddevice as sd
            # If it's a plain integer, use directly
            if self.audio_device.isdigit():
                return int(self.audio_device)
            # Try to find matching device by name
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if (self.audio_device in str(dev.get('name', '')) and
                        dev.get('max_input_channels', 0) > 0):
                    self.get_logger().info(f'Matched audio device: {dev["name"]} (index {i})')
                    return i
            # Default device
            self.get_logger().info(f'Using default input device (could not match "{self.audio_device}")')
            return None
        except Exception:
            return None

    def _is_speech_block(self, block: np.ndarray, vad) -> bool:
        """Determine if an audio block contains speech.

        Uses WebRTC VAD if available (detects human vocal frequencies,
        ignores non-speech noise like lockers/bells). Falls back to RMS
        energy threshold if webrtcvad is not installed.

        Args:
            block: float32 numpy array, mono, 30ms at self.sample_rate
            vad: webrtcvad.Vad instance, or None for RMS fallback
        """
        if vad is not None:
            # WebRTC VAD expects 16-bit PCM bytes
            pcm_bytes = (block * 32767).astype(np.int16).tobytes()
            try:
                return vad.is_speech(pcm_bytes, self.sample_rate)
            except Exception:
                # If VAD fails (wrong frame size, etc.), fall through to RMS
                pass

        # RMS fallback
        rms = float(np.sqrt(np.mean(block ** 2)))
        return rms > self.silence_threshold

    def _vad_process_loop(self, audio_q, block_size, silence_blocks,
                          min_speech_blocks, max_blocks, vad):
        """Main VAD processing loop for continuous capture mode.

        Accumulates audio blocks, detects speech start/end via WebRTC VAD
        (or RMS fallback), and sends complete utterances for transcription.
        """
        speech_buffer = []     # blocks accumulated during speech
        silent_count = 0       # consecutive silent blocks
        is_speaking = False    # currently in a speech region

        while rclpy.ok():
            try:
                block = audio_q.get(timeout=1.0)
            except queue.Empty:
                continue

            # Skip while TTS is playing
            if self._is_speaker_playing:
                speech_buffer.clear()
                silent_count = 0
                is_speaking = False
                continue

            is_speech = self._is_speech_block(block, vad)

            if not is_speaking:
                if is_speech:
                    # Speech started
                    is_speaking = True
                    silent_count = 0
                    speech_buffer.clear()
                    speech_buffer.append(block)
                    self.get_logger().debug('VAD: speech started')
                # else: still silence, do nothing
            else:
                # Currently in speech region
                speech_buffer.append(block)

                if is_speech:
                    silent_count = 0
                else:
                    silent_count += 1

                # Check if speech ended (enough silence after speech)
                if silent_count >= silence_blocks:
                    # Speech ended — process the utterance
                    if len(speech_buffer) >= min_speech_blocks:
                        audio_np = np.concatenate(speech_buffer)
                        self.get_logger().debug(
                            f'VAD: speech ended, {len(audio_np)/self.sample_rate:.1f}s captured'
                        )
                        try:
                            self._transcribe_and_publish(audio_np)
                        except Exception as e:
                            self.get_logger().error(f'Transcription error: {e}')
                    else:
                        self.get_logger().debug('VAD: speech too short, discarding')

                    speech_buffer.clear()
                    silent_count = 0
                    is_speaking = False

                # Safety: cap maximum utterance length
                elif len(speech_buffer) >= max_blocks:
                    audio_np = np.concatenate(speech_buffer)
                    self.get_logger().debug(
                        f'VAD: max length reached, {len(audio_np)/self.sample_rate:.1f}s captured'
                    )
                    try:
                        self._transcribe_and_publish(audio_np)
                    except Exception as e:
                        self.get_logger().error(f'Transcription error: {e}')

                    speech_buffer.clear()
                    silent_count = 0
                    is_speaking = False

    # ══════════════════════════════════════════════════════════════════════════
    # MODE 2: arecord fallback (DEPRECATED — discrete chunks)
    # ══════════════════════════════════════════════════════════════════════════

    def _arecord_listen_loop(self):
        """Fallback: capture fixed-duration WAV chunks via arecord subprocess.

        DEPRECATED: Both Jetson Orin Nano and Raspberry Pi 5 support sounddevice
        with PortAudio natively. This fallback is retained only for edge cases.
        Install sounddevice to use the preferred continuous capture mode:
            pip install sounddevice
        """
        self.get_logger().warning(
            'DEPRECATED: arecord chunked capture is unresponsive during TTS playback '
            'and splits mid-sentence. Install sounddevice for continuous capture mode.'
        )

        tmp_fd, tmp_wav = tempfile.mkstemp(suffix='.wav', prefix='aisha_stt_')
        os.close(tmp_fd)  # arecord will write to the path directly

        while rclpy.ok():
            try:
                if self._is_speaker_playing:
                    time.sleep(0.1)
                    continue

                subprocess.run([
                    'arecord',
                    '-D', self.audio_device,
                    '-f', 'S16_LE',
                    '-r', str(self.sample_rate),
                    '-c', '1',
                    '-d', str(int(self.chunk_duration)),
                    tmp_wav,
                ], capture_output=True, timeout=self.chunk_duration + 5)

                if self._is_speaker_playing:
                    continue

                if not os.path.exists(tmp_wav):
                    continue

                # RMS pre-filter: skip obviously silent chunks
                rms = self._rms_from_file(tmp_wav)
                threshold = self.silence_threshold * 32768.0
                if rms < threshold:
                    continue

                # Read WAV into numpy for unified transcription path
                audio_np = self._wav_to_float32(tmp_wav)
                if audio_np is not None and len(audio_np) > 0:
                    self._transcribe_and_publish(audio_np)

            except subprocess.TimeoutExpired:
                self.get_logger().warning('STT: audio capture timed out, retrying')
            except Exception as e:
                self.get_logger().error(f'STT error: {e}')
                time.sleep(1.0)

    def _rms_from_file(self, wav_path: str) -> float:
        """Compute RMS energy of a WAV file (16-bit PCM, skip 44-byte header)."""
        try:
            with open(wav_path, 'rb') as f:
                f.read(44)
                raw = f.read()
            if not raw:
                return 0.0
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            return float(np.sqrt(np.mean(samples ** 2)))
        except Exception:
            return 0.0

    def _wav_to_float32(self, wav_path: str) -> np.ndarray:
        """Read a 16-bit PCM WAV file and return float32 numpy array in [-1, 1]."""
        try:
            with wave.open(wav_path, 'rb') as wf:
                raw = wf.readframes(wf.getnframes())
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            return samples
        except Exception:
            return np.array([], dtype=np.float32)


def main(args=None):
    rclpy.init(args=args)
    node = STTNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
