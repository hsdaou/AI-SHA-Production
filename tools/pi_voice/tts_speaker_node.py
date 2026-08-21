#!/usr/bin/env python3
"""
ROS2 TTS Speaker Node
Subscribes to /speech/text and speaks the text using Piper TTS
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import subprocess
import os
import time
import tempfile


class TTSSpeakerNode(Node):
    def __init__(self):
        super().__init__('tts_speaker_node')

        # Piper configuration
        self.piper_bin = os.path.expanduser('~/.local/bin/piper')
        self.piper_model = os.path.expanduser('~/piper_models/en_US-lessac-medium.onnx')
        # 'default' (not plughw:0,0) so playback goes through the dmix + softvol
        # chain in ~/.asoundrc: gives a Master volume control and lets other
        # processes share the card. The hifiberry-dac driver has no hw mixer.
        self.audio_device = 'default'

        # Verify piper and model exist
        if not os.path.isfile(self.piper_bin):
            self.get_logger().error('AI-SHARJAH Voice: Piper not found')
            raise FileNotFoundError(f'Piper not found at {self.piper_bin}')

        if not os.path.isfile(self.piper_model):
            self.get_logger().error('AI-SHARJAH Voice: Model not found')
            raise FileNotFoundError(f'Piper model not found at {self.piper_model}')

        # Track speaking state to prevent premature unmuting
        self.is_currently_speaking = False
        self.pending_messages = 0

        # Create subscriber
        self.subscription = self.create_subscription(
            String,
            '/tts_text',
            self.speech_callback,
            10
        )

        # Create publisher for speaking state (to prevent mic feedback)
        self.speaking_publisher = self.create_publisher(Bool, '/robot/speaking', 10)

        # Also publish to /speaker/playing for mic array muting
        self.speaker_playing_publisher = self.create_publisher(Bool, '/speaker/playing', 10)

        self.get_logger().info('AI-SHARJAH Voice: Ready to speak ✓')

    def speech_callback(self, msg):
        """Callback function to handle incoming text messages"""
        text = msg.data.strip()

        if not text:
            return

        self.get_logger().info(f'AI-SHARJAH Voice: Speaking → "{text}"')

        # Track this message
        self.pending_messages += 1

        playback_successful = False

        try:
            # Mute FIRST, before synthesis. The old order was synthesise -> mute ->
            # sleep 1 s so stt_node could react. Synthesis alone takes over a second,
            # so muting first gives MORE settling time than that sleep ever did while
            # costing nothing. (The Jetson also asserts this mute locally before it
            # even POSTs the text, so by here the mic has been muted for seconds.)
            if not self.is_currently_speaking:
                speaking_msg = Bool()
                speaking_msg.data = True
                self.speaking_publisher.publish(speaking_msg)
                self.speaker_playing_publisher.publish(speaking_msg)
                self.is_currently_speaking = True
                self.get_logger().info('AI-SHARJAH Voice: Mic muted')

            # Stream piper STRAIGHT into aplay rather than rendering the whole
            # utterance to a temp file first. Piper runs at ~0.3x real time, so the
            # old approach delayed the first sound by ~30% of the ANSWER'S LENGTH --
            # 1.7 s for a short reply, ~6 s for a long one. Piping makes that latency
            # roughly constant: sound starts on the first chunk while the rest is
            # still being generated.
            piper_process = subprocess.Popen(
                [
                    self.piper_bin,
                    '--model', self.piper_model,
                    '--output-raw'
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            aplay_process = subprocess.Popen(
                [
                    'aplay',
                    '-D', self.audio_device,
                    '-r', '22050',
                    '-f', 'S16_LE',
                    '-t', 'raw',
                    '-q'
                ],
                stdin=piper_process.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            # aplay owns the read end now; closing ours is what lets it see EOF
            # when piper exits, otherwise playback would hang for ever.
            piper_process.stdout.close()

            self.get_logger().info('AI-SHARJAH Voice: Starting playback')
            piper_process.stdin.write(text.encode('utf-8'))
            piper_process.stdin.close()
            piper_process.wait()
            aplay_process.wait()

            if aplay_process.returncode == 0:
                playback_successful = True
                self.get_logger().info('AI-SHARJAH Voice: Playback completed')
            else:
                self.get_logger().warn(f'AI-SHARJAH Voice: Playback may have failed (code: {aplay_process.returncode})')

        except Exception as e:
            self.get_logger().error(f'AI-SHARJAH Voice: TTS Error - {str(e)}')
        finally:
            # Step 5: Mark this message as processed
            self.pending_messages -= 1

            # Only unmute if:
            # 1. No more pending messages
            # 2. We're currently in speaking state
            # 3. Playback completed successfully
            if self.pending_messages == 0 and self.is_currently_speaking:
                speaking_msg = Bool()
                speaking_msg.data = False
                self.speaking_publisher.publish(speaking_msg)
                self.speaker_playing_publisher.publish(speaking_msg)
                self.is_currently_speaking = False
                self.get_logger().info('AI-SHARJAH Voice: All playback finished, mic unmuted')
            elif self.pending_messages > 0:
                self.get_logger().info(f'AI-SHARJAH Voice: Playback finished, but {self.pending_messages} message(s) pending - keeping mic muted')



def main(args=None):
    rclpy.init(args=args)

    try:
        node = TTSSpeakerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
