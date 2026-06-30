#!/usr/bin/env python3
"""
Audio Monitor Node - Monitors actual speaker output and publishes state
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import subprocess
import re
import time


class AudioMonitorNode(Node):
    def __init__(self):
        super().__init__('audio_monitor_node')

        # Publisher for audio state
        self.audio_state_publisher = self.create_publisher(Bool, '/audio/playing', 10)

        # Track last published state to avoid duplicates
        self.last_state = None

        # Audio device to monitor
        self.audio_card = 'MAX98357A'

        # Timer to check audio state
        self.create_timer(0.1, self.check_audio_state)  # Check every 100ms

        self.get_logger().info(f'Audio Monitor started - monitoring {self.audio_card}')
        self.get_logger().info('Publishing to /audio/playing')

    def check_audio_state(self):
        """Check if audio is currently playing on the speaker"""
        is_playing = self.is_audio_active()

        # Only publish if state changed
        if is_playing != self.last_state:
            msg = Bool()
            msg.data = is_playing
            self.audio_state_publisher.publish(msg)
            self.last_state = is_playing

            status = "PLAYING 🔊" if is_playing else "SILENT 🔇"
            self.get_logger().info(f'Audio state: {status}')

    def is_audio_active(self):
        """Check if audio processes are active"""
        try:
            # Check for active audio playback processes
            result = subprocess.run(
                ['ps', 'aux'],
                capture_output=True,
                text=True,
                timeout=0.5
            )

            audio_processes = ['mpg123', 'aplay', 'ffplay', 'mplayer', 'vlc']
            process_found = any(p in result.stdout for p in audio_processes)

            # Check ALSA state
            try:
                alsa_result = subprocess.run(
                    ['cat', f'/proc/asound/{self.audio_card}/pcm0p/sub0/status'],
                    capture_output=True,
                    text=True,
                    timeout=0.5
                )
                alsa_status = alsa_result.stdout

                # DRAINING means the buffer is emptying - audio content has ended
                if 'DRAINING' in alsa_status:
                    return False

                # Only trust RUNNING if an audio process is actually alive.
                # dmix keeps the device RUNNING even when idle, so RUNNING alone
                # is not sufficient evidence of active playback.
                if 'RUNNING' in alsa_status and process_found:
                    return True

            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

            return process_found

        except Exception as e:
            self.get_logger().debug(f'Error checking audio state: {e}')
            return False


def main(args=None):
    rclpy.init(args=args)
    node = AudioMonitorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
