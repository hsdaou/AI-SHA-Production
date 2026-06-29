import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
from gtts import gTTS # Google Text-to-Speech (Simple and high quality)
import os

class RobotSpeaker(Node):
    def __init__(self):
        super().__init__('robot_speaker')
        self.create_subscription(String, '/robot_speech_text', self.callback, 10)

    def callback(self, msg):
        text = msg.data
        self.get_logger().info(f'Speaking: {text}')
        
        # 1. Convert Text to an MP3 file
        tts = gTTS(text=text, lang='en')
        tts.save("response.mp3")
        
        # 2. Play via mpv to your MAX98357A
        # --ao=alsa ensures it uses the I2S hardware
        subprocess.run(['mpv', '--no-video', '--ao=alsa', 'response.mp3'])

def main():
    rclpy.init()
    rclpy.spin(RobotSpeaker())
    rclpy.shutdown()
