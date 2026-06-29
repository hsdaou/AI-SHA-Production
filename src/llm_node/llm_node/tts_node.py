import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import subprocess
import os

class AishaTTSNode(Node):
    def __init__(self):
        super().__init__('tts_node')
        
        self.subscription = self.create_subscription(String, '/tts_text', self.tts_callback, 10)
        self.playing_publisher = self.create_publisher(Bool, '/speaker/playing', 10)
        
        self.piper_dir = '/home/orin-robot/piper'
        self.piper_path = f'{self.piper_dir}/piper'
        self.model_path = f'{self.piper_dir}/en_US-lessac-medium.onnx'
        
        self.get_logger().info('AI-SHA Mouth (TTS) node initialized. Listening to /tts_text...')

    def tts_callback(self, msg):
        text = msg.data
        self.get_logger().info(f'Speaking: "{text}"')
        
        # Tell the microphone to mute
        mute_msg = Bool()
        mute_msg.data = True
        self.playing_publisher.publish(mute_msg)
        
        # Force Ubuntu to use Piper's local bundled libraries via Python OS module
        custom_env = os.environ.copy()
        custom_env['LD_LIBRARY_PATH'] = self.piper_dir
        custom_env['ESPEAK_DATA_PATH'] = f"{self.piper_dir}/espeak-ng-data"
        
        # Play the audio
        command = f'echo "{text}" | {self.piper_path} --model {self.model_path} --output_raw | aplay -r 22050 -f S16_LE -t raw -'
        
        try:
            subprocess.run(command, shell=True, check=True, env=custom_env)
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f'Failed to play audio: {e}')
            
        # Tell the microphone to unmute
        mute_msg.data = False
        self.playing_publisher.publish(mute_msg)

def main(args=None):
    rclpy.init(args=args)
    node = AishaTTSNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
