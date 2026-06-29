# STT Node - Faster-Whisper (GPU Accelerated)

Real-time speech-to-text using CTranslate2-optimized Whisper with ReSpeaker mic array support.

## Features

- **Model**: Faster-Whisper (base model, ~150MB)
- **Performance**: 4-5x faster than standard Whisper with GPU acceleration
- **Hardware**: Optimized for Jetson Orin Nano + ReSpeaker 4-mic array
- **Auto-detection**: Automatically finds ReSpeaker device
- **Voice Activity Detection**: Built-in VAD filter for better accuracy
- **Real-time**: Publishes transcriptions to `/speech/text` topic for LLM

## Installation

### 1. Install system dependencies (requires sudo)

```bash
sudo apt-get update
sudo apt-get install -y libsndfile1 ffmpeg portaudio19-dev
```

### 2. Install Python dependencies

```bash
pip3 install faster-whisper sounddevice soundfile
```

### 3. Build the package

```bash
cd ~/robot_ws
colcon build --packages-select stt_node llm_node
source install/setup.bash
```

## Usage

### Run the node

```bash
ros2 run stt_node stt_node
```

### With custom parameters

```bash
ros2 run stt_node stt_node --ros-args \
  -p sample_rate:=16000 \
  -p chunk_duration:=1.0 \
  -p silence_threshold:=0.015 \
  -p model_size:=base \
  -p compute_type:=float16 \
  -p device_index:=2
```

### Listen to transcriptions

```bash
ros2 topic echo /speech/text
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sample_rate` | int | 16000 | Audio sample rate (Hz) |
| `channels` | int | 1 | Number of audio channels |
| `chunk_duration` | float | 1.0 | Audio chunk size (seconds) |
| `silence_threshold` | float | 0.015 | Voice activity threshold |
| `model_size` | str | base | Whisper model size (tiny, base, small, medium) |
| `compute_type` | str | float16 | Compute type (float16 for GPU, int8 for CPU) |
| `language` | str | en | Language code |
| `device_index` | int | -1 | Audio device index (-1 = auto-detect) |

## Topics

### Published

- `/speech/text` (`std_msgs/String`) - Transcribed text for LLM processing

## Log Messages

| Symbol | Meaning |
|--------|---------|
| ✓ | Success |
| ✗ | Error |
| ⚠ | Warning |
| ⏳ | Loading |
| 🎤 | Speech detected and transcribed |
| ⊘ | No speech detected |

## Example Output

```
[INFO] [stt_node]: Loading Faster-Whisper base on GPU...
[INFO] [stt_node]: Found ReSpeaker: seeed-4mic-voicecard (index 2, 4 channels)
[INFO] [stt_node]: Faster-Whisper base ready on GPU (compute_type=float16)
[INFO] [stt_node]: Expected latency: 0.5-1.5s per utterance (4-5x faster than standard Whisper)
[INFO] [stt_node]: Recording: 16000Hz, 1ch
[INFO] [stt_node]: "Hello, how are you today?" (0.73s)
[INFO] [stt_node]: "What is the weather like?" (0.58s)
```

## Troubleshooting

### ReSpeaker not detected

```bash
# List audio devices
python3 -c "import sounddevice as sd; print(sd.query_devices())"

# Manually set device index
ros2 run stt_node stt_node --ros-args -p device_index:=<INDEX>
```

### Model download fails

Models are downloaded automatically from HuggingFace. Ensure internet connection:
```bash
# Test model loading
python3 -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cuda')"
```

### CUDA/GPU errors

Ensure CUDA is available:
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
```

If GPU fails, it will automatically fallback to CPU with int8 quantization.

### Memory issues

The base model requires ~150MB VRAM. Check GPU memory:
```bash
nvidia-smi
```

## Performance

- **Latency**: 0.5-1.5 seconds per utterance (4-5x faster than standard Whisper)
- **Accuracy**: Same as OpenAI Whisper (state-of-the-art ASR)
- **GPU**: Automatically uses CUDA with CTranslate2 optimization
- **Memory**: ~150MB VRAM (base model), ~1GB for small model

## Integration Example

The STT node is part of the speech pipeline: **STT → LLM → TTS**

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SpeechListenerNode(Node):
    def __init__(self):
        super().__init__('speech_listener')
        # Listen to STT output
        self.sub = self.create_subscription(
            String,
            '/speech/text',
            self.speech_callback,
            10
        )

    def speech_callback(self, msg):
        self.get_logger().info(f'Heard: {msg.data}')
        # This will be picked up by the LLM node automatically

def main():
    rclpy.init()
    node = SpeechListenerNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
```

## Topic Flow

```
STT Node (Jetson) → /speech/text → LLM Node (Jetson) → /tts_text → TTS Node (RPi5)
```

## References

- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
- [CTranslate2](https://github.com/OpenNMT/CTranslate2)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [ReSpeaker 4-Mic Array](https://wiki.seeedstudio.com/ReSpeaker_4_Mic_Array_for_Raspberry_Pi/)

## License

MIT
