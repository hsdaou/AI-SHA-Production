# LLM Node - ROS 2 Humble

## Overview

The LLM node provides natural language processing capabilities for the robot system. It acts as the central reasoning node for speech-based interaction, running on the Jetson Orin Nano.

## Architecture

- **Platform**: Jetson Orin Nano
- **Model**: Llama-3.2-8B-Instruct-Q4_K_M (GGUF format, quantized)
- **Model Size**: 2.1 GB
- **Inference**: GPU-accelerated using llama.cpp

## Topics

### Subscribed Topics
- `/speech_rec` (std_msgs/String): Receives transcribed speech text from RPi5's STT node

### Published Topics
- `/speech/text` (std_msgs/String): Publishes LLM-generated responses for RPi5's TTS node

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | string | `/home/orin-robot/models/Llama-3.2-8B-Instruct-Q4_K_M.gguf` | Path to GGUF model file |
| `n_ctx` | int | 2048 | Context window size |
| `n_gpu_layers` | int | -1 | GPU layers (-1 = all layers on GPU) |
| `temperature` | float | 0.7 | Sampling temperature for generation |
| `max_tokens` | int | 150 | Maximum tokens per response |

## Usage

### Running the Node

```bash
source /opt/ros/humble/setup.bash
source ~/robot_ws/install/setup.bash
ros2 run llm_node llm_node
```

### Testing

```bash
# Publish a test message (simulating RPi5 STT output)
ros2 topic pub --once /speech_rec std_msgs/msg/String "{data: 'Hello robot'}"

# Monitor responses (that would go to RPi5 TTS)
ros2 topic echo /speech/text
```

Or use the provided test script:
```bash
~/test_llm_pipeline.sh
```

## Performance Characteristics

- **Non-blocking Design**: Model loading and response generation occur in separate daemon threads
- **Thread Safety**: Uses threading locks for state management
- **GPU Acceleration**: All model layers offloaded to GPU for optimal performance
- **Response Time**: ~5-10 seconds per inference (depending on input length)

## Integration

The LLM node is part of the speech interaction pipeline:

```
[RPi5] Microphone → STT → /speech_rec
                              ↓
[Jetson] LLM Node processes input (Llama 3.2 8B)
                              ↓
[Jetson] LLM Node → /speech/text
                              ↓
[RPi5] TTS → Speaker
```

## Verification Status

✅ **Updated Configuration** (2026-01-24)
- Subscribes to `/speech_rec` from RPi5's STT node
- Publishes to `/speech/text` for RPi5's TTS node
- Model: Llama 3.2 8B (2.1GB)
- GPU-accelerated inference
- Non-blocking executor verified
