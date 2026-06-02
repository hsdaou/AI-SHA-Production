# Robot Brain - Project Cerebro

Multimodal ROS2 intelligence node for Jetson Orin Nano.

## Overview

Robot Brain is the central intelligence system that:
1. Receives speech commands from `/speech_rec`
2. Gathers visual context from YOLO detections
3. Routes queries intelligently:
   - **Simple queries** → Local Llama LLM (fast, offline)
   - **Complex queries** → Gemini 2.0 Flash Vision (accurate, requires camera + cloud)
4. Publishes responses to `/speech/text` for TTS

## Dependencies

### ROS2 Packages
- rclpy
- std_msgs
- sensor_msgs
- vision_msgs
- cv_bridge

### Python Packages
```bash
pip install llama-cpp-python google-generativeai pillow numpy
```

## Configuration

### Environment Variables

```bash
# Required for cloud vision
export GEMINI_API_KEY="your-gemini-api-key"

# ROS2 configuration (for cross-distro compatibility)
export RMW_IMPLEMENTATION="rmw_fastrtps_cpp"
export ROS_DOMAIN_ID="1"
```

### Launch Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `local_model_path` | `/home/orin-robot/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf` | Path to local LLM model |
| `gemini_api_key` | `$GEMINI_API_KEY` | Gemini API key |
| `gemini_model` | `gemini-2.0-flash-exp` | Gemini model name |
| `n_ctx` | 2048 | Context window size |
| `n_gpu_layers` | -1 | GPU layers (-1 = all) |
| `temperature` | 0.7 | LLM temperature |
| `max_tokens` | 200 | Max response tokens |
| `complexity_threshold` | 0.7 | Threshold for cloud routing |

## Building

```bash
cd ~/robot_ws
colcon build --packages-select robot_brain
source install/setup.bash
```

## Running

### Basic Usage

```bash
# Set API key
export GEMINI_API_KEY="your-key"

# Run node
ros2 run robot_brain robot_brain
```

### With Custom Parameters

```bash
ros2 run robot_brain robot_brain \
  --ros-args \
  -p n_gpu_layers:=-1 \
  -p temperature:=0.5 \
  -p complexity_threshold:=0.6
```

## Testing

### Test Local LLM

```bash
# Terminal 1: Run brain
ros2 run robot_brain robot_brain

# Terminal 2: Send simple query
ros2 topic pub --once /speech_rec std_msgs/msg/String "{data: 'What do you see?'}"

# Terminal 3: Monitor response
ros2 topic echo /speech/text
```

### Test Cloud Vision

```bash
# Ensure camera is running
ros2 launch realsense2_camera rs_launch.py

# Send complex query
ros2 topic pub --once /speech_rec std_msgs/msg/String "{data: 'What color is the object?'}"
```

## Query Routing Logic

### Simple Queries (Local LLM)
- Location: "where is", "find", "locate"
- Listing: "what do you see", "list objects"
- Counting: "how many", "count"
- Distance: "how far", "distance"

### Complex Queries (Cloud Vision)
- Color: "what color", "is it red"
- OCR: "read this", "what does it say"
- State: "is it on", "is it open"
- Quality: "is it healthy", "condition"
- Details: "describe", "appearance"

## Architecture

```
/speech_rec (input)
     ↓
┌────────────────────────┐
│   Complexity Analyzer   │
└────────┬───────────────┘
         │
    ┌────┴────┐
    │ Score?  │
    └─┬────┬──┘
      │    │
   <0.7  ≥0.7
      │    │
      ↓    ↓
  ┌─────┐ ┌────────┐
  │Local│ │ Gemini │
  │ LLM │ │ Vision │
  └──┬──┘ └───┬────┘
     │        │
     └────┬───┘
          ↓
    /speech/text (output)
```

## Topics

### Subscribed

| Topic | Type | Description |
|-------|------|-------------|
| `/speech_rec` | `std_msgs/String` | User speech input |
| `/detection/objects_simple` | `std_msgs/String` | YOLO detections (JSON) |
| `/camera/camera/color/image_raw` | `sensor_msgs/Image` | RGB camera feed |
| `/camera/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | Depth image |

### Published

| Topic | Type | Description |
|-------|------|-------------|
| `/speech/text` | `std_msgs/String` | Robot response |

## Troubleshooting

### "Model not yet loaded"
- Wait 10-30 seconds for Llama model to load into GPU
- Check logs for "Local LLM loaded and ready"
- Verify model file exists at `local_model_path`

### "Gemini API error"
- Check `GEMINI_API_KEY` is set correctly
- Verify internet connectivity
- Check API quota/billing

### No camera frames
- Verify RealSense is running: `ros2 topic hz /camera/camera/color/image_raw`
- Launch camera: `ros2 launch realsense2_camera rs_launch.py`

### Responses too slow
- Lower `n_ctx` to 1024 for faster inference
- Reduce `max_tokens` to 100-150
- Ensure GPU is used: `nvidia-smi` should show llama.cpp process

## Performance

- **Local LLM**: 5-10s response time
- **Cloud Vision**: 1-3s response time (internet dependent)
- **Memory**: ~3GB GPU VRAM for Llama + YOLO
- **CPU**: 4-6 cores utilized

## Full Documentation

See `/home/orin-robot/robot_ws/CEREBRO.md` for complete system architecture.

## License

MIT
