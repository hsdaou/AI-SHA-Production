# RPi5 <-> Jetson LLM Integration

## ✅ Setup Complete!

Your RPi5 is now configured to communicate with the Jetson Orin Nano LLM node.

## 📋 What's Installed

1. **Cyclone DDS** - ROS2 middleware for cross-distro communication (Jazzy ↔ Humble)
2. **DDS Configuration** - Copied from Jetson at `~/cyclonedds.xml`
3. **Environment Variables** - Configured in `~/.bashrc`
4. **STT Package** - Speech-to-Text node (publishes to `/speech_rec`)
5. **TTS Package** - Text-to-Speech node (subscribes to `/speech/text`)

## 🔧 Configuration

### Environment Variables (Auto-loaded)
```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
export CYCLONEDDS_URI=file:///home/pi5/cyclonedds.xml
```

These are already in your `~/.bashrc` and will load automatically on new terminals.

### Network
- **Jetson IP**: 172.41.40.47
- **Connection**: ✓ Verified

## 🚀 Quick Start

### Option 1: Using Launcher Scripts

**Terminal 1 - Start TTS:**
```bash
cd ~/ros2_ws
./launch_tts.sh
```

**Terminal 2 - Start STT:**
```bash
cd ~/ros2_ws
./launch_stt.sh
```

### Option 2: Manual Launch

**Terminal 1 - Start TTS:**
```bash
source ~/.bashrc
cd ~/ros2_ws
source install/setup.bash
ros2 run tts_speaker tts_speaker_node
```

**Terminal 2 - Start STT:**
```bash
source ~/.bashrc
cd ~/ros2_ws
source install/setup.bash
ros2 run speech_recognition stt_node
```

## 🧪 Testing

### Run Integration Test
```bash
cd ~/ros2_ws
./test_jetson_integration.sh
```

This will check:
- Environment variables
- Network connectivity to Jetson
- ROS2 node discovery
- Topic availability

### Manual Test (without microphone)

**Terminal 1 - Monitor LLM responses:**
```bash
source ~/.bashrc
source ~/ros2_ws/install/setup.bash
ros2 topic echo /speech/text
```

**Terminal 2 - Start TTS:**
```bash
cd ~/ros2_ws
./launch_tts.sh
```

**Terminal 3 - Simulate speech input:**
```bash
source ~/.bashrc
source ~/ros2_ws/install/setup.bash
ros2 topic pub --once /speech_rec std_msgs/msg/String "{data: 'Hello robot, what is your name?'}"
```

Wait 10-20 seconds. You should:
1. See the response in Terminal 1
2. Hear the TTS speak the response

## 📡 Communication Flow

```
User speaks → Microphone (RPi5)
    ↓
STT Node (RPi5) → /speech_rec topic
    ↓
Jetson LLM Node receives /speech_rec
    ↓
LLM processes (10-20 sec)
    ↓
Jetson publishes → /speech/text topic
    ↓
TTS Node (RPi5) receives /speech/text
    ↓
Speaker plays response
```

## 🔍 Troubleshooting

### Check if Jetson LLM is running
```bash
source ~/.bashrc
ros2 node list | grep llm_node
```

If you don't see `/llm_node`, the Jetson LLM service is not running.

### Check topics
```bash
source ~/.bashrc
ros2 topic list
```

You should see:
- `/speech_rec` (STT publishes here)
- `/speech/text` (TTS subscribes here)
- `/parameter_events`
- `/rosout`

### Check topic connections
```bash
source ~/.bashrc
ros2 topic info /speech_rec
ros2 topic info /speech/text
```

### Monitor all messages

**Monitor speech recognition:**
```bash
source ~/.bashrc
source ~/ros2_ws/install/setup.bash
ros2 topic echo /speech_rec
```

**Monitor LLM responses:**
```bash
source ~/.bashrc
source ~/ros2_ws/install/setup.bash
ros2 topic echo /speech/text
```

### Network issues

**Ping Jetson:**
```bash
ping 172.41.40.47
```

**Check environment:**
```bash
echo $RMW_IMPLEMENTATION  # Should be: rmw_cyclonedds_cpp
echo $ROS_DOMAIN_ID       # Should be: 42
echo $CYCLONEDDS_URI      # Should be: file:///home/pi5/cyclonedds.xml
```

## 📁 File Locations

- **DDS Config**: `~/cyclonedds.xml`
- **STT Node**: `~/ros2_ws/src/speech_recognition/speech_recognition/stt_node.py`
- **TTS Node**: `~/ros2_ws/src/tts_speaker/tts_speaker/tts_speaker_node.py`
- **Test Script**: `~/ros2_ws/test_jetson_integration.sh`
- **Launcher Scripts**: `~/ros2_ws/launch_stt.sh`, `~/ros2_ws/launch_tts.sh`

## 🎤 STT Node Details

- **Publishes to**: `/speech_rec`
- **Message type**: `std_msgs/msg/String`
- **Speech recognition**: Faster Whisper (base.en model)
- **Audio device**: plughw:1,0 (ReSpeaker)
- **Features**:
  - Voice Activity Detection (VAD)
  - Automatic muting during TTS playback
  - 3-second audio chunks

## 🔊 TTS Node Details

- **Subscribes to**: `/speech/text`
- **Message type**: `std_msgs/msg/String`
- **TTS engine**: Piper (en_US-lessac-medium)
- **Audio output**: plughw:0,0
- **Features**:
  - Publishes speaking state to `/robot/speaking`
  - Prevents mic feedback during speech

## ⚙️ Performance

- **LLM Response Time**: 10-20 seconds (Llama 3.2 8B on Jetson)
- **Network Latency**: <10ms (same network)
- **Total Pipeline**: ~10-25 seconds from speech to response

## 🔗 Jetson Information

- **IP Address**: 172.41.40.47
- **Username**: orin-robot
- **ROS2 Distro**: Humble
- **LLM Model**: Llama 3.2 8B Instruct (Q4_K_M)
- **LLM Node Name**: `/llm_node`

## 🆘 Getting Help

If you encounter issues:

1. Run the test script: `./test_jetson_integration.sh`
2. Check if Jetson LLM is running: `ros2 node list`
3. Verify network: `ping 172.41.40.47`
4. Check environment variables (see above)
5. Monitor topics: `ros2 topic echo /speech_rec` and `ros2 topic echo /speech/text`

---

**Setup Date**: 2026-01-24
**Integration**: RPi5 (ROS2 Jazzy) ↔ Jetson Orin Nano (ROS2 Humble)
**Communication**: Cyclone DDS with Domain ID 42
