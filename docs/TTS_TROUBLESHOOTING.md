# RPi5 TTS Node Troubleshooting Guide

**Date**: 2026-01-26
**Issue**: STT and LLM working, but no audio output from speakers
**System**: Cross-machine ROS 2 setup (RPi5 + Jetson Orin Nano)

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         SPEECH PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [RPi5]                    [Jetson]                   [RPi5]     │
│                                                                   │
│  Microphone                                           Speakers   │
│      │                                                    ▲       │
│      ▼                                                    │       │
│  STT Node ──────────────► LLM Node ───────────────► TTS Node    │
│             /speech/text            /tts_text                    │
│  (std_msgs/String)                  (std_msgs/String)            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Topic Flow

1. **STT Node (RPi5)** → Publishes to `/speech/text` (std_msgs/String)
2. **LLM Node (Jetson)** → Subscribes to `/speech/text`, Publishes to `/tts_text`
3. **TTS Node (RPi5)** → Subscribes to `/tts_text`, Plays audio through speakers

---

## Critical Information for TTS Node

### Topic Details

**Topic Name**: `/tts_text`
**Message Type**: `std_msgs/msg/String`
**Publisher**: Jetson Orin Nano LLM node
**Expected Subscriber**: RPi5 TTS node

### Expected TTS Node Implementation

The TTS node MUST:
1. Subscribe to `/tts_text` topic
2. Receive messages of type `std_msgs/msg/String`
3. Extract the `data` field from the message
4. Convert text to speech (using pyttsx3, espeak, or similar)
5. Play audio through the default audio output device

**Minimal Python Example**:
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import pyttsx3  # or subprocess for espeak

class TTSNode(Node):
    def __init__(self):
        super().__init__('tts_node')
        self.subscription = self.create_subscription(
            String,
            '/tts_text',      # CRITICAL: Must match this topic name
            self.tts_callback,
            10
        )
        self.engine = pyttsx3.init()
        self.get_logger().info('TTS Node initialized, subscribed to /tts_text')

    def tts_callback(self, msg):
        text = msg.data  # CRITICAL: Extract .data field
        self.get_logger().info(f'Received text: {text}')
        self.engine.say(text)
        self.engine.runAndWait()

def main(args=None):
    rclpy.init(args=args)
    node = TTSNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

---

## Step-by-Step Verification

### Step 1: Verify Network Connectivity

```bash
# On RPi5, ping the Jetson
ping 172.41.40.47

# Expected: Successful pings with <1ms latency
```

**Expected Output**:
```
64 bytes from 172.41.40.47: icmp_seq=1 ttl=64 time=0.5 ms
```

---

### Step 2: Verify ROS 2 Node Discovery

```bash
# List all nodes (should see nodes from both machines)
ros2 node list
```

**Expected Output**:
```
/llm_node        # From Jetson
/stt_node        # From RPi5
/tts_node        # From RPi5 (YOUR NODE)
```

**If you DON'T see /llm_node**:
- Check ROS_DOMAIN_ID matches: `echo $ROS_DOMAIN_ID`
- Check Cyclone DDS is configured
- Check firewall isn't blocking multicast

---

### Step 3: Verify Topic Publications

```bash
# Check if /tts_text exists
ros2 topic list | grep tts_text

# Check topic info (publishers and subscribers)
ros2 topic info /tts_text
```

**Expected Output**:
```
Type: std_msgs/msg/String
Publisher count: 1    # Jetson LLM node
Subscription count: 1 # RPi5 TTS node (YOURS)
```

**If Subscription count is 0**:
- Your TTS node is NOT running or NOT subscribed correctly
- Check the topic name in your code (must be exactly `/tts_text`)

---

### Step 4: Monitor Topic Messages

```bash
# Listen to what the LLM is sending
ros2 topic echo /tts_text
```

**Expected Output** (when you speak to the mic):
```
data: 'I am a helpful robot assistant...'
---
data: 'The weather is sunny today'
---
```

**If you see messages**:
- ✅ LLM is working and publishing correctly
- ❌ Your TTS node is not processing them

**If you DON'T see messages**:
- The LLM might not be responding
- Check `/speech/text` is being published by STT

---

### Step 5: Verify Audio Hardware

```bash
# List audio playback devices
aplay -l

# Test speaker with a beep
speaker-test -t wav -c 2 -l 1

# Check ALSA mixer levels
alsamixer
```

**Expected**:
- You should hear a test sound from `speaker-test`
- Volume in `alsamixer` should be >50% and not muted

**If no sound**:
```bash
# Check default audio device
pactl info | grep "Default Sink"

# List all sinks
pactl list sinks short

# Set default (replace X with sink number)
pactl set-default-sink X
```

---

### Step 6: Test TTS Engine Manually

```bash
# Test pyttsx3
python3 << EOF
import pyttsx3
engine = pyttsx3.init()
engine.say("Testing TTS engine")
engine.runAndWait()
EOF

# OR test espeak
espeak "Testing espeak TTS"
```

**Expected**: You should hear audio

**If no sound**:
```bash
# Install/reinstall TTS engines
sudo apt update
sudo apt install espeak espeak-ng festival
pip3 install pyttsx3 --upgrade
```

---

### Step 7: Check TTS Node is Running

```bash
# Check if TTS node is active
ros2 node list | grep tts_node

# Check node info
ros2 node info /tts_node
```

**Expected Output**:
```
/tts_node
  Subscribers:
    /tts_text: std_msgs/msg/String    # CRITICAL: Must be here
```

**If /tts_node doesn't exist**:
- The node is not running
- Start it manually or check your launch files

---

### Step 8: Check TTS Node Logs

```bash
# If running natively
ros2 run <package_name> tts_node --ros-args --log-level debug

# If running in Docker
docker logs tts_node

# Or check systemd logs if it's a service
journalctl -u tts_node -f
```

**Look for**:
- "Subscribed to /tts_text" message
- "Received text: ..." when messages arrive
- Any error messages about audio devices

---

### Step 9: Manual End-to-End Test

```bash
# Terminal 1: Monitor /tts_text
ros2 topic echo /tts_text

# Terminal 2: Publish test message
ros2 topic pub --once /tts_text std_msgs/msg/String "{data: 'Testing one two three'}"
```

**Expected**:
- Terminal 1 shows the message
- You hear "Testing one two three" from speakers

**If you see message but NO audio**:
- TTS node is not subscribed OR
- TTS node callback is not working OR
- Audio device is misconfigured

---

## Common Issues and Fixes

### Issue 1: TTS Node Not Subscribed to /tts_text

**Symptoms**: `ros2 topic info /tts_text` shows 0 subscribers

**Fix**:
```python
# Check your TTS node code has:
self.subscription = self.create_subscription(
    String,
    '/tts_text',  # NOT 'tts_text' or '/tts' - must be exact
    self.callback,
    10
)
```

### Issue 2: Wrong Message Type

**Symptoms**: No errors but no audio

**Fix**:
```python
# In your callback:
def callback(self, msg):
    text = msg.data  # NOT msg.text or msg itself
    print(f"Received: {text}")  # Debug print
    self.engine.say(text)
    self.engine.runAndWait()
```

### Issue 3: Audio Device Not Set

**Symptoms**: TTS engine works in terminal but not in node

**Fix**:
```bash
# Set pulse audio environment for ROS node
export PULSE_SERVER=unix:/run/user/1000/pulse/native

# OR in your launch file:
<env name="PULSE_SERVER" value="unix:/run/user/1000/pulse/native"/>
```

### Issue 4: TTS Engine Blocking

**Symptoms**: First message works, then hangs

**Fix**:
```python
# Use threading for TTS
import threading

def callback(self, msg):
    text = msg.data
    # Non-blocking TTS
    thread = threading.Thread(target=self.speak, args=(text,))
    thread.start()

def speak(self, text):
    self.engine.say(text)
    self.engine.runAndWait()
```

### Issue 5: Permission Issues

**Symptoms**: "Cannot access audio device" errors

**Fix**:
```bash
# Add user to audio group
sudo usermod -a -G audio $USER

# Reboot required
sudo reboot
```

---

## Complete Diagnostic Script

Save this as `test_tts_pipeline.sh` on RPi5:

```bash
#!/bin/bash

echo "=== RPi5 TTS Pipeline Diagnostics ==="
echo ""

echo "1. Network Check:"
ping -c 2 172.41.40.47 && echo "✓ Jetson reachable" || echo "✗ Cannot reach Jetson"
echo ""

echo "2. ROS Nodes:"
ros2 node list
echo ""

echo "3. TTS Topic Info:"
ros2 topic info /tts_text
echo ""

echo "4. Audio Hardware:"
aplay -l
echo ""

echo "5. Testing speaker:"
speaker-test -t wav -c 2 -l 1
echo ""

echo "6. Publishing test message to /tts_text:"
ros2 topic pub --once /tts_text std_msgs/msg/String "{data: 'TTS test message'}"
echo ""

echo "7. Monitoring /tts_text for 5 seconds:"
timeout 5 ros2 topic echo /tts_text
echo ""

echo "8. TTS Node Subscribers:"
ros2 topic info /tts_text | grep -A 5 "Subscription count"
echo ""

echo "=== Diagnostics Complete ==="
echo ""
echo "Expected Results:"
echo "  - Ping successful (✓)"
echo "  - /tts_node in node list"
echo "  - 1 publisher, 1 subscriber on /tts_text"
echo "  - Audio from speaker-test"
echo "  - Audio from test message"
```

Run it:
```bash
chmod +x test_tts_pipeline.sh
./test_tts_pipeline.sh
```

---

## Quick Reference Commands

```bash
# Check everything is connected
ros2 node list

# Check topic has subscriber
ros2 topic info /tts_text

# Listen to LLM responses
ros2 topic echo /tts_text

# Test TTS manually
ros2 topic pub --once /tts_text std_msgs/msg/String "{data: 'Test'}"

# Test speaker hardware
speaker-test -t wav -c 2 -l 1

# Check TTS node logs
ros2 node info /tts_node
```

---

## TTS Node Checklist

- [ ] Node is running: `ros2 node list | grep tts_node`
- [ ] Subscribed to `/tts_text`: `ros2 topic info /tts_text`
- [ ] Message type is `std_msgs/msg/String`
- [ ] Callback extracts `msg.data` (not `msg.text`)
- [ ] TTS engine installed: `espeak` or `pyttsx3`
- [ ] Audio device working: `speaker-test` produces sound
- [ ] Volume not muted: `alsamixer` shows >50%
- [ ] User in audio group: `groups | grep audio`
- [ ] Can see /llm_node: `ros2 node list | grep llm_node`
- [ ] Messages appear on /tts_text: `ros2 topic echo /tts_text`

---

## Contact Information

- **Jetson IP**: 172.41.40.47 (running LLM node)
- **RPi5 IP**: Check with `hostname -I`
- **ROS Domain**: Check with `echo $ROS_DOMAIN_ID` (should match Jetson)
- **DDS**: Cyclone DDS (auto-discovery enabled)

---

## Need More Help?

1. Share the output of `test_tts_pipeline.sh`
2. Share TTS node logs: `ros2 run <pkg> tts_node --ros-args --log-level debug`
3. Share TTS node source code
4. Confirm TTS engine works: `espeak "test"`

**Most likely issue**: TTS node is not subscribed to `/tts_text` or callback is not extracting `msg.data` correctly.
