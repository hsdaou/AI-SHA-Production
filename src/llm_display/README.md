# LLM Display Node

Beautiful animated display application for Raspberry Pi 5 connected to Elecrow display.

## Features

- **Real-time Chat Display**: Shows conversations between user and AI with animated message bubbles
- **System Dashboard**: Displays statistics including message counts and uptime
- **Smooth Animations**: Fade-in effects and smooth transitions
- **Touch-Friendly UI**: Large buttons and clear interface for touch interaction
- **Kiosk Mode**: Runs fullscreen automatically on startup
- **Modern Design**: Gradient backgrounds, rounded corners, and professional styling

## Installation

### 1. Build the package

```bash
cd ~/ros2_ws
colcon build --packages-select llm_display
source install/setup.bash
```

### 2. Test the display

```bash
ros2 run llm_display llm_display
```

### 3. Setup auto-start (optional)

To make the display start automatically on boot:

```bash
# Install unclutter for hiding cursor
sudo apt-get install unclutter -y

# Copy service file
sudo cp ~/llm-display.service /etc/systemd/system/

# Enable the service
sudo systemctl daemon-reload
sudo systemctl enable llm-display.service
sudo systemctl start llm-display.service

# Check status
sudo systemctl status llm-display.service
```

### 4. Alternative: Auto-start with .desktop file

If you prefer using the desktop autostart:

```bash
# Create autostart directory
mkdir -p ~/.config/autostart

# Create autostart entry
cat > ~/.config/autostart/llm-display.desktop << 'EOF'
[Desktop Entry]
Type=Application
Name=LLM Display
Exec=/home/pi5/start_llm_display.sh
Terminal=false
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
EOF
```

## Usage

### Navigation

- **Chat Button**: View the conversation interface
- **Dashboard Button**: View system statistics
- **Clear Chat Button**: Clear all messages from the display

### Topics

The display subscribes to:
- `/speech_rec`: User input from speech recognition
- `/speech/text`: AI responses from the LLM node

### Testing

You can test the display by publishing messages manually:

```bash
# In terminal 1 - Start the display
ros2 run llm_display llm_display

# In terminal 2 - Send test user message
ros2 topic pub --once /speech_rec std_msgs/msg/String "data: 'Hello robot!'"

# In terminal 3 - Send test AI response
ros2 topic pub --once /speech/text std_msgs/msg/String "data: 'Hello! How can I help you today?'"
```

## Customization

### Changing Colors

Edit `/home/pi5/ros2_ws/src/llm_display/llm_display/display_node.py`:

- Background gradient: Search for `qlineargradient` in `setup_ui()`
- Button colors: Look for the button StyleSheet sections
- Message bubble colors: Check `MessageBubble.setup_ui()`

### Adjusting Font Sizes

Search for `QFont("Ubuntu", SIZE)` in the code and adjust the SIZE parameter.

### Screen Resolution

The app automatically adapts to your screen resolution. If you need to force a specific resolution:

```python
# In display_node.py, in setup_ui():
self.resize(1920, 1080)  # Set your resolution
self.showFullScreen()
```

## Troubleshooting

### Display not showing

1. Check if the node is running:
   ```bash
   ros2 node list
   ```

2. Check if topics are publishing:
   ```bash
   ros2 topic list
   ros2 topic echo /speech/text
   ```

### Black screen on startup

1. Check X server is running:
   ```bash
   echo $DISPLAY
   ```

2. Make sure script has correct permissions:
   ```bash
   chmod +x ~/start_llm_display.sh
   ```

### Service won't start

Check the logs:
```bash
sudo journalctl -u llm-display.service -f
```

## Architecture

```
┌─────────────────┐
│  Jetson Orin    │
│  (LLM Node)     │
│                 │
│  Publishes to:  │
│  /speech/text   │
└────────┬────────┘
         │
         │ ROS2 Network
         │
┌────────▼────────┐
│ Raspberry Pi 5  │
│ (Display Node)  │
│                 │
│ Subscribes:     │
│ /speech_rec     │
│ /speech/text    │
│                 │
│ PyQt5 GUI ──────┼───> Elecrow Display
└─────────────────┘
```

## Dependencies

- ROS2 Jazzy
- Python 3.12+
- PyQt5 (already installed)
- rclpy
- std_msgs

## License

MIT
