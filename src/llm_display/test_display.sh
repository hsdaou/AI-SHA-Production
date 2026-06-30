#!/bin/bash
# Test script for LLM Display

echo "Testing LLM Display..."
echo ""
echo "Make sure the display is running in another terminal:"
echo "  ros2 run llm_display llm_display"
echo ""
echo "Press Enter to start sending test messages..."
read

# Source ROS2
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

echo ""
echo "Sending user message..."
ros2 topic pub --once /speech_rec std_msgs/msg/String "data: 'Hello robot, how are you today?'"

sleep 2

echo "Sending AI response..."
ros2 topic pub --once /speech/text std_msgs/msg/String "data: 'Hello! I am doing great, thank you for asking. How can I assist you?'"

sleep 2

echo ""
echo "Sending another user message..."
ros2 topic pub --once /speech_rec std_msgs/msg/String "data: 'What is the weather like?'"

sleep 2

echo "Sending AI response..."
ros2 topic pub --once /speech/text std_msgs/msg/String "data: 'I do not have access to real-time weather data, but I can help you find weather information if you provide a location.'"

sleep 2

echo ""
echo "Sending user message..."
ros2 topic pub --once /speech_rec std_msgs/msg/String "data: 'Tell me a joke'"

sleep 2

echo "Sending AI response..."
ros2 topic pub --once /speech/text std_msgs/msg/String "data: 'Why do programmers prefer dark mode? Because light attracts bugs!'"

echo ""
echo "Test complete! Check the display for the messages."
