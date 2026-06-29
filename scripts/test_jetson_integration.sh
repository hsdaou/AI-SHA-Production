#!/bin/bash
# Test script for RPi5 <-> Jetson Integration
# This script tests the ROS2 communication pipeline

echo "========================================="
echo "RPi5 <-> Jetson Integration Test"
echo "========================================="
echo ""

# Set environment variables
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
export CYCLONEDDS_URI=file:///home/pi5/cyclonedds.xml
source /opt/ros/jazzy/setup.bash
source /home/pi5/ros2_ws/install/setup.bash

echo "1. Checking environment variables..."
echo "   RMW_IMPLEMENTATION: $RMW_IMPLEMENTATION"
echo "   ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "   CYCLONEDDS_URI: $CYCLONEDDS_URI"
echo ""

echo "2. Testing network connectivity to Jetson..."
if ping -c 1 172.41.40.47 &> /dev/null; then
    echo "   ✓ Jetson is reachable at 172.41.40.47"
else
    echo "   ✗ Jetson is NOT reachable at 172.41.40.47"
    echo "   Please check network connection"
    exit 1
fi
echo ""

echo "3. Checking ROS2 nodes..."
echo "   Available nodes:"
ros2 node list
echo ""

echo "4. Checking for Jetson LLM node..."
if ros2 node list | grep -q llm_node; then
    echo "   ✓ Jetson LLM node is running"
else
    echo "   ✗ Jetson LLM node not found"
    echo "   Please start the LLM node on the Jetson"
fi
echo ""

echo "5. Checking ROS2 topics..."
echo "   Available topics:"
ros2 topic list
echo ""

echo "6. Testing /speech_rec topic..."
echo "   Publishing test message to /speech_rec..."
timeout 5 ros2 topic pub --once /speech_rec std_msgs/msg/String "{data: 'Hello robot'}" 2>&1 | head -n 3
echo ""

echo "7. Checking topic info..."
echo "   /speech_rec:"
ros2 topic info /speech_rec
echo ""
echo "   /speech/text:"
ros2 topic info /speech/text
echo ""

echo "========================================="
echo "Test Complete!"
echo ""
echo "To test the full pipeline:"
echo "  1. Start TTS node: ros2 run tts_speaker tts_speaker_node"
echo "  2. In another terminal, publish a test message:"
echo "     ros2 topic pub --once /speech_rec std_msgs/msg/String \"{data: 'test message'}\""
echo "  3. Wait 10-20 seconds for LLM response"
echo "  4. TTS should speak the response"
echo "========================================="
