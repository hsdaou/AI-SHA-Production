# Jetson ↔ Raspberry Pi 5 Integration (two-tier)

How the two SBCs talk to each other in the current architecture. This supersedes
the old single-machine / CycloneDDS notes.

> Middleware is **FastDDS** (`rmw_fastrtps_cpp`) with **static unicast** discovery,
> **not** CycloneDDS. Both boards run **ROS 2 Humble** — a Humble↔Jazzy mesh
> silently drops Nav2 action goals, so do not mix distros.

## Roles

| Board | IP (default, see `.env`) | Runs |
|---|---|---|
| Jetson Orin Nano | `172.23.23.20` | RealSense + YOLOv8 (TensorRT), STT (Whisper/CUDA), brain_node (router), admin_node (RAG: Ollama + ChromaDB), action_node |
| Raspberry Pi 5   | `172.23.23.51` | LiDAR (LD19), mecanum serial bridge → `/odom` + `/imu/data`, SLAM Toolbox + odom source, Piper TTS, face display |
| Arduino Mega     | — (USB serial) | encoders + BNO055 (I2C) → unified `ODOM` packet; motor PWM |

The mic is on the **Jetson** (Whisper runs on CUDA); the speaker is on the **Pi 5**
(Piper). Echo is handled by a software mute bridge: `tts_node` publishes
`/speaker/playing` and `stt_node` drops audio while it is true.

## Network / DDS setup (run on BOTH boards)

1. Set static IPs (DHCP reservations or static config) and put them in `.env`
   (`JETSON_IP`, `RPI_IP`, `WS_IP`) with `ROS_DOMAIN_ID=42`.
2. Generate the FastDDS profiles and enlarge UDP buffers (else LiDAR packets drop):
   ```bash
   ROS_DOMAIN_ID=42 bash scripts/generate_fastdds_configs.sh
   sudo sysctl -w net.core.rmem_max=4194304 net.core.rmem_default=4194304 \
                   net.core.wmem_max=1048576 net.core.wmem_default=1048576
   ```
3. Load the DDS env (add to `~/.bashrc`) — pass the device tag:
   ```bash
   source config/fastdds_env.sh jetson     # on the Jetson
   source config/fastdds_env.sh rpi        # on the Pi 5
   # exports RMW_IMPLEMENTATION=rmw_fastrtps_cpp, FASTRTPS_DEFAULT_PROFILES_FILE, ROS_DOMAIN_ID=42
   ```

## Launch

```bash
# Jetson:
ros2 launch aisha_integration jetson_launch.py
# Pi 5:
ros2 launch aisha_integration rpi_launch.py
```

## Verify cross-host discovery

```bash
# Jetson:
ros2 run demo_nodes_cpp talker
# Pi 5 (should print "I heard: Hello World"):
ros2 run demo_nodes_cpp listener

ros2 node list        # should list nodes from BOTH boards
```

## Key topics

| Topic | Type | Direction |
|---|---|---|
| `/speech/text` | `std_msgs/String` | STT out (Jetson) + wake-word in |
| `/robot_speech` | `std_msgs/String` | brain → TTS (Pi 5 speaks) |
| `/aisha/mode` | `std_msgs/String` | `gpu_arbiter` broadcasts NAVIGATING / CONVERSING |
| `/aisha/set_conversing` | `std_srvs/SetBool` | manual mode trigger |
| `/scan` | `sensor_msgs/LaserScan` | LD19 LiDAR (Pi 5) |
| `/odom`, `/imu/data` | `nav_msgs/Odometry`, `sensor_msgs/Imu` | from the Arduino `ODOM` packet via the serial bridge |
| `/cmd_vel` | `geometry_msgs/Twist` | mecanum velocity command |

## Troubleshooting

- **Nodes invisible across boards:** confirm identical `ROS_DOMAIN_ID=42`, that the
  UDP sysctl was applied, that `FASTRTPS_DEFAULT_PROFILES_FILE` points at the
  generated `config/fastdds_<device>.xml`, and that no host firewall blocks ports
  ≥ 17910. Re-run `generate_fastdds_configs.sh` after changing any IP.
- **Speaker self-transcribes:** verify the `/speaker/playing` mute bridge (TTS on
  Pi 5, STT on Jetson) — see `docs/TTS_TROUBLESHOOTING.md`.
- **OOM on the 8 GB Jetson:** keep the RAG LLM on CPU, bound `OLLAMA_NUM_CTX`, enable
  swap. See the GPU-multiplexing rationale in
  `src/aisha_brain/docs/adr/0001-gpu-multiplexing-navigating-conversing.md`.

See `MERGE_NOTES.md` for provenance and the bring-up checklist for step-by-step
first-boot instructions.
