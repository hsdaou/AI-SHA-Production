# Pi 5 voice, sensors and the two-board link

Everything in this directory runs on the **Raspberry Pi 5**, except
`deploy/10-pi-direct.conf`, which is a systemd drop-in for the **Jetson** console.

## Why the boards talk over HTTP and not ROS

The Jetson runs ROS 2 **Humble**, the Pi runs **Jazzy**. Cross-distro DDS *discovers*
endpoints but never delivers to a real subscription — Iron+ added type hashes that Humble
does not supply — and parking Jazzy nodes on the Jetson's domain floods its logs with
`sequence size exceeds remaining buffer`.

So nothing crosses over DDS. The console POSTs answer text to `speech_bridge.py`, which
republishes it on `/tts_text` **locally on the Pi**. LiDAR, map and volume are read the same
way. Only HTTP crosses the network.

```
console _push("AI-SHA", …)  ──HTTP POST──▶  speech_bridge.py :8093
                                                    │ publishes /tts_text (local)
                                                    ▼
                                            tts_speaker_node ──▶ piper ──▶ speaker
```

Pi-side ROS nodes deliberately sit on **`ROS_DOMAIN_ID=44`**, never the mesh's 42.

## Services

| Unit | Port | Purpose |
|---|---|---|
| `aisha-imu` | – | BNO055 → `/imu/data` + `sensor_msgs/Imu` on `/imu/data_raw` |
| `aisha-lidar` | 8090 | LD06 → `/scan` + `scan.json` |
| `aisha-slam` | 8091 | slam_toolbox + map bridge |
| `aisha-volume` | 8092 | speaker volume, read/set |
| `aisha-tts` | – | `/tts_text` → piper → speaker |
| `aisha-speech-bridge` | 8093 | HTTP from the Jetson → `/tts_text`; mirrors `/robot/speaking` |

## Install

```bash
sudo cp deploy/aisha-*.service /etc/systemd/system/
cp deploy/asoundrc ~/.asoundrc
sudo cp deploy/99-aisha-eth0-static.yaml /etc/netplan/   # chmod 600, root:root
sudo systemctl daemon-reload
sudo systemctl enable --now aisha-imu aisha-volume aisha-tts aisha-speech-bridge
```

On the Jetson: `sudo cp deploy/10-pi-direct.conf /etc/systemd/system/aisha-console.service.d/`

## Gotchas that cost real time

- **Speaker: use `dtoverlay=hifiberry-dac`, NOT `dtoverlay=max98357a`.** With the stock overlay
  on Ubuntu's raspi kernel the card binds but **every** PCM open fails with `Invalid argument`
  (`snd_pcm_hw_constraints_complete failed`). The tell is `/proc/asound/card0/pcm0p/sub0/`
  missing `prealloc`/`prealloc_max` — no DMA buffer was ever allocated. `hifiberry-dac` drives
  the same pins via `snd-soc-rpi-simple-soundcard`, which does allocate.
- **`hifiberry-dac` exposes no hardware mixer**, hence the softvol plugin in `asoundrc`. A softvol
  control does not exist until something opens the plugin once, so `volume_service.py` primes it
  with 0.1 s of silence at startup.
- **`WorkingDirectory=/home/pi5` is load-bearing for `aisha-imu`.** `adafruit_blinka`'s
  `import board` pulls in `lgpio`, which opens its `.lgd-nfy` pipe in the CWD; systemd's default
  `/` is not writable by `pi5`.
- **`tts_speaker_node.py` streams piper straight into aplay.** Do not "restore" the buffered
  version: rendering the whole utterance first delayed the first sound by ~30% of the answer's
  length. `piper_process.stdout.close()` in the parent is required or aplay never sees EOF.
- **`~/ros2_ws` has no `src/` for tts_speaker** — the copy here is the authoritative source.
- **After re-seating the Jetson↔Pi cable, bounce both PHYs** (`ip link set <if> down; … up`).
  Both ends can show carrier and correct IPs while passing zero traffic; `rx_packets: 0` with
  `tx_packets` climbing is the signature.

## Network

Direct cable, no router and no DHCP, so the robot works with no internet:
Jetson `enP8p1s0` **192.168.77.1** ↔ Pi `eth0` **192.168.77.2**.
Both sides set `never-default` and carry no gateway, so this link can never become the
default route.
