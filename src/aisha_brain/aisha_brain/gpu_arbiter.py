#!/usr/bin/env python3
"""GPU arbiter — time-multiplex the Jetson GPU between vision and the LLM.

Implements the state machine from ADR 0001
(docs/adr/0001-gpu-multiplexing-navigating-conversing.md):

  NAVIGATING : YOLO inference active (owns GPU); cmd_vel passes through;
               admin LLM on CPU (num_gpu=0).
  CONVERSING : YOLO killed (engine + CUDA context freed, ~1.2 GB); motors
               locked; optional LLM router unloaded; admin LLM may run on GPU
               (num_gpu=99) for ~2 s answers.

Why kill, not just pause: the full-stack integration test showed pausing
inference leaves the engine's ~1.2 GB GPU reservation resident, and the stack
then has too little free for a GPU llama. In-process release reclaims ~0 MB
(CUDA context persists until the process exits). So the only thing that frees
the GPU is killing yolov8_node — which this arbiter supervises as a managed
subprocess. RealSense is a SEPARATE process and is never touched, so resume is
just a ~3-5 s engine reload, not a USB re-enumeration.

Transition sequence (enter CONVERSING):
  1. lock motors (zero cmd_vel; safety tick keeps it zero)
  2. pause_inference service -> drain in-flight GPU work cleanly
  3. kill yolov8_node            -> frees the GPU
  4. unload LLM router (if any)  -> not co-resident with GPU llama
  5. broadcast admin num_gpu=99  -> via /aisha/mode
Exit (NAVIGATING):
  1. respawn yolov8_node, wait until ready (engine reload)  [motors stay locked]
  2. broadcast admin num_gpu=0; unload llama
  3. clear motion-inhibit ONLY once vision is confirmed back

Triggers:
    - Wake word "Hey Aisha stop" on /speech/text (auto; param wake_enabled).
    - Manual: ros2 service call /aisha/set_conversing std_srvs/srv/SetBool "{data: true}"
"""
import os
import re
import signal
import subprocess
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy,
                       QoSDurabilityPolicy)
from std_msgs.msg import String
from std_srvs.srv import SetBool
from geometry_msgs.msg import Twist

try:
    import requests
except ImportError:  # requests is used by admin_node too; should be present
    requests = None

NAVIGATING = 'NAVIGATING'
CONVERSING = 'CONVERSING'


class GpuArbiter(Node):
    def __init__(self):
        super().__init__('gpu_arbiter')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('yolo_pause_service',
                               '/detection_node/pause_inference')
        self.declare_parameter('cmd_vel_in', '/cmd_vel_nav')
        self.declare_parameter('cmd_vel_out', '/cmd_vel')
        self.declare_parameter('ollama_url', 'http://127.0.0.1:11434')
        # Optional LLM router to unload on CONVERSING. Empty by default since
        # brain_node now uses a rule-based router (no LLM model resident).
        self.declare_parameter('router_model', '')
        self.declare_parameter('llm_model', 'llama3.2:1b')
        # Manage yolov8_node as a child process (spawn/kill to free the GPU).
        # If False, the arbiter only pauses (insufficient to free GPU under the
        # full stack — see ADR). When True, do NOT also launch yolov8_node from
        # cerebro_aisha; the arbiter owns it.
        self.declare_parameter('manage_yolo', True)
        # Command used to (re)start yolov8_node. Node default name is
        # 'detection_node', so the pause service stays /detection_node/...
        self.declare_parameter('yolo_cmd',
                               ['ros2', 'run', 'yolov8_ros', 'yolov8_node'])
        # Max seconds to wait for a respawned YOLO to be ready before giving up
        # (motors stay locked while we wait / on failure).
        self.declare_parameter('yolo_ready_timeout_s', 25.0)
        # Auto-return to NAVIGATING after this many seconds (never stuck blind).
        self.declare_parameter('conversation_timeout_s', 90.0)
        # Wake word: a phrase on the STT stream that switches NAVIGATING ->
        # CONVERSING ("Hey Aisha stop" = stop moving + start a conversation).
        # Matched tolerantly against Whisper output (name variants + "stop").
        self.declare_parameter('wake_enabled', True)
        self.declare_parameter('speech_topic', '/speech/text')

        self.pause_srv_name = self.get_parameter('yolo_pause_service').value
        self.cmd_in = self.get_parameter('cmd_vel_in').value
        self.cmd_out = self.get_parameter('cmd_vel_out').value
        self.ollama_url = self.get_parameter('ollama_url').value
        self.router_model = self.get_parameter('router_model').value
        self.llm_model = self.get_parameter('llm_model').value
        self.manage_yolo = bool(self.get_parameter('manage_yolo').value)
        self.yolo_cmd = list(self.get_parameter('yolo_cmd').value)
        self.yolo_ready_timeout = float(
            self.get_parameter('yolo_ready_timeout_s').value)
        self.conv_timeout = float(
            self.get_parameter('conversation_timeout_s').value)
        self.wake_enabled = bool(self.get_parameter('wake_enabled').value)
        self.speech_topic = self.get_parameter('speech_topic').value

        # ── State ───────────────────────────────────────────────────────────
        self.state = NAVIGATING
        self._vision_ready = False          # gates motion even within NAVIGATING
        self._yolo_proc = None
        self._busy = threading.Lock()        # serialize transitions
        self._conv_deadline = None           # monotonic auto-return deadline
        self._last_wake = 0.0                # debounce repeated wake triggers

        # ── I/O ─────────────────────────────────────────────────────────────
        latched = QoSProfile(
            depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.mode_pub = self.create_publisher(String, '/aisha/mode', latched)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_out, 10)
        self.create_subscription(Twist, self.cmd_in, self._on_cmd_in, 10)
        self.create_service(SetBool, '/aisha/set_conversing',
                            self._on_set_conversing)
        self.pause_client = self.create_client(SetBool, self.pause_srv_name)

        # Wake word "Hey Aisha stop": tolerant match for an Aisha-name variant
        # (Whisper spells it many ways) followed by "stop" in the same line.
        # Requires the NAME — a bare "stop" is brain_node's emergency halt, not
        # a conversation trigger.
        self._wake_re = re.compile(
            r'\b(a[iye]{1,3}sha|asha|aesha)\b.{0,30}\bstop\b')
        if self.wake_enabled:
            self.create_subscription(
                String, self.speech_topic, self._on_speech, 10)

        # Safety: republish zero velocity at 10 Hz whenever motion is not
        # allowed (CONVERSING, or NAVIGATING before vision is confirmed back).
        self.create_timer(0.1, self._safety_tick)

        self._publish_mode()
        wake = (f'wake word "Hey Aisha stop" on {self.speech_topic}'
                if self.wake_enabled else 'wake word disabled')
        self.get_logger().info(
            f'GPU arbiter up in {self.state} (manage_yolo={self.manage_yolo}, '
            f'{wake}). Manual trigger: ros2 service call /aisha/set_conversing '
            f'std_srvs/srv/SetBool "{{data: true}}"')

        # Own the YOLO process from the start so we can kill/respawn it.
        if self.manage_yolo:
            threading.Thread(target=self._navigating_worker,
                             daemon=True).start()
        else:
            self._vision_ready = True  # external YOLO assumed already up

    # ── cmd_vel gating ──────────────────────────────────────────────────────
    def _motion_allowed(self):
        return self.state == NAVIGATING and self._vision_ready

    def _on_cmd_in(self, msg: Twist):
        if self._motion_allowed():
            self.cmd_pub.publish(msg)
        # else: dropped — robot must be stationary while blind.

    def _safety_tick(self):
        if not self._motion_allowed():
            self.cmd_pub.publish(Twist())  # all-zero
        # Auto-return from a stuck CONVERSING (checked on the executor thread).
        if (self.state == CONVERSING and self._conv_deadline is not None
                and time.time() > self._conv_deadline and not self._busy.locked()):
            self._conv_deadline = None
            self.get_logger().warn(
                f'[gpu-mux] CONVERSING timed out -> auto-return to NAVIGATING')
            threading.Thread(target=self._navigating_worker, daemon=True).start()

    # ── Wake word ───────────────────────────────────────────────────────────
    def _on_speech(self, msg: String):
        """Watch the STT stream for the wake phrase -> enter CONVERSING.

        "Hey Aisha stop" both halts the robot (CONVERSING locks motion) and
        starts a conversation (frees the GPU for a fast answer). Only fires
        while NAVIGATING; a 3 s debounce avoids re-triggering on echoes.
        """
        text = (msg.data or '').lower()
        if not self._wake_re.search(text):
            return
        now = time.time()
        if now - self._last_wake < 3.0:
            return
        self._last_wake = now
        if self.state == CONVERSING or self._busy.locked():
            return
        self.get_logger().info(f'[gpu-mux] wake word detected: "{msg.data[:50]}" '
                               f'-> CONVERSING')
        threading.Thread(target=self._conversing_worker, daemon=True).start()

    # ── Trigger ─────────────────────────────────────────────────────────────
    def _on_set_conversing(self, request, response):
        target = CONVERSING if request.data else NAVIGATING
        if self._busy.locked():
            response.success = False
            response.message = 'transition already in progress'
            return response
        if target == self.state:
            response.success = True
            response.message = f'already {self.state}'
            return response
        # Run the (slow) transition in a worker so the executor keeps spinning
        # (safety_tick + cmd_vel gating stay live during the YOLO reload).
        worker = (self._conversing_worker if target == CONVERSING
                  else self._navigating_worker)
        threading.Thread(target=worker, daemon=True).start()
        response.success = True
        response.message = f'transition to {target} started'
        return response

    # ── State workers (run off the executor thread) ─────────────────────────
    def _conversing_worker(self):
        with self._busy:
            self.get_logger().info(f'[gpu-mux] {self.state} -> {CONVERSING}')
            # Enter CONVERSING immediately so motion locks right away.
            self.state = CONVERSING
            self._vision_ready = False
            self._publish_mode()
            self.cmd_pub.publish(Twist())

            # 1) Drain in-flight YOLO GPU work before we kill the process.
            self._call_pause(pause=True)
            # 2) Kill yolov8_node -> frees engine + CUDA context (~1.2 GB).
            self._stop_yolo()
            # 3) Unload the LLM router if one is configured (rule-based router
            #    keeps this empty -> nothing to unload).
            if self.router_model:
                self._set_keepalive(self.router_model, 0)
            # 4) Admin LLM target -> GPU (broadcast via /aisha/mode).
            self._set_admin_num_gpu(99)
            self._conv_deadline = (time.time() + self.conv_timeout
                                   if self.conv_timeout > 0 else None)
            self.get_logger().info(
                '[gpu-mux] CONVERSING ready: GPU freed, llama may use num_gpu=99')

    def _navigating_worker(self):
        with self._busy:
            self.get_logger().info(f'[gpu-mux] -> {NAVIGATING} '
                                   f'(respawning vision)')
            self._conv_deadline = None
            self.state = NAVIGATING
            self._vision_ready = False   # stay motion-locked until vision back
            self._publish_mode()

            # 1) Admin LLM back to CPU; unload llama so it frees the GPU.
            self._set_admin_num_gpu(0)
            self._set_keepalive(self.llm_model, 0)
            # 2) Respawn YOLO and wait until it is actually ready.
            if self.manage_yolo:
                self._start_yolo()
                ready = self._wait_yolo_ready(self.yolo_ready_timeout)
            else:
                # external YOLO: just resume its inference.
                self._call_pause(pause=False)
                ready = True
            # 3) Only now allow motion.
            if ready:
                self._vision_ready = True
                self.get_logger().info(
                    '[gpu-mux] NAVIGATING ready: vision up, motion enabled')
            else:
                self.get_logger().error(
                    '[gpu-mux] YOLO did NOT come ready in time — MOTION STAYS '
                    'LOCKED (robot blind). Will retry on next NAVIGATING.')

    # ── YOLO process supervision ────────────────────────────────────────────
    def _start_yolo(self):
        if not self.manage_yolo:
            return
        if self._yolo_proc is not None and self._yolo_proc.poll() is None:
            return  # already running
        try:
            self._yolo_proc = subprocess.Popen(
                self.yolo_cmd, start_new_session=True, env=os.environ.copy())
            self.get_logger().info(
                f'[gpu-mux] spawned yolov8_node pid {self._yolo_proc.pid} '
                f'({" ".join(self.yolo_cmd)})')
        except Exception as e:
            self.get_logger().error(f'[gpu-mux] failed to spawn YOLO: {e}')
            self._yolo_proc = None

    def _stop_yolo(self):
        if not self.manage_yolo or self._yolo_proc is None:
            return
        if self._yolo_proc.poll() is not None:
            self._yolo_proc = None
            return
        pid = self._yolo_proc.pid
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                self._yolo_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                self._yolo_proc.wait(timeout=5)
        except ProcessLookupError:
            pass
        except Exception as e:
            self.get_logger().warn(f'[gpu-mux] stop YOLO error: {e}')
        self.get_logger().info('[gpu-mux] yolov8_node terminated, GPU reclaimed')
        self._yolo_proc = None

    def _wait_yolo_ready(self, timeout_s):
        """Ready == the node's pause service is up (engine loaded)."""
        deadline = time.time() + timeout_s
        # Recreate a fresh client so discovery isn't stuck on the dead node.
        while time.time() < deadline:
            if self._yolo_proc is not None and self._yolo_proc.poll() is not None:
                self.get_logger().error('[gpu-mux] YOLO process exited early')
                return False
            if self.pause_client.wait_for_service(timeout_sec=1.0):
                time.sleep(0.5)  # small settle after the service appears
                return True
        return False

    # ── Effects ─────────────────────────────────────────────────────────────
    def _call_pause(self, pause: bool):
        if not self.pause_client.service_is_ready():
            if not self.pause_client.wait_for_service(timeout_sec=2.0):
                self.get_logger().warn(
                    f'[gpu-mux] pause service unavailable; '
                    f'{"pause" if pause else "resume"} skipped')
                return
        req = SetBool.Request()
        req.data = pause
        # Best-effort, fire-and-forget; we kill the process right after anyway.
        self.pause_client.call_async(req)

    def _set_keepalive(self, model, seconds):
        if requests is None:
            return
        try:
            requests.post(f'{self.ollama_url}/api/generate',
                          json={'model': model, 'keep_alive': seconds},
                          timeout=10)
            verb = 'unloaded' if seconds == 0 else f'pinned {seconds}s'
            self.get_logger().info(f'[gpu-mux] Ollama {model} {verb}')
        except Exception as e:
            self.get_logger().warn(
                f'[gpu-mux] Ollama {model} keep_alive failed: {e}')

    def _set_admin_num_gpu(self, num_gpu: int):
        """Broadcast desired admin LLM offload; admin_node maps /aisha/mode.

        admin_node currently fixes num_gpu at startup (ADR open item), so the
        /aisha/mode topic is the contract until it is wired to rebuild its
        Ollama client on mode change.
        """
        self.get_logger().info(
            f'[gpu-mux] admin LLM target num_gpu={num_gpu} (via /aisha/mode)')

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _publish_mode(self):
        m = String()
        m.data = self.state
        self.mode_pub.publish(m)

    def shutdown(self):
        """Kill the managed YOLO child so we don't leave an orphan."""
        self.manage_yolo and self._stop_yolo()


def main(args=None):
    rclpy.init(args=args)
    node = GpuArbiter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
