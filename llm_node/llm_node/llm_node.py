#!/usr/bin/env python3
"""
Plant Health LLM Node - Local Llama with sensor integration.

Subscribes to environmental sensor topics and plant disease detection,
builds a live sensor context, and uses it to inform plant health responses.
Uses local Llama model via llama-cpp-python for offline inference.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import Temperature, RelativeHumidity, FluidPressure
from llama_cpp import Llama
import threading
import os
import time


class PlantLLMNodeLocal(Node):
    def __init__(self):
        super().__init__('llm_node')

        # ── Parameters ──────────────────────────────────────────────
        self.declare_parameter('model_path',
                               '/home/orin-robot/models/qwen2.5-1.5b-instruct-q4_k_m.gguf')
        self.declare_parameter('system_prompt_path',
                               '/home/orin-robot/robot_ws/src/llm_node/prompts/sabis_robot_concise.txt')
        self.declare_parameter('n_ctx', 4096)
        self.declare_parameter('n_gpu_layers', -1)
        self.declare_parameter('n_batch', 512)
        self.declare_parameter('temperature', 0.7)
        self.declare_parameter('max_tokens', 200)

        model_path = self.get_parameter('model_path').value
        system_prompt_path = self.get_parameter('system_prompt_path').value
        n_ctx = self.get_parameter('n_ctx').value
        n_gpu_layers = self.get_parameter('n_gpu_layers').value
        n_batch = self.get_parameter('n_batch').value
        self.temperature = self.get_parameter('temperature').value
        self.max_tokens = self.get_parameter('max_tokens').value

        # Load system prompt
        self.system_prompt = self._load_system_prompt(system_prompt_path)

        # ── Sensor state (thread-safe) ──────────────────────────────
        self.sensor_lock = threading.Lock()
        self.sensor_data = {
            'temperature': None,
            'humidity': None,
            'pressure': None,
            'raining': None,
            'soil_moisture': None,
            'soil_wet': None,
            'disease_bottom': None,
            'disease_top': None,
            # ── Future sensor placeholders ──
            # 'light_intensity': None,
            # 'uv_index': None,
            # 'wind_speed': None,
            # 'npk_nitrogen': None,
            # 'npk_phosphorus': None,
            # 'npk_potassium': None,
            # 'soil_ph': None,
            # 'co2_ppm': None,
            # 'leaf_wetness': None,
        }
        self.sensor_timestamps = {}

        # ── Sensor subscribers ──────────────────────────────────────
        self.sub_temperature = self.create_subscription(
            Temperature, '/dht11/temperature', self._cb_temperature, 10)
        self.sub_humidity = self.create_subscription(
            RelativeHumidity, '/dht11/humidity', self._cb_humidity, 10)
        self.sub_pressure = self.create_subscription(
            FluidPressure, '/bmp180/pressure', self._cb_pressure, 10)
        self.sub_raining = self.create_subscription(
            Bool, '/rain_sensor/raining', self._cb_raining, 10)
        self.sub_soil_moisture = self.create_subscription(
            Float32, '/soil_moisture/moisture', self._cb_soil_moisture, 10)
        self.sub_soil_wet = self.create_subscription(
            Bool, '/soil_moisture/wet', self._cb_soil_wet, 10)
        self.sub_disease_bottom = self.create_subscription(
            String, '/plant_disease/bottom', self._cb_disease_bottom, 10)
        self.sub_disease_top = self.create_subscription(
            String, '/plant_disease/top', self._cb_disease_top, 10)

        # ── Future sensor subscribers (uncomment when connected) ────
        # self.sub_light = self.create_subscription(
        #     Float32, '/light_sensor/intensity', self._cb_light, 10)
        # self.sub_uv = self.create_subscription(
        #     Float32, '/uv_sensor/index', self._cb_uv, 10)
        # self.sub_wind = self.create_subscription(
        #     Float32, '/wind_sensor/speed', self._cb_wind, 10)
        # self.sub_npk_n = self.create_subscription(
        #     Float32, '/npk_sensor/nitrogen', self._cb_npk_n, 10)
        # self.sub_npk_p = self.create_subscription(
        #     Float32, '/npk_sensor/phosphorus', self._cb_npk_p, 10)
        # self.sub_npk_k = self.create_subscription(
        #     Float32, '/npk_sensor/potassium', self._cb_npk_k, 10)
        # self.sub_ph = self.create_subscription(
        #     Float32, '/soil_ph/value', self._cb_ph, 10)
        # self.sub_co2 = self.create_subscription(
        #     Float32, '/co2_sensor/ppm', self._cb_co2, 10)
        # self.sub_leaf_wet = self.create_subscription(
        #     Bool, '/leaf_wetness/wet', self._cb_leaf_wet, 10)

        # ── Speech I/O ──────────────────────────────────────────────
        self.publisher = self.create_publisher(String, '/tts_text', 10)
        self.subscription = self.create_subscription(
            String, '/speech/text', self.speech_callback, 10)
        # Create the TTS Publisher
        self.tts_publisher = self.create_publisher(String, '/tts_text', 10)

        # Conversation reset
        self.reset_subscription = self.create_subscription(
            Bool, '/conversation/reset', self.reset_callback, 10)
        self.conversation_history = []

        # ── Load LLM model ──────────────────────────────────────────
        self.get_logger().info(f'Loading LLM model from {model_path}...')
        self.llm = None
        self.model_loaded = False
        self.lock = threading.Lock()       # protects llm calls (prevent segfault)
        self.state_lock = threading.Lock() # protects model_loaded, conversation_history
        self.generating = False            # guard against concurrent generation

        threading.Thread(target=self._load_model,
                         args=(model_path, n_ctx, n_gpu_layers, n_batch),
                         daemon=True).start()

        self.get_logger().info(
            'Plant Health LLM node (local) initialized. '
            'Subscribing to /speech/text + 8 sensor topics')

    # ── System prompt loader ────────────────────────────────────────
    def _load_system_prompt(self, prompt_path):
        try:
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r') as f:
                    prompt = f.read().strip()
                self.get_logger().info(f'Loaded system prompt from {prompt_path} ({len(prompt)} chars)')
                return prompt
            else:
                self.get_logger().warn(f'System prompt file not found: {prompt_path}')
                return "You are a helpful plant health robot. Provide brief responses about plant care."
        except Exception as e:
            self.get_logger().error(f'Error loading system prompt: {e}')
            return "You are a helpful plant health robot. Provide brief responses about plant care."

    # ── Model loader ────────────────────────────────────────────────
    def _load_model(self, model_path, n_ctx, n_gpu_layers, n_batch):
        try:
            self.llm = Llama(
            model_path=self.get_parameter('model_path').value,  # <-- FIXED LINE
            n_gpu_layers=self.get_parameter('n_gpu_layers').value,
            n_ctx=self.get_parameter('n_ctx').value,
            n_threads=6,
            n_batch=512,
            verbose=False
        )
            with self.state_lock:
                self.model_loaded = True
            self.get_logger().info('LLM model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load LLM model: {e}')

    # ── Sensor callbacks ────────────────────────────────────────────
    def _update_sensor(self, key, value):
        with self.sensor_lock:
            self.sensor_data[key] = value
            self.sensor_timestamps[key] = time.time()

    def _cb_temperature(self, msg):
        self._update_sensor('temperature', msg.temperature)

    def _cb_humidity(self, msg):
        self._update_sensor('humidity', msg.relative_humidity * 100.0)

    def _cb_pressure(self, msg):
        self._update_sensor('pressure', msg.fluid_pressure / 100.0)

    def _cb_raining(self, msg):
        self._update_sensor('raining', msg.data)

    def _cb_soil_moisture(self, msg):
        self._update_sensor('soil_moisture', msg.data)

    def _cb_soil_wet(self, msg):
        self._update_sensor('soil_wet', msg.data)

    def _cb_disease_bottom(self, msg):
        self._update_sensor('disease_bottom', msg.data)

    def _cb_disease_top(self, msg):
        self._update_sensor('disease_top', msg.data)

    # ── Future sensor callbacks (uncomment when connected) ──────────
    # def _cb_light(self, msg):
    #     self._update_sensor('light_intensity', msg.data)
    # def _cb_uv(self, msg):
    #     self._update_sensor('uv_index', msg.data)
    # def _cb_wind(self, msg):
    #     self._update_sensor('wind_speed', msg.data)
    # def _cb_npk_n(self, msg):
    #     self._update_sensor('npk_nitrogen', msg.data)
    # def _cb_npk_p(self, msg):
    #     self._update_sensor('npk_phosphorus', msg.data)
    # def _cb_npk_k(self, msg):
    #     self._update_sensor('npk_potassium', msg.data)
    # def _cb_ph(self, msg):
    #     self._update_sensor('soil_ph', msg.data)
    # def _cb_co2(self, msg):
    #     self._update_sensor('co2_ppm', msg.data)
    # def _cb_leaf_wet(self, msg):
    #     self._update_sensor('leaf_wetness', msg.data)

    # ── Build sensor context string ─────────────────────────────────
    def _build_sensor_context(self):
        with self.sensor_lock:
            data = dict(self.sensor_data)
            stamps = dict(self.sensor_timestamps)

        now = time.time()
        lines = []
        stale_threshold = 60.0

        if data['temperature'] is not None:
            age = now - stamps.get('temperature', now)
            stale = " (stale)" if age > stale_threshold else ""
            lines.append(f"Air temperature: {data['temperature']:.1f} C{stale}")

        if data['humidity'] is not None:
            age = now - stamps.get('humidity', now)
            stale = " (stale)" if age > stale_threshold else ""
            lines.append(f"Air humidity: {data['humidity']:.1f}%{stale}")

        if data['pressure'] is not None:
            age = now - stamps.get('pressure', now)
            stale = " (stale)" if age > stale_threshold else ""
            lines.append(f"Atmospheric pressure: {data['pressure']:.1f} hPa{stale}")

        if data['raining'] is not None:
            status = "Yes, it is raining" if data['raining'] else "No rain detected"
            lines.append(f"Rain: {status}")

        if data['soil_moisture'] is not None:
            age = now - stamps.get('soil_moisture', now)
            stale = " (stale)" if age > stale_threshold else ""
            lines.append(f"Soil moisture level: {data['soil_moisture']:.1f}{stale}")

        if data['soil_wet'] is not None:
            status = "Wet" if data['soil_wet'] else "Dry"
            lines.append(f"Soil status: {status}")

        if data['disease_bottom'] is not None and data['disease_bottom']:
            lines.append(f"Bottom camera detection: {data['disease_bottom']}")

        if data['disease_top'] is not None and data['disease_top']:
            lines.append(f"Top camera detection: {data['disease_top']}")

        # Future sensors (uncomment when connected)
        # if data.get('light_intensity') is not None:
        #     lines.append(f"Light intensity: {data['light_intensity']:.0f} lux")
        # ... (same pattern for all future sensors)

        if not lines:
            return ("\n[LIVE SENSOR READINGS]\n"
                    "NO SENSOR DATA AVAILABLE. All sensors are offline or have not reported yet. "
                    "Do NOT invent or guess sensor values. If the user asks about environmental "
                    "conditions or sensor readings, tell them the sensors are not reporting data yet."
                    "\n[END SENSOR READINGS]")

        return "\n[LIVE SENSOR READINGS]\n" + "\n".join(lines) + "\n[END SENSOR READINGS]"

    # ── Conversation reset ──────────────────────────────────────────
    def reset_callback(self, msg):
        if msg.data:
            with self.state_lock:
                self.conversation_history = []
            self.get_logger().info('Conversation history cleared')

    # ── Speech handler ──────────────────────────────────────────────
    def speech_callback(self, msg):
        input_text = msg.data
        self.get_logger().info(f'Received speech: "{input_text}"')

        # 1. Grab the model name to determine the formatting rules
        model_path_str = self.get_parameter('model_path').value

        # 2. BUILD THE PROMPT (This is where your code block goes!)
        if "gemma" in model_path_str.lower():
            # Gemma Architecture Syntax
            prompt = f"<|turn|>system\n{self.system_prompt}<|turn|>\n"
            with self.lock:
                for entry in self.conversation_history:
                    prompt += f"<|turn|>{entry['role']}\n{entry['content']}<|turn|>\n"
            prompt += f"<|turn|>user\n{input_text}<|turn|>\n<|turn|>model\n<|channel|>thought\n"
            stop_tokens = ["<|turn|>", "<|end_of_text|>"]
            
        elif "llama" in model_path_str.lower():
            # Llama 3.2 Architecture Syntax
            prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{self.system_prompt}<|eot_id|>"
            with self.lock:
                for entry in self.conversation_history:
                    role = "assistant" if entry['role'] == "model" else entry['role']
                    prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{entry['content']}<|eot_id|>"
            prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            stop_tokens = ["<|eot_id|>", "<|end_of_text|>"]
            
        else:
            # Qwen / ChatML Fallback Syntax
            prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            with self.lock:
                for entry in self.conversation_history:
                    role = "assistant" if entry['role'] == "model" else entry['role']
                    prompt += f"<|im_start|>{role}\n{entry['content']}<|im_end|>\n"
            prompt += f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
            stop_tokens = ["<|im_end|>", "<|endoftext|>"]

        # 3. GENERATE THE RESPONSE
        self.get_logger().info('Generating LLM response...')
        
        with self.lock:
            # Pass the dynamically generated prompt and stop_tokens to llama.cpp
            response = self.llm(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=stop_tokens,
                echo=False
            )

        raw_text = response['choices'][0]['text'].strip()

        # 4. CLEAN THE OUTPUT (Remove internal thought tags if Gemma is used)
        if "gemma" in model_path_str.lower() and "<|channel|>" in raw_text:
            response_text = raw_text.split("<|channel|>")[-1].strip()
        else:
            response_text = raw_text.strip()

        self.get_logger().info(f'Response: "{response_text}"')

        # 5. PUBLISH TO TEXT-TO-SPEECH
        tts_msg = String()
        tts_msg.data = response_text
        self.tts_publisher.publish(tts_msg)

        # 6. UPDATE CONVERSATION HISTORY
        with self.lock:
            self.conversation_history.append({"role": "user", "content": input_text})
            self.conversation_history.append({"role": "model", "content": response_text})
            # Keep history from getting too long to prevent memory issues
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

    def _generate_response(self, input_text):
        try:
            start_time = time.time()

            # Build sensor context
            sensor_ctx = self._build_sensor_context()

            # Combine system prompt with sensor context (always inject)
            full_system = self.system_prompt + f"\n\n{sensor_ctx}"
            self.get_logger().info(f'Injected sensor context ({len(sensor_ctx)} chars)')

            # Build ChatML prompt (Qwen2.5 format)
            prompt = f"<|im_start|>system\n{full_system}<|im_end|>\n"

            # Add conversation history
            with self.state_lock:
                for entry in self.conversation_history:
                    prompt += f"<|im_start|>{entry['role']}\n{entry['content']}<|im_end|>\n"

            prompt += f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"

            self.get_logger().info(f'Generating LLM response (prompt ~{len(prompt)} chars)...')
            with self.lock:
                response = self.llm(
                    prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stop=["<|im_end|>", "<|endoftext|>"],
                    echo=False
                )

            response_text = response['choices'][0]['text'].strip()
            elapsed = time.time() - start_time

            if response_text:
                self.get_logger().info(f'Response ({elapsed:.1f}s): "{response_text}"')

                with self.state_lock:
                    self.conversation_history.append({'role': 'user', 'content': input_text})
                    self.conversation_history.append({'role': 'assistant', 'content': response_text})

                out = String()
                out.data = response_text
                self.publisher.publish(out)
            else:
                self.get_logger().warn(f'Generated empty response ({elapsed:.1f}s)')

        except Exception as e:
            self.get_logger().error(f'Error generating response: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
        finally:
            with self.state_lock:
                self.generating = False


def main(args=None):
    rclpy.init(args=args)
    try:
        node = PlantLLMNodeLocal()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in local Plant LLM node: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
