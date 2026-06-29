#!/usr/bin/env python3
"""
Plant Health LLM Node - Gemini API with sensor integration.

Subscribes to environmental sensor topics and plant disease detection,
builds a live sensor context, and uses it to inform plant health responses.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import Temperature, RelativeHumidity, FluidPressure
import google.generativeai as genai
import threading
import os
import time


class PlantLLMNode(Node):
    def __init__(self):
        super().__init__('llm_node')

        # ── Parameters ──────────────────────────────────────────────
        self.declare_parameter('api_key', '')
        self.declare_parameter('model_name', 'gemini-2.5-flash')
        self.declare_parameter('system_prompt_path',
                               '/home/orin-robot/robot_ws/src/llm_node/prompts/sabis_robot_concise.txt')
        self.declare_parameter('temperature', 0.4)
        self.declare_parameter('max_tokens', 8192)

        api_key = self.get_parameter('api_key').value
        model_name = self.get_parameter('model_name').value
        system_prompt_path = self.get_parameter('system_prompt_path').value
        self.temperature = self.get_parameter('temperature').value
        self.max_tokens = self.get_parameter('max_tokens').value

        if not api_key:
            api_key = os.environ.get('GEMINI_API_KEY', '')
        if not api_key:
            self.get_logger().error('No API key! Set GEMINI_API_KEY env var or pass api_key parameter')
            raise ValueError('Missing Gemini API key')

        # ── Gemini setup ────────────────────────────────────────────
        genai.configure(api_key=api_key)
        self.system_prompt = self._load_system_prompt(system_prompt_path)

        self.get_logger().info(f'Initializing Gemini model: {model_name}...')
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
            system_instruction=self.system_prompt
        )
        self.get_logger().info('Gemini model initialized')

        # ── Sensor state (thread-safe) ──────────────────────────────
        self.sensor_lock = threading.Lock()
        self.sensor_data = {
            # Environmental sensors
            'temperature': None,       # DHT11 temperature (C)
            'humidity': None,          # DHT11 humidity (%)
            'pressure': None,          # BMP180 pressure (hPa)
            'raining': None,           # Rain sensor (bool)
            'soil_moisture': None,     # Soil moisture (numeric)
            'soil_wet': None,          # Soil wet/dry (bool)
            # Plant disease cameras
            'disease_bottom': None,    # Bottom camera disease detection
            'disease_top': None,       # Top camera disease detection
            # ── Future sensor placeholders ──
            # 'light_intensity': None,   # Light/UV sensor (lux)
            # 'uv_index': None,          # UV index
            # 'wind_speed': None,        # Wind speed (m/s)
            # 'npk_nitrogen': None,      # Soil nitrogen (mg/kg)
            # 'npk_phosphorus': None,    # Soil phosphorus (mg/kg)
            # 'npk_potassium': None,     # Soil potassium (mg/kg)
            # 'soil_ph': None,           # Soil pH
            # 'co2_ppm': None,           # CO2 concentration (ppm)
            # 'leaf_wetness': None,      # Leaf wetness (bool or %)
        }
        self.sensor_timestamps = {}  # Track when each sensor last updated

        # ── Sensor subscribers ──────────────────────────────────────
        # DHT11 - temperature and humidity
        self.sub_temperature = self.create_subscription(
            Temperature, '/dht11/temperature', self._cb_temperature, 10)
        self.sub_humidity = self.create_subscription(
            RelativeHumidity, '/dht11/humidity', self._cb_humidity, 10)

        # BMP180 - atmospheric pressure
        self.sub_pressure = self.create_subscription(
            FluidPressure, '/bmp180/pressure', self._cb_pressure, 10)

        # Rain sensor
        self.sub_raining = self.create_subscription(
            Bool, '/rain_sensor/raining', self._cb_raining, 10)

        # Soil moisture
        self.sub_soil_moisture = self.create_subscription(
            Float32, '/soil_moisture/moisture', self._cb_soil_moisture, 10)
        self.sub_soil_wet = self.create_subscription(
            Bool, '/soil_moisture/wet', self._cb_soil_wet, 10)

        # Plant disease detection cameras
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

        self.get_logger().info(
            'Plant Health LLM node ready! '
            'Subscribing to /speech/text + 8 sensor topics, publishing to /tts_text')

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
                return "You are a helpful plant health robot assistant. Provide brief, clear responses about plant care."
        except Exception as e:
            self.get_logger().error(f'Error loading system prompt: {e}')
            return "You are a helpful plant health robot assistant. Provide brief, clear responses about plant care."

    # ── Sensor callbacks ────────────────────────────────────────────
    def _update_sensor(self, key, value):
        with self.sensor_lock:
            self.sensor_data[key] = value
            self.sensor_timestamps[key] = time.time()

    def _cb_temperature(self, msg):
        self._update_sensor('temperature', msg.temperature)

    def _cb_humidity(self, msg):
        # relative_humidity is 0.0-1.0, convert to percentage
        self._update_sensor('humidity', msg.relative_humidity * 100.0)

    def _cb_pressure(self, msg):
        # fluid_pressure is in Pascals, convert to hPa
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
        """Build a human-readable sensor summary to inject into the prompt."""
        with self.sensor_lock:
            data = dict(self.sensor_data)
            stamps = dict(self.sensor_timestamps)

        now = time.time()
        lines = []
        stale_threshold = 60.0  # seconds before a reading is considered stale

        # Environmental readings
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

        # Plant disease detection
        if data['disease_bottom'] is not None and data['disease_bottom']:
            lines.append(f"Bottom camera detection: {data['disease_bottom']}")

        if data['disease_top'] is not None and data['disease_top']:
            lines.append(f"Top camera detection: {data['disease_top']}")

        # Future sensors (uncomment when connected)
        # if data.get('light_intensity') is not None:
        #     lines.append(f"Light intensity: {data['light_intensity']:.0f} lux")
        # if data.get('uv_index') is not None:
        #     lines.append(f"UV index: {data['uv_index']:.1f}")
        # if data.get('wind_speed') is not None:
        #     lines.append(f"Wind speed: {data['wind_speed']:.1f} m/s")
        # if data.get('npk_nitrogen') is not None:
        #     lines.append(f"Soil nitrogen: {data['npk_nitrogen']:.0f} mg/kg")
        # if data.get('npk_phosphorus') is not None:
        #     lines.append(f"Soil phosphorus: {data['npk_phosphorus']:.0f} mg/kg")
        # if data.get('npk_potassium') is not None:
        #     lines.append(f"Soil potassium: {data['npk_potassium']:.0f} mg/kg")
        # if data.get('soil_ph') is not None:
        #     lines.append(f"Soil pH: {data['soil_ph']:.1f}")
        # if data.get('co2_ppm') is not None:
        #     lines.append(f"CO2: {data['co2_ppm']:.0f} ppm")
        # if data.get('leaf_wetness') is not None:
        #     status = "Wet" if data['leaf_wetness'] else "Dry"
        #     lines.append(f"Leaf surface: {status}")

        if not lines:
            return ("\n[LIVE SENSOR READINGS]\n"
                    "NO SENSOR DATA AVAILABLE. All sensors are offline or have not reported yet. "
                    "Do NOT invent or guess sensor values. If the user asks about environmental "
                    "conditions or sensor readings, tell them the sensors are not reporting data yet."
                    "\n[END SENSOR READINGS]")

        return "\n[LIVE SENSOR READINGS]\n" + "\n".join(lines) + "\n[END SENSOR READINGS]"

    # ── Speech handler ──────────────────────────────────────────────
    def speech_callback(self, msg):
        input_text = msg.data.strip()
        if not input_text:
            return

        self.get_logger().info(f'Received speech: "{input_text}"')
        threading.Thread(target=self._generate_response,
                         args=(input_text,), daemon=True).start()

    def _generate_response(self, input_text):
        try:
            start_time = time.time()

            # Build sensor context and always prepend to user message
            sensor_ctx = self._build_sensor_context()
            full_prompt = f"{sensor_ctx}\n\nUser question: {input_text}"
            self.get_logger().info(f'Injected sensor context ({len(sensor_ctx)} chars)')

            # Detect language for bilingual support
            is_arabic = any(0x0600 <= ord(c) <= 0x06FF for c in input_text)
            if is_arabic:
                full_prompt += "\n\nIMPORTANT: The user spoke in Arabic. Respond ENTIRELY in Arabic."
            else:
                full_prompt += "\n\nIMPORTANT: The user spoke in English. Respond ENTIRELY in English."

            response = self.model.generate_content(full_prompt)
            response_text = response.text.strip()
            elapsed = time.time() - start_time

            if response_text:
                self.get_logger().info(f'Response ({elapsed:.2f}s): "{response_text}"')
                out = String()
                out.data = response_text
                self.publisher.publish(out)
            else:
                self.get_logger().warn('Generated empty response')

        except Exception as e:
            self.get_logger().error(f'Error generating response: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())


def main(args=None):
    rclpy.init(args=args)
    try:
        node = PlantLLMNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in Plant LLM node: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
