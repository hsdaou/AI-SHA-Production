import os
import re
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import re
import requests
import time
import threading
import uuid
import queue
from collections import deque


class BrainNode(Node):
    def __init__(self):
        super().__init__('ai_sha_brain')

        # ── Deduplication: ignore identical messages within this window ─────────
        # /speech/text and /user_speech can both fire for the same WA message.
        self._last_processed_text: str = ''
        self._last_processed_time: float = 0.0
        # 1s debounce: long enough to catch dual-subscription races
        # (/speech/text + /user_speech fire for the same message) but short
        # enough to allow urgent repeated commands ("Stop", "Follow me").
        self._debounce_secs: float = 1.0

        # ── Input subscriptions ────────────────────────────────────────────────
        # Both topics feed the same callback — /speech/text is the Jetson
        # architecture standard; /user_speech is the local alias used by
        # whatsapp_listener and manual ros2 topic pub testing.
        self.create_subscription(String, '/speech/text', self.listener_callback, 10)
        self.create_subscription(String, '/user_speech', self.listener_callback, 10)

        # Vision context from detection_node (YOLOv8) — stored for future routing
        self.create_subscription(String, '/detection/objects_simple', self._on_detection, 10)
        self._last_detection = {}

        # ── Output publishers ──────────────────────────────────────────────────
        self.admin_pub  = self.create_publisher(String, '/admin_task', 10)
        self.nav_pub    = self.create_publisher(String, '/nav_goal', 10)
        self.action_pub = self.create_publisher(String, '/action_request', 10)
        # Single canonical speech output bus — tts_node and whatsapp_listener
        # both subscribe to /robot_speech. Do NOT also publish to /tts_text to
        # avoid double-processing by whatsapp_listener.
        self.speech_pub = self.create_publisher(String, '/robot_speech', 10)

        # ── History: record (user, answer) pairs for follow-up context ─────────
        # admin_node and action_node publish to separate response topics so
        # brain_node can pair each answer with the correct pending question.
        # Without this, a fast ACTION response would steal a slow ADMIN
        # question from a shared FIFO (the "intent race condition").
        # Brain forwards all responses to /robot_speech for TTS + WhatsApp.
        self.create_subscription(
            String, '/admin_response', self._on_admin_response, 10
        )
        self.create_subscription(
            String, '/action_response', self._on_action_response, 10
        )
        # Separate pending dicts per intent — prevents cross-intent pairing.
        # Each entry is {query_uuid: (timestamp, question_text)}.
        # UUID-based lookup prevents FIFO desync: if a question times out
        # and a new question enters, a delayed response carrying the
        # original UUID will NOT match the new question.
        self._pending_admin = {}   # {query_id: (timestamp, question_text)}
        self._pending_action = {}  # {query_id: (timestamp, question_text)}
        # Timeout for pending questions.  If admin_node hangs (Ollama crash,
        # network partition), stale entries latch the VRAM bypass for this
        # entire duration.  30s is long enough for normal RAG inference
        # (Llama 3.2 on Jetson typically responds in 5-20s) but short enough
        # to restore LLM-based routing promptly after a failure.
        self._pending_timeout = 30.0        # seconds before a question expires
        self.history = deque(maxlen=5)   # (user_text, robot_answer) tuples
        # Lock protecting self.history and self._pending_* queues.
        # Response callbacks run on the ROS 2 executor thread while _route()
        # runs on the background worker thread.  Without this lock, a
        # deque.append() at maxlen (which internally poplefts the oldest
        # item) can mutate the deque mid-iteration in _route()'s list
        # comprehension, raising RuntimeError: deque mutated during iteration.
        self._history_lock = threading.Lock()

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('ollama_url', 'http://127.0.0.1:11434/api/generate')
        self.declare_parameter('router_model', 'gemma3:270m')
        self.declare_parameter('router_timeout', 30)

        self.ollama_url     = self.get_parameter('ollama_url').get_parameter_value().string_value
        self.router_model   = self.get_parameter('router_model').get_parameter_value().string_value
        self.router_timeout = self.get_parameter('router_timeout').get_parameter_value().integer_value

        # Ollama per-request VRAM limits.  OLLAMA_NUM_GPU and OLLAMA_NUM_CTX
        # are SERVER-side env vars — setting them on this ROS 2 client process
        # via additional_env does NOT affect the already-running `ollama serve`.
        # We read them from the environment (injected by the launch file) and
        # pass them in the API request's "options" dict, which Ollama honours
        # as per-request overrides.
        self._ollama_num_gpu = int(os.environ.get('OLLAMA_NUM_GPU', '999'))
        self._ollama_num_ctx = int(os.environ.get('OLLAMA_NUM_CTX', '2048'))

        # ── Single worker thread for routing (prevents thread explosion) ────
        # maxsize=3: if the LLM times out (30s), at most 3 commands buffer.
        # Older commands are silently dropped — prevents replaying a long
        # backlog of stale speech after an Ollama timeout resolves.
        self._route_queue = queue.Queue(maxsize=3)
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self._check_ollama()
        self.get_logger().info('AI-SHA Brain: Router Active')

    # ── Startup ────────────────────────────────────────────────────────────────

    def _check_ollama(self):
        try:
            # Derive the /api/tags URL from whatever ollama_url is set to.
            # Works whether ollama_url is a base URL (http://host:11434) or
            # a full endpoint (http://host:11434/api/generate).
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(self.ollama_url)
            tags_url = urlunparse(parsed._replace(path='/api/tags'))
            r = requests.get(tags_url, timeout=5)
            models = [m['name'] for m in r.json().get('models', [])]
            if self.router_model not in models:
                self.get_logger().warning(f'Router model {self.router_model} not found. Available: {models}')
            else:
                self.get_logger().info(f'Ollama OK. Model: {self.router_model}')
        except Exception as e:
            self.get_logger().error(f'Ollama unreachable: {e}')

    # ── Intent classification ──────────────────────────────────────────────────

    # Emergency keywords that must bypass Ollama entirely.
    # If admin_node is mid-inference (15+ seconds), Ollama's request queue
    # is locked (OLLAMA_NUM_PARALLEL=1 default).  A "Stop!" command would
    # hang until RAG finishes — unacceptable for physical safety.  These
    # keywords are routed to NAV immediately without touching Ollama.
    #
    # Word-boundary regex prevents false positives like "bus stop" or
    # "don't stop the music".  Additionally, only short utterances
    # (≤ 4 words) qualify — genuine panic commands are brief.
    _EMERGENCY_STOP_PATTERNS = [
        re.compile(r'\b' + w + r'\b') for w in
        ['stop', 'halt', 'freeze', 'shut up', 'emergency']
    ]

    def classify_intent(self, text):
        """Classify user intent — LLM first, keywords as fallback.

        Gemma 3 (270M) is lightweight enough to serve as the primary router
        on the Jetson Orin Nano.  Keywords only fire if Ollama times out or
        is unreachable, avoiding misroutes from overly broad patterns
        (e.g. "can you tell" matching an ACTION request as ADMIN).

        VRAM safety: If there are pending unanswered questions, admin_node
        is likely mid-inference with Llama 3.2 loaded in VRAM.  Loading the
        router model (Gemma 3) simultaneously would risk OOM on the 8 GB
        Jetson.  In this case, skip the LLM and use keyword-only routing.
        """
        # ── Emergency pre-emption: bypass Ollama entirely ─────────────────
        text_lower = text.lower().strip()
        word_count = len(text_lower.split())
        if word_count <= 4 and any(p.search(text_lower) for p in self._EMERGENCY_STOP_PATTERNS):
            self.get_logger().warning(f'EMERGENCY OVERRIDE -> NAV (stop): "{text[:40]}"')
            return {"intent": "NAV"}

        # Proactively expire stale pending questions so a hung admin_node
        # doesn't latch the VRAM bypass indefinitely.  Response callbacks
        # also expire stale entries, but only when a response message
        # arrives — if admin_node is completely dead, no messages come in
        # and the bypass would stay latched for the full timeout duration.
        with self._history_lock:
            now = time.time()
            for q_name, q_dict in [('ADMIN', self._pending_admin),
                                    ('ACTION', self._pending_action)]:
                stale_ids = [qid for qid, (ts, _) in q_dict.items()
                             if now - ts > self._pending_timeout]
                for qid in stale_ids:
                    stale_ts, stale_q = q_dict.pop(qid)
                    self.get_logger().warning(
                        f'Proactive expiry of pending [{q_name}] '
                        f'({now - stale_ts:.0f}s old): {stale_q[:60]}'
                    )

        # If admin is still processing (ADMIN queue non-empty), Llama 3.2
        # is likely loaded in VRAM.  Skip the LLM router to avoid OOM.
        # ACTION queue alone doesn't block VRAM (action_node doesn't use Ollama).
        if len(self._pending_admin) > 0:
            self.get_logger().info(
                f'Pending ADMIN questions ({len(self._pending_admin)}) — '
                f'skipping LLM classify to avoid VRAM contention'
            )
        else:
            llm_result = self._llm_classify(text)
            if llm_result is not None and llm_result.get("intent") in ("ADMIN", "NAV", "ACTION"):
                return llm_result

        # Fallback to keywords (also used when LLM skipped for VRAM safety)
        keyword_result = self._keyword_classify(text)
        if keyword_result is not None:
            return keyword_result

        return {"intent": "ADMIN"}  # ultimate fallback

    def _keyword_classify(self, text):
        text_lower = text.lower()

        nav_keywords = [
            'go to', 'navigate to', 'move to', 'come to', 'come here',
            'take me to', 'follow me', 'drive to', 'walk to',
            'head to', 'bring me to', 'lead me to',
        ]
        action_keywords = [
            'whatsapp', 'send a message', 'send message', 'text my',
            'call my', 'email', 'remind me', 'set a reminder',
            'send to', 'message my',
        ]

        for kw in nav_keywords:
            if kw in text_lower:
                self.get_logger().info(f'Keyword match "{kw}" -> NAV')
                return {"intent": "NAV"}
        for kw in action_keywords:
            if kw in text_lower:
                self.get_logger().info(f'Keyword match "{kw}" -> ACTION')
                return {"intent": "ACTION"}

        if text.rstrip().endswith('?'):
            self.get_logger().info('Question mark detected -> ADMIN')
            return {"intent": "ADMIN"}

        question_starters = [
            'what ', 'when ', 'where ', 'how ', 'who ', 'which ',
            'is there', 'are there', 'do you', 'can you tell',
            'tell me about', 'i want to know',
        ]
        for qs in question_starters:
            if text_lower.startswith(qs) or text_lower.startswith(qs.lstrip()):
                self.get_logger().info(f'Question pattern "{qs}" -> ADMIN')
                return {"intent": "ADMIN"}

        return None

    def _llm_classify(self, text):
        # Build optional vision context if detection data is available
        vision_ctx = ''
        if self._last_detection and isinstance(self._last_detection, dict):
            det_summary = self._last_detection.get('raw', str(self._last_detection))
            if isinstance(det_summary, str) and det_summary.strip():
                vision_ctx = f'\nThe robot currently sees: {det_summary}\n'

        prompt = f"""Classify this school robot request into exactly one: ADMIN, NAV, or ACTION.

ADMIN = questions about school info, fees, schedule, academics, facilities
NAV = physical movement: go somewhere, navigate, come here
ACTION = send message, whatsapp, call, email, reminder
{vision_ctx}
"{text}"
JSON:"""
        cleaned = ""  # Initialize before try so regex fallback can safely check it
        try:
            r = requests.post(self.ollama_url, json={
                "model": self.router_model,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                # Unload the router model immediately after classification.
                # The 270M Gemma 3 cold-starts in ~1-2s on Jetson CUDA,
                # which is acceptable.  Any keep_alive > 0 risks both
                # models (router + Llama 3.2) occupying VRAM simultaneously,
                # since brain routes to admin_node within milliseconds.
                # On 8 GB shared memory: 2 GB OS + 1.5 GB Whisper + 0.5 GB
                # YOLO + 0.5 GB Gemma + 2-4 GB Llama = 7-9 GB → OOM.
                "keep_alive": 0,
                "options": {
                    "temperature": 0.0,
                    "num_gpu": self._ollama_num_gpu,
                    "num_ctx": self._ollama_num_ctx,
                }
            }, timeout=self.router_timeout)
            raw_response = r.json()['response']
            # Strip markdown code fences that local LLMs sometimes wrap around JSON
            cleaned = raw_response.replace('```json', '').replace('```', '').strip()
            result = json.loads(cleaned)
            intent = result.get("intent", "").upper()
            if intent in ("ADMIN", "NAV", "ACTION"):
                self.get_logger().info(f'LLM classified -> {intent}')
                return {"intent": intent}
            self.get_logger().warning(f'LLM returned unknown intent "{intent}", defaulting to ADMIN')
        except requests.ConnectionError:
            self.get_logger().error('Ollama connection failed')
        except requests.Timeout:
            self.get_logger().error('Ollama timed out')
        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().error(f'Failed to parse LLM response: {e}')
            # Regex fallback: extract intent even from malformed JSON
            if cleaned:
                m = re.search(r'"intent"\s*:\s*"?([A-Z]+)"?', cleaned)
                if m and m.group(1) in ("ADMIN", "NAV", "ACTION"):
                    self.get_logger().info(f'Regex fallback -> {m.group(1)}')
                    return {"intent": m.group(1)}
                self.get_logger().warning(
                    f'Regex fallback also failed on: {cleaned[:80]}'
                )

        # Return None so classify_intent() falls through to keyword fallback
        self.get_logger().info('LLM classification failed — falling back to keywords')
        return None

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _on_detection(self, msg):
        try:
            self._last_detection = json.loads(msg.data)
        except Exception:
            self._last_detection = {'raw': msg.data}

    def _say(self, text):
        """Publish direct brain responses to /robot_speech."""
        msg = String()
        msg.data = text
        self.speech_pub.publish(msg)

    def _handle_response(self, msg_data: str, pending_dict: dict, intent_name: str):
        """Pair an incoming response with its pending question via UUID lookup.

        Supports two message formats:

        1. **Streaming** (sentence-boundary chunks from admin_node):
           - ``is_final=False``: ``chunk`` contains a sentence to speak
             immediately.  History and pending state are NOT touched.
           - ``is_final=True``:  ``answer`` contains the full concatenated
             response for history recording.  ``chunk`` is empty — do NOT
             speak this message (sentences were already spoken).

        2. **Legacy** (single complete response, no ``is_final`` key):
           Treated as a final response — spoken and recorded in one step.
           Backward compatible with older node versions.
        """
        # Parse response — expect JSON with streaming or legacy format
        is_streaming = False
        try:
            payload = json.loads(msg_data)
            chunk = payload.get('chunk', '').strip()
            answer = payload.get('answer', '').strip()
            query_id = payload.get('query_id', '')
            is_final = payload.get('is_final', True)  # Legacy msgs default final
            # 'is_final' key present → streaming protocol; absent → legacy
            is_streaming = 'is_final' in payload
        except (json.JSONDecodeError, AttributeError):
            # Fallback: treat entire message as the answer (backward compat)
            chunk = ''
            answer = msg_data.strip()
            query_id = ''
            is_final = True

        # ── Streaming chunk: speak immediately, don't touch history ──────
        if not is_final:
            if chunk:
                self._say(chunk)
            return

        # ── Final payload: record history, pop pending UUID ──────────────
        if not answer:
            return

        with self._history_lock:
            # Expire stale questions that never received a response
            now = time.time()
            stale_ids = [qid for qid, (ts, _) in pending_dict.items()
                         if now - ts > self._pending_timeout]
            for qid in stale_ids:
                stale_ts, stale_q = pending_dict.pop(qid)
                self.get_logger().warning(
                    f'Expired stale pending [{intent_name}] '
                    f'({now - stale_ts:.0f}s old): {stale_q[:60]}'
                )

            # UUID-based lookup — prevents mispairing after timeout
            if query_id and query_id in pending_dict:
                _, matched_question = pending_dict.pop(query_id)
                self.history.append((matched_question, answer))
            elif not query_id and pending_dict:
                # Backward compat: no UUID, match oldest pending entry
                oldest_id = min(pending_dict, key=lambda k: pending_dict[k][0])
                _, matched_question = pending_dict.pop(oldest_id)
                self.history.append((matched_question, answer))
                self.get_logger().warning(
                    f'[{intent_name}] response without query_id — '
                    f'paired with oldest pending question (FIFO fallback)'
                )

        # Speak the answer — but only for legacy (non-streaming) responses.
        # Streaming final payloads (is_streaming=True) have chunk="" because
        # each sentence was already spoken individually via _say(chunk) above.
        if not is_streaming:
            self._say(answer)

    def _on_admin_response(self, msg):
        """Handle RAG response from admin_node (/admin_response)."""
        self._handle_response(msg.data, self._pending_admin, 'ADMIN')

    def _on_action_response(self, msg):
        """Handle action response from action_node (/action_response)."""
        self._handle_response(msg.data, self._pending_action, 'ACTION')

    # ── Main routing callback ──────────────────────────────────────────────────

    def listener_callback(self, msg):
        """ROS2 subscription callback — returns immediately, work done in thread.

        Blocking the executor here (e.g. with requests.post to Ollama) would
        freeze ALL subscriptions on this node for the duration of the LLM call.
        We deduplicate synchronously (cheap), then hand off to a daemon thread.
        """
        user_input = msg.data.strip()
        if not user_input:
            return

        # ── Deduplication (cheap — done in callback thread) ────────────────────
        now = time.time()
        if (user_input == self._last_processed_text and
                now - self._last_processed_time < self._debounce_secs):
            self.get_logger().debug(
                f'Deduplicated (within {self._debounce_secs}s): {user_input[:60]}'
            )
            return

        self._last_processed_text = user_input
        self._last_processed_time = now

        # ── Hand off to single worker thread (non-blocking) ─────────────────
        # classify_intent() may call requests.post(Ollama) which can take
        # several seconds. Enqueuing keeps the ROS2 executor free and prevents
        # unbounded thread creation under burst traffic.
        # put_nowait: if the queue is full (maxsize=3), silently drop the
        # command rather than blocking — prevents replaying stale backlog.
        try:
            self._route_queue.put_nowait(user_input)
        except queue.Full:
            self.get_logger().warning(
                f'Routing queue full — dropping: {user_input[:60]}'
            )

    def _worker_loop(self):
        """Single background thread that drains the routing queue."""
        while True:
            user_input = self._route_queue.get()
            try:
                self._route(user_input)
            except Exception as e:
                self.get_logger().error(f'Routing error: {e}')

    def _route(self, user_input: str):
        """Classify intent and publish — runs in the single worker thread.

        ADMIN and ACTION intents are added to their respective pending queues.
        Responses arrive on /admin_response and /action_response, where
        intent-specific callbacks pair each answer with the correct question
        and forward to /robot_speech.
        NAV and wake-word responses are generated instantly by brain_node
        itself (_say), so their history is recorded inline.
        """
        self.get_logger().info(f'Heard: {user_input}')

        # Handle wake-word-only triggers (user said just "Hey AISHA")
        if user_input == 'wake_word_triggered':
            self.get_logger().info('Wake word acknowledged — greeting user')
            response = "Yes, how can I help you?"
            self._say(response)
            with self._history_lock:
                self.history.append((user_input, response))
            return

        decision = self.classify_intent(user_input)
        intent = decision.get("intent", "ADMIN")

        out_msg = String()
        if intent == "ADMIN":
            self.get_logger().info("Route -> ADMIN (Knowledge Base)")
            query_id = str(uuid.uuid4())
            with self._history_lock:
                self._pending_admin[query_id] = (time.time(), user_input)
                history_list = [{"user": u, "assistant": a} for u, a in self.history]
            out_msg.data = json.dumps({
                "details": user_input,
                "history": history_list,
                "query_id": query_id,
            })
            self.admin_pub.publish(out_msg)

        elif intent == "NAV":
            # ── NAV intent → waypoint_resolver_node ───────────────────────
            # Architecture:
            #   1. brain_node publishes user_input to /nav_goal (String).
            #   2. waypoint_resolver_node subscribes, resolves the location
            #      name to map coordinates via nav_locations.json, and sends
            #      a NavigateToPose action goal to Nav2.
            #   3. Nav2 plans and publishes /cmd_vel → mecanum_driver (Pi 4b).
            #   4. waypoint_resolver publishes speech feedback on /robot_speech
            #      ("Navigating to...", "I don't know where...", "Arrived at...").
            # brain_node does NOT publish its own speech here — the resolver
            # handles all user feedback to avoid duplicate/conflicting messages.
            self.get_logger().info("Route -> NAV")
            out_msg.data = user_input
            self.nav_pub.publish(out_msg)

        else:
            self.get_logger().info("Route -> ACTION")
            query_id = str(uuid.uuid4())
            with self._history_lock:
                self._pending_action[query_id] = (time.time(), user_input)
            out_msg.data = json.dumps({
                "query_id": query_id,
                "details": user_input,
            })
            self.action_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
