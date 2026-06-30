import re
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
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
        # Timeout for pending questions.  If admin_node hangs (e.g. Ollama
        # crash during RAG, network partition), stale entries are expired
        # after this duration so response-pairing state can't leak.  30s
        # covers normal RAG inference (Llama 3.2 on Jetson responds in 5-20s).
        self._pending_timeout = 30.0        # seconds before a question expires
        self.history = deque(maxlen=5)   # (user_text, robot_answer) tuples
        # Lock protecting self.history and self._pending_* queues.
        # Response callbacks run on the ROS 2 executor thread while _route()
        # runs on the background worker thread.  Without this lock, a
        # deque.append() at maxlen (which internally poplefts the oldest
        # item) can mutate the deque mid-iteration in _route()'s list
        # comprehension, raising RuntimeError: deque mutated during iteration.
        self._history_lock = threading.Lock()

        # ── Intent routing ─────────────────────────────────────────────────────
        # Routing is deterministic keyword/pattern matching (see
        # classify_intent / _keyword_classify).  brain_node uses NO LLM router
        # and has no Ollama dependency.  (admin_node runs the local RAG answer
        # model separately for ADMIN questions.)

        # ── Single worker thread for routing (keeps the ROS callback fast) ──
        # Keyword routing is instant, but a worker thread keeps the
        # subscription callback non-blocking and serializes routing.
        # maxsize=3 bounds the buffer so stale speech can't pile up.
        self._route_queue = queue.Queue(maxsize=3)
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self.get_logger().info('AI-SHA Brain: keyword router active')

    # ── Intent classification ──────────────────────────────────────────────────

    # Emergency keywords are routed to NAV immediately, ahead of all other
    # matching, so a "Stop!" command is never delayed — critical for
    # physical safety.
    #
    # Word-boundary regex prevents false positives like "bus stop" or
    # "don't stop the music".  Additionally, only short utterances
    # (≤ 4 words) qualify — genuine panic commands are brief.
    _EMERGENCY_STOP_PATTERNS = [
        re.compile(r'\b' + w + r'\b') for w in
        ['stop', 'halt', 'freeze', 'shut up', 'emergency']
    ]

    def classify_intent(self, text):
        """Classify user intent with deterministic keyword rules (no LLM).

        Routing uses a fixed keyword / pattern set (see _keyword_classify):
        NAV and ACTION keywords, then question patterns -> ADMIN.  Short
        emergency-stop utterances pre-empt everything and route to NAV.
        Anything unmatched defaults to ADMIN (treated as a school-admin
        question).  No Ollama / LLM call is made for routing.
        """
        text_lower = text.lower().strip()
        word_count = len(text_lower.split())
        if word_count <= 4 and any(p.search(text_lower) for p in self._EMERGENCY_STOP_PATTERNS):
            self.get_logger().warning(f'EMERGENCY OVERRIDE -> NAV (stop): "{text[:40]}"')
            return {"intent": "NAV"}

        keyword_result = self._keyword_classify(text)
        if keyword_result is not None:
            return keyword_result

        return {"intent": "ADMIN"}  # default: treat as a school-admin question

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

        We deduplicate synchronously (cheap), then hand off routing to a
        daemon worker thread so the executor stays free for other callbacks.
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
        # Routing is fast, but enqueuing keeps the ROS2 executor free and
        # serializes work, preventing unbounded thread creation under burst
        # traffic.
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
