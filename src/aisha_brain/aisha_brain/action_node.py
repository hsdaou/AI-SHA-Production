import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import re
import queue
import threading


class ActionNode(Node):
    def __init__(self):
        super().__init__('ai_sha_action')
        self.subscription = self.create_subscription(
            String, '/action_request', self.handle_action, 10)
        # Publish to /action_response — brain_node subscribes here to pair
        # the answer with the correct pending ACTION question, then forwards
        # to /robot_speech.  Avoids the intent race condition where a fast
        # ACTION response steals a slow ADMIN question from the history FIFO.
        self.speech_pub = self.create_publisher(String, '/action_response', 10)

        # Single worker thread — subprocess.run can block for 30s+
        self._action_queue = queue.Queue()
        threading.Thread(target=self._action_worker_loop, daemon=True).start()

        self.get_logger().info('AI-SHA Action Node Online')

    def _say(self, text, query_id=''):
        """Publish feedback to /action_response with query UUID for pairing.

        brain_node uses the query_id to match this response with the
        original question, preventing the FIFO desync race condition.
        """
        msg = String()
        if query_id:
            msg.data = json.dumps({"answer": text, "query_id": query_id})
        else:
            msg.data = text
        self.speech_pub.publish(msg)

    def handle_action(self, msg):
        """ROS 2 callback — enqueues command and returns instantly."""
        raw = msg.data.strip()
        if not raw:
            return
        # Parse JSON payload with query_id; fall back to plain text
        try:
            payload = json.loads(raw)
            command = payload.get('details', raw).strip()
            query_id = payload.get('query_id', '')
        except (json.JSONDecodeError, AttributeError):
            command = raw
            query_id = ''
        if command:
            self._action_queue.put((command, query_id))

    def _action_worker_loop(self):
        """Single background thread that processes action requests sequentially."""
        while True:
            command, query_id = self._action_queue.get()
            try:
                self._dispatch_action(command, query_id)
            except Exception as e:
                self.get_logger().error(f'Action worker error: {e}')

    def _dispatch_action(self, command: str, query_id: str = ''):
        """Route and execute the action — runs in the worker thread."""
        command_lower = command.lower()
        self.get_logger().info(f'Action Request: {command}')

        if "whatsapp" in command_lower or "message" in command_lower:
            self.send_whatsapp(command, query_id)
        elif "calendar" in command_lower or "schedule" in command_lower:
            self._say("Calendar integration is coming soon.", query_id)
            self.get_logger().info("Calendar integration not yet implemented")
        else:
            self._say("I'm not sure how to handle that action yet.", query_id)
            self.get_logger().warning(f'Unrecognized action: {command}')

    def send_whatsapp(self, text, query_id=''):
        try:
            # Extract phone number (UAE format: 971XXXXXXXXX, 10-12 digits)
            phone_match = re.search(r'\b(971\d{8,10})\b', text)
            if not phone_match:
                self._say("I couldn't find a valid phone number. Please include a number starting with 971.", query_id)
                self.get_logger().warning("No valid phone number found in command")
                return

            phone = phone_match.group(1)

            # Extract message content
            message = ""
            text_lower = text.lower()
            for keyword in ["saying", "say", "that says", "message"]:
                if keyword in text_lower:
                    idx = text_lower.index(keyword) + len(keyword)
                    message = text[idx:].strip().strip('"').strip("'")
                    break

            if not message:
                message = "Message from AI-SHA"

            self.get_logger().info(f'Sending to {phone}: {message[:50]}')

            result = subprocess.run(
                ["npx", "mudslide", "send", phone, message],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                self._say("Message sent successfully.", query_id)
                self.get_logger().info("WhatsApp message sent")
            else:
                stderr_lower = (result.stderr or '').lower()
                if any(kw in stderr_lower for kw in ('auth', 'login', 'qr', 'not logged in', 'session')):
                    self._say("My WhatsApp session has expired. Please ask the administrator to re-authenticate me.", query_id)
                    self.get_logger().error(f'WhatsApp auth failure: {result.stderr}')
                else:
                    self._say("I had trouble sending that message. Please try again.", query_id)
                    self.get_logger().error(f'mudslide error: {result.stderr}')

        except subprocess.TimeoutExpired:
            self._say("The message is taking too long to send. Please check the connection.", query_id)
            self.get_logger().error("WhatsApp send timed out")
        except FileNotFoundError:
            self._say("WhatsApp messaging tool is not installed.", query_id)
            self.get_logger().error("npx/mudslide not found")
        except Exception as e:
            self._say("Something went wrong sending the message.", query_id)
            self.get_logger().error(f'WhatsApp error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ActionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
