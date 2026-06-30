"""Unit tests for BrainNode — keyword router."""

import json
import pytest
from unittest.mock import MagicMock, patch
from std_msgs.msg import String

from .conftest import make_string_msg, capture_published, wait_for_call


# ---------------------------------------------------------------------------
# _keyword_classify
# ---------------------------------------------------------------------------

class TestKeywordClassify:

    def test_nav_go_to(self, brain_node):
        node = brain_node
        assert node._keyword_classify('go to the library') == {'intent': 'NAV'}

    def test_nav_navigate_to(self, brain_node):
        node = brain_node
        assert node._keyword_classify('navigate to the clinic') == {'intent': 'NAV'}

    def test_nav_come_here(self, brain_node):
        node = brain_node
        assert node._keyword_classify('come here please') == {'intent': 'NAV'}

    def test_nav_take_me_to(self, brain_node):
        node = brain_node
        assert node._keyword_classify('take me to the cafeteria') == {'intent': 'NAV'}

    def test_action_whatsapp(self, brain_node):
        node = brain_node
        assert node._keyword_classify('send a whatsapp to my dad') == {'intent': 'ACTION'}

    def test_action_send_message(self, brain_node):
        node = brain_node
        assert node._keyword_classify('send a message to my mom') == {'intent': 'ACTION'}

    def test_action_remind_me(self, brain_node):
        node = brain_node
        assert node._keyword_classify('remind me about the meeting') == {'intent': 'ACTION'}

    def test_admin_question_mark(self, brain_node):
        node = brain_node
        assert node._keyword_classify('What are the school fees?') == {'intent': 'ADMIN'}

    def test_admin_what_prefix(self, brain_node):
        node = brain_node
        assert node._keyword_classify('what is the school phone number') == {'intent': 'ADMIN'}

    def test_admin_where_prefix(self, brain_node):
        node = brain_node
        assert node._keyword_classify('where is the swimming pool') == {'intent': 'ADMIN'}

    def test_admin_tell_me(self, brain_node):
        node = brain_node
        assert node._keyword_classify('tell me about admissions') == {'intent': 'ADMIN'}

    def test_ambiguous_returns_none(self, brain_node):
        node = brain_node
        # Short ambiguous input — no keyword match, not a question
        result = node._keyword_classify('hello')
        assert result is None

    def test_case_insensitive(self, brain_node):
        node = brain_node
        assert node._keyword_classify('GO TO THE POOL') == {'intent': 'NAV'}


# ---------------------------------------------------------------------------
# listener_callback routing
# ---------------------------------------------------------------------------

class TestListenerCallback:

    def test_empty_input_ignored(self, brain_node):
        node = brain_node
        admin_pub = capture_published(node, 'admin_pub')
        node.listener_callback(make_string_msg('   '))
        admin_pub.publish.assert_not_called()

    def test_routes_to_admin(self, brain_node):
        node = brain_node
        admin_pub = capture_published(node, 'admin_pub')
        node.listener_callback(make_string_msg('what is the school phone number'))
        assert wait_for_call(admin_pub.publish), 'admin_pub.publish never called'
        admin_pub.publish.assert_called_once()
        payload = json.loads(admin_pub.publish.call_args[0][0].data)
        assert payload['details'] == 'what is the school phone number'

    def test_routes_to_nav(self, brain_node):
        node = brain_node
        nav_pub = capture_published(node, 'nav_pub')
        speech_pub = capture_published(node, 'speech_pub')
        node.listener_callback(make_string_msg('go to the library'))
        assert wait_for_call(nav_pub.publish), 'nav_pub.publish never called'
        nav_pub.publish.assert_called_once()
        # brain_node does NOT speak on NAV — waypoint_resolver_node owns all
        # user feedback ("Navigating to..."/"I don't know where...") to avoid
        # duplicate/conflicting speech (see brain_node._route NAV branch).
        speech_pub.publish.assert_not_called()

    def test_routes_to_action(self, brain_node):
        node = brain_node
        action_pub = capture_published(node, 'action_pub')
        node.listener_callback(make_string_msg('send a whatsapp'))
        assert wait_for_call(action_pub.publish), 'action_pub.publish never called'
        action_pub.publish.assert_called_once()
