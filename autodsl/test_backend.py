"""Tests for backend.py.

No network. What is pinned here is the request shape per protocol, the retry
policy, and that the credential never leaks into anything the loop writes down.
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import backend  # noqa: E402

KEY = "sk-not-a-real-key-0000"


class KeyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.env = mock.patch.dict("os.environ", {}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        __import__("os").environ.pop("BREVIS_LLM_API_KEY", None)

    def write(self, text: str) -> pathlib.Path:
        path = self.dir / "llm.credential"
        path.write_text(text)
        return path

    def test_reads_a_bare_key(self):
        self.assertEqual(KEY, backend.load_api_key(self.write(KEY + "\n")))

    def test_reads_a_name_equals_value_line(self):
        self.assertEqual(KEY, backend.load_api_key(self.write(f"API_KEY={KEY}\n")))

    def test_reads_a_json_object(self):
        self.assertEqual(KEY, backend.load_api_key(
            self.write(json.dumps({"api_key": KEY}))))

    def test_skips_comments_and_blank_lines(self):
        self.assertEqual(KEY, backend.load_api_key(
            self.write(f"# closeai proxy\n\n{KEY}\n")))

    def test_environment_wins_over_the_file(self):
        with mock.patch.dict("os.environ", {"BREVIS_LLM_API_KEY": "sk-from-env"}):
            self.assertEqual("sk-from-env", backend.load_api_key(self.write(KEY)))

    def test_a_missing_file_says_where_to_put_one(self):
        with self.assertRaisesRegex(backend.BackendError, "git-ignored"):
            backend.load_api_key(self.dir / "absent")

    def test_an_empty_file_is_rejected(self):
        with self.assertRaises(backend.BackendError):
            backend.load_api_key(self.write("   \n"))


class RequestShapeTests(unittest.TestCase):
    def capture(self, **kwargs) -> tuple[str, dict, dict]:
        seen = {}

        def fake_post(url, headers, body):
            seen.update(url=url, headers=headers, body=body)
            return {
                "content": [{"type": "text", "text": "hi"}],
                "choices": [{"message": {"content": "hi"}}],
            }

        client = backend.Backend(api_key=KEY, **kwargs)
        with mock.patch.object(backend.Backend, "_post", side_effect=fake_post):
            self.assertEqual("hi", client.complete("sys", "usr"))
        return seen["url"], seen["headers"], seen["body"]

    def test_anthropic_uses_the_native_path_and_header(self):
        url, headers, body = self.capture(model="claude-opus-5")
        self.assertEqual("https://api.openai-proxy.org/anthropic/v1/messages", url)
        self.assertEqual(KEY, headers["x-api-key"])
        self.assertEqual("2023-06-01", headers["anthropic-version"])
        self.assertEqual("sys", body["system"])
        self.assertEqual([{"role": "user", "content": "usr"}], body["messages"])

    def test_openai_compatible_models_use_the_v1_chat_path(self):
        url, headers, body = self.capture(model="qwen3.7-max")
        self.assertEqual("https://api.openai-proxy.org/v1/chat/completions", url)
        self.assertEqual(f"Bearer {KEY}", headers["Authorization"])
        self.assertEqual("system", body["messages"][0]["role"])

    def test_temperature_is_omitted_unless_asked_for(self):
        """Newer reasoning models reject the field outright."""
        _, _, body = self.capture(model="claude-opus-5")
        self.assertNotIn("temperature", body)
        _, _, body = self.capture(model="qwen3.7-max", temperature=0.0)
        self.assertEqual(0.0, body["temperature"])

    def test_an_unknown_model_defaults_to_the_openai_protocol(self):
        url, _, _ = self.capture(model="some-new-model")
        self.assertIn("/v1/chat/completions", url)

    def test_describe_never_includes_the_key(self):
        described = json.dumps(backend.Backend(api_key=KEY).describe())
        self.assertNotIn(KEY, described)
        self.assertIn("claude-opus-5", described)


class TransportTests(unittest.TestCase):
    def client(self) -> backend.Backend:
        return backend.Backend(api_key=KEY, model="qwen3.7-max", retries=3)

    def http_error(self, code: int) -> urllib.error.HTTPError:
        return urllib.error.HTTPError(
            "u", code, "boom", {}, io.BytesIO(b'{"error":"boom"}')
        )

    def test_a_rate_limit_is_retried_then_surfaced(self):
        with mock.patch.object(backend.time, "sleep"), \
             mock.patch.object(backend.urllib.request, "urlopen",
                               side_effect=self.http_error(429)) as urlopen:
            with self.assertRaisesRegex(backend.BackendError, "HTTP 429"):
                self.client().complete("s", "u")
        self.assertEqual(3, urlopen.call_count)

    def test_a_client_error_is_not_retried(self):
        with mock.patch.object(backend.time, "sleep"), \
             mock.patch.object(backend.urllib.request, "urlopen",
                               side_effect=self.http_error(400)) as urlopen:
            with self.assertRaisesRegex(backend.BackendError, "HTTP 400"):
                self.client().complete("s", "u")
        self.assertEqual(1, urlopen.call_count)

    def test_an_empty_response_is_an_error_not_an_empty_macro_list(self):
        with mock.patch.object(backend.Backend, "_post", return_value={"choices": []}):
            with self.assertRaisesRegex(backend.BackendError, "no choices"):
                self.client().complete("s", "u")

    def test_a_reply_that_is_all_thinking_names_the_cause(self):
        """Reasoning models can spend the entire budget before emitting text."""
        response = {"stop_reason": "max_tokens",
                    "content": [{"type": "thinking", "thinking": "", "signature": "x"}]}
        with self.assertRaisesRegex(backend.BackendError, "raise max_tokens"):
            backend._extract_anthropic(response)

    def test_anthropic_text_blocks_are_concatenated(self):
        response = {"content": [{"type": "thinking", "thinking": "hm"},
                                {"type": "text", "text": "a"},
                                {"type": "text", "text": "b"}]}
        self.assertEqual("ab", backend._extract_anthropic(response))


class ScriptedTests(unittest.TestCase):
    def test_replies_are_returned_in_order_and_calls_recorded(self):
        scripted = backend.ScriptedBackend(replies=["one", "two"])
        self.assertEqual("one", scripted.complete("s", "u1"))
        self.assertEqual("two", scripted.complete("s", "u2"))
        self.assertEqual([("s", "u1"), ("s", "u2")], scripted.calls)
        with self.assertRaises(backend.BackendError):
            scripted.complete("s", "u3")


if __name__ == "__main__":
    unittest.main()
