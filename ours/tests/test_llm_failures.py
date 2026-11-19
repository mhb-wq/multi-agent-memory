from __future__ import annotations

import io
import json
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from ours_memory import Settings
from ours_memory.llm import CompatibleJsonModel
from ours_memory.models import InputTooLarge, ModelOutputError, OutputTruncated
from ours_memory.store import Store


class ModelFailures(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)
        self.model = CompatibleJsonModel(Settings(api_key_env="LOCAL_FIXTURE_KEY", retries=2), self.store)
        env = patch.dict("os.environ", {"LOCAL_FIXTURE_KEY": "fixture-key"})
        env.start()
        self.addCleanup(env.stop)

    def invoke(self, content, finish="stop"):
        envelope = {
            "choices": [{"finish_reason": finish, "message": {"content": content}}],
            "usage": {"prompt_tokens": 12},
        }
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value = io.BytesIO(json.dumps(envelope).encode())
        with patch("urllib.request.build_opener", return_value=opener):
            try:
                return self.model.complete("route", {})
            finally:
                self.assertEqual(opener.open.call_count, 1)

    def test_truncation_has_specific_error_and_no_identical_paid_retry(self):
        with self.assertRaises(OutputTruncated):
            self.invoke('{"partial":', "length")
        self.assertEqual(self.store.call_records()[-1]["error"], "OutputTruncated")

    def test_json_code_fence_is_accepted(self):
        self.assertEqual(self.invoke('```json\n{"cards": []}\n```'), {"cards": []})

    def test_null_content_is_diagnosed_without_retry(self):
        with self.assertRaises(ModelOutputError):
            self.invoke(None)

    def test_malformed_json_is_diagnosed_without_retry(self):
        with self.assertRaises(ModelOutputError):
            self.invoke("not json")

    def test_retryable_http_error_is_retried_and_logged(self):
        opener = MagicMock()
        response = MagicMock()
        response.__enter__.return_value = io.BytesIO(
            json.dumps({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}).encode()
        )
        opener.open.side_effect = [urllib.error.HTTPError("http://fixture", 429, "busy", {}, None), response]
        with patch("urllib.request.build_opener", return_value=opener), patch("time.sleep"):
            self.assertEqual(self.model.complete("route", {}), {})
        self.assertEqual(opener.open.call_count, 2)
        self.assertEqual([r["success"] for r in self.store.call_records()], [False, True])

    def test_answer_has_separate_input_and_output_limits(self):
        self.model.settings.max_model_input_chars = 40
        self.model.settings.answer_max_input_chars = 200
        self.model.settings.answer_max_tokens = 8192
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value = io.BytesIO(json.dumps({
            "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}).encode())
        payload = {"text": "x" * 60}
        with patch("urllib.request.build_opener", return_value=opener):
            self.assertEqual(self.model.complete("answer", payload), {})
            request = opener.open.call_args.args[0]
            self.assertEqual(json.loads(request.data)["max_tokens"], 8192)
            with self.assertRaises(InputTooLarge):
                self.model.complete("focused_cards", payload)

    def test_focused_state_uses_increased_output_limit(self):
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value = io.BytesIO(json.dumps({
            "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}).encode())
        with patch("urllib.request.build_opener", return_value=opener):
            self.model.complete("focused_state", {})
        request = opener.open.call_args.args[0]
        self.assertEqual(json.loads(request.data)["max_tokens"], 65536)

    def test_joint_repair_records_its_own_usage_and_elapsed_time(self):
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value = io.BytesIO(json.dumps({
            "model": "fixture-model", "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}}).encode())
        with patch("urllib.request.build_opener", return_value=opener):
            self.model.complete("focused_joint", {"validation_feedback": {"code": "missing_fields"}})
        record = self.store.call_records()[-1]
        self.assertTrue(record["is_repair"])
        self.assertEqual(record["usage"]["total_tokens"], 15)
        self.assertGreaterEqual(record["elapsed_seconds"], 0)
