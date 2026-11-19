from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from local_relay_proxy import start_relay


class RelayAuditTest(unittest.TestCase):
    def test_records_model_usage_without_prompt_or_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "calls.jsonl"
            key = "test-key"
            response = requests.Response()
            response.status_code = 200
            response._content = json.dumps({"model": "served-model", "usage": {"total_tokens": 5},
                                            "choices": [{"finish_reason": "stop", "message": {"content": "answer"}}]}).encode()
            response.headers["Content-Type"] = "application/json"
            server, url = start_relay("https://invalid.example/v1", key, "requested", log)
            try:
                real_post = requests.post
                with patch("local_relay_proxy.requests.post", return_value=response) as upstream:
                    result = real_post(url + "/chat/completions", headers={"Authorization": "Bearer " + key},
                                       json={"model": "requested", "messages": [{"role": "user", "content": "PRIVATE-PROMPT"}]}, timeout=5)
                    self.assertEqual(result.status_code, 200)
                    self.assertEqual(upstream.call_count, 1)
                record_text = log.read_text(encoding="utf-8")
                self.assertNotIn("PRIVATE-PROMPT", record_text)
                self.assertNotIn(key, record_text)
                record = json.loads(record_text)
                self.assertEqual(record["reported_model"], "served-model")
                self.assertEqual(record["usage"]["total_tokens"], 5)
                with patch("local_relay_proxy.requests.post") as upstream:
                    result = real_post(url + "/chat/completions", headers={"Authorization": "Bearer " + key},
                                       json={"model": "different", "messages": []}, timeout=5)
                    self.assertEqual(result.status_code, 400)
                    upstream.assert_not_called()
                response._content = json.dumps({"model": "served-model", "usage": {"total_tokens": 9},
                                                "choices": [{"finish_reason": "length", "message": {"content": "partial"}}]}).encode()
                with patch("local_relay_proxy.requests.post", return_value=response):
                    result = real_post(url + "/chat/completions", headers={"Authorization": "Bearer " + key},
                                       json={"model": "requested", "messages": []}, timeout=5)
                    self.assertEqual(result.status_code, 502)
                self.assertEqual(json.loads(log.read_text(encoding="utf-8").splitlines()[-1])["delivered_status"], 502)
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
