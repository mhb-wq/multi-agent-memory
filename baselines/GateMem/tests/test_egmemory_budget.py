from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "EGMemory"))

from groupmem.memory.reader import ReadTools, run_qa_agentloop
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.agents.egmemory_budget import bounded_read_tools


class FixtureReader(ReadTools):
    def _passage(self, mem):
        return mem.memory_id

    def _decision_candidate(self, mid, exact):
        return mid

    def search_memory(self, query, *args, **kwargs):
        return self._fmt_scored([str(i) for i in range(15)], cap=15)

    def lookup_phrase(self, phrase, *args, **kwargs):
        return self._fmt_scored([str(i) for i in range(10, 30)], cap=12)

    def get_current_decision(self, topic):
        return self._scoped_decisions([str(i) for i in range(15, 30)], True)

    def resolve_date(self, msg_id, phrase, anchor="msg"):
        return f"date for {msg_id}"


def reader():
    memories = {str(i): SimpleNamespace(memory_id=str(i), timestamp=i) for i in range(40)}
    return FixtureReader(SimpleNamespace(memories=memories), SimpleNamespace(), max_passages=12)


class EGMemoryBudgetTest(unittest.TestCase):
    def test_multiple_tools_share_one_budget_and_reset_each_question(self):
        tools = bounded_read_tools(reader(), 20)
        tools.new_session()
        tools.search_memory("first")
        self.assertEqual(len(tools._seen), 15)
        tools.get_current_decision("topic")
        self.assertEqual(len(tools._seen), 20)
        self.assertIn("budget exhausted", tools.lookup_phrase("second"))
        self.assertEqual(len(tools._seen), 20)
        self.assertIn("First retrieve", tools.resolve_date("35", "tomorrow"))
        self.assertEqual(tools.resolve_date("1", "tomorrow"), "date for 1")
        tools.new_session()
        tools.lookup_phrase("next question")
        self.assertEqual(len(tools._seen), 12)

    def test_dedup_and_rewrapping_do_not_expand_budget(self):
        original = reader()
        tools = bounded_read_tools(bounded_read_tools(original, 20), 3)
        tools.new_session()
        tools._fmt_scored(["0", "0", "1", "2", "3"], cap=15)
        self.assertEqual(tools._seen, {"0", "1", "2"})
        self.assertEqual(original._seen, set())
        with self.assertRaises(ValueError):
            bounded_read_tools(original, 0)

    def test_agent_loop_limits_evidence_across_multiple_rounds(self):
        tools = bounded_read_tools(reader(), 20)
        steps = iter([
            {"content": "", "tool_calls": [{"id": "a", "function": {"name": "search_memory", "arguments": '{"query":"x"}'}}]},
            {"content": "", "tool_calls": [{"id": "b", "function": {"name": "get_current_decision", "arguments": '{"topic":"x"}'}}]},
            {"content": "", "tool_calls": [{"id": "c", "function": {"name": "lookup_phrase", "arguments": '{"phrase":"x"}'}}]},
            {"content": "done"},
        ])
        _, trace = run_qa_agentloop(lambda messages, schemas: next(steps), "system", "question", "", tools)
        self.assertEqual(len(tools._seen), 20)
        self.assertIn("budget exhausted", trace[2]["ret"])


if __name__ == "__main__":
    unittest.main()
