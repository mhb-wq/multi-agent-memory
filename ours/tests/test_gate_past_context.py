from pathlib import Path
import sys
from types import SimpleNamespace


def test_shared_replay_uses_only_bounded_earlier_messages():
    directory = Path(__file__).resolve().parents[1] / "experiments" / "_diag_20261027"
    sys.path.insert(0, str(directory))
    try:
        from run_gatemem import OursGateAgent
        agent = OursGateAgent(llm_router=None, out=Path("unused"), domain="medical", access_mode="shared_pool")
        agent.prior_messages, agent.message_count = [], 0
        seen = []
        agent._ingest_turn = lambda turn: seen.append((turn.turn_id, list(agent.prior_messages)))
        agent._sleep = lambda day: None
        for n in range(1, 14):
            agent.ingest(SimpleNamespace(turn_id=f"t{n:03}", text=f"message {n}",
                                         speaker_principal_id="person", speaker_role="user", timestamp=None))
        assert [x["id"] for x in seen[-1][1]] == [f"t{n:03}" for n in range(3, 13)]
        assert all(x["id"] != turn for turn, context in seen for x in context)
        assert all(sum(len(x["text"]) for x in context) <= 6000 for _, context in seen)
    finally:
        sys.path.remove(str(directory))
