from dataclasses import replace

import pytest
import test_memory as legacy

from ours_memory import Caller, ReadRequest
from ours_memory.governance import AccessGraph, GovernedMemoryService
from ours_memory.models import AccessDenied


class TestGovernance(legacy.MemoryTests):
    # Only inherit helpers, not duplicate all legacy test methods in collection below.
    def setUp(self):
        super().setUp()
        self.graph = AccessGraph(self.store)
        self.shared = self.graph.provision(
            "u1", "team", ["dialogue", "planner"], ["dialogue", "planner"], ["dialogue"]
        )
        self.private = self.graph.provision("u1", "team", ["dialogue"], ["dialogue"], ["dialogue"])
        self.planner = self.graph.provision("u1", "team", ["planner"], ["planner"], ["planner"])
        self.service = GovernedMemoryService(self.settings, self.backend, self.model, self.store, self.graph)

    def test_revocation_prevents_search_id_history_and_plan(self):
        card = self.enqueue(self.event())
        reader = Caller("u1", "team", "planner")
        self.route([card], at=legacy.TIME, caller=reader)
        self.service.register_plan("p1", "task2", reader)
        state_id = self.store.states()[0]["id"]
        before = len(self.backend.search_policies)
        self.graph.change(self.shared, "planner", read=False)
        result = self.service.read_memory(ReadRequest("new", "hours", legacy.TIME, "exam"), reader)
        assert not result.cards and not result.states
        assert len(self.backend.search_policies) == before
        with pytest.raises(AccessDenied):
            self.service.get_card(card, reader)
        with pytest.raises(AccessDenied):
            self.service.get_history(state_id, reader)
        assert not self.service.check_plan_dependencies("p1", reader)["ready"]

    def test_restart_does_not_restore_revoked_edges(self):
        self.graph.change(self.shared, "planner", read=False)
        self.graph.provision("u1", "team", ["dialogue", "planner"], ["dialogue", "planner"])
        assert not self.graph.permits(self.shared, Caller("u1", "team", "planner"))

    def test_cannot_grant_new_audience_or_cross_user(self):
        with pytest.raises(AccessDenied):
            self.graph.change(self.private, "planner", read=True)
        assert not self.graph.permits(self.shared, Caller("other", "team", "planner"))

    def test_read_permission_is_not_write_or_delete_permission(self):
        self.graph.change(self.shared, "planner", write=False)
        self.enqueue(self.event())
        who = Caller("u1", "team", "planner")
        with pytest.raises(AccessDenied):
            self.service.ingest_event(replace(self.event("x"), writer_agent_id="planner"), who)
        with pytest.raises(AccessDenied):
            self.service.forget(["e1"], who)

    def test_revoke_source_hides_preexisting_derived_copy(self):
        self.enqueue(self.event("source"))
        self.caller = Caller("u1", "team", "planner")
        derived = replace(
            self.event("copy", readers=("planner",), roots=("source",), speaker="agent"),
            writer_agent_id="planner",
        )
        copy_id = self.enqueue(derived, kind="retelling")
        assert self.service.get_card(copy_id, self.caller)
        self.graph.change(self.shared, "planner", read=False)
        with pytest.raises(AccessDenied):
            self.service.get_card(copy_id, self.caller)
        with pytest.raises(AccessDenied):
            self.service.ingest_event(
                replace(derived, event_id="write-again", source_root_ids=()), self.caller
            )

    def test_owner_can_forget_and_tombstone_blocks_replay(self):
        self.enqueue(self.event())
        result = self.service.forget(["e1"], self.caller)
        assert result["status"] == "forgotten"
        with pytest.raises(AccessDenied):
            self.service.ingest_event(self.event(), self.caller)

    def test_exact_disclosure_and_revocation(self):
        self.enqueue(self.event("secret", readers=("dialogue",)))
        release = replace(self.event("release", roots=("secret",)), disclosure_id="grant-1")
        with pytest.raises(AccessDenied):
            self.service.ingest_event(release, self.caller)
        self.graph.approve_disclosure(release)
        changed = replace(release, text="different private content")
        with pytest.raises(AccessDenied):
            self.service.ingest_event(changed, self.caller)
        card = self.enqueue(release)
        reader = Caller("u1", "team", "planner")
        assert self.service.get_card(card, reader)
        self.graph.revoke_disclosure("grant-1")
        with pytest.raises(AccessDenied):
            self.service.get_card(card, reader)


# unittest subclasses otherwise collect all inherited legacy methods again.
for _name in dir(legacy.MemoryTests):
    if _name.startswith("test_") and _name not in TestGovernance.__dict__:
        setattr(TestGovernance, _name, None)
