"""Joint maintenance wire IDs and deterministic proposal validation."""
from copy import deepcopy

from .models import ValidationError


class JointValidationError(ValidationError):
    def __init__(self, code, detail, *, repairable=True):
        super().__init__(detail)
        self.validation_code = code
        self.repairable = repairable


def fail(code, detail, *, repairable=True):
    raise JointValidationError(code, detail, repairable=repairable)


class JointIds:
    """Map only identifier fields; never replace words in claims or names."""
    def __init__(self, payload):
        self.cards, self.states = {}, {}
        cards = ([payload["new_card"]] if "new_card" in payload else payload["new_cards"])
        for card in [*cards, *payload["old_cards"]]:
            self._add(self.cards, card["id"], "C")
        for state in payload["states"]:
            self._add(self.states, state["id"], "S")
            for key in state.get("evidence", []):
                self._add(self.cards, key, "C")
        self.reverse_cards = {v: k for k, v in self.cards.items()}
        self.reverse_states = {v: k for k, v in self.states.items()}

    @staticmethod
    def _add(mapping, key, prefix):
        if key not in mapping:
            mapping[key] = f"{prefix}{len(mapping) + 1}"

    def encode(self, payload):
        out = deepcopy(payload)
        cards = ([out["new_card"]] if "new_card" in out else out["new_cards"])
        for card in [*cards, *out["old_cards"]]:
            card["id"] = self.cards[card["id"]]
            if "old_card_ids" in card:
                card["old_card_ids"] = [self.cards[k] for k in card["old_card_ids"]]
            if "state_ids" in card:
                card["state_ids"] = [self.states[k] for k in card["state_ids"]]
        for state in out["states"]:
            state["id"] = self.states[state["id"]]
            if "evidence" in state:
                state["evidence"] = [self.cards[k] for k in state["evidence"]]
        out["participant_resolution"] = {
            self.cards[key]: value for key, value in out["participant_resolution"].items()
        }
        out["identifier_instruction"] = (
            "Card IDs C1, C2, ... and state IDs S1, S2, ... are request-local identifiers. "
            "Use only supplied IDs in ID fields. Do not alter user IDs."
        )
        return out

    def decode(self, proposal):
        out = deepcopy(proposal)
        if not isinstance(out, dict):
            return out
        def card_id(key):
            return self.reverse_cards.get(key, key) if isinstance(key, str) else key
        def state_id(key):
            return self.reverse_states.get(key, key) if isinstance(key, str) else key
        def relations(rows):
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict) and "old_card_id" in row:
                        row["old_card_id"] = card_id(row["old_card_id"])
        relations(out.get("relations"))
        if isinstance(out.get("cards"), list):
            for row in out["cards"]:
                if isinstance(row, dict):
                    if "card_id" in row:
                        row["card_id"] = card_id(row["card_id"])
                    relations(row.get("relations"))
        if isinstance(out.get("operations"), list):
            for op in out["operations"]:
                if isinstance(op, dict):
                    if "state_id" in op:
                        op["state_id"] = state_id(op["state_id"])
                    for field in ("evidence", "source_card_ids"):
                        if isinstance(op.get(field), list):
                            op[field] = [card_id(i) for i in op[field]]
        return out


def _fields(value, required, path, *, optional=()):
    if not isinstance(value, dict):
        fail("object_required", f"{path}必须是对象")
    missing = required - value.keys()
    extra = value.keys() - required - set(optional)
    if missing:
        fail("missing_fields", f"{path}缺少字段: {', '.join(sorted(missing))}")
    if extra:
        # Extra model-generated field names can themselves contain source text.
        fail("unexpected_fields", f"{path}包含不允许的多余字段")


def _relation(rel, valid_ids, path, batch):
    required = {"old_card_id", "relation", "changed_part"} | ({"same_matter"} if batch else set())
    _fields(rel, required, path, optional=() if batch else ("same_matter",))
    if not isinstance(rel["old_card_id"], str) or rel["old_card_id"] not in valid_ids:
        fail("old_card_not_candidate", f"{path}.old_card_id不在允许的旧卡片中")
    if not isinstance(rel["relation"], str) or rel["relation"] not in {"update", "supplement", "related", "unrelated", "uncertain"}:
        fail("invalid_relation", f"{path}.relation不是允许的关系类型")
    if rel["relation"] == "uncertain":
        fail("relation_uncertain", f"{path}的新旧关系不确定，需要保留待处理", repairable=False)
    matter = rel.get("same_matter", "uncertain")
    if not isinstance(matter, str) or matter not in {"same", "related", "different", "uncertain"}:
        fail("invalid_same_matter", f"{path}.same_matter不是允许的事项关系")
    if rel["relation"] == "update" and matter != "same":
        fail("update_matter_conflict", f"{path}标记为update时same_matter必须为same")
    if not isinstance(rel["changed_part"], str):
        fail("invalid_changed_part", f"{path}.changed_part必须是文本")


def validate_joint(proposal, payload, *, batch):
    """Check an entire proposal before any state or decision is committed."""
    _fields(proposal, {"cards", "operations"} if batch else {"relations", "operations"}, "输出")
    operations = proposal["operations"]
    if not isinstance(operations, list):
        fail("operations_not_list", "operations必须是列表")
    # An explicit uncertain decision is substantive uncertainty, not a format
    # defect to pressure the model into choosing an identity or state.
    if any(isinstance(op, dict) and op.get("action") == "uncertain" for op in operations):
        fail("state_uncertain", "状态关系不确定，需要保留待处理", repairable=False)
    old_ids = {c["id"] for c in payload["old_cards"]}
    cards = payload["new_cards"] if batch else [payload["new_card"]]
    new_ids = {c["id"] for c in cards}
    if batch:
        rows = proposal["cards"]
        if not isinstance(rows, list) or len(rows) != len(cards):
            fail("batch_card_coverage", "cards必须按顺序覆盖每张新卡片")
        earlier = set()
        for i, (row, card) in enumerate(zip(rows, cards)):
            path = f"cards[{i}]"
            _fields(row, {"card_id", "relations"}, path)
            if row["card_id"] != card["id"]:
                fail("batch_card_order", f"{path}.card_id不符合输入顺序")
            if not isinstance(row["relations"], list):
                fail("relations_not_list", f"{path}.relations必须是列表")
            for j, rel in enumerate(row["relations"]):
                _relation(rel, old_ids | earlier, f"{path}.relations[{j}]", True)
            earlier.add(card["id"])
    else:
        if not isinstance(proposal["relations"], list):
            fail("relations_not_list", "relations必须是列表")
        for i, rel in enumerate(proposal["relations"]):
            _relation(rel, old_ids, f"relations[{i}]", False)

    states = {s["id"]: s for s in payload["states"]}
    allowed_evidence = new_ids | old_ids
    for state in states.values():
        allowed_evidence.update(state.get("evidence", []))
    subjects, touched = set(payload["subjects"]), set()
    required = {"subject", "action", "state_id", "topic", "content", "status", "changed_part", "evidence"}
    if batch:
        required |= {"source_card_ids"}
    for i, op in enumerate(operations):
        path = f"operations[{i}]"
        _fields(op, required, path)
        if not isinstance(op["subject"], str) or op["subject"] not in subjects:
            fail("subject_not_candidate", f"{path}.subject不在候选用户中")
        if not isinstance(op["action"], str) or op["action"] not in {"add", "update", "none"}:
            fail("invalid_action", f"{path}.action不是允许的操作")
        if not isinstance(op["topic"], str) or op["topic"] not in {"preference", "constraint", "plan", "profile", "goal"}:
            fail("invalid_topic", f"{path}.topic不是允许的状态主题")
        if not isinstance(op["status"], str) or op["status"] not in {"confirmed", "uncertain"}:
            fail("invalid_status", f"{path}.status不是允许的状态标记")
        if batch and (not isinstance(op["source_card_ids"], list) or not op["source_card_ids"]
                      or any(not isinstance(x, str) or x not in new_ids for x in op["source_card_ids"])):
            fail("invalid_source_cards", f"{path}.source_card_ids必须引用本批新卡片")
        if op["action"] == "none":
            continue
        if not isinstance(op["content"], str) or not op["content"].strip():
            fail("invalid_state_content", f"{path}.content必须是非空文本")
        if not isinstance(op["changed_part"], str):
            fail("invalid_changed_part", f"{path}.changed_part必须是文本")
        evidence = op["evidence"]
        if not isinstance(evidence, list) or any(not isinstance(x, str) or x not in allowed_evidence for x in evidence):
            fail("invalid_evidence", f"{path}.evidence必须引用允许的卡片")
        required_sources = op["source_card_ids"] if batch else [cards[0]["id"]]
        if any(x not in evidence for x in required_sources):
            fail("missing_new_evidence", f"{path}.evidence必须包含涉及的新卡片")
        if op["action"] == "add":
            if op["state_id"] is not None:
                fail("add_has_state_id", f"{path}新增状态时state_id必须为null")
            continue
        target = states.get(op["state_id"]) if isinstance(op["state_id"], str) else None
        if not target:
            fail("state_not_candidate", f"{path}.state_id不在候选状态中")
        if target["subject"] != op["subject"]:
            fail("state_subject_mismatch", f"{path}目标状态属于另一用户")
        if target["topic"] != op["topic"]:
            fail("state_topic_mismatch", f"{path}目标状态主题不匹配")
        if op["state_id"] in touched:
            fail("duplicate_state_update", f"{path}重复更新同一状态")
        touched.add(op["state_id"])
