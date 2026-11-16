"""Compact model-facing views; storage keeps operational/provenance metadata."""

CARD_FIELDS = (
    "id", "subject", "author_id", "claim_text", "source_quote", "subject_context", "use_hint", "state_topic", "source",
    "evidence_kind", "state_key", "state_aspect", "value", "unit", "event_time",
    "valid_from", "valid_to", "planned_for", "scope", "conditions",
)
STATE_FIELDS = (
    "id", "subject", "state_key", "state_aspect", "unit", "scope", "conditions",
    "revision", "target_time", "status", "value", "has_pending",
)
ASSERTION_FIELDS = (
    "value", "unit", "status", "valid_from", "valid_to", "event_time",
    "evidence_card_ids", "operation", "reason",
)


def compact_card(card, *, for_usage=False):
    if "state_topic" in card:
        return {k: card[k] for k in ("id", "subject", "author_id", "claim_text", "use_hint", "state_topic", "source")
                if k in card}
    # State maintenance still needs effective intervals. Usage reasoning should
    # read the stated appointment/event time from claim and source quote.
    excluded = {"planned_for", "valid_from", "valid_to", "event_time"} if for_usage else set()
    return {k: card[k] for k in CARD_FIELDS if k in card and k not in excluded}


def compact_state(state, *, for_usage=False):
    if "topics" in state:
        return {"subject": state["subject"], "topics": state["topics"]}
    result = {k: state[k] for k in STATE_FIELDS if k in state}
    if for_usage:
        result = {k: v for k, v in result.items() if k not in {"revision", "unit"}}
    if "assertions" in state:
        fields = ("value", "unit", "status", "valid_from", "valid_to", "evidence_card_ids") if for_usage else ASSERTION_FIELDS
        result["assertions"] = [
            {k: a[k] for k in fields if k in a} for a in state["assertions"]
        ]
    return result


def compact_event(event):
    return {k: event[k] for k in ("event_id", "author_id", "speaker", "text", "observed_at") if k in event}
