"""GateMem's per-question evidence budget for EGMemory's native reader."""
from __future__ import annotations


def bounded_read_tools(original, limit: int):
    """Wrap the upstream reader without changing the checked-out baseline.

    One stored memory node counts once across search, phrase and decision
    tools. Date resolution is permitted only for a node already disclosed.
    The upstream QA loop resets ``_seen`` at the start of each question.
    """
    if limit < 1:
        raise ValueError("EGMemory evidence limit must be positive")

    base = getattr(type(original), "_gatemem_unwrapped_type", type(original))

    class BudgetedReadTools(base):
        def _remaining(self):
            return max(0, limit - len(self._seen))

        def _unseen(self, ids):
            unique = set(self._seen)
            result = []
            for mid in ids:
                if mid not in unique and mid in self.store.memories:
                    unique.add(mid)
                    result.append(mid)
            return result

        def _fmt_scored(self, ids, cap=None):
            remaining = self._remaining()
            if not remaining:
                return "Evidence budget exhausted; answer using previously returned memories."
            unseen = self._unseen(ids)
            if not unseen:
                return "No new matching memory; answer using previously returned memories."
            return super()._fmt_scored(unseen, cap=min(cap or self.max_passages, remaining))

        def _scoped_decisions(self, ordered, exact):
            remaining = self._remaining()
            if not remaining:
                return "Evidence budget exhausted; answer using previously returned memories."
            selected = self._unseen(ordered)[:min(remaining, self.max_passages)]
            if not selected:
                return "No new decision evidence; answer using previously returned memories."
            result = super()._scoped_decisions(selected, exact)
            self._seen.update(selected)
            return result

        def resolve_date(self, msg_id, phrase, anchor="msg"):
            if msg_id not in self._seen:
                return "First retrieve this memory within the evidence budget."
            return super().resolve_date(msg_id, phrase, anchor)

    BudgetedReadTools._gatemem_unwrapped_type = base
    return BudgetedReadTools(original.store, original.retr, max_passages=original.max_passages)
