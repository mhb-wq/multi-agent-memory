import unittest

import test_memory as support

from ours_memory.calendar_bounds import complete_month_end
from ours_memory.models import ValidationError


class CalendarBoundsTests(unittest.TestCase):
    def test_uncertain_month_still_has_end(self):
        end, reason = complete_month_end("2026年11月每周4或6小时，还没定", "2026-11-01T00:00:00+08:00", None)
        self.assertEqual(end, "2026-12-01T00:00:00+08:00")
        self.assertIsNotNone(reason)

    def test_december_rollover_and_english(self):
        self.assertEqual(
            complete_month_end("During December 2026 the budget is uncertain", "2026-12-01T00:00:00Z", None)[
                0
            ],
            "2027-01-01T00:00:00+00:00",
        )

    def test_previous_live_output_utc_start_uses_observation_calendar(self):
        # Exact quote/start/observation from live_smoke_2026-10-24_0bd1b68c52.
        quote = "关于2026年11月的每周英语备考时间，我尚未确定是4小时还是6小时；请先保留待确认，不要把6小时当作已经确定的安排。"
        end, reason = complete_month_end(
            quote, "2026-10-31T16:00:00+00:00", None, "2026-10-24T00:00:00+08:00"
        )
        self.assertEqual(end, "2026-12-01T00:00:00+08:00")
        self.assertIsNotNone(reason)

    def test_open_start_day_multiple_months_and_relative_month_unchanged(self):
        for text in [
            "从2026年11月开始改为4小时",
            "2026年11月起每周4小时",
            "2026年11月起预算增加",
            "2026年11月上旬有空",
            "2026年11月以后每周4小时",
            "2026年11月15日改为4小时",
            "2026年11月和2026年12月都是4小时",
            "下个月每周4小时",
        ]:
            with self.subTest(text=text):
                self.assertEqual(complete_month_end(text, "2026-11-01T00:00:00+08:00", None), (None, None))

    def test_contradictory_end_is_not_silently_repaired(self):
        with self.assertRaises(ValidationError):
            complete_month_end(
                "2026年11月每周4小时", "2026-11-01T00:00:00+08:00", "2027-01-01T00:00:00+08:00"
            )


class MonthIntegrationTests(unittest.TestCase):
    setUp = support.MemoryTests.setUp
    tearDown = support.MemoryTests.tearDown
    event = support.MemoryTests.event
    enqueue = support.MemoryTests.enqueue
    current = support.MemoryTests.current

    def test_omitted_end_cannot_extend_month_exception(self):
        self.enqueue(self.event())
        card = self.enqueue(
            self.event("temporary", text="2026年11月每周只有4小时"), 4, start=support.NOV, operation="UPDATE"
        )
        stored = self.store.get("card", card)
        self.assertEqual(stored["valid_to"], "2026-11-30T16:00:00+00:00")
        self.assertEqual(self.current("2026-11-15T00:00:00+08:00")["value"], 4)
        self.assertEqual(self.current("2026-12-01T00:00:00+08:00")["value"], 12)

    def test_month_uncertainty_expires_and_preserves_supported_future_default(self):
        self.enqueue(self.event())
        self.enqueue(
            self.event("future", 4, recorded="2026-10-22T00:00:00+08:00"),
            4,
            start=support.NOV,
            operation="UPDATE",
        )
        card = self.enqueue(
            self.event(
                "uncertain", recorded="2026-10-23T00:00:00+08:00", text="2026年11月每周4或6小时，还没确定"
            ),
            None,
            start=support.NOV,
            operation="CONFLICT",
        )
        self.assertEqual(self.store.get("card", card)["valid_to"], "2026-11-30T16:00:00+00:00")
        self.assertEqual(self.current("2026-10-28T00:00:00+08:00")["value"], 12)
        self.assertEqual(self.current("2026-11-15T00:00:00+08:00")["status"], "conflict")
        self.assertEqual(self.current("2026-12-01T00:00:00+08:00")["value"], 4)
