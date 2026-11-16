"""Conservative, deterministic resolution of relative date expressions.

The extractor is allowed to leave ``valid_from``/``planned_for`` empty; this module
fills only expressions whose meaning is unambiguous from the observation time.
It never invents a date for an open expression such as "soon" or "later".

Generic language support only: weekday names, today/tomorrow, "in N days/weeks",
month end, and the common Chinese equivalents. No domain vocabulary.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
    "一": 0,
    "二": 1,
    "三": 2,
    "四": 3,
    "五": 4,
    "六": 5,
    "日": 6,
    "天": 6,
}

_WEEKDAY_TEXT = "|".join(["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])

_EN_WEEKDAY = re.compile(
    rf"\b(?:(?P<qualifier>next|this|coming|last)\s+)?(?P<weekday>{_WEEKDAY_TEXT})\b",
    re.IGNORECASE,
)
_EN_RELATIVE_DAY = re.compile(r"\b(?P<word>today|tonight|tomorrow|yesterday)\b", re.IGNORECASE)
_EN_INTERVAL = re.compile(r"\bin\s+(?P<count>\d{1,3})\s+(?P<unit>day|days|week|weeks)\b", re.IGNORECASE)
_EN_MONTH_END = re.compile(r"\bend\s+of\s+(?:the\s+)?month\b", re.IGNORECASE)

_CN_RELATIVE_DAY = re.compile(r"(?P<word>今天|今日|明天|明日|后天|昨天|前天|大后天)")
_CN_QUALIFIED_WEEKDAY = re.compile(
    r"(?P<qualifier>本|这|下|上)\s*(?:周|星期|礼拜)\s*(?P<weekday>[一二三四五六日天])"
)
_CN_BARE_WEEKDAY = re.compile(r"(?:周|星期|礼拜)\s*(?P<weekday>[一二三四五六日天])")
_CN_INTERVAL = re.compile(r"(?P<count>\d{1,2})\s*(?P<unit>天|日|周|个?星期)\s*(?:后|之后|以后)")
_CN_MONTH_END = re.compile(r"(?P<which>本月|这个月|下个月|下月)?\s*月底")


def _anchor(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("anchor must include a timezone")
    return parsed.astimezone(parsed.tzinfo)


def _midnight(moment: datetime) -> datetime:
    return moment.replace(hour=0, minute=0, second=0, microsecond=0)


def _weekday_delta(anchor_weekday: int, target_weekday: int, qualifier: str | None) -> int:
    delta = (target_weekday - anchor_weekday) % 7
    if qualifier == "last":
        return -((anchor_weekday - target_weekday) % 7 or 7)
    if qualifier == "next":
        # "next Thursday" means the Thursday of the following week, not today.
        return delta + 7 if delta else 7
    # "this/coming Thursday" and a bare weekday mean the next occurrence, today included.
    return delta


def _month_end(moment: datetime) -> datetime:
    first = moment.replace(day=1)
    following = first.replace(year=first.year + (first.month == 12), month=first.month % 12 + 1)
    return following - timedelta(days=1)


def _candidates(text: str, base: datetime) -> list[tuple[int, int, datetime]]:
    found: list[tuple[int, int, datetime]] = []

    for match in _EN_RELATIVE_DAY.finditer(text):
        word = match.group("word").lower()
        offset = {"today": 0, "tonight": 0, "tomorrow": 1, "yesterday": -1}[word]
        found.append((match.start(), match.end(), _midnight(base + timedelta(days=offset))))
    for match in _EN_WEEKDAY.finditer(text):
        delta = _weekday_delta(
            base.weekday(), _WEEKDAYS[match.group("weekday").lower()], match.group("qualifier")
        )
        found.append((match.start(), match.end(), _midnight(base + timedelta(days=delta))))
    for match in _EN_INTERVAL.finditer(text):
        count = int(match.group("count"))
        days = count * (7 if match.group("unit").lower().startswith("week") else 1)
        found.append((match.start(), match.end(), _midnight(base + timedelta(days=days))))
    for match in _EN_MONTH_END.finditer(text):
        found.append((match.start(), match.end(), _midnight(_month_end(base))))

    for match in _CN_RELATIVE_DAY.finditer(text):
        offset = {"今天": 0, "今日": 0, "明天": 1, "明日": 1, "后天": 2, "昨天": -1, "前天": -2, "大后天": 3}[
            match.group("word")
        ]
        found.append((match.start(), match.end(), _midnight(base + timedelta(days=offset))))
    for match in _CN_QUALIFIED_WEEKDAY.finditer(text):
        qualifier = {"本": "this", "这": "this", "下": "next", "上": "last"}[match.group("qualifier")]
        delta = _weekday_delta(base.weekday(), _WEEKDAYS[match.group("weekday")], qualifier)
        found.append((match.start(), match.end(), _midnight(base + timedelta(days=delta))))
    for match in _CN_INTERVAL.finditer(text):
        count = int(match.group("count"))
        unit = match.group("unit")
        days = count * (7 if unit.endswith("周") or unit.endswith("星期") else 1)
        found.append((match.start(), match.end(), _midnight(base + timedelta(days=days))))
    for match in _CN_MONTH_END.finditer(text):
        which = match.group("which")
        reference = base if which in {None, "本月", "这个月"} else _month_end(base) + timedelta(days=1)
        found.append((match.start(), match.end(), _midnight(_month_end(reference))))
    for match in _CN_BARE_WEEKDAY.finditer(text):
        if any(start <= match.start() < end for start, end, _ in found):
            continue
        delta = _weekday_delta(base.weekday(), _WEEKDAYS[match.group("weekday")], None)
        found.append((match.start(), match.end(), _midnight(base + timedelta(days=delta))))
    return found


def resolve_relative(text: str, anchor: str) -> str | None:
    """Return the first unambiguous calendar date in ``text`` as ISO-8601, else None.

    The returned value is local midnight of the resolved day, in the anchor's offset.
    Longer qualified expressions win over a bare Chinese weekday at the same position.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    base = _anchor(anchor)
    candidates = sorted(_candidates(text, base), key=lambda item: item[0])
    return candidates[0][2].isoformat() if candidates else None
