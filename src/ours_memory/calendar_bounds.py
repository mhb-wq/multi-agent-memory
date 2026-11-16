"""Conservative calendar normalization, using cited text, never benchmark labels."""

from __future__ import annotations

import re
from datetime import datetime

from .models import ValidationError, timestamp

_MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]


def complete_month_end(
    quote: str, start: str | None, end: str | None, observation: str | None = None
) -> tuple[str | None, str | None]:
    """Fill an omitted end only for an explicit year/month aligned with supplied start.

    An open-ended change, a day, several months or an ambiguous relative month is
    deliberately left to the extractor. A conflicting explicit end is rejected.
    This does not infer values, pick state operations, or change source evidence.
    """
    if not start:
        return end, None
    # Validate first, preserving the source calendar's offset for month arithmetic.
    timestamp(start)
    local = datetime.fromisoformat(start.replace("Z", "+00:00"))
    if observation:
        timestamp(observation)
        reference = datetime.fromisoformat(observation.replace("Z", "+00:00"))
        local = local.astimezone(reference.tzinfo)
    if (local.day, local.hour, local.minute, local.second, local.microsecond) != (1, 0, 0, 0, 0):
        return end, None
    text = quote.lower()
    if re.search(
        r"开始|月\s*起|以来|之后|以后|之前|截至|截止|从|自|上旬|中旬|下旬|\b(?:since|starting|onwards?|until|before|after|from)\b",
        text,
    ):
        return end, None
    cn = re.findall(r"(?<!\d)(\d{4})\s*年\s*(1[0-2]|0?[1-9])\s*月(?!\s*\d+\s*[日号])", text)
    en = [
        (y, str(_MONTHS.index(m) + 1))
        for m, y in re.findall(r"\b(" + "|".join(_MONTHS) + r")\s+(\d{4})\b", text)
    ]
    periods = {(int(y), int(m)) for y, m in cn + en}
    # Dates/ranges not supported by this narrow normalizer must not be shortened.
    if len(periods) != 1 or (local.year, local.month) not in periods:
        return end, None
    if re.search(r"\d{4}-\d{2}-\d{2}|\d+\s*[日号]|\b\d{1,2}(?:st|nd|rd|th)\b|至|到|\bthrough\b", text):
        return end, None
    expected = local.replace(year=local.year + (local.month == 12), month=local.month % 12 + 1)
    if end is not None and timestamp(end) != timestamp(expected.isoformat()):
        raise ValidationError("卡片有效期与引用中明确的日历月份不一致")
    return expected.isoformat(), "explicit_calendar_month_exclusive_end" if end is None else None
