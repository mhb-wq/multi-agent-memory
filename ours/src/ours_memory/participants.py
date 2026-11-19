"""Resolve extracted person mentions using explicit, space-local aliases only."""

import unicodedata


def _name(value):
    return unicodedata.normalize("NFKC", value).strip().casefold()


def resolve_participants(card, subjects, aliases=None):
    """Retain ambiguous mentions; never infer identity from a similar name or role."""
    aliases = aliases or {}
    names = {}
    for subject in subjects:
        for name in [subject, *aliases.get(subject, [])]:
            if isinstance(name, str) and name.strip():
                names.setdefault(_name(name), set()).add(subject)
    resolved = []
    for participant in card.get("participants") or []:
        if not isinstance(participant, dict) or not isinstance(participant.get("name"), str):
            continue
        candidates = sorted(names.get(_name(participant["name"]), ()))
        resolved.append({"mention": participant["name"], "role": participant.get("role", ""),
                         "resolved_id": candidates[0] if len(candidates) == 1 else None,
                         "candidate_ids": candidates,
                         "status": "resolved" if len(candidates) == 1 else
                                   "ambiguous" if candidates else "unresolved"})
    return resolved
