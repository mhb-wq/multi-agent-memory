"""Persistent exact-cosine index of focused shared-pool state entries."""

import hashlib
import math

from .models import ValidationError, canonical, stable_id


def _normalize(vector):
    if not isinstance(vector, (list, tuple)) or not vector:
        raise ValidationError("状态嵌入向量为空或格式无效")
    values = [float(x) for x in vector]
    if not all(math.isfinite(x) for x in values):
        raise ValidationError("状态嵌入向量包含非有限数值")
    norm = math.sqrt(sum(x * x for x in values))
    if not norm:
        raise ValidationError("状态嵌入向量不能为零")
    return [x / norm for x in values]


def _embed(backend, space, texts):
    fn = getattr(backend, "embed_state_texts", None)
    if not callable(fn):
        raise ValidationError("记忆后端不支持状态语义向量，请配置支持嵌入的后端")
    vectors = fn(space, texts)
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise ValidationError("状态嵌入结果数量不匹配")
    normalized = [_normalize(v) for v in vectors]
    if normalized and any(len(v) != len(normalized[0]) for v in normalized):
        raise ValidationError("状态嵌入维度不一致")
    return normalized


def entries_for_space(store, space, subjects):
    entries = []
    for subject in sorted(subjects):
        state_key = stable_id("user_state", space, subject)
        state = store.get("user_state", state_key)
        if not state:
            continue
        for topic, rows in state.get("topics", {}).items():
            for index, row in enumerate(rows):
                entry_id = row.get("id") or stable_id("state_item", state_key, topic, str(index))
                content = row.get("content")
                if isinstance(content, str) and content.strip():
                    entries.append({"subject": subject, "topic": topic, "id": entry_id,
                                    "row": row, "state": state, "text": f"{subject} {topic} {content}"})
    return entries


def rank_state_entries(store, backend, space, subjects, query, limit, *, allowed_ids=None, threshold=0.2):
    """Reconcile changed entries, then rank eligible entries with exact cosine similarity.

    Embedding failures abort the search instead of silently returning stale or lexical results.
    Entries and vectors are per-space; the SQLite index trades linear scans for simple recovery.
    """
    identity_fn = getattr(backend, "state_embedding_identity", None)
    if not callable(identity_fn):
        raise ValidationError("记忆后端未提供状态嵌入模型身份")
    identity = identity_fn(space)
    entries = entries_for_space(store, space, subjects)
    stored = {item["id"]: item for item in store.all("state_vector") if item.get("space") == space}
    desired = {}
    changed = []
    for entry in entries:
        vector_id = stable_id("state_vector", space, entry["subject"], entry["topic"], entry["id"])
        digest = hashlib.sha256(canonical([identity, entry["text"]]).encode("utf-8")).hexdigest()
        desired[vector_id] = (entry, digest)
        if vector_id not in stored or stored[vector_id].get("digest") != digest:
            changed.append(vector_id)
    new_vectors = {}
    for start in range(0, len(changed), 64):
        batch = changed[start:start + 64]
        vectors = _embed(backend, space, [desired[key][0]["text"] for key in batch])
        new_vectors.update(zip(batch, vectors))
    with store.transaction():
        for key in set(stored) - set(desired):
            store.remove("state_vector", key)
        for key, vector in new_vectors.items():
            item = {"id": key, "space": space, "digest": desired[key][1], "vector": vector}
            store.put("state_vector", key, item)
            stored[key] = item
        if store.get("state_vector_issue", space):
            store.remove("state_vector_issue", space)
    if not entries or query is None:
        return []
    q = _embed(backend, space, [query])[0]
    ranked = []
    for key, (entry, _) in desired.items():
        if allowed_ids is not None and (entry["subject"], entry["topic"], entry["id"]) not in allowed_ids:
            continue
        v = stored[key]["vector"]
        if len(v) != len(q):
            raise ValidationError("状态索引和查询向量维度不一致，请检查嵌入模型配置")
        score = sum(a * b for a, b in zip(q, v))
        if score >= threshold:
            ranked.append((score, entry))
    ranked.sort(key=lambda item: (-item[0], item[1]["subject"], item[1]["topic"], item[1]["id"]))
    return ranked[:limit]
