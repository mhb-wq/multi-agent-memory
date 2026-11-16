"""Single-process research gateway; tokens belong to trusted agent runners, not LLMs."""

from __future__ import annotations

import hashlib
import hmac
import os

from .models import AccessDenied, Caller, Event, ReadRequest, ValidationError, now


class Authenticator:
    def __init__(self, identities):
        self.identities = []
        seen = set()
        for row in identities:
            token = os.environ.get(row["token_env"])
            if not token or len(token) < 24:
                raise ValidationError(f"{row['token_env']} 需要至少24字符的独立令牌")
            digest = hashlib.sha256(token.encode()).digest()
            if digest in seen:
                raise ValidationError("不同Agent不能共用令牌")
            seen.add(digest)
            self.identities.append((digest, Caller(row["user_id"], row["team_id"], row["agent_id"])))

    def authenticate(self, authorization):
        if not authorization or not authorization.startswith("Bearer "):
            raise AccessDenied("需要已认证Agent身份")
        digest = hashlib.sha256(authorization[7:].encode()).digest()
        for expected, caller in self.identities:
            if hmac.compare_digest(expected, digest):
                return caller
        raise AccessDenied("无效凭证")


def create_app(service, policies, identities):
    from fastapi import Depends, FastAPI, Header
    from fastapi.responses import JSONResponse

    auth = Authenticator(identities)
    app = FastAPI(title="Ours Card-State-Sleep", version="0.1.0")

    @app.exception_handler(AccessDenied)
    async def denied(request, exc):
        return JSONResponse(status_code=403, content={"error": "access_denied"})

    @app.exception_handler(ValidationError)
    async def invalid(request, exc):
        return JSONResponse(status_code=422, content={"error": str(exc)})

    def caller(authorization: str = Header(default="")):
        return auth.authenticate(authorization)

    def only(body, keys):
        if set(body) - set(keys):
            raise ValidationError("请求包含未允许字段（身份和权限不能由请求指定）")

    def required(body, keys):
        if any(not isinstance(body.get(key), str) or not body[key].strip() for key in keys):
            raise ValidationError("缺少必需的非空文本字段：" + ", ".join(keys))

    @app.get("/health")
    def health():
        return {"status": "ok", "backend": "MemOS/general_text", "mode": "single-process"}

    @app.post("/events")
    def ingest(body: dict, who=Depends(caller)):
        only(
            body,
            [
                "event_id",
                "task_id",
                "text",
                "policy",
                "scope",
                "speaker",
                "observed_at",
                "source_root_ids",
                "disclosure_id",
            ],
        )
        required(body, ["event_id", "task_id", "text", "policy"])
        policy_id = policies.get(body.get("policy"))
        if not policy_id or not service.graph.permits(policy_id, who, "write"):
            raise AccessDenied("没有该域的写权限")
        row = service.store.get("policy", policy_id)
        event = Event(
            user_id=who.user_id,
            team_id=who.team_id,
            writer_agent_id=who.agent_id,
            event_id=body["event_id"],
            task_id=body["task_id"],
            text=body["text"],
            recorded_at=now(),
            allowed_reader_ids=tuple(row["original_readers"]),
            scope=body.get("scope", "global"),
            speaker=body.get("speaker", "user"),
            observed_at=body.get("observed_at"),
            source_root_ids=tuple(body.get("source_root_ids", [])),
            disclosure_id=body.get("disclosure_id"),
        )
        # Preserve receipt time on idempotent retries; changed content still fails core validation.
        previous = service.store.get("event", event.event_id)
        if previous:
            from dataclasses import replace

            event = replace(event, recorded_at=previous["recorded_at"])
        ids = service.ingest_event(event, who)
        return {"card_ids": ids, "job": service.job_status(event.event_id, who)}

    @app.post("/read")
    def read(body: dict, who=Depends(caller)):
        only(body, ReadRequest.__dataclass_fields__)
        required(body, ["task_id", "query", "target_time"])
        return service.read_memory(ReadRequest(**body), who).to_dict()

    @app.get("/cards/{card_id}")
    def card(card_id: str, who=Depends(caller)):
        return service.get_card(card_id, who)

    @app.get("/cards/{card_id}/trace")
    def trace(card_id: str, who=Depends(caller)):
        return service.get_trace(card_id, who)

    @app.get("/states/{state_id}/history")
    def history(state_id: str, who=Depends(caller)):
        return service.get_history(state_id, who)

    @app.get("/jobs/{event_id}")
    def job(event_id: str, who=Depends(caller)):
        return service.job_status(event_id, who)

    @app.post("/jobs/{event_id}/retry")
    def retry(event_id: str, who=Depends(caller)):
        return {"card_ids": service.retry_job(event_id, who)}

    @app.post("/tasks/{task_id}/finish")
    def finish(task_id: str, who=Depends(caller)):
        if getattr(service, "scheduler", None):
            return service.scheduler.submit(task_id, who)
        return service.finish_task(task_id, who)

    @app.post("/tasks/{task_id}/sleep-retry")
    def sleep_retry(task_id: str, who=Depends(caller)):
        return service.retry_sleep(who, task_id)

    @app.post("/plans/{plan_id}")
    def plan(plan_id: str, body: dict, who=Depends(caller)):
        only(body, ["task_id"])
        required(body, ["task_id"])
        return service.register_plan(plan_id, body["task_id"], who)

    @app.get("/plans/{plan_id}/check")
    def plan_check(plan_id: str, who=Depends(caller)):
        return service.check_plan_dependencies(plan_id, who)

    @app.post("/forget")
    def forget(body: dict, who=Depends(caller)):
        only(body, ["source_ids"])
        if not isinstance(body.get("source_ids"), list):
            raise ValidationError("source_ids必须为列表")
        return service.forget(body["source_ids"], who)

    @app.post("/maintenance/retry-deletes")
    def cleanup(who=Depends(caller)):
        return service.retry_index_deletes(who)

    return app
