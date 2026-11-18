"""Trusted-runner API: bearer identity binds person, agent and space, not just agent."""

import hashlib
import hmac
import os

from fastapi import Depends, FastAPI, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .control import Actor
from .models import AccessDenied, ValidationError


class TaskBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str = Field(min_length=1)


class EventBody(TaskBody):
    event_id: str = Field(min_length=1)
    subject_id: str
    category: str
    text: str = Field(min_length=1)
    item_id: str | None = None
    scope: str = "global"
    observed_at: str | None = None
    sources: list[str] = Field(default_factory=list)
    speaker: str = "user"
    metadata: dict[str, str | None] = Field(default_factory=dict)
    state_eligible: bool = True


class MessageBody(TaskBody):
    message_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    scope: str = "global"
    item_id: str | None = None
    observed_at: str | None = None
    metadata: dict[str, str | None] = Field(default_factory=dict)
    fallback_category: str | None = None
    prior_context: list[dict] = Field(default_factory=list)


class ReadBody(TaskBody):
    subject_ids: list[str] = Field(min_length=1)
    query: str = Field(min_length=1)
    scope: str = "global"
    target_time: str | None = None
    purpose: str = "current"
    conditions: list[str] = Field(default_factory=list)


class GrantBody(TaskBody):
    grant: dict


class ForgetBody(TaskBody):
    event_ids: list[str] = Field(min_length=1)


class AnswerBody(TaskBody):
    bundle_id: str
    question: str = Field(min_length=1)


class HandoffBody(TaskBody):
    bundle_id: str
    recipient_principal: str
    recipient_agent: str
    recipient_task: str


def create_app(memory, identities):
    tokens, seen = [], set()
    for row in identities:
        token = os.environ.get(row["token_env"], "")
        digest = hashlib.sha256(token.encode()).digest()
        if len(token) < 24 or digest in seen:
            raise ValidationError("每个身份绑定需要独立且至少24字符的令牌")
        seen.add(digest)
        identity = {k: row[k] for k in ("principal_id", "agent_id", "space_id")}
        memory.control.authenticate(Actor(**identity, task_id="startup"))
        tokens.append((digest, identity))
    app = FastAPI(title="Ours Multi-user Card-State-Sleep", version="0.2.0")

    def identity(authorization: str = Header(default="")):
        if authorization.startswith("Bearer "):
            digest = hashlib.sha256(authorization[7:].encode()).digest()
            for expected, who in tokens:
                if hmac.compare_digest(expected, digest):
                    return who
        raise AccessDenied("认证失败")

    def actor(body, who):
        return Actor(**who, task_id=body.task_id)

    @app.exception_handler(AccessDenied)
    async def denied(request, exc):
        return JSONResponse(status_code=403, content={"error": "access_denied"})

    @app.exception_handler(ValidationError)
    async def invalid(request, exc):
        return JSONResponse(status_code=422, content={"error": "invalid_request"})

    @app.get("/health")
    def health():
        immediate = (memory.control.access_mode == "shared_pool" and memory.settings.joint_state_update_enabled
                     and memory.settings.immediate_state_maintenance_enabled)
        return {"mode": "multiuser-v2", "state_visibility": "internal_only",
                "state_updates": "per_card" if immediate else "sleep_only"}

    @app.post("/events")
    def ingest(body: EventBody, who=Depends(identity)):
        return memory.ingest(actor(body, who), **body.model_dump(exclude={"task_id"}))

    @app.post("/messages")
    def message(body: MessageBody, who=Depends(identity)):
        return memory.ingest_message(actor(body, who), **body.model_dump(exclude={"task_id"}))

    @app.post("/read")
    def read(body: ReadBody, who=Depends(identity)):
        return memory.read(actor(body, who), **body.model_dump(exclude={"task_id"}))

    @app.post("/control/grants")
    def grant(body: GrantBody, who=Depends(identity)):
        with memory.lock:
            return memory.control.grant(actor(body, who), body.grant)

    @app.post("/control/grants/{grant_id}/revoke")
    def revoke(grant_id: str, body: TaskBody, who=Depends(identity)):
        with memory.lock:
            return memory.control.revoke(actor(body, who), grant_id)

    @app.post("/control/forget")
    def forget(body: ForgetBody, who=Depends(identity)):
        return memory.forget(actor(body, who), body.event_ids)

    @app.post("/control/retry-deletes")
    def retry(body: TaskBody, who=Depends(identity)):
        return memory.retry_deletes(actor(body, who))

    @app.post("/sleep")
    def sleep(body: TaskBody, who=Depends(identity)):
        return memory.sleep(actor(body, who))

    @app.post("/bundles/{bundle_id}/check")
    def check(bundle_id: str, body: TaskBody, who=Depends(identity)):
        return memory.validate_bundle(actor(body, who), bundle_id)

    @app.post("/bundles/{bundle_id}/read")
    def bundle(bundle_id: str, body: TaskBody, who=Depends(identity)):
        return memory.bundle(actor(body, who), bundle_id)

    @app.post("/handoff")
    def handoff(body: HandoffBody, who=Depends(identity)):
        recipient = Actor(
            body.recipient_principal, body.recipient_agent, who["space_id"], body.recipient_task
        )
        return memory.handoff(actor(body, who), recipient, body.bundle_id)

    @app.post("/answer")
    def answer(body: AnswerBody, who=Depends(identity)):
        return memory.answer(actor(body, who), body.bundle_id, body.question)

    # Intentionally no state/history, raw trace, arbitrary card-ID or legacy admin routes.
    return app
