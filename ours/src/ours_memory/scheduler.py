"""Durable Sleep jobs executed by the upstream MemOS SchedulerDispatcher."""

from __future__ import annotations

from dataclasses import asdict

from .models import AccessDenied, Caller, canonical, now, stable_id


class MemOSSleepScheduler:
    def __init__(self, service):
        from memos.mem_scheduler.task_schedule_modules.dispatcher import SchedulerDispatcher
        from memos.mem_scheduler.utils import metrics

        self.service = service
        self.dispatcher = SchedulerDispatcher(max_workers=1, enable_parallel_dispatch=False, metrics=metrics)
        self.dispatcher.register_handler("ours_sleep", self._handle)

    def submit(self, task_id, caller):
        from memos.mem_scheduler.schemas.message_schemas import ScheduleMessageItem

        self.service._authorize(caller)
        key = stable_id("sleep_job", *asdict(caller).values(), task_id)
        with self.service.lock:
            job = {
                "id": key,
                "caller": asdict(caller),
                "task_id": task_id,
                "status": "pending",
                "submitted_at": now(),
            }
            self.service.store.put("schedule", key, job)
            # Only job identity crosses the queue; no private evidence is serialized here.
            message = ScheduleMessageItem(
                user_id=caller.user_id, mem_cube_id=key, label="ours_sleep", content=key, task_id=task_id
            )
            self.dispatcher.dispatch([message])
            return self.status(key, caller)

    def _handle(self, messages):
        for message in messages:
            with self.service.lock:
                job = self.service.store.get("schedule", message.content)
                if not job or job["caller"]["user_id"] != message.user_id:
                    raise AccessDenied("无效调度消息")
                job["status"] = "running"
                self.service.store.put("schedule", job["id"], job)
                try:
                    result = self.service.finish_task(job["task_id"], Caller(**job["caller"]))
                    job.update(
                        status="completed"
                        if result["status"] in {"completed", "already_completed", "disabled"}
                        else "needs_attention",
                        result=result,
                    )
                except Exception as exc:
                    job.update(status="failed", error=type(exc).__name__)
                job["finished_at"] = now()
                self.service.store.put("schedule", job["id"], job)

    def status(self, job_id, caller):
        self.service._authorize(caller)
        job = self.service.store.get("schedule", job_id)
        if not job or canonical(job["caller"]) != canonical(asdict(caller)):
            raise AccessDenied("无权查看调度任务")
        # Results can mention source identifiers whose permissions changed since the run.
        return {
            k: job[k] for k in ("id", "task_id", "status", "submitted_at", "finished_at", "error", "result") if k in job
        }

    def recover(self):
        """Trusted worker entry; replay only interrupted jobs, not semantic failures forever."""
        jobs = [job for job in self.service.store.all("schedule") if job["status"] in {"pending", "running"}]
        return [self.submit(job["task_id"], Caller(**job["caller"])) for job in jobs]

    def close(self):
        # Upstream shutdown assumes an executor even in synchronous mode.
        # This adapter uses synchronous dispatch and owns no worker threads.
        if self.dispatcher.dispatcher_executor is not None:
            self.dispatcher.shutdown()
