from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

from .models import StaleRevision, canonical, now


class Store:
    """SQLite sidecar. MemOS remains responsible for native memory text/indexing."""

    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = RLock()
        self._transaction_depth = 0
        self.connection.executescript("""
            PRAGMA foreign_keys=ON;
            PRAGMA journal_mode=WAL;
            PRAGMA secure_delete=ON;
            CREATE TABLE IF NOT EXISTS objects (
                kind TEXT NOT NULL, id TEXT NOT NULL, data TEXT NOT NULL,
                PRIMARY KEY(kind,id));
            CREATE TABLE IF NOT EXISTS states (
                id TEXT PRIMARY KEY, revision INTEGER NOT NULL, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS state_history (
                id TEXT NOT NULL, revision INTEGER NOT NULL, data TEXT NOT NULL,
                PRIMARY KEY(id,revision));
            CREATE TABLE IF NOT EXISTS dirty (
                state_id TEXT PRIMARY KEY, generation INTEGER NOT NULL, task_id TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS calls (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, stage TEXT NOT NULL,
                data TEXT NOT NULL);
        """)
        self.connection.commit()

    @contextmanager
    def transaction(self):
        with self.lock:
            if self._transaction_depth:
                self._transaction_depth += 1
                try:
                    yield
                finally:
                    self._transaction_depth -= 1
                return
            self.connection.execute("BEGIN IMMEDIATE")
            self._transaction_depth = 1
            try:
                yield
            except BaseException:
                self.connection.rollback()
                raise
            else:
                self.connection.commit()
            finally:
                self._transaction_depth = 0

    def get(self, kind: str, key: str):
        row = self.connection.execute(
            "SELECT data FROM objects WHERE kind=? AND id=?", (kind, key)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, kind: str, key: str, value: dict):
        with self.transaction():
            self.connection.execute(
                "INSERT OR REPLACE INTO objects VALUES (?,?,?)", (kind, key, canonical(value))
            )

    def remove(self, kind: str, key: str):
        with self.transaction():
            self.connection.execute("DELETE FROM objects WHERE kind=? AND id=?", (kind, key))

    def all(self, kind: str) -> list[dict]:
        return [
            json.loads(row[0])
            for row in self.connection.execute("SELECT data FROM objects WHERE kind=? ORDER BY id", (kind,))
        ]

    def state(self, key: str) -> dict | None:
        row = self.connection.execute("SELECT revision,data FROM states WHERE id=?", (key,)).fetchone()
        if not row:
            return None
        return {**json.loads(row[1]), "revision": row[0]}

    def states(self) -> list[dict]:
        return [
            {**json.loads(row["data"]), "revision": row["revision"]}
            for row in self.connection.execute("SELECT * FROM states ORDER BY id")
        ]

    def commit_state(self, state: dict, expected: int, *, task_id: str, dirty: bool = True) -> int:
        data = {k: v for k, v in state.items() if k != "revision"}
        with self.transaction():
            previous = self.connection.execute(
                "SELECT revision,data FROM states WHERE id=?", (state["id"],)
            ).fetchone()
            current = previous[0] if previous else 0
            if current != expected:
                raise StaleRevision("状态版本已变化，重新读取后再判断")
            if previous and previous[1] == canonical(data):
                return current
            revision = current + 1
            self.connection.execute(
                "INSERT INTO state_history VALUES (?,?,?)", (state["id"], revision, canonical(data))
            )
            self.connection.execute(
                "INSERT OR REPLACE INTO states VALUES (?,?,?)", (state["id"], revision, canonical(data))
            )
            if dirty:
                self.connection.execute(
                    """INSERT INTO dirty VALUES (?,1,?)
                    ON CONFLICT(state_id) DO UPDATE SET generation=generation+1,task_id=excluded.task_id""",
                    (state["id"], task_id),
                )
        return revision

    def dirty_states(self) -> list[dict]:
        return [dict(row) for row in self.connection.execute("SELECT * FROM dirty ORDER BY state_id")]

    def mark_dirty(self, state_id: str, task_id: str):
        with self.transaction():
            self.connection.execute(
                """INSERT INTO dirty VALUES (?,1,?)
                ON CONFLICT(state_id) DO UPDATE SET generation=generation+1,task_id=excluded.task_id""",
                (state_id, task_id),
            )

    def clean_dirty(self, key: str, generation: int):
        with self.transaction():
            self.connection.execute("DELETE FROM dirty WHERE state_id=? AND generation=?", (key, generation))

    def history(self, state_id: str) -> list[dict]:
        return [
            {**json.loads(row["data"]), "revision": row["revision"]}
            for row in self.connection.execute(
                "SELECT * FROM state_history WHERE id=? ORDER BY revision", (state_id,)
            )
        ]

    def call(self, stage: str, **data):
        with self.transaction():
            self.connection.execute(
                "INSERT INTO calls(stage,data) VALUES (?,?)", (stage, canonical({"at": now(), **data}))
            )

    def call_records(self) -> list[dict]:
        return [
            {"stage": row["stage"], **json.loads(row["data"])}
            for row in self.connection.execute("SELECT * FROM calls ORDER BY seq")
        ]

    def close(self):
        self.connection.close()
