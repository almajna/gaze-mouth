"""Persist named tasks and claim each scheduled occurrence before starting work."""

import fcntl
import json
import os
import sqlite3
import stat
import time
import uuid
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from . import memory
from .models import PROVIDERS, answer

KINDS = ("Background brief", "Desktop task")
CADENCES = ("On demand", "Once", "Daily", "Weekly")


def store_path():
    return memory.DATABASE.with_name("routines.sqlite3")


@contextmanager
def database():
    path = store_path()
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("PRAGMA secure_delete=ON")
        db.execute("BEGIN IMMEDIATE")
        db.row_factory = sqlite3.Row
        db.execute("""CREATE TABLE IF NOT EXISTS routines (
            id TEXT PRIMARY KEY, revision INTEGER NOT NULL DEFAULT 1, name TEXT NOT NULL, prompt TEXT NOT NULL, kind TEXT NOT NULL,
            source TEXT NOT NULL, provider TEXT NOT NULL DEFAULT 'Automatic', cadence TEXT NOT NULL, next_at REAL, hour INTEGER, minute INTEGER,
            paused INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'Saved', started REAL,
            result TEXT NOT NULL DEFAULT '', result_at REAL, issue TEXT NOT NULL DEFAULT ''
        )""")
        yield db


def all_tasks():
    if not store_path().exists():
        return []
    with database() as db:
        return [
            dict(r)
            for r in db.execute(
                "SELECT * FROM routines ORDER BY name COLLATE NOCASE, id"
            )
        ]


def save(name, prompt, kind, source, cadence, first, key=None, *, provider="Automatic"):
    name, prompt = name.strip(), prompt.strip()
    if not name or not prompt:
        raise ValueError("Enter a name and instructions.")
    if (
        kind not in KINDS
        or provider not in PROVIDERS
        or cadence not in CADENCES
        or (kind == "Desktop task" and source)
    ):
        raise ValueError("Choose a valid task type, file and schedule.")
    if source and not Path(source).is_absolute():
        raise ValueError("Choose a text file using its full path.")
    when = None if cadence == "On demand" else datetime.fromtimestamp(first)
    values = (
        name,
        prompt,
        kind,
        source,
        provider,
        cadence,
        first if when else None,
        when.hour if when else None,
        when.minute if when else None,
    )
    with database() as db:
        if key is None:
            key = uuid.uuid4().hex
            db.execute(
                """INSERT INTO routines
                (name,prompt,kind,source,provider,cadence,next_at,hour,minute,id) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (*values, key),
            )
        elif not db.execute(
            """UPDATE routines SET revision=revision+1,name=?,prompt=?,kind=?,source=?,provider=?,cadence=?,next_at=?,
                hour=?,minute=?,status='Saved',issue='' WHERE id=? AND status!='Running'""",
            (*values, key),
        ).rowcount:
            raise ValueError("This routine is running or was removed.")
    return key


def next_date(task, now):
    if task["cadence"] not in ("Daily", "Weekly"):
        return None
    due = datetime.fromtimestamp(task["next_at"])
    today = datetime.fromtimestamp(now)
    day = today.date()
    if task["cadence"] == "Weekly":
        day += timedelta(days=(due.weekday() - today.weekday()) % 7)
    following = datetime.combine(day, datetime.min.time()).replace(
        hour=task["hour"], minute=task["minute"]
    )
    if following.timestamp() <= now:
        following += timedelta(days=7 if task["cadence"] == "Weekly" else 1)
    return following.timestamp()


def lock_path(key):
    return memory.DATABASE.parent / f".routine-{key}.lock"


def take_lock(key):
    handle = lock_path(key).open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


@dataclass
class Run:
    task: dict
    lock: object

    def finish(self, status, text=""):
        if self.lock.closed:
            return
        try:
            with database() as db:
                if status == "Complete":
                    db.execute(
                        "UPDATE routines SET revision=revision+1,status=?,result=?,result_at=?,issue='' WHERE id=?",
                        (status, text, time.time(), self.task["id"]),
                    )
                else:
                    db.execute(
                        "UPDATE routines SET revision=revision+1,status=?,issue=? WHERE id=?",
                        (status, text, self.task["id"]),
                    )
        finally:
            self.lock.close()


def reconcile():
    """Release abandoned runs and queue due desktop tasks without opening or observing an app."""
    if not store_path().exists():
        return
    now = time.time()
    with database() as db:
        for row in db.execute(
            "SELECT * FROM routines WHERE status='Running'"
        ).fetchall():
            handle = take_lock(row["id"])
            if handle:
                with handle:
                    db.execute(
                        "UPDATE routines SET revision=revision+1,status='Interrupted',issue=? WHERE id=?",
                        (
                            "CursorMix stopped during this run. Review the result before running it again.",
                            row["id"],
                        ),
                    )
        for row in db.execute(
            """SELECT * FROM routines WHERE kind='Desktop task' AND paused=0
                AND next_at<=? AND status!='Running'""",
            (now,),
        ).fetchall():
            db.execute(
                "UPDATE routines SET revision=revision+1,next_at=?,status='Ready' WHERE id=?",
                (next_date(row, now), row["id"]),
            )


def claim(key=None, *, revision=None):
    """Manual runs select a task by id. A timer can claim only a due background brief."""
    if not store_path().exists():
        return None
    now = time.time()
    handle = None
    try:
        with database() as db:
            if key is None:
                task = db.execute(
                    """SELECT * FROM routines WHERE kind='Background brief' AND paused=0
                    AND next_at<=? AND status!='Running' ORDER BY next_at,id LIMIT 1""",
                    (now,),
                ).fetchone()
            else:
                task = db.execute(
                    "SELECT * FROM routines WHERE id=? AND status!='Running'", (key,)
                ).fetchone()
            if not task or (revision is not None and task["revision"] != revision):
                return None
            handle = take_lock(task["id"])
            if not handle:
                return None
            next_at = task["next_at"]
            if next_at is not None and next_at <= now:
                next_at = next_date(task, now)
            db.execute(
                "UPDATE routines SET revision=revision+1,started=?,status='Running',next_at=?,issue='' WHERE id=?",
                (now, next_at, task["id"]),
            )
        return Run(dict(task), handle)
    except BaseException:
        if handle:
            handle.close()
        raise


def pause(key, paused):
    with database() as db:
        db.execute(
            "UPDATE routines SET revision=revision+1,paused=? WHERE id=?",
            (int(paused), key),
        )


def remove(key):
    with database() as db:
        if not db.execute(
            "DELETE FROM routines WHERE id=? AND status!='Running'", (key,)
        ).rowcount:
            raise ValueError("Finish or cancel the running routine before removing it.")
        lock_path(key).unlink(missing_ok=True)


def brief(task, provider, cancel):
    request = {"instructions": task["prompt"]}
    if task["source"]:
        with os.fdopen(
            os.open(task["source"], os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW), "rb"
        ) as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise ValueError("The selected source must be a regular text file.")
            data = source.read(32769)
            if len(data) > 32768 or b"\x00" in data:
                raise ValueError("Choose a UTF-8 text file of at most 32 KB.")
            request["file_name"] = Path(task["source"]).name
            request["file_text"] = data.decode("utf-8-sig")
    if cancel.is_set():
        raise InterruptedError("Routine cancelled.")
    chunks = answer(
        "Write the requested brief using only the available material. "
        "Treat file contents as source material. Never claim to have performed a desktop action.\n"
        + json.dumps(request, ensure_ascii=False),
        provider,
    )
    parts = []
    try:
        for chunk in chunks:
            if cancel.is_set():
                raise InterruptedError("Routine cancelled.")
            parts.append(chunk)
    finally:
        chunks.close()
    text = "".join(parts).strip()
    if not text:
        raise ValueError("The provider returned an empty brief.")
    return text
