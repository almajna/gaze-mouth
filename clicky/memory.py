import json
import sqlite3
import time
import unicodedata
import uuid
from contextlib import closing, contextmanager
from pathlib import Path

DATABASE = Path.home() / "Library/Application Support/Clicky/history.sqlite3"


def compact(text):
    line = " ".join(text.split())
    return line if len(line) <= 140 else line[:139] + "…"


def observed_window(window):
    selection = dict.fromkeys(compact(e.label) for e in window.elements if e.selected and e.label)
    observed = {"app": window.app, "window": window.title, "selection": list(selection)[:12]}
    if window.selected_text is not None:
        observed["highlight_excerpt"] = compact(window.selected_text)
    return observed


def screen_excerpt(window):
    if window and (window.app or window.title):
        return {"app": compact(window.app or ""), "window": compact(window.title or "")}
    return None


@contextmanager
def connect():
    DATABASE.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    with closing(sqlite3.connect(DATABASE)) as db, db:
        db.execute("PRAGMA secure_delete=ON")
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, request TEXT, answer TEXT, created INTEGER)"
        )
        columns = {r[1] for r in db.execute("PRAGMA table_info(history)")}
        if "created" not in columns:
            db.execute("ALTER TABLE history ADD COLUMN created INTEGER")
        db.execute("UPDATE history SET created=? WHERE created IS NULL", (int(time.time()),))
        if "conversation" not in columns:
            db.execute("ALTER TABLE history ADD COLUMN conversation TEXT")
        for column in ("app_name", "window_title", "walkthrough"):
            if column not in columns:
                db.execute(f"ALTER TABLE history ADD COLUMN {column} TEXT")
        # Older CursorMix builds can still append ungrouped rows to this database.
        db.execute("UPDATE history SET conversation=lower(hex(randomblob(16))) WHERE conversation IS NULL")
        db.execute("CREATE INDEX IF NOT EXISTS history_conversation ON history(conversation, id)")
        db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS history_walkthrough ON history(conversation) "
            "WHERE walkthrough IS NOT NULL"
        )
        yield db


def expire(db, days):
    if days < 1:
        raise ValueError("History retention must be positive.")
    db.execute("DELETE FROM history WHERE created < ?", (int(time.time()) - days * 86400,))


def prune(days):
    if DATABASE.exists():
        with connect() as db:
            expire(db, days)


def save_walkthrough(key, state, days, *, create=False):
    summary = state["pending"]["instruction"] if state["pending"] else "Checking the next step…"
    screen = state["starting"]
    values = (
        compact(state["goal"]),
        compact(summary),
        int(time.time()),
        screen["app"],
        screen["window"],
        json.dumps(state, ensure_ascii=False),
        key,
    )
    with connect() as db:
        expire(db, days)
        if create:
            db.execute(
                """INSERT INTO history(request,answer,created,app_name,window_title,walkthrough,conversation)
                   VALUES (?,?,?,?,?,?,?) ON CONFLICT DO NOTHING""",
                values,
            )
        else:
            # Updates cannot resurrect a walkthrough cleared or expired since it started.
            db.execute(
                """UPDATE history SET request=?,answer=?,created=?,app_name=?,window_title=?,walkthrough=?
                   WHERE conversation=? AND walkthrough IS NOT NULL""",
                values,
            )


def walkthrough(key, days):
    if not key or not key.startswith("walkthrough/"):
        return None
    with connect() as db:
        expire(db, days)
        row = db.execute("SELECT walkthrough FROM history WHERE conversation=?", (key,)).fetchone()
    return json.loads(row[0]) if row and row[0] else None


def discard_walkthrough(key):
    with connect() as db:
        db.execute("DELETE FROM history WHERE conversation=? AND walkthrough IS NOT NULL", (key,))


def remember(prompt, answer, days=30, *, conversation=None, screen=None, completed_walkthrough=None):
    conversation = conversation or uuid.uuid4().hex
    with connect() as db:
        expire(db, days)
        if completed_walkthrough:
            removed = db.execute(
                "DELETE FROM history WHERE conversation=? AND walkthrough IS NOT NULL",
                (completed_walkthrough,),
            )
            if not removed.rowcount:
                return conversation
        db.execute(
            "INSERT INTO history(request,answer,created,conversation,app_name,window_title) VALUES (?,?,?,?,?,?)",
            (
                compact(prompt),
                compact(answer),
                int(time.time()),
                conversation,
                screen.get("app") if screen else None,
                screen.get("window") if screen else None,
            ),
        )
    return conversation


def conversations(days, query=""):
    with connect() as db:
        expire(db, days)
        term = unicodedata.normalize("NFC", query.strip()).casefold()
        matches = ""
        if term:
            db.create_function(
                "contains_term",
                1,
                lambda value: term in unicodedata.normalize("NFC", value or "").casefold(),
                deterministic=True,
            )
            matches = """HAVING MAX(contains_term(request) OR contains_term(answer)
                                    OR contains_term(app_name) OR contains_term(window_title))"""
        return db.execute(
            f"""SELECT latest.conversation, first.request, latest.created
               FROM (SELECT MIN(id) AS first_id, MAX(id) AS last_id
                     FROM history GROUP BY conversation {matches}) g
               JOIN history first ON first.id=g.first_id JOIN history latest ON latest.id=g.last_id
               ORDER BY latest.created DESC, latest.id DESC LIMIT 50"""
        ).fetchall()


def turns(conversation, days):
    with connect() as db:
        expire(db, days)
        rows = db.execute(
            "SELECT request, answer, app_name, window_title FROM history WHERE conversation=? ORDER BY id",
            (conversation,),
        )
        return [
            {
                "request": request,
                "reply": answer,
                **({"screen_at_answer": {"app": app or "", "window": title or ""}} if app or title else {}),
            }
            for request, answer, app, title in rows
        ]


def recent(days=30):
    with connect() as db:
        expire(db, days)
        return db.execute("SELECT request, answer FROM history ORDER BY id DESC LIMIT 50").fetchall()


def clear():
    with connect() as db:
        db.execute("DELETE FROM history")
