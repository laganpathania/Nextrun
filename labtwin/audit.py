"""SQLite audit log with an enforced batch state machine and a hash-chained event
table, so edited or deleted history is detectable."""
import hashlib
import json
import sqlite3
import time

import numpy as np

ALLOWED = {
    "PROPOSED": {"FEASIBILITY_CHECKED"},
    "FEASIBILITY_CHECKED": {"APPROVED", "REJECTED"},
    "APPROVED": {"RUNNING"},
    "RUNNING": {"MEASURED"},
    "MEASURED": {"LEARNED"},
    "LEARNED": set(),
    "REJECTED": set(),
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches(id INTEGER PRIMARY KEY, created_at TEXT, state TEXT,
    planner_version TEXT, seed INTEGER, config_hash TEXT);
CREATE TABLE IF NOT EXISTS experiments(id INTEGER PRIMARY KEY, batch_id INTEGER, idx INTEGER,
    factors_json TEXT, predicted_mean REAL, predicted_std REAL, reason TEXT,
    status TEXT, result REAL, qc_ok INTEGER);
CREATE TABLE IF NOT EXISTS decisions(id INTEGER PRIMARY KEY, batch_id INTEGER, actor TEXT,
    decision TEXT, comment TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, batch_id INTEGER, ts TEXT,
    source TEXT, event_type TEXT, payload_json TEXT, prev_hash TEXT, hash TEXT);
"""


class IllegalTransition(Exception):
    pass


def _j(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


class AuditLog:
    def __init__(self, path=":memory:"):
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.executescript(SCHEMA)

    def new_batch(self, planner_version, seed, config_hash):
        cur = self.con.execute(
            "INSERT INTO batches(created_at,state,planner_version,seed,config_hash) VALUES(?,?,?,?,?)",
            (time.strftime("%Y-%m-%dT%H:%M:%S"), "PROPOSED", planner_version, seed, config_hash))
        self.con.commit()
        return cur.lastrowid

    def state(self, batch_id):
        return self.con.execute("SELECT state FROM batches WHERE id=?", (batch_id,)).fetchone()[0]

    def set_state(self, batch_id, new):
        cur = self.state(batch_id)
        if new not in ALLOWED[cur]:
            raise IllegalTransition(f"batch {batch_id}: {cur} -> {new} is not allowed")
        self.con.execute("UPDATE batches SET state=? WHERE id=?", (new, batch_id))
        self.event(batch_id, "orchestrator", "state_change", {"from": cur, "to": new})

    def add_experiments(self, batch_id, proposals):
        for i, p in enumerate(proposals):
            self.con.execute(
                "INSERT INTO experiments(batch_id,idx,factors_json,predicted_mean,predicted_std,reason,status)"
                " VALUES(?,?,?,?,?,?,?)",
                (batch_id, i, json.dumps(p["factors"]), p["pred_mean"], p["pred_std"], p["reason"], "proposed"))
        self.con.commit()

    def update_experiment(self, batch_id, idx, status, result=None, qc_ok=None):
        self.con.execute("UPDATE experiments SET status=?, result=?, qc_ok=? WHERE batch_id=? AND idx=?",
                         (status, result, None if qc_ok is None else int(qc_ok), batch_id, idx))
        self.con.commit()

    def add_decision(self, batch_id, actor, decision, comment):
        self.con.execute("INSERT INTO decisions(batch_id,actor,decision,comment,ts) VALUES(?,?,?,?,?)",
                         (batch_id, actor, decision, comment, time.strftime("%Y-%m-%dT%H:%M:%S")))
        self.event(batch_id, actor, "decision", {"decision": decision, "comment": comment})

    def _last_hash(self):
        r = self.con.execute("SELECT hash FROM events ORDER BY id DESC LIMIT 1").fetchone()
        return r[0] if r else "GENESIS"

    @staticmethod
    def _digest(prev, batch_id, ts, source, etype, payload):
        return hashlib.sha256("|".join([prev, str(batch_id), ts, source, etype, payload]).encode()).hexdigest()

    def event(self, batch_id, source, event_type, payload):
        ts = f"{time.time():.6f}"
        p = json.dumps(payload, sort_keys=True, default=_j)
        prev = self._last_hash()
        h = self._digest(prev, batch_id, ts, source, event_type, p)
        self.con.execute("INSERT INTO events(batch_id,ts,source,event_type,payload_json,prev_hash,hash)"
                         " VALUES(?,?,?,?,?,?,?)", (batch_id, ts, source, event_type, p, prev, h))
        self.con.commit()

    def verify_chain(self):
        prev = "GENESIS"
        for (bid, ts, src, et, p, ph, h) in self.con.execute(
                "SELECT batch_id,ts,source,event_type,payload_json,prev_hash,hash FROM events ORDER BY id"):
            if ph != prev or self._digest(prev, bid, ts, src, et, p) != h:
                return False
            prev = h
        return True
