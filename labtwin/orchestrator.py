"""Closed loop: propose -> feasibility -> human approval -> simulate -> measure ->
QC -> learn -> next batch. Every step goes to the audit log."""
import numpy as np

from .audit import AuditLog
from .feasibility import check_batch
from .groundtruth import GroundTruth
from .planner import Planner, VERSION
from .scheduler import simulate_batch


def auto_approver(batch_id, proposals, report):
    return True, "auto-run mode"


def cli_approver(batch_id, proposals, report):
    print(f"\n=== Batch {batch_id}: {len(proposals)} proposed experiments ===")
    for i, p in enumerate(proposals):
        f = p["factors"]
        print(f" {i}: " + ", ".join(f"{k}={v:.1f}" for k, v in f.items()) + f"  <- {p['reason']}")
    print("Feasibility:", report.summary())
    while True:
        ans = input("Approve? [y/n]: ").strip().lower()
        if ans in ("y", "n"):
            break
    comment = input("Comment" + (" (required for rejection): " if ans == "n" else " (optional): ")).strip()
    while ans == "n" and not comment:
        comment = input("A rejection needs a comment: ").strip()
    return ans == "y", comment


def run_campaign(cfg, strategy="bo", seed=0, db_path=":memory:", approver=auto_approver,
                 stop_on_stall=False, verbose=False, progress=None):
    gt = GroundTruth(cfg, np.random.default_rng(seed + 1))
    planner = Planner(cfg, strategy, seed + 2)
    sched_rng = np.random.default_rng(seed + 3)
    audit = AuditLog(db_path)
    camp, cons = cfg["campaign"], cfg["constraints"]

    X, y, records, rounds, timeline = [], [], [], [], []
    reagent_used, n_run, clock, best, stall = 0.0, 0, 0.0, -np.inf, 0
    stop_reason = "rounds_complete"

    for rnd in range(camp["rounds"]):
        n = camp["initial_batch"] if rnd == 0 else camp["batch_size"]
        n = min(n, cons["max_experiments"] - n_run)
        if n <= 0:
            stop_reason = "experiment_budget_exhausted"
            break

        proposals = planner.propose(X, y, n)
        batch_id = audit.new_batch(f"{VERSION}/{strategy}", seed, cfg["_hash"])
        audit.add_experiments(batch_id, proposals)
        exps = [p["factors"] for p in proposals]

        report = check_batch(exps, cfg, reagent_used, n_run)
        audit.set_state(batch_id, "FEASIBILITY_CHECKED")
        audit.event(batch_id, "feasibility", "report", {"passed": report.passed, "summary": report.summary()})
        if not report.passed:
            audit.add_decision(batch_id, "system", "rejected", report.summary())
            audit.set_state(batch_id, "REJECTED")
            stop_reason = "feasibility_rejected"
            break

        approved, comment = approver(batch_id, proposals, report)
        audit.add_decision(batch_id, "human" if approver is not auto_approver else "auto",
                           "approved" if approved else "rejected", comment)
        audit.set_state(batch_id, "APPROVED" if approved else "REJECTED")
        if not approved:
            stop_reason = "rejected_by_human"
            break

        audit.set_state(batch_id, "RUNNING")
        sim = simulate_batch(exps, cfg, sched_rng, clock)
        for ev in sim["events"]:
            audit.event(batch_id, ev["resource"], f"instrument_{ev['event']}", ev)
        timeline.extend(sim["timeline"])
        clock += sim["makespan"]

        for i, exp in enumerate(exps):
            out = sim["outcome"][i]
            if out["status"] == "instrument_error":
                audit.update_experiment(batch_id, i, "instrument_error")
                audit.event(batch_id, "exception_queue", "instrument_error",
                            {"exp": i, "failed_at": out["failed_at"]})
                records.append({"factors": exp, "value": None, "status": "instrument_error"})
                continue
            m = gt.measure(exp)
            audit.update_experiment(batch_id, i, m["status"], m["value"], m["qc_ok"])
            if m["status"] == "ok":
                X.append([exp[k] for k in planner.names]); y.append(m["value"])
                records.append({"factors": exp, "value": m["value"], "status": "ok"})
            elif m["status"] == "assay_failed":
                # Informative about chemistry: kept in training with its (no-signal) value.
                audit.event(batch_id, "exception_queue", "assay_failed", {"exp": i, "factors": exp})
                X.append([exp[k] for k in planner.names]); y.append(m["value"])
                records.append({"factors": exp, "value": m["value"], "status": "assay_failed"})
            else:
                # QC failure says nothing about chemistry: excluded from training.
                audit.event(batch_id, "exception_queue", "qc_failed",
                            {"exp": i, "control": m["control"]})
                records.append({"factors": exp, "value": None, "status": "qc_failed"})
        audit.set_state(batch_id, "MEASURED")

        n_run += n
        reagent_used += sum(e["reagent_a"] + e["reagent_b"] for e in exps)
        round_best = max([r["value"] for r in records if r["value"] is not None], default=-np.inf)
        stall = stall + 1 if round_best < best * 1.01 else 0
        best = max(best, round_best)
        audit.set_state(batch_id, "LEARNED")
        rounds.append({"round": rnd, "batch_id": batch_id, "best": best, "makespan": sim["makespan"],
                       "utilization": sim["utilization"], "bottleneck": sim["bottleneck"],
                       "mean_wait": sim["mean_wait"]})
        if progress is not None:
            progress(rnd + 1, camp["rounds"])
        if verbose:
            u = ", ".join(f"{k}={v:.0%}" for k, v in sim["utilization"].items())
            print(f"round {rnd}: best={best:.2f}  makespan={sim['makespan']:.0f} min  "
                  f"bottleneck={sim['bottleneck']}  util: {u}")
        if stop_on_stall and stall >= 2:
            stop_reason = "improvement_stalled"
            break

    valid = [r for r in records if r["value"] is not None]
    top = max(valid, key=lambda r: r["value"]) if valid else None
    return {"records": records, "rounds": rounds, "best": best, "best_factors": top["factors"] if top else None,
            "stop_reason": stop_reason, "audit": audit, "n_run": n_run, "timeline": timeline}


def best_so_far(records, floor=1.0):
    out, cur = [], floor
    for r in records:
        if r["value"] is not None:
            cur = max(cur, r["value"])
        out.append(cur)
    return out
