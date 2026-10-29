"""Feasibility checks. The planner filters candidates through check_experiment
before proposing, and the orchestrator re-checks the whole batch."""
from dataclasses import dataclass, field


@dataclass
class FeasibilityReport:
    rejected: dict = field(default_factory=dict)       # experiment index -> reasons
    batch_reasons: list = field(default_factory=list)  # batch-level problems

    @property
    def passed(self):
        return not self.rejected and not self.batch_reasons

    def summary(self):
        if self.passed:
            return "all checks passed"
        parts = [f"exp {i}: {'; '.join(r)}" for i, r in self.rejected.items()]
        return " | ".join(parts + self.batch_reasons)


def check_experiment(exp, cfg):
    reasons = []
    for k, spec in cfg["factors"].items():
        v = exp[k]
        if not spec["min"] <= v <= spec["max"]:
            reasons.append(f"{k}={v:.2f} outside [{spec['min']}, {spec['max']}]")
    c = cfg["constraints"]
    total = exp["reagent_a"] + exp["reagent_b"]
    if total > c["max_total_reagent"]:
        reasons.append(f"reagent_a+reagent_b={total:.1f} exceeds {c['max_total_reagent']}")
    for k in ("reagent_a", "reagent_b"):
        if exp[k] < c["min_pipette_volume"]:
            reasons.append(f"{k}={exp[k]:.2f} below min pipetting volume {c['min_pipette_volume']}")
    return reasons


def check_batch(batch, cfg, reagent_used=0.0, n_done=0):
    c = cfg["constraints"]
    rep = FeasibilityReport()
    for i, e in enumerate(batch):
        r = check_experiment(e, cfg)
        if r:
            rep.rejected[i] = r
    if n_done + len(batch) > c["max_experiments"]:
        rep.batch_reasons.append(
            f"batch would exceed experiment budget ({n_done}+{len(batch)}>{c['max_experiments']})")
    need = sum(e["reagent_a"] + e["reagent_b"] for e in batch)
    if reagent_used + need > c["reagent_budget_total"]:
        rep.batch_reasons.append(
            f"batch would exceed reagent budget ({reagent_used + need:.0f}>{c['reagent_budget_total']})")
    return rep
