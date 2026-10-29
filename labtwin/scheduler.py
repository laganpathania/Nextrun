"""SimPy discrete-event model of the virtual lab.

Each experiment becomes a workflow:
  prepare+dispense (liquid handler) -> move (plate mover) -> incubate (incubator)
  -> move (plate mover) -> measure (plate reader)

Devices have limited capacity, so jobs queue (first come, first served). A device
can fail mid-step: that experiment is aborted, the device sits in its fault state
for a recovery time, then returns to idle. The incubator can have a planned
maintenance window. Output: per-experiment outcome, event log, device state
timeline, makespan, utilization and queue waits (to name the bottleneck).
"""
import simpy

from .devices import Device

DEVICES = ("liquid_handler", "plate_mover", "incubator", "plate_reader")


def simulate_batch(experiments, cfg, rng, t_offset=0.0):
    env = simpy.Environment()
    ic = cfg["instruments"]
    timeline, events, waits, outcome = [], [], [], {}
    dev = {n: Device(env, n, ic[n]["capacity"], t_offset, timeline) for n in DEVICES}

    def log(resource, idx, what):
        events.append({"t": t_offset + env.now, "resource": resource, "exp": idx, "event": what})

    def use(name, idx, duration):
        d, spec = dev[name], ic[name]
        requested = env.now
        with d.res.request(priority=1) as req:
            yield req
            waits.append(env.now - requested)
            d.acquire(idx)
            log(name, idx, "start")
            failed = rng.random() < spec.get("failure_prob", 0.0)
            dur = duration * (float(rng.uniform(0.3, 0.9)) if failed else 1.0)
            yield env.timeout(dur)
            d.busy_time += dur
            if failed:
                d.fail(idx)
                log(name, idx, "error")
                yield env.timeout(spec.get("recovery_min", 10))
            else:
                log(name, idx, "done")
            d.release(idx)
        return not failed

    def workflow(idx, exp):
        steps = (("liquid_handler", ic["liquid_handler"]["prep_min"]),
                 ("plate_mover", ic["plate_mover"]["move_min"]),
                 ("incubator", exp["reaction_time"]),
                 ("plate_mover", ic["plate_mover"]["move_min"]),
                 ("plate_reader", ic["plate_reader"]["read_min"]))
        for name, dur in steps:
            ok = yield env.process(use(name, idx, dur))
            if not ok:
                outcome[idx] = {"status": "instrument_error", "failed_at": name}
                return
        outcome[idx] = {"status": "completed", "failed_at": None}

    def maintenance(name, start, duration):
        d = dev[name]
        yield env.timeout(start)
        reqs = [d.res.request(priority=0) for _ in range(d.capacity)]
        yield simpy.AllOf(env, reqs)  # waits for running jobs to finish
        d.set_maintenance(True)
        log(name, None, "maintenance_start")
        yield env.timeout(duration)
        d.set_maintenance(False)
        log(name, None, "maintenance_end")
        for r in reqs:
            d.res.release(r)

    procs = [env.process(workflow(i, e)) for i, e in enumerate(experiments)]
    for name, m in cfg.get("maintenance", {}).items():
        env.process(maintenance(name, m["start_min"], m["duration_min"]))
    env.run(until=simpy.AllOf(env, procs))
    makespan = env.now
    util = {n: dev[n].busy_time / (ic[n]["capacity"] * makespan) if makespan else 0.0 for n in DEVICES}
    return {
        "outcome": outcome, "events": events, "timeline": timeline, "makespan": makespan,
        "utilization": util, "bottleneck": max(util, key=util.get),
        "mean_wait": sum(waits) / len(waits) if waits else 0.0,
    }
