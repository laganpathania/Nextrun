"""Virtual instruments as small state machines.

Device       idle        busy          fault     maintenance
liquid_handler  idle     running       error     -
plate_reader    idle     measuring     error     -
plate_mover     idle     transferring  blocked   -
incubator       available occupied (all slots full)  -  maintenance

Every state change is validated against the allowed transitions and recorded in
a timeline, which the OPC UA server replays.
"""
import simpy

SPECS = {
    "liquid_handler": {"idle": "idle", "busy": "running", "fault": "error"},
    "plate_reader": {"idle": "idle", "busy": "measuring", "fault": "error"},
    "plate_mover": {"idle": "idle", "busy": "transferring", "fault": "blocked"},
    "incubator": {"idle": "available", "busy": "occupied", "fault": None, "maint": "maintenance"},
}


class IllegalDeviceTransition(Exception):
    pass


def allowed_transitions(spec):
    a = {(spec["idle"], spec["busy"]), (spec["busy"], spec["idle"])}
    if spec.get("fault"):
        a |= {(spec["busy"], spec["fault"]), (spec["fault"], spec["idle"])}
    if spec.get("maint"):
        a |= {(spec["idle"], spec["maint"]), (spec["maint"], spec["idle"])}
    return a


class Device:
    def __init__(self, env, name, capacity, t_offset=0.0, timeline=None):
        self.env, self.name, self.spec, self.capacity = env, name, SPECS[name], capacity
        self.allowed = allowed_transitions(self.spec)
        self.res = simpy.PriorityResource(env, capacity)
        self.in_use, self.faulted, self.maint = 0, False, False
        self.busy_time = 0.0
        self.t_offset = t_offset
        self.timeline = timeline if timeline is not None else []
        self.state = self.spec["idle"]
        self._record(None)

    def _record(self, exp):
        self.timeline.append({"t": self.t_offset + self.env.now, "device": self.name,
                              "state": self.state, "exp": exp})

    def _refresh(self, exp):
        s = self.spec
        if self.maint:
            new = s["maint"]
        elif self.faulted:
            new = s["fault"]
        elif self.in_use >= self.capacity:
            new = s["busy"]
        else:
            new = s["idle"]
        if new != self.state:
            if (self.state, new) not in self.allowed:
                raise IllegalDeviceTransition(f"{self.name}: {self.state} -> {new}")
            self.state = new
            self._record(exp)

    def acquire(self, exp):
        self.in_use += 1
        self._refresh(exp)

    def fail(self, exp):
        if not self.spec.get("fault"):
            raise IllegalDeviceTransition(f"{self.name} has no fault state")
        self.faulted = True
        self._refresh(exp)

    def release(self, exp):
        self.in_use -= 1
        self.faulted = False
        self._refresh(exp)

    def set_maintenance(self, on):
        self.maint = on
        self._refresh(None)
