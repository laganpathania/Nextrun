"""Hidden synthetic 'assay biology'. The planner only ever sees noisy measurements.

Signal-to-background ratio is a Gaussian bump around a hidden optimum (a smooth
stand-in for a response surface), plus a chemistry failure region, noise, and
occasional QC outliers. This is a made-up function: it tests the *method*, not
real-world performance.
"""
import numpy as np


class GroundTruth:
    def __init__(self, cfg, rng):
        self.g = cfg["groundtruth"]
        self.rng = rng

    def true_signal(self, exp):
        g = self.g
        z = sum(((exp[k] - g["optimum"][k]) / g["width"][k]) ** 2 for k in g["optimum"])
        return g["baseline"] + (g["peak"] - g["baseline"]) * np.exp(-z)

    def in_failure_region(self, exp):
        return exp["temperature"] > self.g["fail_temp"] and exp["ph"] < self.g["fail_ph"]

    def measure(self, exp):
        g = self.g
        control = float(np.exp(self.rng.normal(0, 0.03)))  # control well, nominally 1.0
        if self.in_failure_region(exp):
            value = g["baseline"] * float(np.exp(self.rng.normal(0, g["noise_sd"])))
            return {"status": "assay_failed", "value": value, "control": control, "qc_ok": True}
        value = self.true_signal(exp) * float(np.exp(self.rng.normal(0, g["noise_sd"])))
        if self.rng.random() < g["qc_outlier_prob"]:
            value *= float(self.rng.uniform(0.2, 0.5))
            control *= float(self.rng.uniform(1.3, 1.6))
        qc_ok = abs(control - 1.0) <= g["qc_control_tolerance"]
        return {"status": "ok" if qc_ok else "qc_failed", "value": value,
                "control": control, "qc_ok": qc_ok}
