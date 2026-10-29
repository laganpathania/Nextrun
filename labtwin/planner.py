"""Experiment planner: Latin-hypercube DoE first, then Gaussian-process Bayesian
optimization with Expected Improvement and constant-liar batch selection.
Also a random-search baseline for benchmarking."""
import warnings
import numpy as np
from scipy.stats import norm, qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .feasibility import check_experiment

VERSION = "planner-0.1"


class Planner:
    def __init__(self, cfg, strategy="bo", seed=0):
        assert strategy in ("bo", "random")
        self.cfg, self.strategy = cfg, strategy
        self.rng = np.random.default_rng(seed)
        self.names = list(cfg["factors"])
        self.lo = np.array([cfg["factors"][k]["min"] for k in self.names], float)
        self.hi = np.array([cfg["factors"][k]["max"] for k in self.names], float)

    def to_unit(self, X):
        return (np.asarray(X) - self.lo) / (self.hi - self.lo)

    def from_unit(self, U):
        return self.lo + np.asarray(U) * (self.hi - self.lo)

    def as_dict(self, x):
        return {k: float(v) for k, v in zip(self.names, x)}

    def _feasible(self, x):
        return not check_experiment(self.as_dict(x), self.cfg)

    def _feasible_pool(self, n):
        pool = []
        for _ in range(20):
            sampler = qmc.Sobol(d=len(self.names), scramble=True,
                                seed=int(self.rng.integers(2**31)))
            X = self.from_unit(sampler.random_base2(10))
            pool.extend(x for x in X if self._feasible(x))
            if len(pool) >= n:
                break
        return np.array(pool[:n])

    def _wrap(self, X, stage, mu=None, sd=None, ei=None, reasons=None):
        out = []
        for i, x in enumerate(X):
            out.append({
                "factors": self.as_dict(x), "stage": stage,
                "pred_mean": None if mu is None else float(mu[i]),
                "pred_std": None if sd is None else float(sd[i]),
                "ei": None if ei is None else float(ei[i]),
                "reason": reasons[i] if reasons else stage,
            })
        return out

    def _doe(self, n):
        lhs = qmc.LatinHypercube(d=len(self.names), seed=int(self.rng.integers(2**31)))
        X = self.from_unit(lhs.random(n))
        spare = self._feasible_pool(n)
        k = 0
        for i in range(n):
            if not self._feasible(X[i]):
                X[i] = spare[k]
                k += 1
        return self._wrap(X, "doe", reasons=["initial space-filling design (Latin hypercube)"] * n)

    def _random(self, n):
        pool = self._feasible_pool(n * 4)
        idx = self.rng.choice(len(pool), n, replace=False)
        return self._wrap(pool[idx], "random", reasons=["random baseline"] * n)

    def _fit(self, Xu, y):
        d = Xu.shape[1]
        kernel = (ConstantKernel(1.0, (1e-2, 1e2))
                  * Matern(length_scale=np.full(d, 0.3), length_scale_bounds=(0.05, 5.0), nu=2.5)
                  + WhiteKernel(0.01, (1e-4, 1.0)))
        gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2,
                                      random_state=int(self.rng.integers(2**31)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(Xu, y)
        return gp

    @staticmethod
    def _ei(mu, sd, best):
        sd = np.maximum(sd, 1e-9)
        z = (mu - best) / sd
        return (mu - best) * norm.cdf(z) + sd * norm.pdf(z)

    def propose(self, X, y, n):
        y = np.asarray(y, float)
        if self.strategy == "random":
            return self._random(n)
        if len(y) < self.cfg["planner"]["min_points_for_gp"]:
            return self._doe(n)
        return self._bo(np.asarray(X, float).reshape(-1, len(self.names)), y, n)

    def _bo(self, X, y, n):
        p = self.cfg["planner"]
        pool = self._feasible_pool(p["n_candidates"])
        Up = self.to_unit(pool)
        time_norm = Up[:, self.names.index("reaction_time")]
        cost_norm = (pool[:, self.names.index("reagent_a")]
                     + pool[:, self.names.index("reagent_b")]) / self.cfg["constraints"]["max_total_reagent"]
        Xu, yv = self.to_unit(X), y.copy()
        mask = np.ones(len(pool), bool)
        picks, mus, sds, eis, reasons = [], [], [], [], []
        for _ in range(n):
            gp = self._fit(Xu, yv)
            mu, sd = gp.predict(Up, return_std=True)
            ei = self._ei(mu, sd, yv.max())
            acq = ei / (ei.max() + 1e-12) - p["w_time"] * time_norm - p["w_cost"] * cost_norm
            acq[~mask] = -np.inf
            i = int(np.argmax(acq))
            mask[i] = False
            picks.append(pool[i]); mus.append(mu[i]); sds.append(sd[i]); eis.append(ei[i])
            kind = ("exploration: high-uncertainty region" if sd[i] > np.median(sd)
                    else "exploitation: near predicted optimum")
            reasons.append(f"{kind}; EI={ei[i]:.3f}")
            Xu = np.vstack([Xu, Up[i]])
            yv = np.append(yv, yv.mean())  # constant liar
        return self._wrap(np.array(picks), "bo", np.array(mus), np.array(sds), np.array(eis), reasons)
