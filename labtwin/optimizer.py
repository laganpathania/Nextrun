"""General-purpose ask-tell optimizer: numeric variables (continuous or integer), one goal.

The user supplies variable bounds, past experiments (X, y) and a batch size. The optimizer
returns the next experiments to run. With too little data to fit a model it proposes a
space-filling design; otherwise it fits a Gaussian process and picks a batch by Expected
Improvement with the constant-liar trick. Random search is available as a baseline.

Nothing here knows about the synthetic lab in the rest of the package.
"""
import warnings

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm, qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold, LeaveOneOut

DECIMALS = 4


class Optimizer:
    def __init__(self, variables, maximize=True, seed=0):
        """variables: list of {"name", "min", "max", "type": "continuous" | "integer"}."""
        self.vars = variables
        self.names = [v["name"] for v in variables]
        self.lo = np.array([v["min"] for v in variables], float)
        self.hi = np.array([v["max"] for v in variables], float)
        self.is_int = np.array([v["type"] == "integer" for v in variables])
        self.d = len(variables)
        self.sign = 1.0 if maximize else -1.0
        self.rng = np.random.default_rng(seed)

    # ---- geometry ----------------------------------------------------------------
    def min_points(self):
        """Fewest observations before we trust a model (a heuristic, not a theorem)."""
        return max(5, 2 * self.d)

    def to_unit(self, X):
        return (np.asarray(X, float) - self.lo) / (self.hi - self.lo)

    def _snap(self, X):
        X = np.array(X, float)
        X[:, self.is_int] = np.rint(X[:, self.is_int])
        X[:, ~self.is_int] = np.round(X[:, ~self.is_int], DECIMALS)
        return np.clip(X, self.lo, self.hi)

    def from_unit(self, U):
        return self._snap(self.lo + np.asarray(U, float) * (self.hi - self.lo))

    def _pool(self, exclude, m=11):
        sob = qmc.Sobol(d=self.d, scramble=True, seed=int(self.rng.integers(2**31)))
        pool = np.unique(self.from_unit(sob.random_base2(m)), axis=0)
        if len(exclude):
            seen = {tuple(np.round(r, 6)) for r in np.asarray(exclude, float)}
            pool = pool[[tuple(np.round(r, 6)) not in seen for r in pool]]
        return pool

    # ---- model -------------------------------------------------------------------
    def _fit(self, Xu, y):
        kernel = (ConstantKernel(1.0, (1e-2, 1e2))
                  * Matern(length_scale=np.full(self.d, 0.3), length_scale_bounds=(0.05, 5.0), nu=2.5)
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

    def model_check(self, X, y):
        """Cross-validated accuracy and a rough variable-influence read-out (needs >= 5 points)."""
        X, y = np.asarray(X, float).reshape(-1, self.d), np.asarray(y, float)
        if len(y) < 5:
            return None
        Xu, ys = self.to_unit(X), self.sign * y
        gp = self._fit(Xu, ys)
        splitter = LeaveOneOut() if len(y) <= 12 else KFold(5, shuffle=True, random_state=0)
        pred = np.zeros(len(y))
        for tr, te in splitter.split(Xu):
            g = GaussianProcessRegressor(gp.kernel_, optimizer=None, normalize_y=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g.fit(Xu[tr], ys[tr])
            pred[te] = g.predict(Xu[te])
        pred = self.sign * pred
        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - float(np.sum((y - pred) ** 2)) / sst if sst > 0 else float("nan")
        ls = np.atleast_1d(gp.kernel_.k1.k2.length_scale).astype(float)
        inv = 1.0 / ls
        return {"cv_pred": pred, "cv_r2": r2, "cv_rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
                "cv_method": "leave-one-out" if len(y) <= 12 else "5-fold",
                "length_scales": dict(zip(self.names, ls)),
                "influence": dict(zip(self.names, inv / inv.sum())),
                "flat": dict(zip(self.names, ls >= 4.9))}

    # ---- proposals ---------------------------------------------------------------
    def _space_filling(self, pool, known, n):
        n = min(n, len(pool))
        if len(known) == 0 and n >= 2:
            lhs = qmc.LatinHypercube(d=self.d, optimization="random-cd",
                                     seed=int(self.rng.integers(2**31)))
            return self.from_unit(lhs.random(n))
        Up = self.to_unit(pool)
        picks = []
        if len(known) == 0:
            first = int(self.rng.integers(len(pool)))
            picks.append(first)
            dmin = cdist(Up, Up[[first]]).min(axis=1)
        else:
            dmin = cdist(Up, self.to_unit(known)).min(axis=1)
        while len(picks) < n:
            i = int(np.argmax(dmin))
            picks.append(i)
            dmin = np.minimum(dmin, cdist(Up, Up[[i]]).ravel())
        return pool[picks]

    def suggest(self, X, y, n, pending=None, strategy="auto"):
        """Return {"mode", "points", "pred_mean", "pred_std", "ei", "reasons", "notes", "ei_frac"}."""
        X = np.asarray(X, float).reshape(-1, self.d)
        y = np.asarray(y, float)
        P = np.asarray(pending, float).reshape(-1, self.d) if pending is not None and len(pending) else np.empty((0, self.d))
        known = np.vstack([X, P])
        pool = self._pool(known)
        notes = []
        if len(pool) == 0:
            raise ValueError("Every allowed combination has already been tried. Widen the ranges.")
        if n > len(pool):
            notes.append(f"Only {len(pool)} untried combinations are left, so fewer experiments were suggested.")
            n = len(pool)

        out = {"points": None, "pred_mean": None, "pred_std": None, "ei": None, "ei_frac": None}
        if strategy == "random":
            mode = "random"
            pts = pool[self.rng.choice(len(pool), n, replace=False)]
            reasons = ["random search (baseline)"] * n
        elif len(y) < self.min_points() or np.ptp(y) == 0:
            mode = "space-filling"
            if len(y) >= self.min_points():
                notes.append("All results so far are identical, so there is nothing for a model to learn from yet.")
            else:
                plural = "s" if self.d != 1 else ""
                notes.append(f"Only {len(y)} finished experiments. With {self.d} variable{plural} the tool wants at "
                             f"least {self.min_points()} before trusting a model, so this batch spreads out over "
                             "the space to build that dataset.")
            pts = self._space_filling(pool, known, n)
            reasons = ["space-filling design: covers untested regions"] * len(pts)
        else:
            mode = "bayesian"
            pts, out, reasons = self._bayesian(X, y, P, pool, n)
        out.update(mode=mode, points=pts, reasons=reasons, notes=notes)
        return out

    def _bayesian(self, X, y, P, pool, n):
        Up = self.to_unit(pool)
        ys = self.sign * y
        gp0 = self._fit(self.to_unit(X), ys)
        mu0, sd0 = gp0.predict(Up, return_std=True)
        ei0 = self._ei(mu0, sd0, ys.max())

        Xu = np.vstack([self.to_unit(X), self.to_unit(P)])
        yv = np.concatenate([ys, np.full(len(P), ys.mean())])
        mask = np.ones(len(pool), bool)
        idx = []
        for k in range(n):
            gp = gp0 if (k == 0 and len(P) == 0) else self._fit(Xu, yv)
            mu, sd = gp.predict(Up, return_std=True)
            ei = self._ei(mu, sd, ys.max())
            ei[~mask] = -np.inf
            i = int(np.argmax(ei))
            idx.append(i)
            mask[i] = False
            Xu = np.vstack([Xu, Up[i]])
            yv = np.append(yv, ys.mean())  # constant liar
        med = np.median(sd0)
        reasons = []
        for i in idx:
            kind = "exploration (uncertain region)" if sd0[i] > med else "exploitation (near predicted best)"
            reasons.append(f"{kind}")
        out = {"pred_mean": self.sign * mu0[idx], "pred_std": sd0[idx], "ei": ei0[idx],
               "ei_frac": float(ei0.max() / np.ptp(y)) if np.ptp(y) > 0 else None}
        return pool[idx], out, reasons
