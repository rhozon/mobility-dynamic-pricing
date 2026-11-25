"""Day-to-day evolutionary simulator of one platform's riders and drivers across six areas.

Each day, at the evening peak:
- riders' demand for the platform in area k is q = Q * m**eps * s / s0, where m is the price
  multiplier, eps the price elasticity of ride-hail demand and s the platform's rider share;
- the wait rises when demand per driver rises: w = w0 * ((q / Q) * (x0 / x))**alpha;
- the rider share moves part of the way to its target (logit dynamic with inertia rho):
  L' = rho * L + (1 - rho) * (L0 - beta_p * log m - beta_w * (w - w0));
- drivers move between areas by an exponential replicator on yesterday's earnings per driver,
  e = theta * m * p0 * q / x, measured against what the area usually pays (pi = e / e0, which
  absorbs trip length, idle time and deadheading, so the observed split of drivers is the
  starting rest point), with imitation speed eta: x' proportional to x * exp(eta * (pi - mean pi)).

The competitor's prices and waits stay at their observed levels. Policies:
- fixed: m = 1;
- clearing: the static optimum that brings today's wait back to its usual level given today's
  drivers, m = ((x / x0) / (s / s0))**(1 / eps), within [m_min, m_max];
- smoothed: m moves a fraction lam of the way from yesterday's multiplier to the clearing one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Params:
    Q: np.ndarray          # trips per peak hour at m = 1, by area
    p0: np.ndarray         # mean fare per trip
    theta: np.ndarray      # driver pay / fare
    w0: np.ndarray         # usual wait, minutes
    s0: np.ndarray         # usual rider share of the platform
    eps: float = -0.4      # price elasticity of ride-hail demand
    alpha: float = 0.3     # elasticity of the wait to demand per driver
    beta_p: float = 0.30   # logit share response to log relative price
    beta_w: float = 0.08   # logit share response to extra minutes of wait
    rho: float = 0.61      # daily inertia of the rider share
    m_min: float = 0.7
    m_max: float = 2.5
    x0: np.ndarray = field(init=False)

    def __post_init__(self):
        self.x0 = self.Q / self.Q.sum()


def logit(s):
    return np.log(s / (1 - s))


def expit(x):
    return 1 / (1 + np.exp(-x))


def run(par: Params, policy: str = "clearing", eta: float = 1.0, lam: float = 1.0, days: int = 90,
        shock_area: int = 0, shock: float = 0.25) -> dict:
    """Simulate after a lasting demand shock of `shock` in `shock_area` on day 0."""
    K = len(par.Q)
    Q = par.Q.copy()
    Q[shock_area] *= 1 + shock
    x = par.x0.copy()
    L0 = logit(par.s0)
    L = L0.copy()
    m_prev = np.ones(K)
    e0 = par.theta * par.p0 * par.Q / par.x0
    out = {k: [] for k in ("m", "x", "s", "w", "q", "margin", "revenue", "pay")}
    for _ in range(days):
        s = expit(L)
        if policy == "fixed":
            m = np.ones(K)
        else:
            clear = ((x / par.x0) / (s / par.s0) * (par.Q / Q)) ** (1 / par.eps)
            clear = np.clip(clear, par.m_min, par.m_max)
            m = clear if policy == "clearing" else m_prev + lam * (clear - m_prev)
        q = Q * m ** par.eps * s / par.s0
        w = par.w0 * ((q / par.Q) * (par.x0 / x)) ** par.alpha
        rev = m * par.p0 * q
        pay = par.theta * rev
        e = pay / x
        for k, v in (("m", m), ("x", x), ("s", s), ("w", w), ("q", q)):
            out[k].append(v.copy())
        out["revenue"].append(rev.sum()); out["pay"].append(pay.sum()); out["margin"].append((rev - pay).sum())
        # updates for tomorrow
        L = par.rho * L + (1 - par.rho) * (L0 - par.beta_p * np.log(m) - par.beta_w * (w - par.w0))
        pi = e / e0
        g = x * np.exp(np.clip(eta * (pi - np.average(pi, weights=x)), -5, 5))
        x = np.maximum(g / g.sum(), 1e-4)
        x = x / x.sum()
        m_prev = m
    return {k: np.array(v) for k, v in out.items()}


def metrics(r: dict, burn: int = 30) -> dict:
    tail = slice(burn, None)
    return {
        "margin": float(r["margin"][tail].mean()),
        "revenue": float(r["revenue"][tail].mean()),
        "wait": float((r["w"][tail] * r["q"][tail]).sum() / r["q"][tail].sum()),
        "supply_swing": float(r["x"][tail].std(axis=0).max() / r["x"][tail].mean(axis=0)[r["x"][tail].std(axis=0).argmax()]),
        "price_swing": float(r["m"][tail].std(axis=0).max()),
    }


def phase(par: Params, etas, lams, **kw) -> list[dict]:
    rows = []
    for eta in etas:
        for lam in lams:
            mt = metrics(run(par, "smoothed", eta=eta, lam=lam, **kw))
            rows.append({"eta": float(eta), "lam": float(lam), **mt})
    return rows
