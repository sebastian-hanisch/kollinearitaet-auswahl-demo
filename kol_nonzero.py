"""Die Zahl der Nichtnullen und die Rauschgrenze: ohne Schwellenangabe ist „x Merkmale gewählt“ keine Messung.

Der Koordinatenabstieg (`kol_algorithm.cd_batch`) liefert exakte Nullen; ein **glattes** Verfahren (hier L-BFGS-B auf |b| ≈ √(b² + ε)) liefert nie exakte Nullen, dort
entscheidet nur die Schwelle, welche Koeffizienten als gewählt zählen. Zähleinheit: die Anpassung (Stichprobe)."""

import numpy as np
from scipy.optimize import minimize


def threshold_table(b, thresholds, n_dummy_from=3):
    """b (K, p) Koeffizienten -> je Schwelle: Häufigkeit von x1, x2, x3, den Attrappen x4..x10 (Mittel) und die mittlere Zahl der Nichtnullen (Zähleinheit: Anpassung)."""
    rows = []
    for thr in thresholds:
        sel = np.abs(b) > thr
        rows.append({"thr": float(thr), "x1": float(sel[:, 0].mean()), "x2": float(sel[:, 1].mean()), "x3": float(sel[:, 2].mean()),
                     "dummies": float(sel[:, n_dummy_from:].mean()), "nonzeros": float(sel.sum(axis=1).mean())})
    return rows


def smallest_chosen(b):
    """Kleinster Betrag unter den Koeffizienten ungleich 0, je Anpassung (nan, wenn alle 0)."""
    mag = np.where(b != 0.0, np.abs(b), np.inf)
    m = mag.min(axis=1)
    return np.where(np.isfinite(m), m, np.nan)


def smooth_fit(G, c, alpha, eps):
    """Glatte Näherung des Lasso: minimiere ½ b'Gb - c'b + α Σ √(b² + ε) mit L-BFGS-B (analytischer Gradient). Rückgabe b; keine exakten Nullen."""
    def fun(b):
        r = np.sqrt(b * b + eps)
        return 0.5 * b @ G @ b - c @ b + alpha * r.sum(), G @ b - c + alpha * b / r

    res = minimize(fun, np.zeros(len(c)), jac=True, method="L-BFGS-B", options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-12, "maxcor": 30})
    return res.x


def smooth_agreement(G, c, alpha, b_exact, eps, thresholds):
    """Wie oft wählt das glatte Verfahren dieselbe Menge wie der exakte Abstieg: je Schwelle die Zahl der Anpassungen mit gleicher Menge (und die mittlere Zahl der Nichtnullen).
    G (K, p, p), c (K, p), b_exact (K, p)."""
    K = c.shape[0]
    bs = np.array([smooth_fit(G[k], c[k], alpha, eps) for k in range(K)])
    exact_set = b_exact != 0.0
    out = []
    for thr in thresholds:
        sel = np.abs(bs) > thr
        out.append({"thr": float(thr), "same": int(np.sum(np.all(sel == exact_set, axis=1))), "of": K, "nonzeros": float(sel.sum(axis=1).mean())})
    return out
