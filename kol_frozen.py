"""Die vorgerechnete Messreihe der App (`data/kol_frozen.json`): große Zählungen, die live zu lange dauern würden.

Erzeugt von `tools/kol_make_frozen.py` (Laufzeit einige Minuten); `build(...)` rechnet dieselbe Messreihe neu, `tests/test_frozen.py` prüft die Probe jeder Zelle neu
(Band) und mit KOL_FULL=1 die ganze Messreihe. Zähleinheiten stehen an jedem Block: Wiederholung (neue Fehler), Design (Zufallsdesign), frische Stichprobe,
Originalstichprobe (je B = 100 Bootstrap- und Teilstichproben), Stichprobe (Auswahl der Attrappe)."""

import json
from pathlib import Path

import numpy as np

import kol_algorithm as A
import kol_condition as COND
import kol_constants as C
import kol_inference as I
import kol_nonzero as NZ
import kol_sampling as SA
import kol_stream as S

ROOT = Path(__file__).resolve().parent
FROZEN_PATH = ROOT / "data" / "kol_frozen.json"

N, RHO, SIGMA = 100, 0.95, 2.0
M_ORIG, B_RES, N_FRESH = 200, 100, 2000
PROBE_M, PROBE_FRESH = 10, 200
ALPHA_L, ALPHA_E = 0.09, 0.13
SEED_CELLS = 12000
# Messzellen: Schlüssel -> (Welt, Szenario, ℓ, α, ρ, Zufalls-Seed)
CELLS = {
    "C_S3_L": ("C", "S3", 1.0, ALPHA_L, 0.95, SEED_CELLS + 1),
    "C_S3_L_seed2": ("C", "S3", 1.0, ALPHA_L, 0.95, SEED_CELLS + 2),
    "C_S1_L": ("C", "S1", 1.0, ALPHA_L, 0.95, SEED_CELLS + 3),
    "C_S2_L": ("C", "S2", 1.0, ALPHA_L, 0.95, SEED_CELLS + 4),
    "C_S3_E": ("C", "S3", 0.5, ALPHA_E, 0.95, SEED_CELLS + 5),
    "C_S2_E": ("C", "S2", 0.5, ALPHA_E, 0.95, SEED_CELLS + 6),
    "C_S3_L_rho099": ("C", "S3", 1.0, ALPHA_L, 0.99, SEED_CELLS + 7),
    "G_S3_L": ("G", "S3", 1.0, ALPHA_L, 0.95, SEED_CELLS + 8),
    "G_S1_L": ("G", "S1", 1.0, ALPHA_L, 0.95, SEED_CELLS + 9),
    "G_S2_L": ("G", "S2", 1.0, ALPHA_L, 0.95, SEED_CELLS + 10),
    "E_S2E_L": ("E", "S2E", 1.0, ALPHA_L, 0.95, SEED_CELLS + 11),
    "E_S2E_E": ("E", "S2E", 0.5, ALPHA_E, 0.95, SEED_CELLS + 12),
}


def run_cell(key, m_orig=M_ORIG, n_fresh=N_FRESH):
    world, scen, l1, alpha, rho, seed = CELLS[key]
    out = SA.selection_cell(world, scen, N, rho, SIGMA, alpha, l1, m_orig, B_RES, n_fresh, seed, pi=0.8)
    out["params"] = {"world": world, "scen": scen, "l1": l1, "alpha": alpha, "rho": rho, "n": N, "sigma": SIGMA, "seed": seed, "m_orig": m_orig, "B": B_RES, "n_fresh": n_fresh}
    return out


def selection_block(progress=None):
    cells = {}
    for key in CELLS:
        cell = run_cell(key)
        cell["probe"] = run_cell(key, PROBE_M, PROBE_FRESH)
        cells[key] = cell
        if progress:
            progress(key)
    return cells


def inference_blocks():
    sumdiff = []
    for rho in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999):
        X = I.fixed_design(N, rho, 5)
        row = I.variance_check(X, 1.0, 100000, 1201)
        row.update({"rho": rho, "reps": 100000, "formula_single": 1.0 / (1.0 - rho * rho), "formula_sum": 2.0 / (1.0 + rho), "formula_diff": 2.0 / (1.0 - rho)})
        sumdiff.append(row)
    tests = []
    for b in (0.5, 1.0):
        for rho in (0.0, 0.5, 0.9, 0.95, 0.99):
            X = I.fixed_design(N, rho, 5)
            row = I.test_rates(X, (b, b), SIGMA, 20000, 1202)
            row.update({"b": b, "rho": rho})
            tests.append(row)
    coverage = []
    for rho in (0.0, 0.9, 0.99, 0.999):
        X = I.fixed_design(N, rho, 5)
        row = I.test_rates(X, (0.5, 0.5), SIGMA, 20000, 1203)
        coverage.append({"rho": rho, "cover": row["cover"], "width": row["width"], "reps": 20000})
    base = coverage[0]["width"]
    for row in coverage:
        row["width_ratio"] = row["width"] / base
    spread = []
    for rho in (0.9, 0.99):
        for n in (30, 100, 1000):
            v = I.vif_spread(n, rho, 4000, 1204)
            v.pop("vifs")
            v["rho"] = rho
            spread.append(v)
    return {"sumdiff": sumdiff, "tests": tests, "coverage": coverage, "vif_spread": spread}


def natural_vif_export(n=60000, seed=1):
    """Größte VIF im Export-Zweig des Generators: Vorlauf, Wochenende, Gewicht und elf Kunden-Dummies (zwölf Kunden, einer als Basis); Zähleinheit: ein Design mit n Containern."""
    u = S.uniform_open(S.u64_block(S.child_seeds(seed, 41, 1), n * 4)).reshape(n, 4)
    v = 1.0 + 9.0 * u[:, 0]
    we = (u[:, 1] < 2.0 / 7.0).astype(float)
    g = S.norm_ppf(u[:, 2])
    kunde = np.minimum((u[:, 3] * 12).astype(int), 11)
    dummies = np.column_stack([(kunde == k).astype(float) for k in range(1, 12)])
    X = np.column_stack([v, we, g, dummies])
    vif = A.vif_diag_inverse(X)
    return {"n": n, "max_vif": float(vif.max()), "vif_vorlauf": float(vif[0]), "vif_wochenende": float(vif[1]), "vif_gewicht": float(vif[2])}


def nonzero_block():
    world, scen = "C", "S3"
    n_fresh = 2000
    X, y = SA.draw(world, scen, n_fresh, N, RHO, SIGMA, 1230, SA.TAG_FRESH)
    Z, yc = A.standardize_batch(X, y)
    G, c = A.gram_batch(Z, yc)
    b, _ = A.cd_batch(G, c, ALPHA_L, 1.0)
    thr = (0.0, 1e-8, 1e-4, 1e-3, 1e-2, 5e-2, 0.1)
    table = NZ.threshold_table(b, thr)
    small = NZ.smallest_chosen(b)
    small = small[~np.isnan(small)]
    out = {"n_fresh": n_fresh, "alpha": ALPHA_L, "table": table,
           "smallest_q": [float(np.quantile(small, q)) for q in (0.05, 0.25, 0.5, 0.75)],
           "below": {f"{t:g}": float(np.mean(small < t)) for t in (1e-8, 1e-4, 1e-2)},
           "ridge_nonzeros": float(A.cd_batch(G[:50], c[:50], ALPHA_L, 0.0)[0].astype(bool).sum(axis=1).mean())}
    smooth = []
    for eps in (1e-12, 1e-8):
        smooth.append({"eps": eps, "rows": NZ.smooth_agreement(G[:200], c[:200], ALPHA_L, b[:200], eps, (0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2))})
    out["smooth"] = smooth
    return out


def condition_block():
    rows = []
    for label, n_true, a in COND.DESIGNS:
        row = {"label": label, "n_true": n_true, "a": a, "q": COND.q_value(n_true, a), "pop_coef": COND.population_dummy_coefficient(n_true, a), "reps": 500}
        row["rates"] = {str(n): COND.data_selection_rate(n_true, a, n, 500, seed=1231 + n_true) for n in (100, 1000, 10000)}
        rows.append(row)
    return rows


def build(progress=None):
    out = {"_hinweis": "erzeugt von tools/kol_make_frozen.py; Zufalls-Seeds und Zahl der Wiederholungen stehen in den Einträgen"}
    out.update(inference_blocks())
    out["natural_vif_export"] = natural_vif_export()
    out["selection"] = selection_block(progress)
    out["nonzero"] = nonzero_block()
    out["condition"] = condition_block()
    return out


def load():
    return json.loads(FROZEN_PATH.read_text(encoding="utf-8"))


def write(data):
    FROZEN_PATH.parent.mkdir(exist_ok=True)
    FROZEN_PATH.write_text(json.dumps(data, indent=1, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")
