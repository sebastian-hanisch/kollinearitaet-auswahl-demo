"""Wann ist die richtige Auswahl überhaupt möglich? Die Bedingung von Zhao/Yu (2006) an einer Attrappe, in der Bevölkerung und an Daten.

Entwurf: `n_true` wahre Merkmale (β = 1, untereinander unabhängig) und eine Attrappe x₀ mit Korrelation a zu jedem von ihnen (β = 0). Mit der Korrelationsmatrix C, Index 1 = wahre
Merkmale, 2 = Attrappe, ist q = |C₂₁ C₁₁⁻¹ sign(β₁)| (hier: n_true·a). Zhao/Yu zeigen: q < 1 ist „fast notwendig und hinreichend“ dafür, dass das Lasso die wahre Menge findet
(Auswahlkonsistenz); die Demo prüft die Aussage nur an sieben Entwürfen (kein Beweis; q = 1 ist der Rand und wird nicht gezählt).
Bevölkerung: Lasso auf der Gram-Matrix C mit c = C·β, kleines α, kein Rauschen. Daten: Lasso auf n Beobachtungen (Fehler σ = 2), Zähleinheit: Stichprobe."""

import numpy as np

import kol_algorithm as A
import kol_stream as S

TAG_COND = 31
# (Bezeichnung, Zahl wahrer Merkmale, Korrelation a der Attrappe zu jedem wahren Merkmal)
DESIGNS = (
    ("1 wahr, Attrappe Korrelation 0,5", 1, 0.5),
    ("1 wahr, Attrappe Korrelation 0,9", 1, 0.9),
    ("1 wahr, Attrappe Korrelation 0,95", 1, 0.95),
    ("2 wahr unabhängig, Attrappe Korrelation 0,3", 2, 0.3),
    ("2 wahr unabhängig, Attrappe Korrelation 0,4", 2, 0.4),
    ("2 wahr unabhängig, Attrappe Korrelation 0,5", 2, 0.5),
    ("2 wahr unabhängig, Attrappe Korrelation 0,6", 2, 0.6),
)


def cov_matrix(n_true, a):
    """Korrelationsmatrix (n_true + 1): wahre Merkmale unabhängig, die Attrappe (letzte Zeile) mit Korrelation a zu jedem."""
    Cm = np.eye(n_true + 1)
    Cm[n_true, :n_true] = a
    Cm[:n_true, n_true] = a
    return Cm


def q_value(n_true, a):
    """q = |C₂₁ C₁₁⁻¹ sign(β₁)| mit β₁ = (1, …, 1)."""
    Cm = cov_matrix(n_true, a)
    C11, C21 = Cm[:n_true, :n_true], Cm[n_true:, :n_true]
    return float(abs((C21 @ np.linalg.solve(C11, np.ones(n_true)))[0]))


def population_dummy_coefficient(n_true, a, alpha=0.05):
    """Koeffizient der Attrappe im Lasso der Bevölkerung (Gram-Matrix C, c = C·β, β = (1, …, 1, 0)); exakt 0, wenn sie nicht gewählt wird."""
    Cm = cov_matrix(n_true, a)
    beta = np.append(np.ones(n_true), 0.0)
    b, _ = A.cd_batch(Cm, Cm @ beta, alpha, 1.0, tol=1e-14)
    return float(b[n_true])


def data_selection_rate(n_true, a, n, reps, alpha=0.1, sigma=2.0, seed=1231, chunk=100):
    """Anteil der `reps` Stichproben der Größe n, in denen das Lasso (festes α) die Attrappe wählt."""
    L = np.linalg.cholesky(cov_matrix(n_true, a))
    hits = 0
    for start in range(0, reps, chunk):
        k = min(chunk, reps - start)
        seeds = S.child_seeds(seed + 7919 * start, TAG_COND, k)
        z = S.normals(seeds, n * (n_true + 2)).reshape(k, n, n_true + 2)
        X = z[:, :, : n_true + 1] @ L.T
        y = X[:, :, :n_true].sum(axis=2) + sigma * z[:, :, n_true + 1]
        Zs, yc = A.standardize_batch(X, y)
        G, c = A.gram_batch(Zs, yc)
        b, _ = A.cd_batch(G, c, alpha, 1.0)
        hits += int(np.sum(np.abs(b[:, n_true]) > 1e-8))
    return hits / reps
