"""Kern: Gram-Matrix, VIF, Koordinatenabstieg für Lasso und Elastic Net (gestapelt über viele Anpassungen), Pfad, Optimalitätsbedingungen.

Konvention (die von scikit-learn und glmnet, Faktor 1/(2n)): minimiere über b
    (1/(2n)) |y - Z b|² + α · ( ℓ |b|₁ + (1 - ℓ)/2 · |b|₂² ),     ℓ = 1 ist das Lasso, ℓ = 0 Ridge.
Die Merkmale sind **standardisiert** (Mittel 0, Standardabweichung 1 mit Teiler n), y ist zentriert; der Achsenabschnitt wird nie bestraft und nicht
mitgerechnet. Mit G = Z'Z/n und c = Z'y/n lautet die Aktualisierung einer Koordinate (weiche Schwelle S)
    b_j = S(c_j - Σ_{k≠j} G_jk b_k, αℓ) / (G_jj + α(1 - ℓ)).
Gerechnet wird nur auf (G, c): eine Anpassung kostet O(p²) je Durchlauf, unabhängig von n; viele Anpassungen laufen als ein numpy-Stapel.
Nur numpy."""

import numpy as np


def standardize_batch(X, y):
    """X (K, n, p), y (K, n) -> Z (K, n, p) mit Mittel 0 und Standardabweichung 1 (Teiler n), zentriertes y.
    Eine konstante Spalte bekommt Standardabweichung 1 und bleibt Null (sie kann nie gewählt werden)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mean = X.mean(axis=1, keepdims=True)
    Xc = X - mean
    sd = np.sqrt((Xc * Xc).mean(axis=1, keepdims=True))
    sd = np.where(sd > 0.0, sd, 1.0)
    return Xc / sd, y - y.mean(axis=1, keepdims=True)


def gram_batch(Z, yc):
    """G = Z'Z/n (K, p, p) und c = Z'y/n (K, p) aus standardisierten Daten."""
    n = Z.shape[1]
    G = np.einsum("knp,knq->kpq", Z, Z) / n
    c = np.einsum("knp,kn->kp", Z, yc) / n
    return G, c


def gram_of(X, y):
    """Einzelne Stichprobe: X (n, p), y (n,) -> (G, c) standardisiert."""
    Z, yc = standardize_batch(X[None], y[None])
    G, c = gram_batch(Z, yc)
    return G[0], c[0]


def cd_batch(G, c, alpha, l1_ratio=1.0, tol=1e-12, max_sweeps=20000):
    """Koordinatenabstieg für K Anpassungen gleichzeitig. G (K, p, p) oder (p, p), c (K, p) oder (p,), α und ℓ Skalar oder (K,).
    Rückgabe: b (K, p) (bzw. (p,)), Durchläufe (K,) bis zur Ruhe (größte Koeffizientenänderung < tol). Exakte Nullen sind exakt 0.0."""
    G = np.asarray(G, dtype=float)
    c = np.asarray(c, dtype=float)
    single = c.ndim == 1
    if single:
        c = c[None, :]
    K, p = c.shape
    if G.ndim == 2:
        G = np.broadcast_to(G, (K, p, p))
    alpha = np.broadcast_to(np.asarray(alpha, dtype=float), (K,)).astype(float)
    l1 = np.broadcast_to(np.asarray(l1_ratio, dtype=float), (K,)).astype(float)
    if np.any(alpha < 0.0) or np.any(l1 < 0.0) or np.any(l1 > 1.0):
        raise ValueError("α ≥ 0 und 0 ≤ ℓ ≤ 1 verlangt")
    a1 = alpha * l1
    a2 = alpha * (1.0 - l1)
    b_out = np.zeros((K, p))
    sweeps_out = np.zeros(K, dtype=int)
    idx = np.arange(K)
    Ga = np.ascontiguousarray(G)
    ga = c.copy()                     # Gradient c - G b
    ba = np.zeros((K, p))
    a1a = a1.copy()
    diag = np.einsum("kjj->kj", Ga).copy()
    denom = diag + a2[:, None]
    ok = denom > 0.0
    inv = np.where(ok, 1.0 / np.where(ok, denom, 1.0), 0.0)
    for sweep in range(1, max_sweeps + 1):
        dmax = np.zeros(len(idx))
        for j in range(p):
            bj = ba[:, j]
            z = ga[:, j] + diag[:, j] * bj
            new = np.sign(z) * np.maximum(np.abs(z) - a1a, 0.0) * inv[:, j]
            d = new - bj
            ga -= Ga[:, j, :] * d[:, None]
            ba[:, j] = new
            np.maximum(dmax, np.abs(d), out=dmax)
        done = dmax < tol
        if done.any():
            b_out[idx[done]] = ba[done]
            sweeps_out[idx[done]] = sweep
            keep = ~done
            if not keep.any():
                break
            idx, Ga, ga, ba, a1a, diag, inv = idx[keep], Ga[keep], ga[keep], ba[keep], a1a[keep], diag[keep], inv[keep]
    else:
        b_out[idx] = ba
        sweeps_out[idx] = max_sweeps
    return (b_out[0], sweeps_out[0]) if single else (b_out, sweeps_out)


def alpha_max(c, l1_ratio=1.0):
    """Kleinstes α, ab dem alle Koeffizienten 0 sind: max |c_j| / ℓ (ℓ > 0)."""
    if l1_ratio <= 0.0:
        raise ValueError("ℓ = 0 (Ridge) hat kein α_max: kein Koeffizient wird je exakt 0")
    return float(np.max(np.abs(c))) / float(l1_ratio)


def lasso_path(G, c, l1_ratio=1.0, n_alpha=100, eps=1e-3, tol=1e-12):
    """Pfad über n_alpha logarithmisch gestufte α von α_max bis eps·α_max; alle Punkte als ein Stapel. Rückgabe α (n,), b (n, p)."""
    amax = alpha_max(c, l1_ratio)
    alphas = amax * np.exp(np.linspace(0.0, np.log(eps), n_alpha))
    G = np.asarray(G, dtype=float)
    b, _ = cd_batch(np.broadcast_to(G, (n_alpha,) + G.shape), np.broadcast_to(c, (n_alpha, len(c))), alphas, l1_ratio, tol)
    return alphas, b


def kkt_violation(G, c, b, alpha, l1_ratio=1.0):
    """Größte Verletzung der Optimalitätsbedingungen (ein Vektor b): auf der Menge der Nichtnullen g_j = α·ℓ·sign(b_j) + α(1-ℓ) b_j mit g = c - G b,
    sonst |g_j| ≤ α·ℓ. Rückgabe die größte Abweichung (0 = erfüllt)."""
    g = c - G @ b
    a1, a2 = alpha * l1_ratio, alpha * (1.0 - l1_ratio)
    act = b != 0.0
    viol_act = np.abs(g[act] - a1 * np.sign(b[act]) - a2 * b[act]) if act.any() else np.zeros(1)
    viol_pass = np.maximum(np.abs(g[~act]) - a1, 0.0) if (~act).any() else np.zeros(1)
    return float(max(viol_act.max(), viol_pass.max()))


def nonzero_count(b, threshold):
    """Zahl der Koeffizienten mit |b| > threshold (Schwelle 0: alle ungleich 0). Ohne Schwellenangabe ist die Zahl keine Messung."""
    return int(np.sum(np.abs(b) > threshold))


# ---------------------------------------------------------------------------------------------------------------------------------------
# Kollinearität: VIF
# ---------------------------------------------------------------------------------------------------------------------------------------

def vif_diag_inverse(X):
    """VIF_j = j-tes Diagonalelement der Inversen der Korrelationsmatrix (Spalten von X (n, p), ohne Achsenabschnitt-Spalte)."""
    R = np.corrcoef(np.asarray(X, dtype=float), rowvar=False)
    return np.diag(np.linalg.inv(np.atleast_2d(R))).copy()


def vif_via_r2(X):
    """VIF_j = 1/(1 - R²_j) mit R²_j der Hilfsregression von x_j auf die übrigen Spalten (mit Achsenabschnitt)."""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    out = np.empty(p)
    for j in range(p):
        others = np.column_stack([np.ones(n), np.delete(X, j, axis=1)])
        coef, *_ = np.linalg.lstsq(others, X[:, j], rcond=None)
        res = X[:, j] - others @ coef
        r2 = 1.0 - (res @ res) / (((X[:, j] - X[:, j].mean()) ** 2).sum())
        out[j] = 1.0 / (1.0 - r2)
    return out


def vif_via_variance(X, sigma=1.0):
    """VIF_j = Var(b_j)·S_xx,j / σ² aus der Kovarianzmatrix σ²(X'X)⁻¹ der zentrierten Spalten (S_xx,j = Σ (x_ij - Mittel)²)."""
    Xc = np.asarray(X, dtype=float)
    Xc = Xc - Xc.mean(axis=0)
    var = sigma ** 2 * np.diag(np.linalg.inv(Xc.T @ Xc))
    return var * (Xc ** 2).sum(axis=0) / sigma ** 2


def vif_two(rho):
    """Zwei Merkmale mit Stichprobenkorrelation ρ: VIF = 1/(1 - ρ²); bei |ρ| = 1 unendlich (None)."""
    if abs(rho) >= 1.0:
        return None
    return 1.0 / (1.0 - rho * rho)


def two_feature_factors(rho):
    """Faktoren der Varianzen gegen orthogonale Spalten gleicher Länge: Einzelkoeffizient 1/(1-ρ²), Summe 1/(1+ρ), Differenz 1/(1-ρ); Korrelation -ρ."""
    if abs(rho) >= 1.0:
        return None
    return {"single": 1.0 / (1.0 - rho * rho), "sum": 1.0 / (1.0 + rho), "diff": 1.0 / (1.0 - rho), "corr": -rho}
