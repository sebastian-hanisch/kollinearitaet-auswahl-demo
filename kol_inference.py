"""Zwei fast gleiche Merkmale in der Kleinste-Quadrate-Anpassung: Aufblähung der Varianz, Summe und Differenz, Einzeltest gegen gemeinsamen Test, Intervalle,
Streuung des VIF der Stichprobe. Festes Design (zwei zentrierte Merkmale, Stichprobenkorrelation **genau** ρ), Fehler N(0, σ²) neu gezogen; mit Achsenabschnitt
(Rest-Freiheitsgrade n - 3). Zähleinheit überall: die **Wiederholung** (Neuziehen der Fehler) bzw. das **Design** (Zufallsdesign).

Der exakte Vergleich (Orakel) ist bedingt auf das Design: der Einzeltest ist ein nichtzentrales t mit ncp = β_j/SE_j, der gemeinsame Test ein nichtzentrales F(2, n-3)
mit ncp = β'X'Xβ/σ². scipy wird nur für diese Verteilungen und die kritischen Werte gebraucht."""

import numpy as np
from scipy import stats

import kol_stream as S

TAG_DESIGN, TAG_ERR, TAG_RANDOM = 21, 22, 23


def fixed_design(n, rho, seed=0):
    """X (n, 2): beide Spalten mit Mittel 0 und Σ x² = n, Stichprobenkorrelation genau ρ (Gram-Schmidt auf zwei Normalfolgen)."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("ρ muss zwischen 0 und 1 liegen")
    z = S.stream_normals(seed, TAG_DESIGN, (2, n))
    z = z - z.mean(axis=1, keepdims=True)
    u1 = z[0] * (np.sqrt(n) / np.sqrt(z[0] @ z[0]))
    w = z[1] - (z[1] @ u1) / n * u1
    w = w * (np.sqrt(n) / np.sqrt(w @ w))
    return np.column_stack([u1, rho * u1 + np.sqrt(1.0 - rho * rho) * w])


def simulate_ols(X, beta, sigma, reps, seed, level=0.05):
    """`reps` Wiederholungen: neue Fehler, OLS mit Achsenabschnitt. Rückgabe Koeffizienten, Standardfehler und Testentscheidungen je Wiederholung."""
    n = X.shape[0]
    if np.linalg.matrix_rank(X) < 2:
        raise ValueError("X'X ist singulär (zwei gleiche Spalten, ρ = 1): die Koeffizienten sind nicht einzeln bestimmt, die Varianz jedes Einzelkoeffizienten ist unendlich")
    eps = S.normals(S.child_seeds(seed, TAG_ERR, reps), n) * sigma                                # (reps, n)
    eps = eps - eps.mean(axis=1, keepdims=True)                                                   # Achsenabschnitt: nur die zentrierten Fehler wirken
    V = np.linalg.inv(X.T @ X)
    xe = eps @ X                                                                                  # X'ε je Wiederholung
    d = xe @ V                                                                                    # b̂ - β
    sse = (eps * eps).sum(axis=1) - np.einsum("ri,ij,rj->r", xe, V, xe)
    df = n - 3
    s2 = sse / df
    b = np.asarray(beta, dtype=float)[None, :] + d
    se = np.sqrt(s2[:, None] * np.diag(V)[None, :])
    tcrit = float(stats.t.ppf(1.0 - level / 2.0, df))
    fcrit = float(stats.f.ppf(1.0 - level, 2, df))
    t = b / se
    F = np.einsum("ri,ij,rj->r", b, X.T @ X, b) / 2.0 / s2
    return {"b": b, "se": se, "t": t, "F": F, "d": d, "tcrit": tcrit, "fcrit": fcrit, "df": df, "V": V}


def test_rates(X, beta, sigma, reps, seed, level=0.05):
    """Anteile der Wiederholungen, in denen die Tests verwerfen (Simulation), und die exakte Rechnung (nichtzentrale Verteilungen, bedingt auf das Design)."""
    n = X.shape[0]
    r = simulate_ols(X, beta, sigma, reps, seed, level)
    rej = np.abs(r["t"]) > r["tcrit"]
    rejF = r["F"] > r["fcrit"]
    V = r["V"]
    beta = np.asarray(beta, dtype=float)
    ncp_t = beta / (sigma * np.sqrt(np.diag(V)))
    exact_t = [float(stats.nct.sf(r["tcrit"], r["df"], c) + stats.nct.cdf(-r["tcrit"], r["df"], c)) if c != 0.0 else float(2.0 * stats.t.sf(r["tcrit"], r["df"])) for c in ncp_t]
    ncp_f = float(beta @ (X.T @ X) @ beta / sigma ** 2)
    exact_f = float(stats.ncf.sf(r["fcrit"], 2, r["df"], ncp_f)) if ncp_f > 0.0 else float(stats.f.sf(r["fcrit"], 2, r["df"]))     # ncf mit ncp = 0 ist in scipy nicht verwendbar
    tc = r["tcrit"]
    cover = np.abs(r["d"][:, 0]) <= tc * r["se"][:, 0]
    return {
        "reps": reps, "n": n, "se_formula": float(sigma * np.sqrt(V[0, 0])), "se_mean": float(r["se"][:, 0].mean()),
        "p_single": float(rej[:, 0].mean()), "p_single_exact": exact_t[0], "p_both_single": float((rej[:, 0] & rej[:, 1]).mean()),
        "p_F": float(rejF.mean()), "p_F_exact": exact_f, "p_F_no_single": float((rejF & ~rej[:, 0] & ~rej[:, 1]).mean()),
        "p_F_given_none": float(np.mean(rejF[~rej[:, 0] & ~rej[:, 1]])) if (~rej[:, 0] & ~rej[:, 1]).any() else float("nan"),
        "cover": float(cover.mean()), "width": float(2.0 * tc * r["se"][:, 0].mean()),
    }


def variance_check(X, sigma, reps, seed):
    """n·Var der Koeffizienten, ihrer Summe und Differenz (in Einheiten σ²) aus `reps` Wiederholungen, und die Korrelation von b1 und b2."""
    n = X.shape[0]
    r = simulate_ols(X, (0.0, 0.0), sigma, reps, seed)
    d = r["d"]
    return {"n_var_b1": float(n * d[:, 0].var(ddof=1) / sigma ** 2), "n_var_sum": float(n * (d[:, 0] + d[:, 1]).var(ddof=1) / sigma ** 2),
            "n_var_diff": float(n * (d[:, 0] - d[:, 1]).var(ddof=1) / sigma ** 2), "corr": float(np.corrcoef(d[:, 0], d[:, 1])[0, 1])}


def vif_spread(n, rho, designs, seed):
    """Zufallsdesign: `designs` Stichproben (x1, x2) der Größe n mit Bevölkerungskorrelation ρ; der VIF der Stichprobe 1/(1 - r̂²) gegen die Formel mit ρ.
    Rückgabe die Quantile des VIF relativ zur Formel (5 %, 50 %, 95 %) und den Anteil der Designs mit mehr als dem Doppelten der Formel."""
    z = S.normals(S.child_seeds(seed, TAG_RANDOM, designs), 2 * n).reshape(designs, 2, n)
    x1 = z[:, 0, :]
    x2 = rho * x1 + np.sqrt(1.0 - rho * rho) * z[:, 1, :]
    x1 = x1 - x1.mean(axis=1, keepdims=True)
    x2 = x2 - x2.mean(axis=1, keepdims=True)
    r = (x1 * x2).sum(axis=1) / np.sqrt((x1 * x1).sum(axis=1) * (x2 * x2).sum(axis=1))
    formula = 1.0 / (1.0 - rho * rho)
    vif = 1.0 / (1.0 - r * r)
    q = np.quantile(vif, [0.05, 0.5, 0.95])
    return {"formula": formula, "q05": float(q[0]), "q50": float(q[1]), "q95": float(q[2]), "rel05": float(q[0] / formula - 1.0), "rel95": float(q[2] / formula - 1.0),
            "over2": float(np.mean(vif > 2.0 * formula)), "designs": designs, "n": n, "vifs": vif}
