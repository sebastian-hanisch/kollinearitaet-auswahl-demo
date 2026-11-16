"""Stichproben der Messwelten und die Auswahl über frische Stichproben, Bootstrap und Teilstichproben (gestapelt, nur numpy, Zufall aus `kol_stream`).

Welten (n Container, p = 10 Merkmale, x1 und x2 mit Stichprobenkorrelation ρ, alle Merkmale je Stichprobe standardisiert):
  C  Container-Welt: x1 = Vorlauf, gleichverteilt 1 bis 10; x2 = „Vorlauf bis Ladeschluss“ (erfunden): ρ·x1 (standardisiert) + √(1-ρ²)·Normalrauschen; x3 Bernoulli(2/7) (Wochenende);
     x4 Normal (Gewicht, Rauschmerkmal des Generators); x5 und x6 Bernoulli(0,3); x7 bis x10 Normal; y = X β + σ ε (additiv, konstantes σ).
  G  Gauß-Welt: alle Merkmale normal (Vergleichswelt der Tests).
  E  Export-Zweig des Generators (Szenario S2E): wie C, aber y ist die lognormale Verweilzeit selbst (`kol_generator`), nicht X β + σ ε; σ skaliert die Streuung (σ = 2 ist der Generator).
Szenarien: S1 β = (1; 1; 1; 0 …), S3 β = (0,5; 0,5; 1; 0 …), S2 β = (2; 0; 0 …) (x2 ist dort eine Attrappe), S2E wie S2 in der Welt E.
Zähleinheiten: eine **frische Stichprobe** (neu gezogen) bzw. eine **Originalstichprobe**, deren Bootstrap- oder Teilstichproben je B Stück sind. „Auswahl“ = |Koeffizient| über 1e-8
(der Abstieg liefert exakte Nullen). α ist fest, nie je Stichprobe per Kreuzvalidierung gewählt."""

import numpy as np

import kol_algorithm as A
import kol_constants as C
import kol_generator as GEN
import kol_stream as S

P = C.P_FEATURES
SD_VORLAUF = float(np.sqrt(81.0 / 12.0))   # Standardabweichung der Gleichverteilung 1 bis 10
MEAN_VORLAUF = 5.5
TAG_FRESH, TAG_ORIG, TAG_RESAMPLE = 1, 2, 3


def _uniform_cols(seed, tag, K, n, ncols):
    seeds = S.child_seeds(seed, tag, K)
    return S.uniform_open(S.u64_block(seeds, n * ncols)).reshape(K, n, ncols)


def draw(world, scen, K, n, rho, sigma, seed, tag):
    """K Stichproben der Größe n. Rückgabe X (K, n, 10) standardisiert, y (K, n) zentriert."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("ρ muss zwischen 0 und 1 liegen")
    ncols = 12 if world == "E" else 11
    u = _uniform_cols(seed, tag, K, n, ncols)
    norm = lambda j: S.norm_ppf(u[:, :, j])
    root = float(np.sqrt(1.0 - rho * rho))
    X = np.empty((K, n, P))
    if world == "G":
        for j in range(P):
            X[:, :, j] = norm(j)
        X[:, :, 1] = rho * X[:, :, 0] + root * X[:, :, 1]
        eps = norm(10)
    else:
        v = 1.0 + 9.0 * u[:, :, 0]
        X[:, :, 0] = v
        X[:, :, 1] = rho * (v - MEAN_VORLAUF) / SD_VORLAUF + root * norm(1)
        X[:, :, 2] = (u[:, :, 2] < 2.0 / 7.0).astype(float)
        if world == "E":                      # Export: x4 = Gewicht, x5..x10 weitere Normalmerkmale
            for j in range(3, P):
                X[:, :, j] = norm(j)
        else:
            X[:, :, 3] = norm(3)
            X[:, :, 4] = (u[:, :, 4] < 0.3).astype(float)
            X[:, :, 5] = (u[:, :, 5] < 0.3).astype(float)
            for j in range(6, P):
                X[:, :, j] = norm(j)
        eps = norm(10)
    Xs, _ = A.standardize_batch(X, np.zeros((K, n)))
    if world == "E":
        kunde = np.minimum((u[:, :, 10] * GEN.K_KUNDEN).astype(int), GEN.K_KUNDEN - 1)
        z = S.norm_ppf(u[:, :, 11])
        y = GEN.export_dwell(X[:, :, 0], kunde, z, noise_scale=sigma / 2.0)
    else:
        beta = np.asarray(C.BETAS[scen])
        y = Xs @ beta + sigma * eps
    return Xs, y - y.mean(axis=1, keepdims=True)


def fit_batch(X, y, alpha, l1=1.0):
    """Standardisieren (je Anpassung) und Koordinatenabstieg: Koeffizienten (K, p) auf der standardisierten Skala."""
    Z, yc = A.standardize_batch(X, y)
    G, c = A.gram_batch(Z, yc)
    b, _ = A.cd_batch(G, c, alpha, l1)
    return b


def select_batch(X, y, alpha, l1=1.0, threshold=C.NZ_THRESHOLD):
    return np.abs(fit_batch(X, y, alpha, l1)) > threshold


def summarize(sel):
    """sel (K, p) boolesch -> Häufigkeiten über die K Anpassungen (Zähleinheit: eine Anpassung)."""
    s1, s2 = sel[:, 0], sel[:, 1]
    only1, only2 = float(np.mean(s1 & ~s2)), float(np.mean(~s1 & s2))
    share = only1 / (only1 + only2) if only1 + only2 > 0 else float("nan")
    return {"p1": float(s1.mean()), "p2": float(s2.mean()), "both": float(np.mean(s1 & s2)), "only1": only1, "only2": only2, "share1": share,
            "att": float(sel[:, 3:].mean()), "nz": float(sel.sum(axis=1).mean()), "n_one": int(np.sum(s1 ^ s2))}


def fresh_selection(world, scen, n, rho, sigma, alpha, l1, n_fresh, seed, chunk=2000):
    """Auswahl über `n_fresh` frische Stichproben; Rückgabe die Auswahlmatrix (n_fresh, p)."""
    out = []
    for start in range(0, n_fresh, chunk):
        k = min(chunk, n_fresh - start)
        X, y = draw(world, scen, k, n, rho, sigma, seed + 1000003 * start, TAG_FRESH)
        out.append(select_batch(X, y, alpha, l1))
    return np.concatenate(out)


def resample_indices(M, B, n, scheme, seed):
    """Indizes (M, B, n) bzw. (M, B, n//2): Bootstrap = n Ziehungen mit Zurücklegen, Teilstichprobe = n//2 ohne Zurücklegen (n Zufallsschlüssel sortieren)."""
    seeds = S.child_seeds(seed, TAG_RESAMPLE, M * B)
    u64 = S.u64_block(seeds, n)
    if scheme == "boot":
        return S.below(u64, n).reshape(M, B, n)
    keys = S.uniform_open(u64)
    return np.argsort(keys, axis=1, kind="stable")[:, : n // 2].reshape(M, B, n // 2)


def original_selections(X, y, alpha, l1, scheme, B, seed):
    """Auswahl über B Bootstrap- bzw. Teilstichproben je Originalstichprobe: X (M, n, p), y (M, n) -> Auswahl (M, B, p)."""
    M, n, _ = X.shape
    idx = resample_indices(M, B, n, scheme, seed)
    m_idx = np.arange(M)[:, None, None]
    Xb = X[m_idx, idx]                      # (M, B, n', p)
    yb = y[m_idx, idx]
    n2 = Xb.shape[2]
    sel = select_batch(Xb.reshape(M * B, n2, P), yb.reshape(M * B, n2), alpha, l1)
    return sel.reshape(M, B, P)


def per_original(sel_mbp, pi=0.8):
    """Kennzahlen je Originalstichprobe aus der Auswahl über ihre B Wiederholungen (M, B, p)."""
    p1 = sel_mbp[:, :, 0].mean(axis=1)
    p2 = sel_mbp[:, :, 1].mean(axis=1)
    only1 = np.mean(sel_mbp[:, :, 0] & ~sel_mbp[:, :, 1], axis=1)
    only2 = np.mean(~sel_mbp[:, :, 0] & sel_mbp[:, :, 1], axis=1)
    denom = only1 + only2
    share = np.where(denom > 0, only1 / np.where(denom > 0, denom, 1.0), np.nan)
    freq = sel_mbp.mean(axis=1)             # (M, p)
    stable = freq >= pi - 1e-12
    return {"p1": p1, "p2": p2, "both": np.mean(sel_mbp[:, :, 0] & sel_mbp[:, :, 1], axis=1), "att": sel_mbp[:, :, 3:].mean(axis=(1, 2)),
            "nz": sel_mbp.sum(axis=2).mean(axis=1), "share": share, "stable": stable}


def original_summary(po, pi=0.8):
    """Mittel und Zählungen über die M Originalstichproben (Zähleinheit: Originalstichprobe)."""
    share = po["share"]
    ok = ~np.isnan(share)
    st = po["stable"]
    return {
        "p1": float(po["p1"].mean()), "p2": float(po["p2"].mean()), "both": float(po["both"].mean()), "att": float(po["att"].mean()), "nz": float(po["nz"].mean()),
        "absdiff": float(np.mean(np.abs(po["p1"] - po["p2"]))), "share_mean": float(share[ok].mean()) if ok.any() else float("nan"),
        "share_sd": float(share[ok].std()) if ok.any() else float("nan"), "n_share": int(ok.sum()),
        "clear": float(np.mean((share[ok] > C.CLEAR_HI) | (share[ok] < C.CLEAR_LO))) if ok.any() else float("nan"),
        "only1_st": float(np.mean(st[:, 0] & ~st[:, 1])), "only2_st": float(np.mean(~st[:, 0] & st[:, 1])),
        "both_st": float(np.mean(st[:, 0] & st[:, 1])), "none_st": float(np.mean(~st[:, 0] & ~st[:, 1])), "att_st": float(np.mean(st[:, 3:].any(axis=1))),
    }


def selection_cell(world, scen, n, rho, sigma, alpha, l1, m_orig, B, n_fresh, seed, pi=0.8, chunk_orig=10):
    """Eine Messzelle: frische Stichproben, Bootstrap- und Teilstichproben von `m_orig` Originalstichproben (Zähleinheit Originalstichprobe)."""
    fresh = summarize(fresh_selection(world, scen, n, rho, sigma, alpha, l1, n_fresh, seed))
    out = {"fresh": fresh}
    Xo, yo = draw(world, scen, m_orig, n, rho, sigma, seed + 7, TAG_ORIG)
    for scheme in ("boot", "sub"):
        parts = []
        for start in range(0, m_orig, chunk_orig):
            sl = slice(start, min(start + chunk_orig, m_orig))
            parts.append(original_selections(Xo[sl], yo[sl], alpha, l1, scheme, B, seed + 13 + 100003 * start))
        po = per_original(np.concatenate(parts), pi)
        out[scheme] = original_summary(po, pi)
    return out
