"""Auswertung: Einstellungen, die Rechnungen hinter den Experimenten und die Verdikte als reine Funktionen (`verdict_*(...) -> (Art, Text)`).

Jede Rechnung nennt ihre Zähleinheit: eine Wiederholung (neue Fehler, festes Design), ein Design, eine frische Stichprobe, eine Originalstichprobe (je B Bootstrap- bzw.
Teilstichproben). Die vorgerechneten Zählungen (`data/kol_frozen.json`) stehen in README und Expandern, nicht in den Verdikten: ein Verdikt nennt nur, was die Rechnung der
aktuellen Einstellung ergibt."""

from dataclasses import dataclass
from fractions import Fraction as F

import numpy as np

import kol_algorithm as A
import kol_condition as COND
import kol_constants as C
import kol_exact as E
import kol_inference as I
import kol_sampling as SA


@dataclass(frozen=True)
class Settings:
    scen: str = "S3"
    rho: float = 0.95
    n: int = 100
    sigma: float = 2.0
    alpha: float = 0.09
    l1: float = 1.0
    seed: int = 1
    hand_swap: bool = False
    hand_alpha: str = "2"
    vif_b: float = 0.5
    vif_reps: int = 2000
    path_rel: float = 0.1
    path_thr: float = 1e-8
    fresh: int = 500
    scheme: str = C.SCHEME_OPTIONS[0]
    pi: float = 0.8
    m_orig: int = 20
    en_l1: float = 0.5


def world_of(scen):
    return C.WORLD_OF[scen]


def de(x, digits=2):
    return C.pretty(x, digits)


def thr_text(t):
    """Schwelle als Text: 0, 1e-8, 1e-4, 1e-2 (nicht 0,00000001)."""
    return "0" if t == 0 else f"{t:.0e}".replace("e-0", "e-")


def frac_text(x):
    x = F(x)
    return f"{x.numerator}" if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


# ---------------------------------------------------------------------------------------------------------------------------------------
# Handbeispiel (Brüche)
# ---------------------------------------------------------------------------------------------------------------------------------------

def hand_analysis(swapped, alpha_text):
    """Das Handbeispiel mit vier Containern: Brüche, Lasso- und Elastic-Net-Lösung bei α, Pfad (Brüche) und die Ellipse der Quadratsumme."""
    rep = E.hand_report(swapped)
    alpha = F(alpha_text)
    G, c = E.hand_data(swapped)
    lasso = E.unique_solution(G, c, alpha)
    enet = E.unique_solution(G, c, alpha, F(1, 2))
    grid = [F(k, 20) for k in range(0, 71)]                      # α von 0 bis 7/2 in Schritten von 1/20
    path = [E.unique_solution(G, c, a) for a in grid]
    first, second, k = E.hand_entry_points(swapped)
    return {"rep": rep, "alpha": alpha, "lasso": lasso, "enet": enet, "grid": grid, "path": path, "first": first, "second": second, "first_feature": k}


def hand_bootstrap(swapped):
    return E.bootstrap_counts(swapped)


def ellipse_points(rep, k=80):
    """Höhenlinie (b - b̂)'X'X(b - b̂) = s² der Quadratsumme (s² = SSE/Freiheitsgrade, ein s² über dem Minimum) als Punkte (float)."""
    xtx = np.array([[float(v) for v in row] for row in rep["xtx"]])
    s2 = float(rep["sse"]) / rep["df"]
    w, V = np.linalg.eigh(xtx)
    t = np.linspace(0, 2 * np.pi, k)
    unit = np.vstack([np.cos(t), np.sin(t)])
    pts = V @ (np.sqrt(s2 / w)[:, None] * unit)
    b = np.array([float(v) for v in rep["b"]])
    return b[0] + pts[0], b[1] + pts[1]


# ---------------------------------------------------------------------------------------------------------------------------------------
# Eine Stichprobe: was die Anpassung sagt
# ---------------------------------------------------------------------------------------------------------------------------------------

def one_sample(s: Settings):
    """Eine Stichprobe (Zufalls-Seed der Seitenleiste): VIF, Kleinste Quadrate mit Standardfehlern, Lasso bei α. Koeffizienten auf der standardisierten Skala."""
    X, y = SA.draw(world_of(s.scen), s.scen, 1, s.n, s.rho, s.sigma, s.seed, SA.TAG_ORIG)
    Z, yc = A.standardize_batch(X, y)
    G, c = A.gram_batch(Z, yc)
    n = s.n
    out = {"X": X[0], "y": y[0], "G": G[0], "c": c[0], "n": n}
    r_hat = float(G[0][0, 1])
    out["r_hat"] = r_hat
    out["vif_pair"] = A.vif_two(r_hat)
    out["vif"] = A.vif_diag_inverse(X[0])
    out["ols"] = None
    if n > C.P_FEATURES + 1:
        try:
            ols = np.linalg.solve(G[0], c[0])
            resid = yc[0] - Z[0] @ ols
            s2 = float(resid @ resid) / (n - C.P_FEATURES - 1)
            se = np.sqrt(s2 * np.diag(np.linalg.inv(G[0])) / n)
            out["ols"], out["ols_se"] = ols, se
        except np.linalg.LinAlgError:
            pass
    b, _ = A.cd_batch(G[0], c[0], s.alpha, s.l1)
    out["lasso"] = b
    out["beta"] = np.asarray(C.BETAS["S2" if s.scen == "S2E" else s.scen])
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 2: Aufblähung
# ---------------------------------------------------------------------------------------------------------------------------------------

def vif_analysis(n, rho, sigma, b, reps, seed):
    X = I.fixed_design(n, rho, 5)
    rates = I.test_rates(X, (b, b), sigma, reps, seed)
    rates["se_orth"] = sigma / np.sqrt(n)
    rates["rho"], rates["b"], rates["sigma"] = rho, b, sigma
    rates["sumdiff"] = I.variance_check(X, sigma, reps, seed + 1)
    rates["factors"] = A.two_feature_factors(rho)
    return rates


def vif_spread_analysis(n, rho, designs, seed):
    return I.vif_spread(n, rho, designs, seed)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 3: Pfad
# ---------------------------------------------------------------------------------------------------------------------------------------

def path_analysis(s: Settings, n_alpha=60):
    smp = one_sample(s)
    G, c = smp["G"], smp["c"]
    amax = A.alpha_max(c, s.l1)
    alphas, b = A.lasso_path(G, c, s.l1, n_alpha)
    alpha_sel = s.path_rel * amax
    b_sel, _ = A.cd_batch(G, c, alpha_sel, s.l1)
    entry = {}
    for j in range(C.P_FEATURES):
        hit = np.nonzero(np.abs(b[:, j]) > s.path_thr)[0]
        entry[j] = float(alphas[hit[0]]) if len(hit) else None
    order = sorted((j for j in entry if entry[j] is not None), key=lambda j: -entry[j])
    counts = {thr: A.nonzero_count(b_sel, thr) for thr in C.THRESHOLD_OPTIONS}
    kkt = max(A.kkt_violation(G, c, b[i], alphas[i], s.l1) for i in range(n_alpha))
    return {"alphas": alphas, "b": b, "alpha_max": amax, "alpha_sel": alpha_sel, "b_sel": b_sel, "entry": entry, "order": order, "counts": counts, "kkt": kkt,
            "n_zero_exact": int(np.sum(b_sel == 0.0))}


# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 4 bis 6
# ---------------------------------------------------------------------------------------------------------------------------------------

def fresh_analysis(world, scen, n, rho, sigma, alpha, l1, n_fresh, seed):
    sel = SA.fresh_selection(world, scen, n, rho, sigma, alpha, l1, n_fresh, seed)
    out = SA.summarize(sel)
    out["freq"] = sel.mean(axis=0)
    out["n_fresh"] = n_fresh
    return out


def condition_table():
    """Die sieben Entwürfe: q und die Attrappe im Lasso der Bevölkerung (α = 0,05, kein Rauschen)."""
    rows = []
    for label, n_true, a in COND.DESIGNS:
        coef = COND.population_dummy_coefficient(n_true, a)
        rows.append({"label": label, "q": COND.q_value(n_true, a), "coef": coef, "chosen": abs(coef) > 1e-12})
    return rows


def bootstrap_analysis(s: Settings):
    """m Originalstichproben mit je B Bootstrap- oder Teilstichproben (festes α) und als Bezug frische Stichproben; Zähleinheit der Zählungen: die Originalstichprobe."""
    world = world_of(s.scen)
    scheme = "boot" if s.scheme == C.SCHEME_OPTIONS[0] else "sub"
    Xo, yo = SA.draw(world, s.scen, s.m_orig, s.n, s.rho, s.sigma, s.seed + 7, SA.TAG_ORIG)
    sel = SA.original_selections(Xo, yo, s.alpha, s.l1, scheme, C.B_RESAMPLES, s.seed + 13)
    po = SA.per_original(sel, s.pi)
    return {"po": po, "summary": SA.original_summary(po, s.pi), "scheme": scheme, "m": s.m_orig, "B": C.B_RESAMPLES, "pi": s.pi}


# ---------------------------------------------------------------------------------------------------------------------------------------
# Verdikte
# ---------------------------------------------------------------------------------------------------------------------------------------

def verdict_q1(v):
    """Kernfrage 1: Wie viel erfahren die Daten über zwei fast gleiche Merkmale? `v` = `vif_analysis`."""
    rho, n, sigma, b, reps = v["rho"], v["n"], v["sigma"], v["b"], v["reps"]
    head = (f"Bei ρ = {de(rho)} (n = {n}, β = ({de(b, 1)}; {de(b, 1)}), σ = {de(sigma, 1)}, {reps} Wiederholungen): Standardfehler von b₁ {de(v['se_formula'])} statt {de(v['se_orth'])} bei "
            f"orthogonalen Spalten; ein Einzeltest verwirft in {C.pct(v['p_single'])} (exakt {C.pct(v['p_single_exact'])}), beide Einzeltests in {C.pct(v['p_both_single'])}, der gemeinsame F-Test in "
            f"{C.pct(v['p_F'])} (exakt {C.pct(v['p_F_exact'])}).")
    if v["p_F_no_single"] >= 0.2:
        return "warning", head + (f" In **{C.pct(v['p_F_no_single'])}** der Wiederholungen verwirft der F-Test, **ohne dass ein einziger Einzeltest verwirft**: die Daten kennen die Summe, "
                                  "nicht die Aufteilung. Die 95-%-Intervalle bleiben gültig, sie sind nur breit.")
    return "info", head + f" Der F-Test verwirft ohne jeden Einzeltest in {C.pct(v['p_F_no_single'])} der Wiederholungen: bei dieser Einstellung ist der Unterschied zwischen Einzel- und gemeinsamem Test klein."


def verdict_q2(f, scen, n, rho, sigma, alpha, l1):
    """Kernfrage 2: Welches von beiden wählt das Lasso? `f` = `fresh_analysis` (Zähleinheit: frische Stichprobe)."""
    method = "Lasso" if l1 >= 1.0 else f"Elastic Net (ℓ = {de(l1, 2)})"
    N = f["n_fresh"]
    one = f["only1"] + f["only2"]
    head = (f"{method}, Szenario {scen} (n = {n}, ρ = {de(rho)}, σ = {de(sigma, 1)}, α = {de(alpha, 3)} fest, {N} frische Stichproben): x₁ in {C.pct(f['p1'])}, x₂ in {C.pct(f['p2'])}, "
            f"beide in {C.pct(f['both'])}.")
    if one < 0.25:
        return "info", head + f" Genau eines von beiden wird nur in {C.pct(one)} der Stichproben gewählt; meist werden beide oder keines gewählt."
    share = f["share1"]
    if 0.4 <= share <= 0.6 and abs(f["p1"] - f["p2"]) < 0.1:
        return "warning", head + (f" Wenn genau eines gewählt wird (in {C.pct(one)} der Stichproben), ist es x₁ in {C.pct(share)}: **wie ein Münzwurf**; welches von beiden, entscheidet "
                                  "der Zufall der Stichprobe.")
    if share > 0.6:
        return "info", head + f" Wenn genau eines gewählt wird (in {C.pct(one)}), ist es x₁ in {C.pct(share)}: x₁ wird bevorzugt."
    return "info", head + f" Wenn genau eines gewählt wird (in {C.pct(one)}), ist es x₁ in {C.pct(share)}: x₂ wird bevorzugt."


def verdict_q3(fresh, boot, scen, scheme_label, pi):
    """Kernfrage 3: Was zeigt der Bootstrap derselben Stichprobe, was frische Stichproben nicht zeigen? Zähleinheit: Originalstichprobe."""
    d_fresh = abs(fresh["p1"] - fresh["p2"])
    d_boot = boot["absdiff"]
    m = boot["n_share"]
    head = (f"{scheme_label}, Szenario {scen}: über frische Stichproben unterscheiden sich x₁ und x₂ um |P(x₁) − P(x₂)| = {de(d_fresh, 3)}; je Originalstichprobe unterscheiden sie sich im Bootstrap "
            f"im Mittel um {de(d_boot, 3)}.")
    if d_boot - d_fresh >= 0.15:
        return "warning", head + (f" In {C.pct(boot['clear'])} der {m} Originalstichproben (in denen überhaupt eine Wiederholung genau eines wählt) liegt der Anteil von x₁ über 0,75 oder "
                                  f"unter 0,25, während er über frische Stichproben bei {C.pct(fresh['share1'])} liegt: **der Bootstrap zeigt eine Entscheidung, die es über Stichproben nicht gibt**. "
                                  f"Die Regel „Häufigkeit ≥ {de(pi, 1)} gilt als stabil“ meldet genau x₁ in {C.pct(boot['only1_st'])}, genau x₂ in {C.pct(boot['only2_st'])}, beide in "
                                  f"{C.pct(boot['both_st'])} der Originalstichproben.")
    return "info", head + (f" Beide stimmen hier überein (Unterschied der Beträge unter 0,15): ist ein Merkmal klar stärker, bildet der Bootstrap die Verhältnisse ab. Die Regel „Häufigkeit ≥ {de(pi, 1)}“ meldet genau x₁ in "
                           f"{C.pct(boot['only1_st'])}, genau x₂ in {C.pct(boot['only2_st'])}, beide in {C.pct(boot['both_st'])} der {m} Originalstichproben.")


def verdict_hand(h):
    rep = h["rep"]
    b = rep["b"]
    text = (f"VIF = {frac_text(rep['vif'])}; Var(b₁ + b₂) = {frac_text(rep['var_sum'])}·σ² (orthogonal {frac_text(rep['orth_sum'])}), Var(b₁ − b₂) = {frac_text(rep['var_diff'])}·σ². "
            f"Kleinste Quadrate b = ({frac_text(b[0])}; {frac_text(b[1])}), Summe {frac_text(b[0] + b[1])}. Lasso bei α = {frac_text(h['alpha'])}: "
            f"({frac_text(h['lasso'][0])}; {frac_text(h['lasso'][1])}), Elastic Net (ℓ = 1/2): ({frac_text(h['enet'][0])}; {frac_text(h['enet'][1])}). "
            f"Das Lasso nimmt zuerst x{h['first_feature'] + 1} (α = {frac_text(h['first'])}).")
    return "info", text


def verdict_path(p, s: Settings):
    c = p["counts"]
    order = ", ".join(f"x{j + 1}" for j in p["order"][:5])
    parts = " / ".join(f"{thr_text(thr)}: {c[thr]}" for thr in C.THRESHOLD_OPTIONS)
    same = len(set(c.values())) == 1
    text = (f"α = {de(p['alpha_sel'], 4)} ({de(s.path_rel, 3)} von α_max = {de(p['alpha_max'], 3)}): Nichtnullen nach Schwelle (Schwelle: Zahl) {parts}. Eintrittsreihenfolge (Schwelle {thr_text(s.path_thr)}), "
            f"erste fünf: {order}. Größte Verletzung der Optimalitätsbedingungen auf dem Pfad: {p['kkt']:.1e}.")
    if same:
        return "info", text + " Bei dieser Stichprobe und diesem α ändert die Schwelle die Zahl nicht; trotzdem gehört die Schwelle zu jeder Zahl der Nichtnullen."
    return "warning", text + " **Die Zahl der Nichtnullen hängt an der Schwelle**: eine Zahl ohne Schwellenangabe ist keine Messung."


def verdict_elastic(fl, fe, scen, l1_en):
    """Experiment 6: Lasso gegen Elastic Net (frische Stichproben, gleiches α)."""
    head = (f"Szenario {scen}, {fl['n_fresh']} frische Stichproben, gleiches α: Lasso wählt x₁ in {C.pct(fl['p1'])}, x₂ in {C.pct(fl['p2'])}, beide in {C.pct(fl['both'])}; Elastic Net (ℓ = {de(l1_en, 2)}) "
            f"x₁ in {C.pct(fe['p1'])}, x₂ in {C.pct(fe['p2'])}, beide in {C.pct(fe['both'])}. Die Attrappen x₄ bis x₁₀ werden im Mittel in {C.pct(fl['att'])} (Lasso) gegen {C.pct(fe['att'])} (Elastic Net) gewählt.")
    if scen in ("S2", "S2E"):
        return "warning", head + f" Nur x₁ wirkt: **das Elastic Net nimmt die Kopie x₂ mit** ({C.pct(fe['p2'])} gegen {C.pct(fl['p2'])} beim Lasso)."
    return "info", head + " Beide Zwillinge wirken: das Elastic Net wählt sie öfter gemeinsam."
