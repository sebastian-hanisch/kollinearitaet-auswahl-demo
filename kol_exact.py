"""Exakte Rechnung in Brüchen (`fractions`): das Handbeispiel mit vier Containern, Lasso und Elastic Net durch Aufzählen aller Aktivmengen und Vorzeichen,
die vollständig aufgezählte Bootstrap-Zählung, die Bedingung der Auswahl (Zhao/Yu) in der Bevölkerung.

Die Aufzählung ist ein **anderer Rechenweg** als der Koordinatenabstieg in `kol_algorithm.py`: Für jede Menge A von Merkmalen und jedes Vorzeichenmuster s löst man
(G_AA + α(1-ℓ)·I) b_A = c_A - αℓ·s und prüft, ob b_A genau die Vorzeichen s hat und die Optimalitätsbedingung außerhalb von A gilt (|c_k - G_kA b_A| ≤ αℓ).
Ohne Rauschen und ohne Näherung; für höchstens drei Merkmale (3^p Muster) bleibt es klein. Nur Standardbibliothek."""

import itertools
from fractions import Fraction as F

# Vier erfundene Container (ganze Zahlen, damit die Brüche exakt bleiben; keine Daten des Generators), Abweichungen vom Mittel:
X1 = (-3, -1, 1, 3)   # Vorlauf laut Reederei
X2 = (-3, -1, 3, 1)   # Vorlauf bis Ladeschluss
Y = (-2, -1, 1, 2)    # Verweilzeit
Y_SWAPPED = (-2, -1, 2, 1)  # y mit vertauschten Zeilen 3 und 4


def sgn(v):
    return (v > 0) - (v < 0)


def dot(a, b):
    return sum(F(p) * F(q) for p, q in zip(a, b))


def solve(M, v):
    """Lineares Gleichungssystem M b = v in Brüchen (Gauß-Jordan mit Zeilentausch); `None`, falls M singulär ist."""
    n = len(v)
    A = [[F(x) for x in row] + [F(v[i])] for i, row in enumerate(M)]
    for col in range(n):
        piv = next((r for r in range(col, n) if A[r][col] != 0), None)
        if piv is None:
            return None
        A[col], A[piv] = A[piv], A[col]
        pv = A[col][col]
        A[col] = [x / pv for x in A[col]]
        for r in range(n):
            if r != col and A[r][col] != 0:
                fac = A[r][col]
                A[r] = [x - fac * y for x, y in zip(A[r], A[col])]
    return [A[i][n] for i in range(n)]


def gram_and_c(cols, y, center=False):
    """G = X'X/n und c = X'y/n in Brüchen, ohne Standardisierung (die Zahlen bleiben Brüche); mit `center` werden vorher die Mittel abgezogen."""
    n = len(y)
    cols = [[F(v) for v in col] for col in cols]
    yy = [F(v) for v in y]
    if center:
        cols = [[v - sum(col) / n for v in col] for col in cols]
        yy = [v - sum(yy) / n for v in yy]
    G = [[dot(a, b) / n for b in cols] for a in cols]
    c = [dot(a, yy) / n for a in cols]
    return G, c


def lasso_exact(G, c, alpha, l1_ratio=F(1)):
    """Alle Lösungen der Optimalitätsbedingungen (Liste von Koeffizientenlisten, ohne Doppelte). Bei nichtsingulärem G + α(1-ℓ)I genau eine."""
    p = len(c)
    alpha, l1 = F(alpha), F(l1_ratio)
    a1, a2 = alpha * l1, alpha * (1 - l1)
    sols = []
    for size in range(p + 1):
        for A in itertools.combinations(range(p), size):
            for signs in itertools.product((1, -1), repeat=size):
                b = [F(0)] * p
                if size:
                    M = [[G[i][j] + (a2 if i == j else 0) for j in A] for i in A]
                    sol = solve(M, [c[i] - a1 * s for i, s in zip(A, signs)])
                    if sol is None or any(sgn(v) != s for v, s in zip(sol, signs)):
                        continue
                    for i, v in zip(A, sol):
                        b[i] = v
                if all(abs(c[k] - sum(G[k][j] * b[j] for j in range(p))) <= a1 for k in range(p) if k not in A):
                    if b not in sols:
                        sols.append(b)
    return sols


def unique_solution(G, c, alpha, l1_ratio=F(1)):
    sols = lasso_exact(G, c, alpha, l1_ratio)
    if len(sols) != 1:
        raise ValueError(f"nicht eindeutig: {len(sols)} Lösungen")
    return sols[0]


def hand_data(swapped=False):
    y = Y_SWAPPED if swapped else Y
    G, c = gram_and_c([X1, X2], y)
    return G, c


def hand_report(swapped=False):
    """Die Zahlen des Handbeispiels: Gram-Matrix, VIF, Varianzfaktoren, Kleinste Quadrate, Residuen (alle als Brüche)."""
    y = Y_SWAPPED if swapped else Y
    n = len(y)
    xtx = [[dot(X1, X1), dot(X1, X2)], [dot(X2, X1), dot(X2, X2)]]
    det = xtx[0][0] * xtx[1][1] - xtx[0][1] * xtx[1][0]
    inv = [[xtx[1][1] / det, -xtx[0][1] / det], [-xtx[1][0] / det, xtx[0][0] / det]]
    rho = xtx[0][1] / xtx[0][0]
    var_single = inv[0][0]
    var_sum = inv[0][0] + inv[1][1] + 2 * inv[0][1]
    var_diff = inv[0][0] + inv[1][1] - 2 * inv[0][1]
    Xty = [dot(X1, y), dot(X2, y)]
    b = [inv[0][0] * Xty[0] + inv[0][1] * Xty[1], inv[1][0] * Xty[0] + inv[1][1] * Xty[1]]
    resid = [F(yy) - b[0] * a - b[1] * cc for yy, a, cc in zip(y, X1, X2)]
    sse = sum(r * r for r in resid)
    return {
        "xtx": xtx, "rho": rho, "vif": 1 / (1 - rho * rho), "var_single": var_single, "var_sum": var_sum, "var_diff": var_diff,
        "orth_single": 1 / xtx[0][0], "orth_sum": F(2) / xtx[0][0], "factor_single": var_single * xtx[0][0],
        "factor_sum": var_sum / (F(2) / xtx[0][0]), "factor_diff": var_diff / (F(2) / xtx[0][0]), "corr_b": inv[0][1] / var_single,
        "b": b, "resid": resid, "sse": sse, "df": n - 3, "n": n,
    }


def hand_path(alpha, l1_ratio=F(1), swapped=False):
    G, c = hand_data(swapped)
    return unique_solution(G, c, alpha, l1_ratio)


def hand_entry_points(swapped=False):
    """α, bei dem das erste und das zweite Merkmal eintritt (Lasso), und der Index des ersten: (7/2, 1, 0) bei y, (7/2, 1, 1) bei y' (Zeilen 3 und 4 vertauscht)."""
    G, c = hand_data(swapped)
    first = max(abs(v) for v in c)
    k = 0 if abs(c[0]) >= abs(c[1]) else 1
    # zweites Merkmal tritt ein, sobald |c_o - G_ok b_k(α)| > α mit b_k = (c_k - α)/G_kk; löse c_o - G_ok (c_k - α)/G_kk = α (G_ok > 0)
    o = 1 - k
    cok = G[o][k] / G[k][k]
    second = (c[o] - cok * c[k]) / (1 - cok)
    return first, second, k


def bootstrap_counts(swapped=False, alpha=F(2)):
    """Alle 4⁴ = 256 Bootstrap-Stichproben der vier Container (Zentrieren in jeder Stichprobe, ohne Standardisieren), Lasso bei α:
    (nur x1, nur x2, beide, keines, nicht eindeutig). „Nicht eindeutig“: höchstens zwei verschiedene Zeilen (Gram-Matrix singulär), nicht gewertet."""
    y = Y_SWAPPED if swapped else Y
    only1 = only2 = both = none = degenerate = 0
    for idx in itertools.product(range(4), repeat=4):
        if len(set(idx)) < 3:
            degenerate += 1
            continue
        G, c = gram_and_c([[X1[i] for i in idx], [X2[i] for i in idx]], [y[i] for i in idx], center=True)
        b = unique_solution(G, c, alpha)
        s1, s2 = b[0] != 0, b[1] != 0
        if s1 and s2:
            both += 1
        elif s1:
            only1 += 1
        elif s2:
            only2 += 1
        else:
            none += 1
    return only1, only2, both, none, degenerate


def condition_q(a):
    """Zwei wahre unabhängige Merkmale (β = (1, 1)), eine Attrappe mit Korrelation a zu beiden: q = 2a (Größe der Bedingung von Zhao/Yu).
    Population: die Lasso-Koeffizienten der wahren Merkmale sind β - α(1, 1); der Zielwert der Attrappe ist 2aα = q·α; sie tritt bei beliebig kleinem α ein genau dann, wenn q > 1."""
    q = 2 * F(a)
    return q, ("gewählt" if q > 1 else "nicht gewählt" if q < 1 else "Rand")


def stability_bound(q, p, pi):
    """Schranke der erwarteten Zahl falsch gewählter Merkmale der Stabilitätsauswahl: q²/((2π - 1)p) (Meinshausen/Bühlmann 2010; sie setzt Austauschbarkeit der
    Nichtprädiktoren voraus), q = Zahl der je Teilstichprobe gewählten Merkmale, p = Zahl aller, π = Schwelle."""
    return F(q) ** 2 / ((2 * F(pi) - 1) * p)
