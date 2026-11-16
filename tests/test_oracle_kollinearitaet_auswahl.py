"""Orakel-Tests (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §5): der Kern gegen einen anderen Rechenweg.

* Koordinatenabstieg (`kol_algorithm.cd_batch`) gegen scikit-learn `Lasso`, `ElasticNet`, `lasso_path` (Toleranzen: Streuung des Orakels bei tol 1e-12 gemessen, nicht knapp darüber),
* gegen die **Aufzählung aller Aktivmengen und Vorzeichen in Brüchen** (`kol_exact`, drei Merkmale),
* gegen die geschlossene weiche Schwelle im orthonormalen Fall und die Optimalitätsbedingungen am eigenen Ergebnis,
* die Auswahl der Bevölkerung an der Bedingung von Zhao/Yu (q > 1) gegen die Aufzählung,
* VIF auf drei Wegen (Diagonale der inversen Korrelationsmatrix, 1/(1 − R²), Var·S_xx/σ²), Summe und Differenz geschlossen gegen die Matrix.
Die Testraten gegen nichtzentrales t und F stehen in `test_inference.py`, die Aufzählung des Bootstraps in `test_exact.py`. scikit-learn steht nur in `requirements-dev.txt`."""

from fractions import Fraction as F

import numpy as np
import pytest

import kol_algorithm as A
import kol_condition as COND
import kol_exact as E
import kol_sampling as SA

sk = pytest.importorskip("sklearn.linear_model")


def _instances(k, n=50, p=6, rho=0.9, seed=1):
    rng = np.random.default_rng(seed)
    for _ in range(k):
        X = rng.normal(size=(n, p))
        X[:, 1] = rho * X[:, 0] + np.sqrt(1 - rho ** 2) * X[:, 1]
        y = X @ (rng.normal(size=p) * rng.integers(0, 2, size=p)) + 2.0 * rng.normal(size=n)
        yield X, y


def test_lasso_and_elastic_net_equal_scikit_learn_on_standardized_data():
    """60 Instanzen mit korrelierten Spalten; eigener Abstieg tol 1e-12 gegen scikit-learn tol 1e-12: Koeffizienten auf 1e-7, gleiche gewählte Menge."""
    worst, same = 0.0, 0
    cases = list(_instances(30))
    for X, y in cases:
        Z, yc = A.standardize_batch(X[None], y[None])
        G, c = A.gram_batch(Z, yc)
        for alpha, l1 in ((0.15, 1.0), (0.2, 0.5)):
            b, _ = A.cd_batch(G[0], c[0], alpha, l1)
            if l1 == 1.0:
                ref = sk.Lasso(alpha=alpha, fit_intercept=False, tol=1e-12, max_iter=500000).fit(Z[0], yc[0]).coef_
            else:
                ref = sk.ElasticNet(alpha=alpha, l1_ratio=l1, fit_intercept=False, tol=1e-12, max_iter=500000).fit(Z[0], yc[0]).coef_
            worst = max(worst, float(np.max(np.abs(b - ref))))
            same += int(np.array_equal(np.abs(b) > 1e-8, np.abs(ref) > 1e-8))
    assert worst < 1e-7, worst
    assert same == 2 * len(cases)


def test_path_equals_lasso_path_and_active_sets_agree_at_every_point():
    X, y = next(_instances(1, n=100, p=8, rho=0.5, seed=402))
    Z, yc = A.standardize_batch(X[None], y[None])
    G, c = A.gram_batch(Z, yc)
    alphas, b = A.lasso_path(G[0], c[0], 1.0, 60, eps=1e-3, tol=1e-13)
    _, ref, _ = sk.lasso_path(Z[0], yc[0], alphas=alphas, tol=1e-13, max_iter=500000)
    ref = ref.T
    assert np.max(np.abs(b - ref)) < 1e-8
    assert np.array_equal(np.abs(b) > 1e-8, np.abs(ref) > 1e-8)                  # aktive Menge in allen 60 Punkten gleich (Schwelle 1e-8)


def test_enumeration_of_all_supports_and_signs_in_fractions_equals_the_float_descent():
    """Zufällige ganzzahlige Instanzen mit drei Merkmalen (3³ Vorzeichenmuster je Menge): Brüche gegen Gleitkomma, 40 Instanzen, α und ℓ zufällig."""
    rng = np.random.default_rng(7)
    done = 0
    while done < 40:
        M = rng.integers(-3, 4, size=(6, 3))
        y = rng.integers(-4, 5, size=6)
        cols = [[int(v) for v in M[:, j]] for j in range(3)]
        G, c = E.gram_and_c(cols, [int(v) for v in y], center=True)
        if E.solve([[G[i][j] + F(1, 100) * (i == j) for j in range(3)] for i in range(3)], [F(1)] * 3) is None:
            continue
        alpha = F(int(rng.integers(0, 40)), 20)
        l1 = [F(1), F(1, 2), F(1, 4)][int(rng.integers(0, 3))]
        try:
            exact = E.unique_solution(G, c, alpha, l1)
        except ValueError:
            continue                                                             # singuläres G (gleiche Spalten): nicht eindeutig, nicht gewertet
        Gf = np.array([[float(v) for v in row] for row in G])
        cf = np.array([float(v) for v in c])
        b, _ = A.cd_batch(Gf, cf, float(alpha), float(l1), tol=1e-14)
        assert np.max(np.abs(b - np.array([float(v) for v in exact]))) < 1e-9
        assert np.array_equal(b != 0.0, np.array([v != 0 for v in exact]))
        done += 1


@pytest.mark.parametrize("l1", [1.0, 0.5, 0.2])
def test_orthonormal_case_is_the_closed_form_soft_threshold(l1):
    rng = np.random.default_rng(11)
    for _ in range(20):
        z = rng.normal(0, 2, size=6)
        alpha = float(rng.uniform(0.1, 2.0))
        b, _ = A.cd_batch(np.eye(6), z, alpha, l1)
        closed = np.sign(z) * np.maximum(np.abs(z) - alpha * l1, 0.0) / (1.0 + alpha * (1.0 - l1))
        assert np.max(np.abs(b - closed)) < 1e-14


def test_optimality_conditions_hold_at_the_own_result_for_many_instances():
    worst = 0.0
    for X, y in _instances(40, n=40, p=8, rho=0.95, seed=5):
        Z, yc = A.standardize_batch(X[None], y[None])
        G, c = A.gram_batch(Z, yc)
        for alpha, l1 in ((0.1, 1.0), (0.3, 0.6), (0.02, 1.0)):
            b, _ = A.cd_batch(G[0], c[0], alpha, l1)
            worst = max(worst, A.kkt_violation(G[0], c[0], b, alpha, l1))
    assert worst < 1e-9, worst


@pytest.mark.parametrize("label,n_true,a", COND.DESIGNS)
def test_population_choice_of_the_dummy_follows_q_greater_than_one(label, n_true, a):
    """Alle sieben Entwürfe: der Koeffizient der Attrappe im Lasso der Bevölkerung ist genau dann ungleich 0, wenn q > 1 (hier ist keiner der Entwürfe am Rand); Brüche als Gegenprobe."""
    q = COND.q_value(n_true, a)
    coef = COND.population_dummy_coefficient(n_true, a)
    assert (abs(coef) > 1e-12) == (q > 1.0)
    # Aufzählung in Brüchen (a als Bruch aus dem Dezimaltext)
    af = F(str(a))
    Cm = [[F(int(i == j)) for j in range(n_true + 1)] for i in range(n_true + 1)]
    for j in range(n_true):
        Cm[n_true][j] = Cm[j][n_true] = af
    beta = [F(1)] * n_true + [F(0)]
    c = [sum(Cm[i][j] * beta[j] for j in range(n_true + 1)) for i in range(n_true + 1)]
    sol = E.lasso_exact(Cm, c, F(1, 20))
    assert len(sol) == 1 and (sol[0][n_true] != 0) == (q > 1.0)
    assert float(sol[0][n_true]) == pytest.approx(coef, abs=1e-9)


def test_the_edge_case_q_equals_one_is_not_counted():
    q = COND.q_value(2, 0.5)
    assert q == pytest.approx(1.0)
    assert abs(COND.population_dummy_coefficient(2, 0.5)) < 1e-12          # am Rand nicht gewählt (nicht gewertet: Zhao/Yu: fast notwendig und hinreichend)


def test_selection_pipeline_agrees_with_scikit_learn_on_generated_samples():
    """Die ganze Kette Welt → Standardisieren → Abstieg gegen scikit-learn, 60 Stichproben der Container-Welt mit ρ = 0,95."""
    X, y = SA.draw("C", "S3", 60, 100, 0.95, 2.0, 3, SA.TAG_FRESH)
    sel = SA.select_batch(X, y, 0.09, 1.0)
    for k in range(60):
        ref = sk.Lasso(alpha=0.09, fit_intercept=False, tol=1e-12, max_iter=500000).fit(X[k], y[k]).coef_
        assert np.array_equal(sel[k], np.abs(ref) > 1e-8), k
