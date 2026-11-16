"""Kern (`kol_algorithm`): Koordinatenabstieg gegen die Brüche des Handbeispiels, Randfälle der Eingabe, VIF, Optimalitätsbedingungen."""

from fractions import Fraction as F

import numpy as np
import pytest

import kol_algorithm as A
import kol_exact as E


def _hand_float(swapped=False):
    G, c = E.hand_data(swapped)
    return np.array([[float(v) for v in row] for row in G]), np.array([float(v) for v in c])


@pytest.mark.parametrize("alpha", [F(7, 2), F(3), F(2), F(1), F(1, 2), F(1, 4), F(0)])
def test_coordinate_descent_equals_the_exact_fractions_of_the_hand_example(alpha):
    G, c = _hand_float()
    b, _ = A.cd_batch(G, c, float(alpha), 1.0, tol=1e-14)
    assert np.max(np.abs(b - np.array([float(v) for v in E.hand_path(alpha)]))) < 1e-11


def test_elastic_net_descent_equals_7_20_and_1_10():
    G, c = _hand_float()
    b, _ = A.cd_batch(G, c, 2.0, 0.5, tol=1e-14)
    assert np.max(np.abs(b - np.array([0.35, 0.1]))) < 1e-12
    b, _ = A.cd_batch(*_hand_float(True), 2.0, 0.5, tol=1e-14)
    assert np.max(np.abs(b - np.array([0.1, 0.35]))) < 1e-12


def test_exact_zeros_are_exactly_zero_and_alpha_max_is_the_entry_point():
    G, c = _hand_float()
    assert A.alpha_max(c) == 3.5
    b, _ = A.cd_batch(G, c, 3.5, 1.0)
    assert np.all(b == 0.0)                                   # genau bei α_max alles null (exakt)
    b, _ = A.cd_batch(G, c, 3.5 - 1e-9, 1.0)
    assert b[1] == 0.0 and b[0] > 0.0                          # knapp darunter tritt nur x₁ ein
    b, _ = A.cd_batch(G, c, 10.0, 1.0)
    assert np.all(b == 0.0)                                   # darüber alles null
    assert A.alpha_max(c, 0.5) == 7.0                          # ℓ = 1/2: α_max = max|c|/ℓ
    with pytest.raises(ValueError):
        A.alpha_max(c, 0.0)


def test_alpha_zero_gives_ordinary_least_squares():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 4))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0]) + rng.normal(size=60)
    G, c = A.gram_of(X, y)
    b, _ = A.cd_batch(G, c, 0.0, 1.0, tol=1e-14)
    assert np.max(np.abs(b - np.linalg.solve(G, c))) < 1e-9


def test_a_stacked_fit_equals_the_single_fits():
    rng = np.random.default_rng(2)
    Gs, cs = [], []
    for _ in range(6):
        X = rng.normal(size=(50, 5))
        X[:, 1] = 0.9 * X[:, 0] + 0.1 * X[:, 1]
        y = X[:, 0] + rng.normal(size=50)
        G, c = A.gram_of(X, y)
        Gs.append(G)
        cs.append(c)
    stack, sweeps = A.cd_batch(np.array(Gs), np.array(cs), 0.1, 0.7)
    assert sweeps.shape == (6,) and np.all(sweeps >= 1)
    for k in range(6):
        single, _ = A.cd_batch(Gs[k], cs[k], 0.1, 0.7)
        assert np.array_equal(single, stack[k])                # gestapelt gleich einzeln, bitgleich (je Zeile eigene Rechnung)


def test_standardization_gives_unit_columns_and_handles_a_constant_column():
    rng = np.random.default_rng(3)
    X = rng.normal(5.0, 3.0, size=(2, 40, 3))
    X[1, :, 2] = 7.0                                           # konstante Spalte in der zweiten Stichprobe
    y = rng.normal(size=(2, 40))
    Z, yc = A.standardize_batch(X, y)
    assert np.allclose(Z[0].mean(axis=0), 0.0, atol=1e-12) and np.allclose((Z[0] ** 2).mean(axis=0), 1.0)
    assert np.all(Z[1][:, 2] == 0.0)                           # bleibt null und kann nie gewählt werden
    assert np.allclose(yc.mean(axis=1), 0.0, atol=1e-12)
    G, c = A.gram_batch(Z, yc)
    assert np.allclose(np.diagonal(G[0]), 1.0) and G[1][2, 2] == 0.0 and c[1][2] == 0.0
    b, _ = A.cd_batch(G, c, 0.05, 1.0)
    assert b[1][2] == 0.0
    b, _ = A.cd_batch(G, c, 0.05, 0.5)                         # auch mit quadratischem Anteil (Nenner α/2 > 0): null bleibt null
    assert b[1][2] == 0.0


def test_vif_three_ways_agree_and_two_columns_give_one_over_one_minus_r_squared():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 4))
    X[:, 1] = 0.9 * X[:, 0] + 0.3 * X[:, 1]
    X[:, 3] = X[:, 2] + 0.5 * X[:, 0] + 0.4 * rng.normal(size=200)
    v1, v2, v3 = A.vif_diag_inverse(X), A.vif_via_r2(X), A.vif_via_variance(X)
    assert np.max(np.abs(v1 - v2)) < 1e-10 and np.max(np.abs(v1 - v3)) < 1e-10 and np.all(v1 >= 1.0)
    X2 = X[:, :2]
    r = np.corrcoef(X2.T)[0, 1]
    assert np.allclose(A.vif_diag_inverse(X2), 1.0 / (1.0 - r * r), rtol=1e-12)


def test_vif_does_not_depend_on_scale_or_shift_of_the_columns():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(100, 3))
    X[:, 1] += 0.8 * X[:, 0]
    X2 = X * np.array([1000.0, 0.001, 3.0]) + np.array([5.0, -7.0, 1e4])
    assert np.allclose(A.vif_diag_inverse(X), A.vif_diag_inverse(X2), rtol=1e-8)


def test_two_feature_factors_and_the_unit_correlation_case():
    f = A.two_feature_factors(0.8)
    assert f["single"] == pytest.approx(25 / 9) and f["sum"] == pytest.approx(5 / 9) and f["diff"] == pytest.approx(5.0) and f["corr"] == -0.8
    assert A.vif_two(1.0) is None and A.two_feature_factors(1.0) is None and A.two_feature_factors(-1.0) is None
    assert A.vif_two(0.0) == 1.0


def test_kkt_violation_is_zero_at_the_solution_and_large_elsewhere():
    G, c = _hand_float()
    b, _ = A.cd_batch(G, c, 2.0, 1.0, tol=1e-14)
    assert A.kkt_violation(G, c, b, 2.0, 1.0) < 1e-12
    assert A.kkt_violation(G, c, b + np.array([0.05, 0.0]), 2.0, 1.0) > 0.1
    assert A.kkt_violation(G, c, np.zeros(2), 2.0, 1.0) > 1.0             # b = 0 ist bei α = 2 < α_max = 3,5 keine Lösung
    b0, _ = A.cd_batch(G, c, 2.0, 0.5, tol=1e-14)
    assert A.kkt_violation(G, c, b0, 2.0, 0.5) < 1e-12


def test_nonzero_count_depends_on_the_threshold():
    b = np.array([0.0, 1e-9, 1e-5, 0.02, -0.5])
    assert [A.nonzero_count(b, t) for t in (0.0, 1e-8, 1e-4, 1e-2)] == [4, 3, 2, 2]


def test_path_starts_at_zero_and_ends_near_least_squares():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(80, 4))
    y = X @ np.array([1.0, 0.0, -1.0, 0.5]) + rng.normal(size=80)
    G, c = A.gram_of(X, y)
    alphas, b = A.lasso_path(G, c, 1.0, 40, eps=1e-6)
    assert np.all(b[0] == 0.0) and alphas[0] == pytest.approx(A.alpha_max(c)) and np.all(np.diff(alphas) < 0)
    assert np.max(np.abs(b[-1] - np.linalg.solve(G, c))) < 1e-4
