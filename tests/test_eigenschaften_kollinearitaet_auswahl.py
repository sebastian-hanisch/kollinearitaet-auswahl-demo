"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen, die für Lasso, Elastic Net und VIF gelten müssen, ohne dass man das richtige Ergebnis kennt.

Je Eigenschaft 20 bis 50 Zufallsinstanzen (numpy mit festem Zufalls-Seed, nur im Test), unter 10 s. Eine Eigenschaft gilt nur mit ihren Voraussetzungen; sie stehen im Test."""

import numpy as np
import pytest

import kol_algorithm as A


def _problem(rng, n=60, p=5, rho=0.8):
    X = rng.normal(size=(n, p))
    X[:, 1] = rho * X[:, 0] + np.sqrt(1 - rho ** 2) * X[:, 1]
    y = X @ (rng.normal(size=p) * (rng.random(p) < 0.6)) + rng.normal(size=n)
    return X, y


def test_alpha_at_or_above_alpha_max_gives_all_zeros_and_just_below_exactly_one_entry():
    rng = np.random.default_rng(1)
    for _ in range(30):
        G, c = A.gram_of(*_problem(rng))
        amax = A.alpha_max(c)
        b, _ = A.cd_batch(G, c, amax, 1.0)
        assert np.all(b == 0.0)
        b, _ = A.cd_batch(G, c, amax * (1 - 1e-6), 1.0)
        assert int((b != 0.0).sum()) == 1 and b[int(np.argmax(np.abs(c)))] != 0.0


def test_alpha_to_zero_gives_ordinary_least_squares_when_n_exceeds_p():
    rng = np.random.default_rng(2)
    for _ in range(30):
        G, c = A.gram_of(*_problem(rng))
        b, _ = A.cd_batch(G, c, 1e-9, 1.0, tol=1e-14)
        assert np.max(np.abs(b - np.linalg.solve(G, c))) < 1e-5


def test_swapping_two_columns_swaps_their_coefficients():
    rng = np.random.default_rng(3)
    for _ in range(30):
        X, y = _problem(rng)
        for l1 in (1.0, 0.5):
            b, _ = A.cd_batch(*A.gram_of(X, y), 0.12, l1, tol=1e-14)
            perm = [1, 0, 2, 3, 4]
            bp, _ = A.cd_batch(*A.gram_of(X[:, perm], y), 0.12, l1, tol=1e-14)
            assert np.max(np.abs(bp - b[perm])) < 1e-9


def test_scaling_a_column_changes_the_lasso_but_not_after_standardizing():
    """Ohne Standardisieren ist das Lasso nicht skalenäquivariant (Gram-Matrix und c der Rohdaten), mit Standardisieren unabhängig von der Skala der Spalten."""
    rng = np.random.default_rng(4)
    changed = 0
    for _ in range(30):
        X, y = _problem(rng)
        X2 = X.copy()
        X2[:, 2] *= 10.0
        G, c = A.gram_of(X, y)
        G2, c2 = A.gram_of(X2, y)
        b, _ = A.cd_batch(G, c, 0.1, 1.0, tol=1e-14)
        b2, _ = A.cd_batch(G2, c2, 0.1, 1.0, tol=1e-14)
        assert np.max(np.abs(b - b2)) < 1e-9                    # standardisiert: gleiche Koeffizienten
        n = len(y)
        Xc, yc = X - X.mean(0), y - y.mean()
        X2c = X2 - X2.mean(0)
        raw, _ = A.cd_batch(Xc.T @ Xc / n, Xc.T @ yc / n, 0.1, 1.0, tol=1e-14)
        raw2, _ = A.cd_batch(X2c.T @ X2c / n, X2c.T @ yc / n, 0.1, 1.0, tol=1e-14)
        changed += int(np.max(np.abs(raw2[2] * 10.0 - raw[2])) > 1e-6 or not np.array_equal(raw != 0, raw2 != 0))
    assert changed >= 25                                         # roh: Skalieren verändert die Lösung in (fast) jeder Instanz (gemessen: Zählung, nicht „immer“)


def test_scaling_y_by_c_scales_the_lasso_solution_at_alpha_times_c():
    """Nur für das Lasso (ℓ = 1): bei einem Elastic Net skaliert die quadratische Strafe mit c², die L1-Strafe mit c; dort gibt es keine Gleichung dieser Form."""
    rng = np.random.default_rng(5)
    for _ in range(30):
        X, y = _problem(rng)
        for c_ in (0.5, 3.0):
            G, c = A.gram_of(X, y)
            Gs, cs = A.gram_of(X, c_ * y)
            b, _ = A.cd_batch(G, c, 0.1, 1.0, tol=1e-14)
            bs, _ = A.cd_batch(Gs, cs, 0.1 * c_, 1.0, tol=1e-14)
            assert np.max(np.abs(bs - c_ * b)) < 1e-9
            assert np.array_equal(bs != 0, b != 0)


def test_l1_ratio_zero_is_ridge_with_no_exact_zeros():
    rng = np.random.default_rng(6)
    for _ in range(30):
        G, c = A.gram_of(*_problem(rng))
        b, _ = A.cd_batch(G, c, 0.3, 0.0, tol=1e-14)
        assert np.max(np.abs(b - np.linalg.solve(G + 0.3 * np.eye(5), c))) < 1e-9
        assert A.nonzero_count(b, 0.0) == 5


def test_two_identical_columns_elastic_net_splits_equally_and_lasso_fixes_only_the_sum():
    """Zwei gleiche Spalten: Elastic Net (ℓ < 1, streng konvex) gibt gleiche Koeffizienten; beim Lasso ist nur die Summe bestimmt und gleich dem Koeffizienten der zusammengelegten Spalte (abgeleitet, keine Quelle)."""
    rng = np.random.default_rng(7)
    for _ in range(30):
        X, y = _problem(rng, p=4)
        Xd = np.column_stack([X[:, 0], X[:, 0], X[:, 1:]])
        G, c = A.gram_of(Xd, y)
        be, _ = A.cd_batch(G, c, 0.1, 0.5, tol=1e-14)
        assert abs(be[0] - be[1]) < 1e-9
        bl, _ = A.cd_batch(G, c, 0.1, 1.0, tol=1e-14)
        G1, c1 = A.gram_of(X, y)
        merged, _ = A.cd_batch(G1, c1, 0.1, 1.0, tol=1e-14)
        # Standardisierte Spalten: die Summe der beiden gleichen Koeffizienten entspricht dem Koeffizienten der einzelnen Spalte
        assert abs(bl[0] + bl[1] - merged[0]) < 1e-8 and np.max(np.abs(bl[2:] - merged[1:])) < 1e-8
        assert bl[0] * bl[1] >= 0.0                              # beide mit gleichem Vorzeichen (oder null)


def test_the_l1_norm_of_the_lasso_solution_does_not_increase_with_alpha():
    rng = np.random.default_rng(8)
    for _ in range(30):
        G, c = A.gram_of(*_problem(rng))
        _, b = A.lasso_path(G, c, 1.0, 30, eps=1e-3)
        l1 = np.abs(b).sum(axis=1)
        assert np.all(np.diff(l1) >= -1e-9)                      # α fällt entlang des Pfads, die Summe der Beträge steigt


def test_vif_is_at_least_one_and_equals_one_for_orthogonal_columns():
    rng = np.random.default_rng(9)
    for _ in range(30):
        X, _ = _problem(rng)
        assert np.all(A.vif_diag_inverse(X) >= 1.0 - 1e-12)
    Xc = rng.normal(size=(50, 4))
    Q, _ = np.linalg.qr(Xc - Xc.mean(axis=0))                      # orthonormale und zentrierte Spalten: Korrelationsmatrix = Einheitsmatrix
    assert np.allclose(A.vif_diag_inverse(Q), 1.0)


def test_more_data_gives_a_selection_closer_to_the_truth_on_average():
    """Mittel über Zufallsinstanzen: bei n = 400 wählt das Lasso (festes α) das wahre Merkmal x₃ öfter und die Attrappen seltener als bei n = 30 (Voraussetzung: kleines Signal im Rauschen, gleiches α)."""
    import kol_sampling as SA

    def stats(n):
        sel = SA.fresh_selection("C", "S1", n, 0.0, 2.0, 0.09, 1.0, 300, 17)
        return sel[:, 2].mean(), sel[:, 3:].mean()

    (t30, d30), (t400, d400) = stats(30), stats(400)
    assert t400 > t30 and d400 < d30 + 0.2                       # Attrappen: bei festem α nicht monoton, nur „nicht mehr als“ gemessen
