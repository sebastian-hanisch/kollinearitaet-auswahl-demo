"""Experiment 2 (`kol_inference`): Design, geschlossene Varianzen, Testraten gegen die exakte nichtzentrale Rechnung, unabhängige Vollsimulation, VIF-Streuung.

Zählungen aus Neu-Rechnung nur als Band (Plattform, Monte-Carlo-Fehler); exakt nur, was geschlossen oder in Brüchen gerechnet ist."""

import numpy as np
import pytest
from scipy import stats

import kol_algorithm as A
import kol_inference as I
from kol_rng import SplitMix64


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9, 0.999, 1.0])
def test_fixed_design_has_exact_mean_scale_and_correlation(rho):
    X = I.fixed_design(100, rho, 3)
    assert np.allclose(X.mean(axis=0), 0.0, atol=1e-12) and np.allclose((X ** 2).sum(axis=0), 100.0)
    assert np.corrcoef(X.T)[0, 1] == pytest.approx(rho, abs=1e-12)
    if rho == 1.0:
        assert np.allclose(X[:, 0], X[:, 1])


def test_fixed_design_rejects_correlations_outside_the_range():
    for bad in (-0.2, 1.01):
        with pytest.raises(ValueError):
            I.fixed_design(50, bad)


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999])
def test_variance_formulas_equal_the_matrix(rho):
    """Var(b₁) = σ²/(n(1−ρ²)), Var(b₁ ± b₂) = 2σ²/(n(1 ± ρ)), Korr(b₁, b₂) = −ρ, gegen σ²(X'X)⁻¹ (Orakel: anderer Rechenweg als die Formel)."""
    n = 100
    X = I.fixed_design(n, rho, 1)
    V = np.linalg.inv(X.T @ X)
    assert V[0, 0] == pytest.approx(1.0 / (n * (1 - rho ** 2)), rel=1e-9)
    assert V[0, 0] + V[1, 1] + 2 * V[0, 1] == pytest.approx(2.0 / (n * (1 + rho)), rel=1e-9)
    assert V[0, 0] + V[1, 1] - 2 * V[0, 1] == pytest.approx(2.0 / (n * (1 - rho)), rel=1e-9)
    assert V[0, 1] / V[0, 0] == pytest.approx(-rho, abs=1e-9)
    f = A.two_feature_factors(rho)
    assert V[0, 0] * n == pytest.approx(f["single"], rel=1e-9)


def test_simulated_variances_agree_with_the_formulas_in_a_band():
    X = I.fixed_design(100, 0.9, 5)
    r = I.variance_check(X, 1.0, 20000, 11)
    assert r["n_var_b1"] == pytest.approx(1 / (1 - 0.81), rel=0.05)
    assert r["n_var_sum"] == pytest.approx(2 / 1.9, rel=0.05)
    assert r["n_var_diff"] == pytest.approx(2 / 0.1, rel=0.05)
    assert r["corr"] == pytest.approx(-0.9, abs=0.02)


@pytest.mark.parametrize("rho,b", [(0.0, 0.5), (0.9, 0.5), (0.95, 0.5), (0.95, 1.0)])
def test_test_rates_agree_with_the_exact_noncentral_distributions(rho, b):
    """Simulation (6000 Wiederholungen, Monte-Carlo-Fehler höchstens 0,0065) gegen nichtzentrales t und F; Band 0,03."""
    X = I.fixed_design(100, rho, 5)
    r = I.test_rates(X, (b, b), 2.0, 6000, 17)
    assert abs(r["p_single"] - r["p_single_exact"]) < 0.03
    assert abs(r["p_F"] - r["p_F_exact"]) < 0.03
    assert r["se_formula"] == pytest.approx(2.0 / np.sqrt(100 * (1 - rho ** 2)), rel=1e-9)
    assert abs(r["se_mean"] - r["se_formula"]) / r["se_formula"] < 0.02


def test_under_the_null_both_tests_keep_their_level_and_intervals_cover():
    X = I.fixed_design(60, 0.95, 2)
    r = I.test_rates(X, (0.0, 0.0), 1.0, 20000, 23)
    assert r["p_single"] == pytest.approx(0.05, abs=0.01) and r["p_F"] == pytest.approx(0.05, abs=0.01)
    assert r["p_single_exact"] == pytest.approx(0.05, abs=1e-9) and r["p_F_exact"] == pytest.approx(0.05, abs=1e-9)
    assert r["cover"] == pytest.approx(0.95, abs=0.01)


def test_collinearity_does_not_break_the_intervals_it_only_widens_them():
    """Überdeckung 0,95 ± 0,015 bei ρ = 0 und ρ = 0,99; Breite im Verhältnis √VIF (Band)."""
    w = {}
    for rho in (0.0, 0.99):
        r = I.test_rates(I.fixed_design(100, rho, 5), (0.5, 0.5), 2.0, 5000, 31)
        assert r["cover"] == pytest.approx(0.95, abs=0.015)
        w[rho] = r["width"]
    assert w[0.99] / w[0.0] == pytest.approx(np.sqrt(1 / (1 - 0.99 ** 2)), rel=0.03)


def test_independent_full_simulation_with_the_reference_generator_agrees():
    """Anderer Rechenweg: jede Wiederholung zieht alle n Fehler mit `kol_rng.SplitMix64.normal()` und rechnet OLS mit Achsenabschnitt über `lstsq`; die Raten liegen im Band um die exakte Rechnung."""
    n, reps, rho, b, sigma = 30, 400, 0.9, 0.5, 2.0
    X = I.fixed_design(n, rho, 8)
    rng = SplitMix64(4242)
    A1 = np.column_stack([np.ones(n), X])
    tcrit = stats.t.ppf(0.975, n - 3)
    fcrit = stats.f.ppf(0.95, 2, n - 3)
    rej_t = rej_f = 0
    for _ in range(reps):
        y = b * X[:, 0] + b * X[:, 1] + sigma * np.array([rng.normal() for _ in range(n)])
        coef, *_ = np.linalg.lstsq(A1, y, rcond=None)
        res = y - A1 @ coef
        s2 = res @ res / (n - 3)
        cov = s2 * np.linalg.inv(A1.T @ A1)
        rej_t += abs(coef[1] / np.sqrt(cov[1, 1])) > tcrit
        Fstat = ((coef[1:] @ np.linalg.inv(cov[1:, 1:]) @ coef[1:]) / 2.0)
        rej_f += Fstat > fcrit
    exact = I.test_rates(X, (b, b), sigma, 10, 1)
    assert abs(rej_t / reps - exact["p_single_exact"]) < 0.08 and abs(rej_f / reps - exact["p_F_exact"]) < 0.1


def test_vif_spread_median_equals_the_formula_for_large_n_and_scatters_for_small_n():
    big = I.vif_spread(1000, 0.9, 1500, 7)
    assert big["q50"] / big["formula"] == pytest.approx(1.0, abs=0.03) and abs(big["rel05"]) < 0.15 and abs(big["rel95"]) < 0.15
    small = I.vif_spread(30, 0.9, 1500, 7)
    assert small["rel05"] < -0.3 and small["rel95"] > 0.6                      # bei n = 30 streut der VIF der Stichprobe stark
    assert big["formula"] == pytest.approx(1 / (1 - 0.81))
    assert I.vif_spread(50, 0.0, 500, 3)["formula"] == 1.0


def test_vif_spread_agrees_with_explicit_samples_by_the_reference_correlation():
    """Gegenprobe der Quantile mit unabhängig gezogenen Stichproben (`numpy.random`, nur im Test)."""
    rng = np.random.default_rng(1)
    n, rho = 100, 0.9
    z = rng.normal(size=(3000, 2, n))
    x2 = rho * z[:, 0] + np.sqrt(1 - rho ** 2) * z[:, 1]
    r = np.array([np.corrcoef(z[k, 0], x2[k])[0, 1] for k in range(3000)])
    q_ref = np.quantile(1 / (1 - r ** 2), [0.05, 0.5, 0.95])
    got = I.vif_spread(n, rho, 3000, 9)
    assert got["q05"] == pytest.approx(q_ref[0], rel=0.06) and got["q50"] == pytest.approx(q_ref[1], rel=0.06) and got["q95"] == pytest.approx(q_ref[2], rel=0.08)
