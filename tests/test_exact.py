"""Handbeispiel mit vier Containern, alles in Brüchen (`fractions`): jede Erwartung ist von Hand gerechnet und steht als Bruch im Test (DEMO-PLAYBOOK §1b, §5).

Erfundene ganze Zahlen (Abweichungen vom Mittel): x₁ = (−3, −1, 1, 3), x₂ = (−3, −1, 3, 1), y = (−2, −1, 1, 2); y′ mit vertauschten Zeilen 3 und 4. Gerechnet ohne Achsenabschnitt und ohne Standardisieren.
Von Hand: X'X = [[20, 16], [16, 20]] (Σx₁² = 9+1+1+9, Σx₁x₂ = 9+1+3+3), Determinante 400 − 256 = 144, (X'X)⁻¹ = (1/144)·[[20, −16], [−16, 20]]."""

import itertools
from fractions import Fraction as F

import pytest

import kol_exact as E


def test_gram_matrix_vif_and_variance_factors_by_hand():
    r = E.hand_report()
    assert r["xtx"] == [[20, 16], [16, 20]]
    assert r["rho"] == F(4, 5)
    assert r["var_single"] == F(20, 144) == F(5, 36)
    assert r["vif"] == F(25, 9) == 1 / (1 - F(4, 5) ** 2)
    assert r["factor_single"] == F(25, 9)                     # Var(b₁)/(1/20)
    assert r["var_sum"] == F(1, 18) and r["factor_sum"] == F(5, 9) == 1 / (1 + F(4, 5))
    assert r["var_diff"] == F(1, 2) and r["factor_diff"] == 5 == 1 / (1 - F(4, 5))
    assert r["corr_b"] == -F(4, 5)


def test_ols_coefficients_residuals_and_the_swap_of_two_rows():
    r, rs = E.hand_report(), E.hand_report(swapped=True)
    assert r["b"] == [F(11, 18), F(1, 9)]
    assert r["resid"] == [F(1, 6), F(-5, 18), F(1, 18), F(1, 18)] and sum(r["resid"]) == 0
    assert r["sse"] == F(1, 9) and r["df"] == 1
    assert rs["b"] == [F(1, 9), F(11, 18)]                    # die Koeffizienten tauschen die Plätze
    assert r["b"][0] + r["b"][1] == rs["b"][0] + rs["b"][1] == F(13, 18)   # die Summe bleibt
    assert rs["sse"] == r["sse"]


def _path_closed_form(alpha):
    """Von Hand: G = X'X/4 = [[5, 4], [4, 5]], c = X'y/4 = (7/2, 3). x₁ tritt bei α = 7/2 ein (b₁ = (7/2 − α)/5), x₂ bei α < 1 (|c₂ − 4b₁| = (1 + 4α)/5 > α)."""
    if alpha >= F(7, 2):
        return [F(0), F(0)]
    if alpha >= 1:
        return [(F(7, 2) - alpha) / 5, F(0)]
    return [F(11, 18) - alpha / 9, (1 - alpha) / 9]


@pytest.mark.parametrize("alpha", [F(4), F(7, 2), F(3), F(2), F(3, 2), F(1), F(1, 2), F(1, 4), F(0)])
def test_lasso_path_enumeration_equals_the_hand_formula(alpha):
    assert E.hand_path(alpha) == _path_closed_form(alpha)
    assert E.hand_path(alpha, swapped=True) == _path_closed_form(alpha)[::-1]    # mit vertauschten Zeilen: erst x₂, der Pfad ist das Spiegelbild


def test_hand_values_named_in_the_text():
    assert E.hand_path(F(3)) == [F(1, 10), 0] and E.hand_path(F(2)) == [F(3, 10), 0] and E.hand_path(F(1, 2)) == [F(5, 9), F(1, 18)]
    assert E.hand_path(F(2), swapped=True) == [0, F(3, 10)]    # bei α = 2 wählt y das Merkmal x₁, y′ das Merkmal x₂
    assert E.hand_path(F(0)) == E.hand_report()["b"]
    first, second, k = E.hand_entry_points()
    assert (first, second, k) == (F(7, 2), F(1), 0)
    assert E.hand_entry_points(swapped=True) == (F(7, 2), F(1), 1)


def test_elastic_net_takes_both_where_the_lasso_takes_one():
    G, c = E.hand_data()
    # (G + (α(1−ℓ)) I) b = c − αℓ (1, 1) bei α = 2, ℓ = 1/2: (6, 4; 4, 6)·b = (5/2, 2)  →  b = (7/20, 1/10)
    en = E.unique_solution(G, c, F(2), F(1, 2))
    assert en == [F(7, 20), F(1, 10)]
    assert 6 * en[0] + 4 * en[1] == F(5, 2) and 4 * en[0] + 6 * en[1] == 2
    assert E.unique_solution(G, c, F(2)) == [F(3, 10), 0]
    assert E.unique_solution(*E.hand_data(True), F(2), F(1, 2)) == [F(1, 10), F(7, 20)]


@pytest.mark.parametrize("z,alpha", [([F(3), F(1, 2)], F(1)), ([F(-2), F(5, 2)], F(3, 2)), ([F(1, 3), F(-7, 4)], F(1, 2))])
def test_orthonormal_case_is_the_soft_threshold(z, alpha):
    I2 = [[F(1), F(0)], [F(0), F(1)]]

    def soft(v, g):
        return ((v > 0) - (v < 0)) * max(abs(v) - g, F(0))

    assert E.unique_solution(I2, z, alpha) == [soft(v, alpha) for v in z]
    assert E.unique_solution(I2, z, alpha, F(1, 2)) == [soft(v, alpha / 2) / (1 + alpha / 2) for v in z]


def test_bootstrap_counts_all_256_samples_and_the_mirror():
    cy, cs = E.bootstrap_counts(), E.bootstrap_counts(swapped=True)
    assert cy == (108, 36, 0, 24, 88)                         # nur x₁, nur x₂, beide, keines, nicht eindeutig
    assert cs == (36, 108, 0, 24, 88)                         # mit y′ genau das Spiegelbild
    assert sum(cy) == 256 == 4 ** 4
    assert 88 == 4 + 6 * 14                                   # 4 Stichproben mit einer Zeile, 6·14 mit genau zwei verschiedenen Zeilen
    used = 256 - cy[4]
    assert (F(cy[0] + cy[2], used), F(cy[1] + cy[2], used)) == (F(9, 14), F(3, 14))
    assert F(cy[0], cy[0] + cy[1]) == F(3, 4)


def test_the_count_of_non_unique_samples_by_direct_enumeration():
    n_two_or_fewer = sum(1 for idx in itertools.product(range(4), repeat=4) if len(set(idx)) < 3)
    assert n_two_or_fewer == 88


@pytest.mark.parametrize("a,q,verdict", [(F(2, 5), F(4, 5), "nicht gewählt"), (F(3, 5), F(6, 5), "gewählt"), (F(1, 2), F(1), "Rand")])
def test_condition_q_and_the_population_choice_by_hand(a, q, verdict):
    assert E.condition_q(a) == (q, verdict)
    # Bevölkerung, β = (1, 1), α = 1/10: wahre Koeffizienten β − α(1, 1) = 9/10 je Merkmal; Zielwert der Attrappe c₀ − C₀₁b = a·2 − a·2·(9/10) = 2aα = q·α
    alpha = F(1, 10)
    target = a * 2 - a * (2 * (1 - alpha))
    assert target == q * alpha and (target > alpha) == (verdict == "gewählt")


def test_stability_bound_examples():
    assert E.stability_bound(7, 10, F(4, 5)) == F(49, 6) and E.stability_bound(7, 10, F(4, 5)) > 7
    assert E.stability_bound(2, 10, F(9, 10)) == F(1, 2) and E.stability_bound(8, 100, F(9, 10)) == F(4, 5)


def test_solve_handles_singular_and_unique_systems():
    assert E.solve([[F(2), F(1)], [F(1), F(3)]], [F(5), F(10)]) == [F(1), F(3)]
    assert E.solve([[1, 2], [2, 4]], [1, 2]) is None
    with pytest.raises(ValueError):
        E.unique_solution([[F(1), F(1)], [F(1), F(1)]], [F(1), F(1)], F(0))      # zwei gleiche Spalten, α = 0: nicht eindeutig
