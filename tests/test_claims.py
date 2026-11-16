"""Claims-Tests: jede Zahl der Befunde B1 bis B13 der README (B14 ist eine Messung der README, siehe dort) mit ihren Voraussetzungen (DEMO-PLAYBOOK §4, §11a).

Plattform (Lehre 2026-11-15): Zählungen aus Neu-Rechnung als Band, Brüche exakt. Die vorgerechneten Zahlen stehen in `data/kol_frozen.json` (erzeugt von `tools/kol_make_frozen.py`);
dieser Test vergleicht sie mit den in der README gedruckten Zahlen (halbe Einheit der letzten gedruckten Stelle) und rechnet die Hauptaussagen mit anderen Zufalls-Seeds und kleinerer
Zahl neu (Band). Zufall nur aus `kol_stream` (bitgleich auf jeder Plattform)."""

from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest

import kol_constants as C
import kol_evaluation as ev
import kol_exact as E
import kol_frozen as FZ
import kol_inference as I
import kol_sampling as SA


@pytest.fixture(scope="module")
def fz():
    return FZ.load()


def _row(rows, **kw):
    hit = [r for r in rows if all(r[k] == v for k, v in kw.items())]
    assert len(hit) == 1, kw
    return hit[0]


def pct(x, digits=1):
    return pytest.approx(x, abs=0.5 * 10 ** -(digits + 2) + 1e-12)       # Prozentwert mit `digits` Stellen gedruckt (Anteil 0,xyz): halbe Einheit der letzten Stelle


# --- B1: Aufblähung, Summe, Differenz (festes Design n = 100, 100 000 Wiederholungen, Zufalls-Seed 1201) -----------------------------------------

@pytest.mark.parametrize("rho,single,f_single,summ,f_sum,diff,f_diff", [
    (0.9, 5.28, 5.26, 1.063, 1.053, 20.07, 20.0), (0.95, 10.29, 10.26, 1.035, 1.026, 40.14, 40.0),
    (0.99, 50.43, 50.25, 1.014, 1.005, 200.7, 200.0), (0.999, 502.1, 500.25, 1.010, 1.0005, 2007.7, 2000.0)])
def test_b1_variance_of_single_coefficient_sum_and_difference(fz, rho, single, f_single, summ, f_sum, diff, f_diff):
    r = _row(fz["sumdiff"], rho=rho)
    assert r["n_var_b1"] == pytest.approx(single, abs=0.006 if single < 100 else 0.06) and r["formula_single"] == pytest.approx(f_single, abs=0.006 if single < 100 else 0.006)
    assert r["n_var_sum"] == pytest.approx(summ, abs=0.0006) and r["formula_sum"] == pytest.approx(f_sum, abs=0.00051)
    assert r["n_var_diff"] == pytest.approx(diff, abs=0.006 if diff < 100 else 0.06 if diff < 1000 else 0.06) and r["formula_diff"] == pytest.approx(f_diff)
    assert r["corr"] == pytest.approx(-rho, abs=0.001) and r["reps"] == 100000


def test_b1_formulas_by_hand_for_rho_095():
    """Die Formeln selbst, ohne Simulation: n·Var(b₁) = 1/(1 − 0,95²), Summe 2/(1 + 0,95), Differenz 2/(1 − 0,95) in Einheiten σ² (hier die Faktoren gegen orthogonal: Summe 1/(1 + ρ) mal 2)."""
    assert 1 / (1 - 0.95 ** 2) == pytest.approx(10.2564, abs=5e-5) and 2 / 1.95 == pytest.approx(1.0256, abs=5e-5) and 2 / 0.05 == pytest.approx(40.0)


def test_b1_recomputed_with_another_seed_and_fewer_repetitions_in_a_band():
    X = I.fixed_design(100, 0.95, 5)
    r = I.variance_check(X, 1.0, 20000, 4711)
    assert r["n_var_b1"] == pytest.approx(10.2564, rel=0.04) and r["n_var_sum"] == pytest.approx(1.0256, rel=0.04) and r["n_var_diff"] == pytest.approx(40.0, rel=0.04)


# --- B2: Einzeltest gegen gemeinsamen Test (festes Design, n = 100, σ = 2, 20 000 Wiederholungen, Zufalls-Seed 1202) -----------------------------

@pytest.mark.parametrize("b,rho,se,single,single_exact,both,F_,F_exact,F_no", [
    (0.5, 0.0, 0.200, 0.697, 0.697, 0.489, 0.886, 0.886, 0.010), (0.5, 0.9, 0.459, 0.190, 0.190, 0.003, 0.994, 0.994, 0.617),
    (0.5, 0.95, 0.641, 0.123, 0.121, 0.008, 0.994, 0.995, 0.759), (0.5, 0.99, 1.418, 0.065, 0.064, 0.022, 0.995, 0.995, 0.887),
    (1.0, 0.95, 0.641, 0.336, 0.340, 0.003, 1.0, 1.0, 0.325), (1.0, 0.99, 1.418, 0.111, 0.107, 0.009, 1.0, 1.0, 0.790)])
def test_b2_single_tests_against_the_joint_test(fz, b, rho, se, single, single_exact, both, F_, F_exact, F_no):
    r = _row(fz["tests"], b=b, rho=rho)
    assert r["se_formula"] == pytest.approx(se, abs=0.0006)
    assert r["p_single"] == pct(single, 1) and r["p_single_exact"] == pct(single_exact, 1) and r["p_both_single"] == pct(both, 1)
    assert r["p_F"] == pct(F_, 1) and r["p_F_exact"] == pct(F_exact, 1) and r["p_F_no_single"] == pct(F_no, 1)
    assert abs(r["p_single"] - r["p_single_exact"]) < 0.015 and abs(r["p_F"] - r["p_F_exact"]) < 0.015      # Simulation gegen exakt: Band 0,015


def test_b2_headline_rho_095_recomputed_with_another_seed_in_a_band():
    r = I.test_rates(I.fixed_design(100, 0.95, 5), (0.5, 0.5), 2.0, 8000, 90210)
    assert 0.09 < r["p_single"] < 0.15 and r["p_both_single"] < 0.03 and r["p_F"] > 0.98 and 0.72 < r["p_F_no_single"] < 0.80


# --- B3: Intervalle (β = (0,5; 0,5), 20 000 Wiederholungen, Zufalls-Seed 1203) -----------------------------------------------------------------

def test_b3_intervals_stay_valid_and_get_wide(fz):
    cover = {r["rho"]: r for r in fz["coverage"]}
    assert [round(cover[k]["cover"], 3) for k in (0.0, 0.9, 0.99, 0.999)] == [0.952, 0.948, 0.948, 0.948]
    assert all(abs(r["cover"] - 0.95) < 0.0075 for r in fz["coverage"])                                    # 0,95 ± 0,0075 (Monte-Carlo-Fehler 0,0015)
    assert [round(cover[k]["width_ratio"], 2) for k in (0.9, 0.99, 0.999)] == [2.29, 7.09, 22.37]
    assert cover[0.0]["width"] == pytest.approx(0.792, abs=0.0006)


# --- B4: Streuung des VIF der Stichprobe (Zufallsdesign, 4000 Designs, Zufalls-Seed 1204) -----------------------------------------------------

@pytest.mark.parametrize("rho,n,formula,q05,q95,rel05,rel95", [
    (0.9, 30, 5.26, 3.13, 9.66, -40, 84), (0.9, 100, 5.26, 3.99, 7.25, -24, 38), (0.9, 1000, 5.26, 4.81, 5.78, -9, 10),
    (0.99, 100, 50.25, 36.92, 71.25, -27, 42)])
def test_b4_vif_of_the_sample_scatters_around_the_formula(fz, rho, n, formula, q05, q95, rel05, rel95):
    r = _row(fz["vif_spread"], rho=rho, n=n)
    assert r["formula"] == pytest.approx(formula, abs=0.006) and r["q05"] == pytest.approx(q05, abs=0.006) and r["q95"] == pytest.approx(q95, abs=0.006)
    assert round(100 * r["rel05"]) == rel05 and round(100 * r["rel95"]) == rel95


def test_b4_share_of_designs_with_more_than_twice_the_formula(fz):
    assert _row(fz["vif_spread"], rho=0.9, n=30)["over2"] == pytest.approx(0.031, abs=0.0006) and _row(fz["vif_spread"], rho=0.99, n=30)["over2"] == pytest.approx(0.042, abs=0.0006)
    assert all(r["over2"] == 0.0 for r in fz["vif_spread"] if r["n"] >= 100)


# --- B5: Kollinearität im Export-Zweig --------------------------------------------------------------------------------------------------------

def test_b5_the_export_branch_alone_has_no_high_collinearity(fz):
    nat = fz["natural_vif_export"]
    assert nat["n"] == 60000 and nat["max_vif"] == pytest.approx(1.87, abs=0.005)
    assert max(nat["vif_vorlauf"], nat["vif_wochenende"], nat["vif_gewicht"]) < 1.001                       # die drei Merkmale außer den Kunden-Dummies: VIF unter 1,001


# --- B6: Auswahl über frische Stichproben (n = 100, p = 10, ρ = 0,95, σ = 2, α = 0,09, 2000 frische Stichproben) ---------------------------------

@pytest.mark.parametrize("key,p1,p2,both,share,att", [
    ("C_S3_L", 0.761, 0.766, 0.527, 0.495, 0.640), ("C_S3_L_seed2", 0.753, 0.762, 0.515, 0.490, 0.633), ("C_S1_L", 0.931, 0.924, 0.856, 0.522, 0.647),
    ("C_S2_L", 0.999, 0.468, 0.467, 0.998, 0.640), ("E_S2E_L", 1.000, 0.480, 0.480, 1.000, 0.570), ("C_S3_L_rho099", 0.613, 0.628, 0.241, 0.490, 0.637)])
def test_b6_selection_frequencies_of_the_lasso(fz, key, p1, p2, both, share, att):
    f = fz["selection"][key]["fresh"]
    assert f["p1"] == pct(p1, 1) and f["p2"] == pct(p2, 1) and f["both"] == pct(both, 1) and f["share1"] == pct(share, 1) and f["att"] == pct(att, 1)


def test_b6_exactly_one_of_the_twins_is_chosen_only_sometimes(fz):
    """S3: in 945 von 2000 Stichproben genau eines (47,3 %), S1: 289 (14,5 %), ρ = 0,99 in S3: 1517 (75,9 %): „wie ein Münzwurf“ gilt für S3 (Anteil 49,5 %), nicht als Aussage über S1 (dort wird fast immer beides gewählt)."""
    assert [fz["selection"][k]["fresh"]["n_one"] for k in ("C_S3_L", "C_S1_L", "C_S3_L_rho099")] == [945, 289, 1517]
    assert 289 / 2000 == pytest.approx(0.1445) and 945 / 2000 == pytest.approx(0.4725) and 1517 / 2000 == pytest.approx(0.7585)
    assert fz["selection"]["C_S1_L"]["fresh"]["both"] == pytest.approx(0.856, abs=0.0005)


def test_b6_recomputed_live_with_another_seed_in_a_band():
    f = ev.fresh_analysis("C", "S3", 100, 0.95, 2.0, 0.09, 1.0, 600, 2468)
    assert 0.68 < f["p1"] < 0.82 and 0.68 < f["p2"] < 0.82 and 0.42 < f["share1"] < 0.58 and 0.40 < f["both"] < 0.60
    f2 = ev.fresh_analysis("C", "S2", 100, 0.95, 2.0, 0.09, 1.0, 600, 2468)
    assert f2["p1"] > 0.99 and 0.40 < f2["p2"] < 0.55 and f2["share1"] > 0.99          # S2: x₂ ist die Attrappe, in fast der Hälfte der Stichproben mitgewählt


# --- B7: die Falle des Bootstraps (200 Originalstichproben je Zelle, B = 100) ---------------------------------------------------------------------

def test_b7_bootstrap_difference_per_original_sample_against_fresh_samples_in_s3(fz):
    c = fz["selection"]["C_S3_L"]
    d_fresh = abs(c["fresh"]["p1"] - c["fresh"]["p2"])
    assert d_fresh == pytest.approx(0.005, abs=0.0006) and c["boot"]["absdiff"] == pytest.approx(0.392, abs=0.0006) and c["sub"]["absdiff"] == pytest.approx(0.374, abs=0.0006)
    assert c["boot"]["clear"] == 0.65 and c["sub"]["clear"] == 0.58 and c["boot"]["n_share"] == 200
    assert c["boot"]["share_mean"] == pytest.approx(0.477, abs=0.0006) and c["boot"]["share_sd"] == pytest.approx(0.343, abs=0.0006)
    c2 = fz["selection"]["C_S3_L_seed2"]
    assert c2["boot"]["clear"] == 0.59 and c2["sub"]["clear"] == 0.565 and c2["boot"]["absdiff"] == pytest.approx(0.395, abs=0.0006)
    # Gegenprobe: die Zählung „klar“ schwankt zwischen zwei Zufalls-Seeds um 6 Punkte (Monte-Carlo-Fehler bei m = 200: ± 3,4 Punkte je Wert)
    assert abs(c["boot"]["clear"] - c2["boot"]["clear"]) <= 0.07


def test_b7_the_stability_rule_in_s3(fz):
    r = fz["selection"]["C_S3_L"]["boot"]
    assert (r["only1_st"], r["only2_st"], r["both_st"], r["none_st"]) == (0.395, 0.455, 0.05, 0.10)
    assert r["att_st"] == 0.85
    s = fz["selection"]["C_S3_L"]["sub"]
    assert (s["only1_st"], s["only2_st"], s["both_st"], s["none_st"], s["att_st"]) == (0.385, 0.425, 0.025, 0.165, 0.825)


def test_b7_bootstrap_underestimates_the_twins_and_overestimates_the_dummies_in_s3(fz):
    c = fz["selection"]["C_S3_L"]
    assert (c["boot"]["p1"], c["boot"]["p2"], c["boot"]["both"]) == (pytest.approx(0.689, abs=0.0006), pytest.approx(0.717, abs=0.0006), pytest.approx(0.407, abs=0.0006))
    assert c["boot"]["p1"] < c["fresh"]["p1"] and c["boot"]["p2"] < c["fresh"]["p2"] and c["boot"]["both"] < c["fresh"]["both"]
    assert c["boot"]["att"] == pytest.approx(0.717, abs=0.0006) and c["sub"]["att"] == pytest.approx(0.724, abs=0.0006) and c["fresh"]["att"] == pytest.approx(0.640, abs=0.0006)


def test_b7_the_mean_share_is_not_biased_but_each_original_is_decided(fz):
    c = fz["selection"]["C_S3_L"]
    assert abs(c["boot"]["share_mean"] - c["fresh"]["share1"]) < 0.03                                  # im Mittel gleich (0,477 gegen 0,495): nicht schief
    assert c["boot"]["share_sd"] > 0.3                                                                   # je Original entschieden (Streuung 0,343 über Originale)


@pytest.mark.parametrize("key,absdiff_boot,clear_boot,clear_sub", [("C_S1_L", 0.259, 0.76, 0.715), ("C_S3_L_rho099", 0.508, 0.595, 0.60), ("G_S3_L", 0.439, 0.71, 0.64), ("G_S1_L", 0.267, 0.819, 0.745)])
def test_b7_other_cells_with_two_equally_strong_twins_show_the_same_trap(fz, key, absdiff_boot, clear_boot, clear_sub):
    c = fz["selection"][key]
    assert c["boot"]["absdiff"] == pytest.approx(absdiff_boot, abs=0.0006) and c["boot"]["clear"] == pytest.approx(clear_boot, abs=0.0006) and c["sub"]["clear"] == pytest.approx(clear_sub, abs=0.0006)
    assert c["boot"]["absdiff"] - abs(c["fresh"]["p1"] - c["fresh"]["p2"]) > 0.2                    # Unterschied je Original gegen Unterschied über frische Stichproben


def test_b8_the_trap_does_not_apply_when_one_twin_is_clearly_stronger(fz):
    """Gegenprobe S2 und Export-Zweig: Bootstrap und frische Stichproben unterscheiden x₁ und x₂ gleich stark (Unterschiede der Beträge höchstens 0,05)."""
    for key, boot_diff, fresh_diff in (("C_S2_L", 0.493, 0.531), ("G_S2_L", 0.488, 0.520), ("E_S2E_L", 0.490, 0.519)):
        c = fz["selection"][key]
        assert c["boot"]["absdiff"] == pytest.approx(boot_diff, abs=0.0006) and abs(c["fresh"]["p1"] - c["fresh"]["p2"]) == pytest.approx(fresh_diff, abs=0.0011)
        assert abs(c["boot"]["absdiff"] - abs(c["fresh"]["p1"] - c["fresh"]["p2"])) < 0.05
    c = fz["selection"]["C_S2_L"]
    assert c["boot"]["share_mean"] == pytest.approx(0.912, abs=0.0006) and c["fresh"]["share1"] == pytest.approx(0.998, abs=0.0006)       # Anteil x₁: 0,912 gegen 0,998
    assert fz["selection"]["E_S2E_L"]["boot"]["share_mean"] == pytest.approx(0.989, abs=0.0006)
    assert (fz["selection"]["C_S2_L"]["boot"]["only1_st"], fz["selection"]["C_S2_L"]["boot"]["only2_st"], fz["selection"]["C_S2_L"]["boot"]["both_st"]) == (0.77, 0.01, 0.22)


def test_b7_live_trap_in_s3_and_no_trap_in_s2_and_the_export_branch():
    """Je Preset-Szenario die erwartete Verdikt-Art aus der Live-Rechnung mit anderem Zufalls-Seed (m = 30 Originale)."""
    s3 = ev.Settings(scen="S3", seed=777, m_orig=30)
    kind3, _ = ev.verdict_q3(ev.fresh_analysis("C", "S3", 100, 0.95, 2.0, 0.09, 1.0, 500, 777), ev.bootstrap_analysis(s3)["summary"], "S3", "Bootstrap", 0.8)
    assert kind3 == "warning"
    for scen, world in (("S2", "C"), ("S2E", "E")):
        s = ev.Settings(scen=scen, seed=777, m_orig=30)
        kind, _ = ev.verdict_q3(ev.fresh_analysis(world, scen, 100, 0.95, 2.0, 0.09, 1.0, 500, 777), ev.bootstrap_analysis(s)["summary"], scen, "Bootstrap", 0.8)
        assert kind == "info", scen


# --- B9: Elastic Net (ℓ = 1/2, α = 0,13) --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key,p1,p2,both,att", [("C_S3_E", 0.938, 0.941, 0.879, 0.735), ("C_S2_E", 1.000, 0.952, 0.952, 0.739), ("E_S2E_E", 1.000, 0.992, 0.992, 0.678)])
def test_b9_elastic_net_takes_both_twins_including_the_copy(fz, key, p1, p2, both, att):
    f = fz["selection"][key]["fresh"]
    assert f["p1"] == pct(p1, 1) and f["p2"] == pct(p2, 1) and f["both"] == pct(both, 1) and f["att"] == pct(att, 1)


def test_b9_copy_chosen_much_more_often_than_by_the_lasso_and_dummies_too(fz):
    s = fz["selection"]
    assert s["C_S2_E"]["fresh"]["p2"] - s["C_S2_L"]["fresh"]["p2"] > 0.4 and s["E_S2E_E"]["fresh"]["p2"] - s["E_S2E_L"]["fresh"]["p2"] > 0.4
    assert s["C_S3_E"]["fresh"]["both"] - s["C_S3_L"]["fresh"]["both"] > 0.3
    assert s["C_S2_E"]["fresh"]["att"] > s["C_S2_L"]["fresh"]["att"] and s["E_S2E_E"]["fresh"]["att"] > s["E_S2E_L"]["fresh"]["att"]
    assert s["C_S2_E"]["boot"]["p2"] == pytest.approx(0.906, abs=0.0006) and s["C_S2_E"]["fresh"]["p2"] == pytest.approx(0.952, abs=0.0006)     # Bootstrap 90,6 % gegen 95,2 % frisch


def test_b9_hand_example_alpha_2():
    G, c = E.hand_data()
    assert E.unique_solution(G, c, F(2)) == [F(3, 10), 0] and E.unique_solution(G, c, F(2), F(1, 2)) == [F(7, 20), F(1, 10)]


# --- B10: Nichtnullen und Rauschgrenze (S3, Lasso, α = 0,09, 2000 frische Stichproben, Zufalls-Seed 1230) -------------------------------------------

def test_b10_the_count_of_non_zeros_depends_on_the_threshold(fz):
    t = {r["thr"]: r for r in fz["nonzero"]["table"]}
    assert t[0.0]["dummies"] == t[1e-8]["dummies"] == pct(0.635, 1) and t[0.0]["nonzeros"] == pytest.approx(6.94, abs=0.005)
    assert t[1e-4]["dummies"] == pytest.approx(0.635, abs=0.0006) and t[0.01]["dummies"] == pct(0.600, 1) and t[0.05]["dummies"] == pct(0.471, 1) and t[0.1]["dummies"] == pct(0.341, 1)
    assert [round(t[k]["nonzeros"], 2) for k in (0.01, 0.05, 0.1)] == [6.69, 5.75, 4.78]
    below = fz["nonzero"]["below"]
    assert below["1e-08"] == 0.0 and below["0.0001"] == pytest.approx(0.0025) and below["0.01"] == pytest.approx(0.227)
    assert fz["nonzero"]["ridge_nonzeros"] == 10.0                                                       # Ridge: kein Koeffizient exakt null


def test_b10_threshold_up_to_1e_4_changes_nothing_for_the_exact_solver_in_this_cell(fz):
    """Gegenprobe zum Satz „die Schwelle ändert die Zahl“: für den exakten Abstieg ändert eine Schwelle bis 1e-4 die Zahl der Nichtnullen in dieser Zelle kaum (6,94 gegen 6,94), erst ab 1e-3."""
    t = {r["thr"]: r for r in fz["nonzero"]["table"]}
    assert round(t[0.0]["nonzeros"], 2) == round(t[1e-8]["nonzeros"], 2) == 6.94 and round(t[1e-4]["nonzeros"], 2) == 6.94 and round(t[1e-3]["nonzeros"], 2) == 6.91


# --- B11: glattes Verfahren (L-BFGS-B auf √(b² + ε), 200 Stichproben) --------------------------------------------------------------------------

def test_b11_smooth_solver_needs_a_threshold_far_above_square_root_of_epsilon(fz):
    by = {blk["eps"]: {r["thr"]: r["same"] for r in blk["rows"]} for blk in fz["nonzero"]["smooth"]}
    assert by[1e-12] == {0.0: 6, 1e-8: 7, 1e-6: 89, 1e-4: 199, 1e-3: 195, 1e-2: 150}
    assert by[1e-8] == {0.0: 6, 1e-8: 6, 1e-6: 7, 1e-4: 89, 1e-3: 196, 1e-2: 150}
    assert 1e-6 == pytest.approx(1.0 * np.sqrt(1e-12)) and 1e-4 == pytest.approx(1.0 * np.sqrt(1e-8))     # Schwelle gleich √ε: nur 89 von 200


# --- B12: Bedingung von Zhao/Yu ----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("idx,q,chosen,r100,r1000,r10000", [(0, 0.5, False, 0.576, 0.196, 0.0), (2, 0.95, False, 0.448, 0.402, 0.212), (3, 0.6, False, 0.59, 0.226, 0.012),
                                                          (5, 1.0, False, 0.584, 0.474, 0.494), (6, 1.2, True, 0.596, 0.726, 0.976)])
def test_b12_population_choice_and_data_selection_of_the_dummy(fz, idx, q, chosen, r100, r1000, r10000):
    r = fz["condition"][idx]
    assert r["q"] == pytest.approx(q, abs=1e-9) and (abs(r["pop_coef"]) > 1e-12) == chosen
    assert (r["rates"]["100"], r["rates"]["1000"], r["rates"]["10000"]) == (r100, r1000, r10000) and r["reps"] == 500


def test_b12_population_choice_equals_q_greater_than_one_in_all_seven_designs(fz):
    rows = fz["condition"]
    assert len(rows) == 7
    for r in rows:
        if abs(r["q"] - 1.0) > 1e-9:
            assert (abs(r["pop_coef"]) > 1e-12) == (r["q"] > 1.0), r["label"]
    assert sum(abs(r["q"] - 1.0) < 1e-9 for r in rows) == 1                                              # ein Entwurf am Rand (nicht gewertet)


def test_b12_probability_falls_with_n_for_q_below_one_and_rises_above_one_in_the_chosen_designs(fz):
    rows = fz["condition"]
    for r in rows:
        if r["q"] < 0.95:
            assert r["rates"]["10000"] < r["rates"]["100"], r["label"]
    last = rows[6]
    assert last["rates"]["100"] < last["rates"]["1000"] < last["rates"]["10000"]


# --- B13: Handbeispiel --------------------------------------------------------------------------------------------------------------------------

def test_b13_hand_example_numbers():
    r = E.hand_report()
    assert (r["vif"], r["var_sum"], r["var_diff"], r["b"], r["sse"]) == (F(25, 9), F(1, 18), F(1, 2), [F(11, 18), F(1, 9)], F(1, 9))
    assert E.hand_report(True)["b"] == [F(1, 9), F(11, 18)] and E.bootstrap_counts() == (108, 36, 0, 24, 88)


def test_b13_stability_bound_example():
    assert E.stability_bound(7, 10, F(4, 5)) == F(49, 6) and float(F(49, 6)) == pytest.approx(8.17, abs=0.005)


# --- B6 (Fortsetzung): ρ = 0,99 gegen ρ = 0,95 ---------------------------------------------------------------------------------------------------

def test_b6_rho_099_has_fewer_chosen_pairs_than_rho_095(fz):
    assert fz["selection"]["C_S3_L_rho099"]["fresh"]["both"] < fz["selection"]["C_S3_L"]["fresh"]["both"] - 0.2
    assert fz["selection"]["C_S3_L_rho099"]["fresh"]["n_one"] > fz["selection"]["C_S3_L"]["fresh"]["n_one"]


# --- Zählaussagen über das ganze Raster -------------------------------------------------------------------------------------------------------

def test_count_over_all_cells_in_which_the_trap_applies(fz):
    """Die Falle (Unterschied je Original mindestens 0,2 über dem über frische Stichproben) gilt in allen 7 Zellen mit zwei gleich starken Zwillingen (S1, S3, ρ = 0,99, Elastic Net) und in keiner der 5 Zellen mit klar stärkerem Merkmal."""
    trap, no_trap = [], []
    for key, cell in fz["selection"].items():
        diff = cell["boot"]["absdiff"] - abs(cell["fresh"]["p1"] - cell["fresh"]["p2"])
        (trap if diff >= 0.15 else no_trap).append(key)
    assert sorted(no_trap) == ["C_S2_E", "C_S2_L", "E_S2E_E", "E_S2E_L", "G_S2_L"]
    assert len(trap) == 7 and all(k.split("_")[1] in ("S1", "S3") for k in trap)


def test_the_selection_cell_for_the_default_preset_equals_the_live_default():
    """Die Live-Rechnung der App mit Standardwerten ist dieselbe Funktion wie die Messzelle (kleiner gerechnet): gleiche Welt, Parameter und Seitenleisten-Seed liefern eine Probe im Band."""
    s = ev.Settings()
    f = ev.fresh_analysis(ev.world_of(s.scen), s.scen, s.n, s.rho, s.sigma, s.alpha, s.l1, 400, 4242)
    cell = FZ.load()["selection"]["C_S3_L"]["fresh"]
    assert abs(f["p1"] - cell["p1"]) < 0.08 and abs(f["both"] - cell["both"]) < 0.09
    assert replace(s, seed=1) == s
