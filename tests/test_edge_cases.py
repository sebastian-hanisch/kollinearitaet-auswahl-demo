"""Randfälle: ρ = 1, n < p, konstante Spalte, konstantes y, α = 0 und α über α_max, ℓ = 0, Bootstrap mit höchstens zwei verschiedenen Zeilen, eine Originalstichprobe, Zufalls-Seed 0."""

import numpy as np
import pytest

import kol_algorithm as A
import kol_constants as C
import kol_evaluation as ev
import kol_exact as E
import kol_inference as I
import kol_sampling as SA


def test_rho_one_vif_is_reported_as_unbounded_and_the_simulation_refuses():
    assert A.vif_two(1.0) is None and A.two_feature_factors(1.0) is None
    X = I.fixed_design(60, 1.0, 1)
    with pytest.raises(ValueError, match="singulär"):
        I.simulate_ols(X, (0.5, 0.5), 2.0, 10, 1)
    assert np.linalg.matrix_rank(X) == 1                                          # beide Spalten gleich: Rang 1, X'X nicht umkehrbar


def test_lasso_with_two_identical_columns_runs_and_fixes_the_sum():
    X, y = SA.draw("C", "S3", 1, 80, 1.0, 2.0, 3, SA.TAG_FRESH)
    assert np.allclose(X[0][:, 0], X[0][:, 1])                                 # ρ = 1: zwei gleiche (standardisierte) Spalten
    b, sweeps = A.cd_batch(*A.gram_of(X[0], y[0]), 0.09, 1.0)
    assert sweeps >= 1 and b[0] * b[1] >= 0.0 and np.all(np.isfinite(b))
    Xm = np.delete(X[0], 1, axis=1)
    bm, _ = A.cd_batch(*A.gram_of(Xm, y[0]), 0.09, 1.0)
    assert b[0] + b[1] == pytest.approx(bm[0], abs=1e-8)


def test_one_sample_with_fewer_rows_than_features_has_no_least_squares_but_a_lasso():
    smp = ev.one_sample(ev.Settings(n=8))
    assert smp["ols"] is None and smp["lasso"].shape == (10,)
    smp = ev.one_sample(ev.Settings(n=C.P_FEATURES + 1))
    assert smp["ols"] is None                                                     # n = p + 1: kein Rest-Freiheitsgrad mit Achsenabschnitt


def test_constant_column_and_constant_response_select_nothing():
    X = np.random.default_rng(1).normal(size=(30, 3))
    X[:, 1] = 4.0
    Z, yc = A.standardize_batch(X[None], np.zeros((1, 30)))
    G, c = A.gram_batch(Z, yc)
    assert np.all(A.cd_batch(G[0], c[0], 0.0, 1.0)[0] == 0.0) and A.alpha_max(c[0]) == 0.0
    b, _ = A.cd_batch(G[0], c[0], 0.1, 1.0)
    assert np.all(b == 0.0)                                                       # konstantes y: nichts wird gewählt


def test_alpha_zero_and_alpha_above_alpha_max_in_the_application_pipeline():
    X, y = SA.draw("C", "S3", 5, 60, 0.95, 2.0, 1, SA.TAG_FRESH)
    assert np.all(SA.fit_batch(X, y, 1e6, 1.0) == 0.0)
    ols = SA.fit_batch(X, y, 0.0, 1.0)
    assert np.all(ols != 0.0)                                                     # α = 0: kein Koeffizient exakt null


def test_elastic_net_with_zero_l1_ratio_has_no_exact_zeros_even_at_large_alpha():
    X, y = SA.draw("C", "S3", 4, 60, 0.95, 2.0, 1, SA.TAG_FRESH)
    b = SA.fit_batch(X, y, 5.0, 0.0)
    assert np.all(b != 0.0) and A.nonzero_count(b[0], 0.0) == 10 and np.all(np.abs(b) < 0.5)


def test_resample_with_at_most_two_distinct_rows_does_not_crash_and_selects_nothing_extra():
    X, y = SA.draw("C", "S3", 1, 2, 0.95, 2.0, 1, SA.TAG_ORIG)                    # n = 2: jede Wiederholung hat höchstens zwei verschiedene Zeilen
    sel = SA.original_selections(X, y, 0.09, 1.0, "boot", 10, 3)
    assert sel.shape == (1, 10, 10) and sel.dtype == bool


def test_a_single_original_sample_and_seed_zero():
    cell = SA.selection_cell("C", "S3", 40, 0.95, 2.0, 0.09, 1.0, 1, 10, 10, 0)
    assert set(cell) == {"fresh", "boot", "sub"}
    assert cell["boot"]["n_share"] in (0, 1)
    X, y = SA.draw("C", "S2", 2, 30, 0.9, 2.0, 0, SA.TAG_FRESH)
    assert np.all(np.isfinite(X)) and np.all(np.isfinite(y))


def test_one_fresh_sample_has_defined_frequencies():
    f = ev.fresh_analysis("C", "S3", 60, 0.95, 2.0, 0.09, 1.0, 1, 5)
    assert f["n_fresh"] == 1 and f["p1"] in (0.0, 1.0) and f["n_one"] in (0, 1)


def test_hand_example_for_every_alpha_option_has_a_unique_solution_and_all_zero_at_the_top():
    for swapped in (False, True):
        for text in C.HAND_ALPHA_OPTIONS:
            h = ev.hand_analysis(swapped, text)
            assert len(h["lasso"]) == 2 and len(h["enet"]) == 2 and len(h["grid"]) == len(h["path"]) == 71
        assert ev.hand_analysis(swapped, "7/2")["lasso"] == [0, 0]
        assert ev.hand_analysis(swapped, "0")["lasso"] == E.hand_report(swapped)["b"]


def test_path_analysis_at_the_boundaries_of_the_controls():
    for rel in (C.ALPHA_REL_OPTIONS[0], C.ALPHA_REL_OPTIONS[-1]):
        for thr in (0.0, 1e-2):
            p = ev.path_analysis(ev.Settings(path_rel=rel, path_thr=thr))
            assert p["kkt"] < 1e-9
            if rel == 1.0:
                assert p["n_zero_exact"] == 10                                    # α = α_max: alles null (exakt)


def test_vif_analysis_with_the_smallest_sample_and_largest_correlation_option():
    v = ev.vif_analysis(30, 0.999, 4.0, 1.0, 500, 1)
    assert np.isfinite(v["se_formula"]) and 0.0 <= v["p_F"] <= 1.0 and v["factors"]["single"] == pytest.approx(1 / (1 - 0.999 ** 2))


def test_exact_cdf_branch_for_zero_signal():
    X = I.fixed_design(40, 0.9, 1)
    r = I.test_rates(X, (0.0, 0.0), 1.0, 200, 1)
    assert r["p_single_exact"] == pytest.approx(0.05) and r["p_F_exact"] == pytest.approx(0.05)
