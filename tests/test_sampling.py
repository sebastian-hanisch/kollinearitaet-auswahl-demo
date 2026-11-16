"""Messwelten und Auswahl (`kol_sampling`, `kol_generator`): Form, Bestimmtheit, Unabhängigkeit vom Stapel, Eigenschaften der Welten, Zählungen."""

import numpy as np
import pytest

import kol_algorithm as A
import kol_constants as C
import kol_generator as GEN
import kol_sampling as SA


def test_customer_effects_equal_the_numpy_stream_they_were_frozen_from():
    """Die zwölf Kundeneffekte sind `default_rng(12345).normal(0, 0,30, 12)` der Quelle (Herkunft im Modulkopf); nur dieser Test und die Gegenproben benutzen den numpy-Zufall."""
    assert np.array_equal(np.array(GEN.KUNDEN_EFFEKTE), np.random.default_rng(12345).normal(0.0, 0.30, 12))
    assert GEN.K_KUNDEN == 12


def test_export_dwell_mean_is_linear_in_the_lead_time_and_sd_over_mean_is_constant():
    """Median 0,6 + 0,85 v; Mittel = Faktor · Median (linear in v); sd/Mittel konstant (aus s = 0,25 und den zwölf Kundeneffekten, Zahl des Verhältnisses unten)."""
    v = np.array([1.0, 2.5, 5.5, 8.5, 10.0])
    fac = GEN.export_mean_factor()
    mean = fac * GEN.export_median(v)
    assert np.allclose(np.diff(mean) / np.diff(v), 0.85 * fac, rtol=1e-12)
    u = np.asarray(GEN.KUNDEN_EFFEKTE) * 0.3
    s = 0.25
    ratio = np.sqrt(np.exp(s * s) * np.mean(np.exp(2 * u)) / np.mean(np.exp(u)) ** 2 - 1.0)      # sd/Mittel von exp(u + s z), z standardnormal, u über die Kunden
    assert ratio == pytest.approx(0.279, abs=0.002)


def test_export_dwell_is_lognormal_and_positive():
    z = np.random.default_rng(1).normal(size=20000)
    d = GEN.export_dwell(np.full(20000, 5.0), np.zeros(20000, dtype=int), z)
    assert np.all(d > 0)
    logs = np.log(d)
    assert abs(logs.std() - 0.25) < 0.01
    assert abs(logs.mean() - (np.log(0.6 + 0.85 * 5.0) + GEN.KUNDEN_EFFEKTE[0] * 0.3)) < 0.01


@pytest.mark.parametrize("world,scen", [("C", "S3"), ("G", "S1"), ("E", "S2E")])
def test_draw_is_deterministic_standardized_and_shaped(world, scen):
    X, y = SA.draw(world, scen, 6, 50, 0.95, 2.0, 7, SA.TAG_FRESH)
    assert X.shape == (6, 50, 10) and y.shape == (6, 50)
    assert np.allclose(X.mean(axis=1), 0.0, atol=1e-12) and np.allclose((X ** 2).mean(axis=1), 1.0)
    assert np.allclose(y.mean(axis=1), 0.0, atol=1e-12)
    X2, y2 = SA.draw(world, scen, 6, 50, 0.95, 2.0, 7, SA.TAG_FRESH)
    assert np.array_equal(X, X2) and np.array_equal(y, y2)
    X3, _ = SA.draw(world, scen, 6, 50, 0.95, 2.0, 8, SA.TAG_FRESH)
    assert not np.array_equal(X, X3)


def test_a_sample_does_not_depend_on_how_many_are_drawn_with_it():
    X5, y5 = SA.draw("C", "S3", 5, 40, 0.9, 2.0, 3, SA.TAG_FRESH)
    X9, y9 = SA.draw("C", "S3", 9, 40, 0.9, 2.0, 3, SA.TAG_FRESH)
    assert np.array_equal(X5, X9[:5]) and np.array_equal(y5, y9[:5])


@pytest.mark.parametrize("world", ["C", "G"])
@pytest.mark.parametrize("rho", [0.0, 0.5, 0.95])
def test_the_twin_columns_have_the_requested_population_correlation(world, rho):
    X, _ = SA.draw(world, "S3", 400, 100, rho, 2.0, 11, SA.TAG_FRESH)
    r = (X[:, :, 0] * X[:, :, 1]).mean(axis=1)                  # Stichprobenkorrelation je Stichprobe (Spalten standardisiert)
    assert abs(r.mean() - rho) < 0.01
    assert abs(np.mean((X[:, :, 0] * X[:, :, 2]).mean(axis=1))) < 0.02         # x₃ unabhängig von x₁


def test_container_world_has_the_generator_feature_types():
    X, _ = SA.draw("C", "S3", 1, 4000, 0.95, 2.0, 1, SA.TAG_FRESH)
    n_values = lambda j: len(np.unique(np.round(X[0, :, j], 9)))
    assert [n_values(j) for j in (2, 4, 5)] == [2, 2, 2] and min(n_values(j) for j in (0, 1, 3, 6, 7, 8, 9)) > 100
    for j, p in ((2, 2 / 7), (4, 0.3), (5, 0.3)):                 # Bernoulli(2/7) und Bernoulli(0,3): Anteil der selteneren Ausprägung
        share = float(np.mean(X[0, :, j] > 0))
        assert abs(share - p) < 0.03


def test_signal_and_noise_follow_beta_and_sigma():
    beta = np.array(C.BETAS["S3"])
    X, y = SA.draw("C", "S3", 300, 200, 0.95, 2.0, 5, SA.TAG_FRESH)
    resid = y - np.einsum("knp,p->kn", X, beta)
    assert abs(resid.std() - 2.0) < 0.05
    X1, y1 = SA.draw("C", "S3", 300, 200, 0.95, 1.0, 5, SA.TAG_FRESH)
    assert (y1 - np.einsum("knp,p->kn", X1, beta)).std() == pytest.approx(0.5 * resid.std(), rel=0.02)


def test_export_world_response_depends_on_the_lead_time_only():
    X, y = SA.draw("E", "S2E", 200, 200, 0.95, 2.0, 5, SA.TAG_FRESH)
    corr = np.array([np.mean([np.corrcoef(X[k, :, j], y[k])[0, 1] for k in range(200)]) for j in range(10)])
    assert 0.75 < corr[0] < 0.9 and corr[1] == pytest.approx(0.95 * corr[0], abs=0.03)       # x₂ ist die Kopie: Korrelation 0,95 mal die des Vorlaufs, nicht die Ursache
    assert np.all(np.abs(corr[2:]) < 0.1)


def test_rho_out_of_range_is_rejected():
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            SA.draw("C", "S3", 1, 20, bad, 2.0, 1, SA.TAG_FRESH)


def test_summarize_probabilities_are_consistent():
    sel = SA.fresh_selection("C", "S3", 100, 0.95, 2.0, 0.09, 1.0, 300, 21)
    f = SA.summarize(sel)
    assert f["only1"] + f["only2"] + f["both"] + float(np.mean(~sel[:, 0] & ~sel[:, 1])) == pytest.approx(1.0)     # vier Fälle, Summe 1
    assert f["p1"] == pytest.approx(f["only1"] + f["both"]) and f["p2"] == pytest.approx(f["only2"] + f["both"])
    assert 0.0 <= f["share1"] <= 1.0 and f["n_one"] == int(np.sum(sel[:, 0] ^ sel[:, 1]))
    assert f["nz"] == pytest.approx(sel.sum(axis=1).mean())


def test_fresh_selection_is_a_prefix_and_chunk_independent():
    a = SA.fresh_selection("C", "S3", 60, 0.95, 2.0, 0.09, 1.0, 100, 4, chunk=2000)
    b = SA.fresh_selection("C", "S3", 60, 0.95, 2.0, 0.09, 1.0, 200, 4, chunk=2000)
    assert np.array_equal(a, b[:100])


def test_resample_indices_ranges_and_without_replacement():
    boot = SA.resample_indices(3, 20, 40, "boot", 5)
    assert boot.shape == (3, 20, 40) and boot.min() >= 0 and boot.max() < 40
    assert any(len(set(row)) < 40 for row in boot[0])                              # Zurücklegen: Wiederholungen kommen vor
    sub = SA.resample_indices(3, 20, 40, "sub", 5)
    assert sub.shape == (3, 20, 20) and all(len(set(row)) == 20 for row in sub.reshape(-1, 20))      # ohne Zurücklegen: lauter verschiedene Zeilen
    assert np.array_equal(SA.resample_indices(3, 20, 40, "boot", 5), boot)


def test_bootstrap_expected_share_of_distinct_rows():
    boot = SA.resample_indices(2, 200, 100, "boot", 9).reshape(-1, 100)
    distinct = np.mean([len(set(r)) for r in boot]) / 100
    assert abs(distinct - (1 - (1 - 1 / 100) ** 100)) < 0.005                      # etwa 63,4 % verschiedene Zeilen


def test_original_selection_with_a_single_original_and_per_original_statistics():
    X, y = SA.draw("C", "S3", 1, 60, 0.95, 2.0, 3, SA.TAG_ORIG)
    sel = SA.original_selections(X, y, 0.09, 1.0, "boot", 30, 4)
    assert sel.shape == (1, 30, 10)
    po = SA.per_original(sel, 0.8)
    s = SA.original_summary(po, 0.8)
    assert 0.0 <= s["p1"] <= 1.0 and s["n_share"] in (0, 1)
    assert po["stable"].shape == (1, 10)
    assert s["only1_st"] + s["only2_st"] + s["both_st"] + s["none_st"] == pytest.approx(1.0)


def test_selection_cell_shape_and_reproducibility():
    a = SA.selection_cell("C", "S3", 60, 0.95, 2.0, 0.09, 1.0, 4, 20, 40, 3)
    b = SA.selection_cell("C", "S3", 60, 0.95, 2.0, 0.09, 1.0, 4, 20, 40, 3)
    assert a == b and set(a) == {"fresh", "boot", "sub"}


def test_select_batch_equals_the_pipeline_by_hand():
    X, y = SA.draw("C", "S3", 3, 50, 0.9, 2.0, 2, SA.TAG_FRESH)
    for k in range(3):
        G, c = A.gram_of(X[k], y[k])
        b, _ = A.cd_batch(G, c, 0.09, 1.0)
        assert np.array_equal(np.abs(b) > C.NZ_THRESHOLD, SA.select_batch(X[k : k + 1], y[k : k + 1], 0.09, 1.0)[0])
