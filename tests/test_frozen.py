"""Die vorgerechnete Messreihe (`data/kol_frozen.json`): Aufbau, innere Widerspruchsfreiheit und Neu-Rechnung.

* Aufbau: jeder Block ist da, mit seinen Zähleinheiten und Zufalls-Seeds.
* Probe: je Auswahlzelle wird die kleine Probe (200 frische Stichproben, 10 Originalstichproben, gleiche Zufalls-Seeds) neu gerechnet und mit der Datei verglichen
  (Band: Gleitkommawerte relativ 1e-9, Zählungen exakt; die Rechnung benutzt nur exakt gerundete Grundoperationen, siehe `kol_stream`).
* KOL_FULL=1: die ganze Messreihe wird neu gerechnet und verglichen (Laufzeit wie beim Erzeugen, einige Minuten)."""

import importlib.util
import os
from pathlib import Path

import pytest

import kol_frozen as FZ

TOOL = Path(__file__).resolve().parent.parent / "tools" / "kol_make_frozen.py"


def _tool():
    spec = importlib.util.spec_from_file_location("kol_make_frozen", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def frozen():
    return FZ.load()


def test_every_block_is_present_with_its_size(frozen):
    assert len(frozen["sumdiff"]) == 7 and all(r["reps"] == 100000 for r in frozen["sumdiff"])
    assert len(frozen["tests"]) == 10 and all(r["reps"] == 20000 for r in frozen["tests"])
    assert len(frozen["coverage"]) == 4 and len(frozen["vif_spread"]) == 6 and all(r["designs"] == 4000 for r in frozen["vif_spread"])
    assert set(frozen["selection"]) == set(FZ.CELLS) and len(frozen["selection"]) == 12
    assert len(frozen["condition"]) == 7 and all(set(r["rates"]) == {"100", "1000", "10000"} and r["reps"] == 500 for r in frozen["condition"])
    assert frozen["nonzero"]["n_fresh"] == 2000 and len(frozen["nonzero"]["table"]) == 7 and len(frozen["nonzero"]["smooth"]) == 2
    assert frozen["natural_vif_export"]["n"] == 60000


def test_selection_cells_state_their_counting_units_and_seeds(frozen):
    for key, cell in frozen["selection"].items():
        p = cell["params"]
        assert (p["m_orig"], p["B"], p["n_fresh"], p["n"], p["sigma"]) == (200, 100, 2000, 100, 2.0), key
        assert p["seed"] == FZ.CELLS[key][5] and (p["world"], p["scen"], p["l1"], p["alpha"], p["rho"]) == FZ.CELLS[key][:5]
        assert cell["probe"]["params"]["m_orig"] == FZ.PROBE_M and cell["probe"]["params"]["n_fresh"] == FZ.PROBE_FRESH


def test_probabilities_are_consistent_inside_every_cell(frozen):
    for key, cell in frozen["selection"].items():
        f = cell["fresh"]
        assert f["only1"] + f["only2"] + f["both"] <= 1.0 + 1e-12 and f["p1"] == pytest.approx(f["only1"] + f["both"]) and f["p2"] == pytest.approx(f["only2"] + f["both"]), key
        assert f["n_one"] == pytest.approx(2000 * (f["only1"] + f["only2"])), key
        for scheme in ("boot", "sub"):
            r = cell[scheme]
            assert 0.0 <= r["clear"] <= 1.0 and 0 <= r["n_share"] <= 200 and 0.0 <= r["absdiff"] <= 1.0, (key, scheme)
            assert r["only1_st"] + r["only2_st"] + r["both_st"] + r["none_st"] == pytest.approx(1.0), (key, scheme)
            assert r["att"] >= 0.0 and r["nz"] <= 10.0


def test_frozen_values_agree_with_closed_forms_in_a_band(frozen):
    for r in frozen["sumdiff"]:
        assert r["n_var_b1"] == pytest.approx(r["formula_single"], rel=0.02 if r["rho"] < 0.999 else 0.03)       # Monte-Carlo-Fehler bei 100 000 Wiederholungen etwa 0,5 %
        assert r["n_var_sum"] == pytest.approx(r["formula_sum"], rel=0.03)
        assert r["n_var_diff"] == pytest.approx(r["formula_diff"], rel=0.03)
        assert r["corr"] == pytest.approx(-r["rho"], abs=0.01)
    for r in frozen["tests"]:
        assert abs(r["p_single"] - r["p_single_exact"]) < 0.015 and abs(r["p_F"] - r["p_F_exact"]) < 0.015     # Band 0,015 (Monte-Carlo-Fehler höchstens 0,0035)
    for r in frozen["coverage"]:
        assert abs(r["cover"] - 0.95) < 0.01


def test_probe_of_every_selection_cell_is_reproduced():
    tool = _tool()
    old = FZ.load()["selection"]
    bad = []
    for key in FZ.CELLS:
        bad += tool.compare(FZ.run_cell(key, FZ.PROBE_M, FZ.PROBE_FRESH), old[key]["probe"], key)
    assert not bad, bad[:10]


@pytest.mark.skipif(os.environ.get("KOL_FULL") != "1", reason="ganze Messreihe nur mit KOL_FULL=1 (einige Minuten)")
def test_the_whole_measurement_series_is_reproduced():
    tool = _tool()
    bad = tool.compare(FZ.build(), FZ.load())
    assert not bad, bad[:10]
