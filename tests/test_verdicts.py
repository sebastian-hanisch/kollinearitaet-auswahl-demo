"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zeilen, nicht Zufalls-Seeds. Damit prüft der Test die Grenzen (>= 0,2, 0,4 bis 0,6, 0,15) und jede genannte Zahl direkt.
Am Ende: je Preset die erwartete Verdikt-Zeile aus einer Live-Rechnung (ein Preset ohne Effekt, S2 beim Bootstrap, meldet **keinen** Unterschied)."""

from fractions import Fraction as F

import pytest

import kol_constants as C
import kol_evaluation as ev


def _v(p_single=0.122, p_F_no_single=0.76, **kw):
    base = dict(rho=0.95, n=100, sigma=2.0, b=0.5, reps=2000, se_formula=0.641, se_orth=0.2, p_single=p_single, p_single_exact=0.121, p_both_single=0.007, p_F=0.995, p_F_exact=0.995,
                p_F_no_single=p_F_no_single)
    base.update(kw)
    return base


def test_q1_warns_when_the_f_test_rejects_without_any_single_test_in_20_percent_or_more():
    kind, text = ev.verdict_q1(_v())
    assert kind == "warning"
    assert "ρ = 0,95 (n = 100, β = (0,5; 0,5), σ = 2,0, 2000 Wiederholungen)" in text
    assert "Standardfehler von b₁ 0,64 statt 0,20" in text
    assert "ein Einzeltest verwirft in 12,2 % (exakt 12,1 %), beide Einzeltests in 0,7 %" in text and "gemeinsame F-Test in 99,5 % (exakt 99,5 %)" in text
    assert "**76,0 %**" in text and "ohne dass ein einziger Einzeltest verwirft" in text and "Intervalle bleiben gültig" in text


def test_q1_threshold_is_inclusive_at_20_percent():
    assert ev.verdict_q1(_v(p_F_no_single=0.20))[0] == "warning"
    kind, text = ev.verdict_q1(_v(p_F_no_single=0.199))
    assert kind == "info" and "19,9 %" in text and "klein" in text


def _f(p1=0.745, p2=0.758, both=0.502, only1=0.2430, only2=0.2570, share1=0.486, n_fresh=2000):
    return dict(p1=p1, p2=p2, both=both, only1=only1, only2=only2, share1=share1, n_fresh=n_fresh, att=0.638, nz=7.0, n_one=1000)


def test_q2_coin_flip_only_when_share_between_40_and_60_and_twins_equally_often_chosen():
    kind, text = ev.verdict_q2(_f(), "S3", 100, 0.95, 2.0, 0.09, 1.0)
    assert kind == "warning" and "wie ein Münzwurf" in text
    assert "Lasso, Szenario S3 (n = 100, ρ = 0,95, σ = 2,0, α = 0,090 fest, 2000 frische Stichproben)" in text
    assert "x₁ in 74,5 %, x₂ in 75,8 %, beide in 50,2 %" in text and "in 50,0 % der Stichproben), ist es x₁ in 48,6 %" in text
    assert ev.verdict_q2(_f(share1=0.40), "S3", 100, 0.95, 2.0, 0.09, 1.0)[0] == "warning"       # Rand 0,4 inklusive
    assert ev.verdict_q2(_f(share1=0.60), "S3", 100, 0.95, 2.0, 0.09, 1.0)[0] == "warning"       # Rand 0,6 inklusive
    assert ev.verdict_q2(_f(share1=0.61), "S3", 100, 0.95, 2.0, 0.09, 1.0)[0] == "info"
    kind, text = ev.verdict_q2(_f(p1=0.80, p2=0.69, share1=0.5), "S3", 100, 0.95, 2.0, 0.09, 1.0)  # Unterschied der Häufigkeiten 0,11: kein Münzwurf
    assert kind == "info" and "Münzwurf" not in text


def test_q2_rare_exactly_one_and_preferred_twin_and_elastic_net_label():
    kind, text = ev.verdict_q2(_f(p1=0.847, p2=0.847, both=0.847 - 0.0, only1=0.07, only2=0.08, share1=0.467), "S1", 100, 0.95, 2.0, 0.09, 1.0)
    assert kind == "info" and "nur in 15,0 %" in text and "meist werden beide oder keines gewählt" in text
    kind, text = ev.verdict_q2(_f(p1=0.996, p2=0.442, both=0.438, only1=0.558, only2=0.004, share1=0.993), "S2", 100, 0.95, 2.0, 0.09, 1.0)
    assert kind == "info" and "x₁ wird bevorzugt" in text and "99,3 %" in text
    kind, text = ev.verdict_q2(_f(p1=0.44, p2=0.99, both=0.43, only1=0.01, only2=0.56, share1=0.02), "S2", 100, 0.95, 2.0, 0.09, 0.5)
    assert "x₂ wird bevorzugt" in text and "Elastic Net (ℓ = 0,50)" in text
    assert ev.verdict_q2(_f(only1=0.249, only2=0.0), "S3", 100, 0.95, 2.0, 0.09, 1.0)[1].count("meist werden beide") == 1     # genau eines selten: Rand 0,25 exklusiv
    assert "meist werden beide" not in ev.verdict_q2(_f(only1=0.25, only2=0.0, share1=0.5), "S3", 100, 0.95, 2.0, 0.09, 1.0)[1]


def _b(absdiff=0.405, clear=0.61, n_share=200, only1_st=0.355, only2_st=0.425, both_st=0.05):
    return dict(absdiff=absdiff, clear=clear, n_share=n_share, only1_st=only1_st, only2_st=only2_st, both_st=both_st)


def test_q3_trap_when_bootstrap_difference_exceeds_fresh_by_015_inclusive():
    fresh = dict(p1=0.745, p2=0.758, share1=0.486)
    kind, text = ev.verdict_q3(fresh, _b(), "S3", "Bootstrap (Größe n)", 0.8)
    assert kind == "warning"
    assert "|P(x₁) − P(x₂)| = 0,013" in text and "im Mittel um 0,405" in text
    assert "In 61,0 % der 200 Originalstichproben" in text and "über frische Stichproben bei 48,6 %" in text
    assert "genau x₁ in 35,5 %, genau x₂ in 42,5 %, beide in 5,0 %" in text and "Häufigkeit ≥ 0,8 gilt als stabil" in text
    assert ev.verdict_q3(fresh, _b(absdiff=0.013 + 0.15), "S3", "Bootstrap (Größe n)", 0.8)[0] == "warning"      # Rand inklusive (Differenz auf Rundung genau)
    assert ev.verdict_q3(fresh, _b(absdiff=0.013 + 0.149), "S3", "Bootstrap (Größe n)", 0.8)[0] == "info"


def test_q3_reports_no_difference_when_one_twin_is_clearly_stronger():
    fresh = dict(p1=0.996, p2=0.442, share1=0.993)
    kind, text = ev.verdict_q3(fresh, _b(absdiff=0.513, clear=0.885, only1_st=0.79, only2_st=0.04, both_st=0.17), "S2", "Teilstichproben (Größe n/2)", 0.8)
    assert kind == "info" and "stimmen hier überein" in text and "Teilstichproben (Größe n/2), Szenario S2" in text and "Falle" not in text


def test_hand_verdict_names_every_exact_number():
    kind, text = ev.verdict_hand(ev.hand_analysis(False, "2"))
    assert kind == "info"
    for piece in ("VIF = 25/9", "Var(b₁ + b₂) = 1/18·σ² (orthogonal 1/10)", "Var(b₁ − b₂) = 1/2·σ²", "b = (11/18; 1/9), Summe 13/18", "Lasso bei α = 2: (3/10; 0)", "Elastic Net (ℓ = 1/2): (7/20; 1/10)", "zuerst x1 (α = 7/2)"):
        assert piece in text, piece
    assert "zuerst x2 (α = 7/2)" in ev.verdict_hand(ev.hand_analysis(True, "2"))[1]
    assert "b = (1/9; 11/18), Summe 13/18" in ev.verdict_hand(ev.hand_analysis(True, "0"))[1]


def _p(counts, **kw):
    base = dict(alpha_sel=0.1726, alpha_max=1.726, counts=counts, order=[0, 2, 1, 4, 5, 6], kkt=9.2e-13)
    base.update(kw)
    return base


def test_path_verdict_warns_when_the_count_depends_on_the_threshold():
    s = ev.Settings(path_rel=0.1, path_thr=1e-8)
    kind, text = ev.verdict_path(_p({0.0: 8, 1e-8: 8, 1e-4: 7, 1e-2: 5}), s)
    assert kind == "warning" and "0: 8 / 1e-8: 8 / 1e-4: 7 / 1e-2: 5" in text and "Schwelle 1e-8" in text
    assert "erste fünf: x1, x3, x2, x5, x6" in text and "α = 0,1726 (0,100 von α_max = 1,726)" in text and "9.2e-13" in text
    assert "keine Messung" in text
    kind, text = ev.verdict_path(_p({0.0: 5, 1e-8: 5, 1e-4: 5, 1e-2: 5}), ev.Settings(path_thr=0.0))
    assert kind == "info" and "ändert die Schwelle die Zahl nicht" in text and "(Schwelle 0)" in text


def test_threshold_text():
    assert [ev.thr_text(t) for t in C.THRESHOLD_OPTIONS] == ["0", "1e-8", "1e-4", "1e-2"]


def test_elastic_verdict_warns_about_the_copy_when_only_x1_acts():
    fl = dict(p1=1.0, p2=0.442, both=0.442, att=0.638, n_fresh=500)
    fe = dict(p1=1.0, p2=0.966, both=0.966, att=0.737, n_fresh=500)
    kind, text = ev.verdict_elastic(fl, fe, "S2", 0.5)
    assert kind == "warning" and "**das Elastic Net nimmt die Kopie x₂ mit** (96,6 % gegen 44,2 % beim Lasso)" in text
    assert "63,8 % (Lasso) gegen 73,7 % (Elastic Net)" in text and "Elastic Net (ℓ = 0,50)" in text and "500 frische Stichproben" in text
    assert ev.verdict_elastic(fl, fe, "S2E", 0.5)[0] == "warning"
    kind, text = ev.verdict_elastic(dict(fl, p2=0.75, both=0.5), dict(fe, p2=0.94), "S3", 0.25)
    assert kind == "info" and "öfter gemeinsam" in text


def test_fraction_text():
    assert ev.frac_text(F(3, 10)) == "3/10" and ev.frac_text(F(4, 2)) == "2" and ev.frac_text(0) == "0" and ev.frac_text(F(-5, 18)) == "-5/18"


# --- je Preset die erwartete Verdikt-Zeile aus der Live-Rechnung --------------------------------------------------------------------------------

@pytest.mark.parametrize("name,expect_trap", [("Zwei Angaben, gleich schwach (S3)", True), ("Zwei Angaben, gleich stark (S1)", True), ("Starke Kollinearität (ρ = 0,99)", True),
                                              ("Nur der Vorlauf wirkt (S2, echter Generator)", False)])
def test_each_preset_reports_the_expected_bootstrap_verdict(name, expect_trap):
    from dataclasses import replace

    preset = C.PRESETS[name]
    s = ev.Settings(**{k: v for k, v in preset.items() if k in ev.Settings.__dataclass_fields__})
    f = ev.fresh_analysis(ev.world_of(s.scen), s.scen, s.n, s.rho, s.sigma, s.alpha, s.l1, 300, s.seed)
    b = ev.bootstrap_analysis(replace(s, m_orig=20))
    kind, _ = ev.verdict_q3(f, b["summary"], s.scen, C.SCHEME_OPTIONS[0], s.pi)
    assert (kind == "warning") == expect_trap, (name, kind)


def test_the_elastic_net_preset_names_the_copy_and_the_hand_preset_has_no_effect_claim():
    s = ev.Settings(scen="S2E", l1=0.5, seed=34)
    fl = ev.fresh_analysis("E", "S2E", s.n, s.rho, s.sigma, s.alpha, 1.0, 300, s.seed)
    fe = ev.fresh_analysis("E", "S2E", s.n, s.rho, s.sigma, s.alpha, 0.5, 300, s.seed)
    kind, text = ev.verdict_elastic(fl, fe, "S2E", 0.5)
    assert kind == "warning" and fe["p2"] > fl["p2"] + 0.3 and "nimmt die Kopie x₂ mit" in text
