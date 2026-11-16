"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# --- Zusätze dieser Demo (der Anfang der Datei bleibt die Vorlage) ------------------------------------------------------------------------------

def test_every_select_slider_at_its_first_and_last_option_does_not_raise():
    """Alle Auswahlregler (Seitenleiste und Experimente): jeder an seinen beiden Rändern, mit dem Schlüssel angesprochen."""
    import kol_presets as P

    at0 = _fresh_app()
    keys = [w.key for w in list(at0.sidebar.select_slider) + list(at0.main.select_slider)]
    assert set(keys) == {k for k, spec in P.SETTING_SPECS.items() if spec.options is not None} - {"scen_select", "scheme_select"}, keys      # die beiden Ausnahmen sind Auswahlfelder
    for key in keys:
        options = P.SETTING_SPECS[key].options
        for value in (options[0], options[-1]):
            at = _fresh_app()
            (at.sidebar if key in {w.key for w in at.sidebar.select_slider} else at.main).select_slider(key=key).set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (key, value, [str(e) for e in at.exception])


def test_every_scenario_and_scheme_and_the_swap_toggle_does_not_raise():
    import kol_constants as C

    for scen in C.SCENARIOS:
        at = _fresh_app()
        at.sidebar.selectbox(key="scen_select").select(scen).run(timeout=TIMEOUT)
        assert not at.exception, (scen, [str(e) for e in at.exception])
    for scheme in C.SCHEME_OPTIONS:
        at = _fresh_app()
        at.main.selectbox(key="scheme_select").select(scheme).run(timeout=TIMEOUT)
        assert not at.exception, (scheme, [str(e) for e in at.exception])
    at = _fresh_app()
    at.main.toggle(key="hand_swap_toggle").set_value(True).run(timeout=TIMEOUT)
    assert not at.exception


def test_extreme_combination_smallest_sample_largest_correlation_every_scenario():
    import kol_constants as C

    for scen in C.SCENARIOS:
        at = _fresh_app()
        at.sidebar.select_slider(key="rho_select").set_value(0.999)
        at.sidebar.select_slider(key="n_select").set_value(30)
        at.sidebar.select_slider(key="l1_select").set_value(0.1)
        at.sidebar.selectbox(key="scen_select").select(scen).run(timeout=TIMEOUT)
        assert not at.exception, (scen, [str(e) for e in at.exception])
        assert len(at.metric) >= 9 and len(list(at.warning) + list(at.info)) >= 4


def test_permalink_sets_and_snaps_the_controls():
    at = AppTest.from_file(APP_PATH)
    at.query_params["rho"] = "0.93"            # rastet auf die Stufe 0,95 ein
    at.query_params["scen"] = "S2"
    at.query_params["seed"] = "-4"             # wird auf 0 begrenzt
    at.query_params["vr"] = "1234"             # rastet auf die Schrittweite 500 ein
    at.run(timeout=TIMEOUT)
    assert not at.exception
    assert at.sidebar.select_slider(key="rho_select").value == 0.95
    assert at.sidebar.selectbox(key="scen_select").value == "S2"
    assert at.sidebar.number_input(key="seed_input").value == 0
    assert at.main.slider(key="vif_reps_slider").value == 1000


def test_the_first_load_shows_three_core_question_verdicts_and_the_expected_sections():
    at = _fresh_app()
    texts = [m.value for m in list(at.warning) + list(at.info) + list(at.success)]
    assert any("Standardfehler von b₁" in t for t in texts) and any("Münzwurf" in t for t in texts) and any("Bootstrap" in t for t in texts)
    subheaders = [h.value for h in at.subheader]
    assert [h for h in subheaders if h.startswith("🔬 Experiment")] == [
        "🔬 Experiment 1: Von Hand, vier Container",
        "🔬 Experiment 2: Wie viel wissen die Daten? Varianzaufblähung und die Illusion „kein Effekt“",
        "🔬 Experiment 3: Der Lasso-Pfad – wer tritt zuerst ein?",
        "🔬 Experiment 4: Welches von beiden? Auswahl über frische Stichproben",
        "🔬 Experiment 5: Die Falle – der Bootstrap derselben Stichprobe",
        "🔬 Experiment 6: Elastic Net – der Gruppeneffekt und sein Preis",
    ]
    assert "🚧 Wo die Annahmen enden" in subheaders
    assert any(e.label == "📐 Mathematische Formulierung" for e in at.expander)
