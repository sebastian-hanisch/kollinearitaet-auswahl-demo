"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Verschiedenheit; Permalink (Einrasten, Begrenzen)."""

import kol_constants as C
import kol_evaluation as ev
import kol_presets as P


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid_options_of_their_controls():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[P.PRESET_KEYS[key]]
            if spec.options is not None:
                assert value in spec.options, (name, key, value)
            else:
                assert spec.lo <= value <= spec.hi, (name, key, value)


def test_default_preset_equals_the_default_settings_and_every_preset_has_its_own_seed():
    first = next(iter(C.PRESETS))
    assert first == "Von Hand (vier Container)"
    s = ev.Settings(**{k: v for k, v in C.PRESETS[first].items() if k in ev.Settings.__dataclass_fields__})
    assert s == ev.Settings()
    seeds = [p["seed"] for p in C.PRESETS.values()]
    assert len(set(seeds)) == len(seeds)                                  # sonst wären zwei Presets nicht zu unterscheiden (Hervorhebung)


def test_settings_defaults_equal_the_constants_and_the_specs():
    d = ev.Settings()
    for key, value in C.DEFAULTS.items():
        assert getattr(d, key) == value, key
        assert P.SETTING_SPECS[P.PRESET_KEYS[key]].default == value
    assert set(P.PRESET_KEYS) == set(C.DEFAULTS)
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_defaults_are_inside_their_options():
    for state_key, spec in P.SETTING_SPECS.items():
        if spec.options is not None:
            assert spec.default in spec.options, state_key
        elif spec.lo is not None:
            assert spec.lo <= spec.default <= spec.hi, state_key


def test_presets_show_the_scenarios_named_in_their_titles():
    assert C.PRESETS["Zwei Angaben, gleich schwach (S3)"]["scen"] == "S3"
    assert C.PRESETS["Zwei Angaben, gleich stark (S1)"]["scen"] == "S1"
    assert C.PRESETS["Nur der Vorlauf wirkt (S2, echter Generator)"]["scen"] == "S2E"
    assert C.PRESETS["Elastic Net: Preis des Gruppeneffekts"]["l1"] == 0.5 and C.PRESETS["Elastic Net: Preis des Gruppeneffekts"]["scen"] == "S2E"
    assert C.PRESETS["Starke Kollinearität (ρ = 0,99)"]["rho"] == 0.99


def test_snap_rounds_to_options_and_steps_and_clamps():
    rho = P.SETTING_SPECS["rho_select"]
    assert P._snap(rho, 0.93) == 0.95 and P._snap(rho, 5.0) == 0.999 and P._snap(rho, -1.0) == 0.0
    scen = P.SETTING_SPECS["scen_select"]
    assert P._snap(scen, "S1") == "S1" and P._snap(scen, "unbekannt") == scen.default
    reps = P.SETTING_SPECS["vif_reps_slider"]
    assert P._snap(reps, 1234) == 1000 and P._snap(reps, 99999) == 5000 and P._snap(reps, 1) == 500 and isinstance(P._snap(reps, 1700), int)
    seed = P.SETTING_SPECS["seed_input"]
    assert P._snap(seed, -5) == 0 and P._snap(seed, 10 ** 9) == C.SEED_MAX
    assert P._to_bool("1") and P._to_bool("true") and not P._to_bool("0") and not P._to_bool("nein")


def test_scenario_labels_and_betas_are_consistent():
    assert set(C.SCENARIO_LABELS) == set(C.SCENARIOS) == set(C.WORLD_OF)
    assert set(C.BETAS) == {"S1", "S2", "S3"} and all(len(b) == C.P_FEATURES for b in C.BETAS.values())
    assert C.BETAS["S3"][:3] == (0.5, 0.5, 1.0) and C.BETAS["S1"][:3] == (1.0, 1.0, 1.0) and C.BETAS["S2"][:3] == (2.0, 0.0, 0.0)
    assert all(v == 0.0 for b in C.BETAS.values() for v in b[3:])
