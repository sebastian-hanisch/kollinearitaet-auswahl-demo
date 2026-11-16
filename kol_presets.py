"""Permalink (Begrenzen, Einrasten), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben) und Zufalls-Seed-Button."""

import math
import random
from dataclasses import dataclass

import streamlit as st

import kol_constants as C


def _to_bool(text) -> bool:
    return str(text).strip().lower() in ("1", "true", "ja", "yes")


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: object
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None     # Auswahlregler: erlaubte Werte (Permalink rastet darauf ein)
    step: float | None = None        # Schieberegler: Schrittweite (Permalink rastet darauf ein)


D = C.DEFAULTS
SETTING_SPECS = {
    "scen_select": SettingSpec("scen", str, D["scen"], options=C.SCENARIOS),
    "rho_select": SettingSpec("rho", float, D["rho"], min(C.RHO_OPTIONS), max(C.RHO_OPTIONS), options=C.RHO_OPTIONS),
    "n_select": SettingSpec("n", int, D["n"], min(C.N_OPTIONS), max(C.N_OPTIONS), options=C.N_OPTIONS),
    "sigma_select": SettingSpec("sigma", float, D["sigma"], min(C.SIGMA_OPTIONS), max(C.SIGMA_OPTIONS), options=C.SIGMA_OPTIONS),
    "alpha_select": SettingSpec("alpha", float, D["alpha"], min(C.ALPHA_OPTIONS), max(C.ALPHA_OPTIONS), options=C.ALPHA_OPTIONS),
    "l1_select": SettingSpec("l1", float, D["l1"], min(C.L1_OPTIONS), max(C.L1_OPTIONS), options=C.L1_OPTIONS),
    "seed_input": SettingSpec("seed", int, D["seed"], 0, C.SEED_MAX),
    "hand_swap_toggle": SettingSpec("hs", _to_bool, D["hand_swap"]),
    "hand_alpha_select": SettingSpec("ha", str, D["hand_alpha"], options=C.HAND_ALPHA_OPTIONS),
    "vif_b_select": SettingSpec("vb", float, D["vif_b"], min(C.VIF_B_OPTIONS), max(C.VIF_B_OPTIONS), options=C.VIF_B_OPTIONS),
    "vif_reps_slider": SettingSpec("vr", int, D["vif_reps"], C.VIF_REPS_MIN, C.VIF_REPS_MAX, step=C.VIF_REPS_STEP),
    "path_rel_select": SettingSpec("pr", float, D["path_rel"], min(C.ALPHA_REL_OPTIONS), max(C.ALPHA_REL_OPTIONS), options=C.ALPHA_REL_OPTIONS),
    "path_thr_select": SettingSpec("pt", float, D["path_thr"], min(C.THRESHOLD_OPTIONS), max(C.THRESHOLD_OPTIONS), options=C.THRESHOLD_OPTIONS),
    "fresh_select": SettingSpec("fr", int, D["fresh"], min(C.FRESH_OPTIONS), max(C.FRESH_OPTIONS), options=C.FRESH_OPTIONS),
    "scheme_select": SettingSpec("sc", str, D["scheme"], options=C.SCHEME_OPTIONS),
    "pi_select": SettingSpec("pi", float, D["pi"], min(C.PI_OPTIONS), max(C.PI_OPTIONS), options=C.PI_OPTIONS),
    "m_orig_select": SettingSpec("mo", int, D["m_orig"], min(C.M_ORIG_OPTIONS), max(C.M_ORIG_OPTIONS), options=C.M_ORIG_OPTIONS),
    "en_l1_select": SettingSpec("el", float, D["en_l1"], min(C.EN_L1_OPTIONS), max(C.EN_L1_OPTIONS), options=C.EN_L1_OPTIONS),
}
PRESET_KEYS = {"scen": "scen_select", "rho": "rho_select", "n": "n_select", "sigma": "sigma_select", "alpha": "alpha_select", "l1": "l1_select", "seed": "seed_input",
               "hand_swap": "hand_swap_toggle", "hand_alpha": "hand_alpha_select", "vif_b": "vif_b_select", "vif_reps": "vif_reps_slider", "path_rel": "path_rel_select",
               "path_thr": "path_thr_select", "fresh": "fresh_select", "scheme": "scheme_select", "pi": "pi_select", "m_orig": "m_orig_select", "en_l1": "en_l1_select"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _snap(spec, value):
    """Wert auf die erlaubten Stufen einrasten (Auswahlregler) bzw. auf die Reglergrenzen begrenzen."""
    if spec.options is not None:
        if spec.caster is str:
            return value if value in spec.options else spec.default
        return min(spec.options, key=lambda o: abs(o - value))
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    if spec.step is not None and spec.lo is not None:
        value = type(spec.default)(round((value - spec.lo) / spec.step) * spec.step + spec.lo)
        value = min(spec.hi, max(spec.lo, value))
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf die erlaubten Stufen eingerastet."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if isinstance(value, float) and not math.isfinite(value):
                continue
            st.session_state[state_key] = _snap(spec, value) if not isinstance(value, bool) else value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Ein Preset setzt jeden Regler: die genannten auf ihren Wert, alle übrigen auf den Standard (damit zwei Presets sich nie überlagern)."""
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name].get(key, SETTING_SPECS[state_key].default)


def _gleich(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str):
        return a == b
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-9)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen (Gleitkommawerte auf 1e-9), sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State.
    Der Zufalls-Seed gehört dazu. tests/test_layout.py prüft für jeden Regler, dass seine Änderung die Markierung aufhebt."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(_gleich(st.session_state.get(state_key, spec.default), preset.get(preset_key_of.get(state_key), spec.default)) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
