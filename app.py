"""Zwei Angaben, die fast dasselbe messen: Kollinearität und Lasso - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Sechs Experimente: Handbeispiel mit vier Containern (Brüche), Aufblähung der Varianz und die Illusion „kein Effekt“, der Lasso-Pfad, die Auswahl über frische Stichproben,
die Falle des Bootstraps derselben Stichprobe, das Elastic Net und sein Preis. Zufall nur aus `kol_stream` (SplitMix64); vorgerechnete Zählungen in data/kol_frozen.json.

Lauffähig mit: streamlit run app.py
"""

from dataclasses import replace

import pandas as pd
import streamlit as st

import kol_constants as C
import kol_evaluation as ev
import kol_frozen as FZ
from kol_evaluation import Settings, de
from kol_layout import dunkles_design, kopf_css
from kol_presets import SETTING_SPECS, init_session_state_defaults, load_permalink_settings, randomize_seed, sidebar_presets, sync_query_params
from kol_visualization import (
    build_compare, build_hand_path, build_hand_plane, build_originals, build_path, build_rates, build_sample, build_selection, build_vif_hist,
)

L = C.repo_link

st.set_page_config(page_title="Zwei Angaben, die fast dasselbe messen – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _hand(swapped, alpha_text):
    return ev.hand_analysis(swapped, alpha_text)


@st.cache_data(show_spinner=False)
def _hand_boot(swapped):
    return ev.hand_bootstrap(swapped)


@st.cache_data(show_spinner=False)
def _sample(s):
    return ev.one_sample(s)


@st.cache_data(show_spinner=False)
def _vif(n, rho, sigma, b, reps, seed):
    return ev.vif_analysis(n, rho, sigma, b, reps, seed)


@st.cache_data(show_spinner=False)
def _spread(n, rho, designs, seed):
    return ev.vif_spread_analysis(n, rho, designs, seed)


@st.cache_data(show_spinner=False)
def _path(s):
    return ev.path_analysis(s)


@st.cache_data(show_spinner=False)
def _fresh(scen, n, rho, sigma, alpha, l1, n_fresh, seed):
    return ev.fresh_analysis(ev.world_of(scen), scen, n, rho, sigma, alpha, l1, n_fresh, seed)


@st.cache_data(show_spinner=False)
def _boot(s):
    return ev.bootstrap_analysis(s)


@st.cache_data(show_spinner=False)
def _condition():
    return ev.condition_table()


@st.cache_data(show_spinner=False)
def _frozen():
    return FZ.load()


def fresh_of(s: Settings, l1=None, n_fresh=None):
    return _fresh(s.scen, s.n, s.rho, s.sigma, s.alpha, s.l1 if l1 is None else l1, s.fresh if n_fresh is None else n_fresh, s.seed)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kol_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧮 Zwei Angaben, die fast dasselbe messen: Kollinearität und Lasso")
st.markdown(
    f"""
Zwei Angaben zu einem Container messen fast dasselbe: etwa der **Vorlauf laut Reederei** und der **Vorlauf bis Ladeschluss** (Stichprobenkorrelation **ρ** nahe 1). Die Demo zeigt, **was dann aus den
Koeffizienten, den Tests und der Merkmalsauswahl wird**: Die Varianz jedes Einzelkoeffizienten bläht sich um **1/(1 − ρ²)** auf, die ihrer Summe nicht; die Daten kennen die Summe, nicht die Aufteilung. Das **Lasso**
wählt dann je Stichprobe eines von beiden, und die Häufigkeit, mit der **Bootstrap oder Teilstichproben einer Stichprobe** ein Merkmal wählen, sagt darüber weniger, als sie aussieht. Alle Zahlen sind
**eigene Nachrechnung in diesem Modell, bei den genannten Parametern**; VIF, Lasso und Bootstrap sind Lehrbuchstoff, keine Neuheitsaussage. Der Containerstrom ist erfunden.

Ridge-Regression und die Wahl von λ stehen in der {L('tikhonov-demo')} und der {L('parameterwahl-demo')}; dass die Koeffizienten bei
Kollinearität wandern und die Wichtigkeit im Wald sich aufteilt, zeigt die {L('korrelierte-merkmale-demo')}. **Zeilen gegen Spalten:** die {L('hebel-diagnose-demo')}
prüft, ob eine **Zeile** über die Gerade entscheidet; diese Demo fragt, ob jede **Spalte** eigene Information trägt.

Weiter unten: die **drei Kernfragen** mit Verdikt, sechs Experimente (🔬), der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung** mit der Konventionstabelle (Faktor ½, Standardisierung,
Stichproben- gegen Bevölkerungskorrelation).
"""
)

with st.expander("So funktioniert es", expanded=True):
    st.markdown(
        """
1. **Zwei Spalten, fast gleich.** Liegen x₁ und x₂ fast übereinander (Korrelation ρ), kann die Anpassung nur noch gut sagen, **was beide zusammen** bewirken; wie sich der Effekt auf beide **aufteilt**, ist unscharf.
   Der **VIF** (1/(1 − R²) der Hilfsregression, bei zwei Merkmalen 1/(1 − ρ²)) misst die Aufblähung der Varianz.
2. **Lasso:** minimiert Quadratsumme plus α mal Summe der Beträge der Koeffizienten; viele Koeffizienten werden **exakt null**. Bei zwei fast gleichen Merkmalen reicht eines von beiden, und **welches**, entscheidet die Stichprobe.
   **Elastic Net** mischt die Strafe mit einer quadratischen und nimmt beide Zwillinge mit.
3. **Bootstrap und Teilstichproben:** Man zieht aus **einer** Stichprobe wiederholt neu und zählt, wie oft ein Merkmal gewählt wird. Ob diese Häufigkeit zeigt, was **frische** Stichproben zeigen würden, prüft Experiment 5.
4. **Gegenprobe:** jeder Kern ist gegen einen anderen Rechenweg geprüft (Aufzählung aller Aktivmengen in Brüchen, scikit-learn, Optimalitätsbedingungen, exakte nichtzentrale Verteilungen).
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Zwillinge und Szenario**")
    scen = st.selectbox("Szenario", list(C.SCENARIOS), key="scen_select", format_func=lambda k: C.SCENARIO_LABELS[k],
                        help="Welche Merkmale wirken: S3 beide Zwillinge schwach, S1 beide stark, S2 nur x₁, S2E wie S2 mit der lognormalen Verweilzeit der Export-Container.")
    rho = st.select_slider("Korrelation ρ von x₁ und x₂", options=list(C.RHO_OPTIONS), key="rho_select", format_func=lambda v: de(v, 3).rstrip("0").rstrip(",") if v not in (0.0,) else "0",
                           help="Bevölkerungskorrelation der beiden Zwillingsmerkmale (x₂ ist eine erfundene zweite Vorlaufangabe).")
    n = st.select_slider("Stichprobengröße n", options=list(C.N_OPTIONS), key="n_select", help="Zahl der Container je Stichprobe.")
    sigma = st.select_slider("Streuung σ", options=list(C.SIGMA_OPTIONS), key="sigma_select", format_func=lambda v: de(v, 1),
                             help="Streuung des Fehlers (in den Szenarien S1 bis S3 additiv und konstant); bei S2E skaliert σ/2 die Streuung der log-Verweilzeit des Generators (σ = 2 ist der Generator).")
    st.markdown("**Lasso und Elastic Net**")
    alpha = st.select_slider("Strafe α (fest)", options=list(C.ALPHA_OPTIONS), key="alpha_select", format_func=lambda v: de(v, 3),
                             help="Faktor der Strafe in der Konvention von scikit-learn und glmnet. Fest: α wird nie je Stichprobe per Kreuzvalidierung gewählt.")
    l1 = st.select_slider("Mischung ℓ (1 = Lasso)", options=list(C.L1_OPTIONS), key="l1_select", format_func=lambda v: de(v, 2),
                          help="Strafe α·(ℓ·|b|₁ + (1 − ℓ)/2·|b|₂²): ℓ = 1 ist das Lasso, kleiner ist ein Elastic Net. Gilt für die Experimente 3 bis 5; Experiment 6 vergleicht beide.")
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed", *(0, C.SEED_MAX), key="seed_input", step=1)
    st.button("🎲 Neue Stichprobe ziehen", width="stretch", on_click=randomize_seed)

s = Settings(scen=scen, rho=float(rho), n=int(n), sigma=float(sigma), alpha=float(alpha), l1=float(l1), seed=int(seed))
D = Settings()  # Standardwerte: die Kernfragen rechnen mit den Standardwerten der Experimente, die weiter unten stehen

# --- Ergebnis: eine Stichprobe --------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis: eine Stichprobe")
smp = _sample(s)
st.caption(
    f"Eine Stichprobe mit n = {s.n} Containern (Zufalls-Seed {s.seed}), Szenario {s.scen}, ρ = {de(s.rho, 3)}, σ = {de(s.sigma, 1)}; Koeffizienten auf der standardisierten Skala "
    "(Merkmale mit Mittel 0 und Standardabweichung 1, y zentriert, kein Achsenabschnitt gerechnet)."
)
m1, m2, m3 = st.columns(3)
m1.metric("Stichprobenkorrelation r̂(x₁, x₂)", de(smp["r_hat"], 3), help="Korrelation der beiden Spalten in dieser Stichprobe; sie weicht von ρ ab (Streuung bei kleinem n).")
m2.metric("VIF von x₁", de(smp["vif"][0], 2), help="1/(1 − R²) der Hilfsregression von x₁ auf die übrigen neun Merkmale; bei zwei Merkmalen allein gleich 1/(1 − r̂²).")
m3.metric("Gewählt vom Lasso/Elastic Net (Zahl der Nichtnullen)", f"{int((abs(smp['lasso']) > C.NZ_THRESHOLD).sum())} von {C.P_FEATURES}", help="Schwelle 1e-8; der Abstieg liefert exakte Nullen.")
if smp["ols"] is not None:
    m4, m5, m6 = st.columns(3)
    m4.metric("Kleinste Quadrate b₁ (± SE)", f"{de(smp['ols'][0], 2)} ± {de(smp['ols_se'][0], 2)}")
    m5.metric("Kleinste Quadrate b₂ (± SE)", f"{de(smp['ols'][1], 2)} ± {de(smp['ols_se'][1], 2)}")
    m6.metric("Lasso/Elastic Net b₁ | b₂", f"{de(smp['lasso'][0], 2)} | {de(smp['lasso'][1], 2)}", help="Null heißt: nicht gewählt.")
else:
    st.caption("Bei n ≤ 11 Containern ist die Anpassung mit zehn Merkmalen nicht eindeutig (Kleinste Quadrate entfallen).")
st.plotly_chart(build_sample(smp), width="stretch", key="sample_chart")
if s.scen == "S2E":
    st.caption("Szenario S2E: die Antwort ist die lognormale Verweilzeit (Tage), das „wahre β“ ist nur ein Bezug für den Lesenden (der Vorlauf wirkt, x₂ nicht).")

# --- Kernfragen -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
st.markdown("**1. Wie viel erfahren die Daten über zwei fast gleiche Merkmale?**")
v_main = _vif(s.n, s.rho, s.sigma, D.vif_b, D.vif_reps, s.seed)
kind, text = ev.verdict_q1(v_main)
getattr(st, kind)(text)
st.markdown("**2. Welches von beiden wählt das Lasso, und wovon hängt das ab?**")
f_main = fresh_of(s, n_fresh=D.fresh)
kind, text = ev.verdict_q2(f_main, s.scen, s.n, s.rho, s.sigma, s.alpha, s.l1)
getattr(st, kind)(text)
st.markdown("**3. Was zeigt der Bootstrap derselben Stichprobe, was frische Stichproben nicht zeigen?**")
b_main = _boot(replace(s, scheme=D.scheme, pi=D.pi, m_orig=D.m_orig))
kind, text = ev.verdict_q3(f_main, b_main["summary"], s.scen, C.SCHEME_OPTIONS[0], D.pi)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 1: Handbeispiel -------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Von Hand, vier Container")
st.markdown(
    "Vier **erfundene** Container mit ganzen Zahlen (Abweichungen vom Mittel; keine Daten des Generators, damit die Brüche exakt bleiben): Vorlauf laut Reederei **x₁ = (−3, −1, 1, 3)**, "
    "Vorlauf bis Ladeschluss **x₂ = (−3, −1, 3, 1)**, Verweilzeit **y = (−2, −1, 1, 2)**. Gerechnet ohne Achsenabschnitt und ohne Standardisieren, alles in Brüchen."
)
c1, c2 = st.columns(2)
with c1:
    swap = st.toggle("Zeilen 3 und 4 von y vertauschen: y′ = (−2, −1, 2, 1)", key="hand_swap_toggle", help="Dieselben Spalten, zwei Werte von y getauscht: die Koeffizienten tauschen die Plätze.")
with c2:
    hand_alpha = st.select_slider("Strafe α des Handbeispiels", options=list(C.HAND_ALPHA_OPTIONS), key="hand_alpha_select", help="α von 0 (Kleinste Quadrate) bis 7/2 (alles null).")
h = _hand(bool(swap), hand_alpha)
rep = h["rep"]
fr = ev.frac_text
h1, h2, h3 = st.columns(3)
h1.metric("VIF = 1/(1 − ρ²)", fr(rep["vif"]), help=f"Gram-Matrix X'X = [[{fr(rep['xtx'][0][0])}, {fr(rep['xtx'][0][1])}], [{fr(rep['xtx'][1][0])}, {fr(rep['xtx'][1][1])}]], ρ = {fr(rep['rho'])}.")
h2.metric("Var(b₁ + b₂) | Var(b₁ − b₂) in σ²", f"{fr(rep['var_sum'])} | {fr(rep['var_diff'])}", help=f"Bei orthogonalen Spalten gleicher Länge: {fr(rep['orth_sum'])} für beide; Faktoren 1/(1 + ρ) = {fr(rep['factor_sum'])} und 1/(1 − ρ) = {fr(rep['factor_diff'])}.")
h3.metric("Kleinste Quadrate b = (b₁; b₂)", f"({fr(rep['b'][0])}; {fr(rep['b'][1])})", help=f"Summe {fr(rep['b'][0] + rep['b'][1])}; SSE = {fr(rep['sse'])} bei {rep['df']} Rest-Freiheitsgrad.")
h4, h5, h6 = st.columns(3)
h4.metric(f"Lasso bei α = {fr(h['alpha'])}", f"({fr(h['lasso'][0])}; {fr(h['lasso'][1])})")
h5.metric(f"Elastic Net (ℓ = 1/2) bei α = {fr(h['alpha'])}", f"({fr(h['enet'][0])}; {fr(h['enet'][1])})")
h6.metric("Korrelation von b₁ und b₂", fr(rep["corr_b"]), help="Gleich −ρ (beide Spalten gleich lang).")
ex, ey = ev.ellipse_points(rep)
g1, g2 = st.columns(2)
g1.plotly_chart(build_hand_path(h), width="stretch", key="hand_path_chart")
g2.plotly_chart(build_hand_plane(h, ex, ey), width="stretch", key="hand_plane_chart")
kind, text = ev.verdict_hand(h)
getattr(st, kind)(text)
cnt_y, cnt_ys = _hand_boot(False), _hand_boot(True)
st.markdown("**Der Bootstrap dieser vier Container, vollständig aufgezählt** (alle 4⁴ = 256 Stichproben, Lasso bei α = 2, in jeder Stichprobe zentriert; Zähleinheit: eine der 256 Bootstrap-Stichproben):")
st.dataframe(
    pd.DataFrame(
        [{"Daten": "y", "nur x₁": cnt_y[0], "nur x₂": cnt_y[1], "beide": cnt_y[2], "keines": cnt_y[3], "nicht eindeutig (höchstens zwei verschiedene Zeilen)": cnt_y[4]},
         {"Daten": "y′ (Zeilen 3 und 4 vertauscht)", "nur x₁": cnt_ys[0], "nur x₂": cnt_ys[1], "beide": cnt_ys[2], "keines": cnt_ys[3], "nicht eindeutig (höchstens zwei verschiedene Zeilen)": cnt_ys[4]}]
    ),
    hide_index=True, width="stretch",
)
st.caption("Mit y wählt der Bootstrap fast immer x₁, mit y′ fast immer x₂: die Zählung ist ein Spiegel der Daten, nicht der Welt (x₁ und x₂ sind in der Konstruktion austauschbar).")

st.markdown("---")

# --- Experiment 2: Aufblähung ---------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Wie viel wissen die Daten? Varianzaufblähung und die Illusion „kein Effekt“")
st.caption(
    "Festes Design mit zwei zentrierten Merkmalen (Stichprobenkorrelation genau ρ), Fehler N(0, σ²) in jeder Wiederholung neu gezogen, mit Achsenabschnitt (n − 3 Rest-Freiheitsgrade), "
    "Niveau 5 %, β = (b, b). ρ, n und σ stehen in der Seitenleiste. Zähleinheit: die Wiederholung."
)
e1, e2 = st.columns(2)
with e1:
    vif_b = st.select_slider("Signal b (β = (b, b))", options=list(C.VIF_B_OPTIONS), key="vif_b_select", format_func=lambda v: de(v, 1), help="Wirkung jedes der beiden Merkmale.")
with e2:
    vif_reps = st.slider("Wiederholungen", C.VIF_REPS_MIN, C.VIF_REPS_MAX, step=C.VIF_REPS_STEP, key="vif_reps_slider", help="Zahl der Wiederholungen der Simulation; der Monte-Carlo-Fehler einer Rate ist etwa 1 bis 2 Prozentpunkte bei 2000.")
v = _vif(s.n, s.rho, s.sigma, float(vif_b), int(vif_reps), s.seed)
x1, x2, x3 = st.columns(3)
x1.metric("Standardfehler von b₁", de(v["se_formula"], 3), help=f"σ·√(V₁₁) mit V = (X'X)⁻¹; bei orthogonalen Spalten wäre er {de(v['se_orth'], 3)}; Faktor √VIF.")
fac = v["factors"]
x2.metric("VIF = 1/(1 − ρ²)", de(fac["single"], 2) if fac else "unendlich", help="ρ = 1 macht X'X singulär.")
x3.metric("Summe | Differenz: Faktor der Varianz", f"{de(fac['sum'], 3)} | {de(fac['diff'], 1)}" if fac else "–", help="Gegen orthogonale Spalten: Summe 1/(1 + ρ), Differenz 1/(1 − ρ). Die Daten kennen die Summe, nicht die Aufteilung.")
t1, t2 = st.columns(2)
t1.plotly_chart(build_rates(v), width="stretch", key="rates_chart")
spread = _spread(s.n, s.rho, 2000, s.seed)
t2.plotly_chart(build_vif_hist(spread["vifs"], spread["formula"]), width="stretch", key="vif_hist_chart")
kind, text = ev.verdict_q1(v)
getattr(st, kind)(text)
st.caption(
    f"Rechts: der VIF der **Stichprobe** (Zufallsdesign, 2000 Designs, Zähleinheit: ein Design) liegt im 5-bis-95-%-Bereich bei {de(100 * spread['rel05'], 0)} % bis +{de(100 * spread['rel95'], 0)} % um die Formel mit der Bevölkerungskorrelation; "
    "die Formel gilt für die Stichprobe nur im Mittel."
)
fz = _frozen()
with st.expander("📊 Vorgerechnet: Summe und Differenz, Intervalle, VIF-Streuung (große Zählungen)"):
    st.markdown("**Summe und Differenz** (festes Design n = 100, Fehler N(0, 1), 100 000 Wiederholungen je ρ, Zufalls-Seed 1201; n·Var in Einheiten σ²; Zähleinheit: Wiederholung):")
    st.dataframe(
        pd.DataFrame([{"ρ": de(r["rho"], 3), "n·Var(b₁)": de(r["n_var_b1"], 3), "Formel": de(r["formula_single"], 3), "n·Var(b₁ + b₂)": de(r["n_var_sum"], 3), "Formel (Summe)": de(r["formula_sum"], 3),
                       "n·Var(b₁ − b₂)": de(r["n_var_diff"], 2), "Formel (Differenz)": de(r["formula_diff"], 1), "Korrelation b₁, b₂": de(r["corr"], 3)} for r in fz["sumdiff"]]),
        hide_index=True, width="stretch",
    )
    st.markdown("**Überdeckung und Breite des 95-%-Intervalls für b₁** (β = (0,5; 0,5), σ = 2, n = 100, 20 000 Wiederholungen je ρ, Zufalls-Seed 1203; Zähleinheit: Wiederholung): Kollinearität verletzt keine Annahme des Modells, die Antwort wird nur breit.")
    st.dataframe(pd.DataFrame([{"ρ": de(r["rho"], 3), "Überdeckung": C.pct(r["cover"]), "mittlere Breite": de(r["width"], 2), "Breite gegen ρ = 0": de(r["width_ratio"], 2)} for r in fz["coverage"]]), hide_index=True, width="stretch")
    st.markdown("**Streuung des VIF der Stichprobe** (Zufallsdesign, 4000 Designs je Zeile, Zufalls-Seed 1204; Zähleinheit: Design):")
    st.dataframe(pd.DataFrame([{"ρ": de(r["rho"], 2), "n": r["n"], "Formel": de(r["formula"], 2), "5 %": de(r["q05"], 2), "Median": de(r["q50"], 2), "95 %": de(r["q95"], 2),
                                "5 % relativ": f"{de(100 * r['rel05'], 0)} %", "95 % relativ": f"+{de(100 * r['rel95'], 0)} %", "Anteil über dem Doppelten": C.pct(r["over2"])} for r in fz["vif_spread"]]), hide_index=True, width="stretch")
nat = fz["natural_vif_export"]
st.caption(
    f"Der Generator selbst bringt von Haus aus keine hohe Kollinearität mit: im Export-Zweig (Vorlauf, Wochenende, Gewicht, elf Kunden-Dummies; {nat['n']} Container, Zähleinheit: ein Design) ist die größte VIF {de(nat['max_vif'], 2)}. "
    "Hohe Kollinearität kommt erst mit einem zweiten, fast gleichen Merkmal."
)

st.markdown("---")

# --- Experiment 3: Pfad ---------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Der Lasso-Pfad – wer tritt zuerst ein?")
st.caption("Die Stichprobe von oben (Szenario, ρ, n, σ, Zufalls-Seed der Seitenleiste), Mischung ℓ der Seitenleiste. α läuft von α_max (alle Koeffizienten 0) abwärts; alle 60 Pfadpunkte werden als ein Stapel gerechnet.")
p1c, p2c = st.columns(2)
with p1c:
    path_rel = st.select_slider("Gewähltes α in Anteilen von α_max", options=list(C.ALPHA_REL_OPTIONS), key="path_rel_select", format_func=lambda v: de(v, 3), help="Die gestrichelte Linie im Pfad.")
with p2c:
    path_thr = st.select_slider("Schwelle für „Nichtnull“", options=list(C.THRESHOLD_OPTIONS), key="path_thr_select", format_func=lambda v: "0" if v == 0 else f"{v:.0e}".replace("e-0", "e-"),
                                help="Ein Koeffizient zählt als gewählt, wenn sein Betrag über der Schwelle liegt. Der Abstieg liefert exakte Nullen, ein glattes Verfahren nicht.")
sp = replace(s, path_rel=float(path_rel), path_thr=float(path_thr))
pa = _path(sp)
st.plotly_chart(build_path(pa, sp.path_thr), width="stretch", key="path_chart")
st.markdown("**Nichtnullen je Schwelle bei dem gewählten α** (diese Stichprobe):")
st.dataframe(pd.DataFrame([{"Schwelle": "0" if t == 0 else f"{t:.0e}".replace("e-0", "e-"), "Nichtnullen": pa["counts"][t]} for t in C.THRESHOLD_OPTIONS]), hide_index=True, width="stretch")
kind, text = ev.verdict_path(pa, sp)
getattr(st, kind)(text)
nz = fz["nonzero"]
with st.expander("📊 Vorgerechnet: Nichtnullen und Rauschgrenze über 2000 frische Stichproben, glatter Löser"):
    st.markdown(f"**Szenario S3, Lasso, α = {de(nz['alpha'], 2)}, n = 100, ρ = 0,95, σ = 2, {nz['n_fresh']} frische Stichproben** (Zähleinheit: frische Stichprobe): Auswahlhäufigkeit je Schwelle.")
    st.dataframe(pd.DataFrame([{"Schwelle": "0" if r["thr"] == 0 else f"{r['thr']:.0e}".replace("e-0", "e-"), "x₁": C.pct(r["x1"]), "x₂": C.pct(r["x2"]), "x₃": C.pct(r["x3"]),
                                "Attrappen x₄–x₁₀ (Mittel)": C.pct(r["dummies"]), "Nichtnullen im Mittel": de(r["nonzeros"], 2)} for r in nz["table"]]), hide_index=True, width="stretch")
    st.markdown(f"Anteil der Stichproben, deren kleinster gewählter Betrag unter 1e-8 / 1e-4 / 1e-2 liegt: {C.pct(nz['below']['1e-08'], 2)} / {C.pct(nz['below']['0.0001'], 2)} / {C.pct(nz['below']['0.01'], 1)}: "
                "es gibt keine Lücke zwischen „null“ und „klein“; eine Zahl der Nichtnullen ohne Schwellenangabe ist keine Messung.")
    st.markdown("**Glattes Verfahren** (L-BFGS-B auf √(b² + ε) statt |b|, 200 Stichproben): Zahl der Stichproben, in denen die gewählte Menge mit der des exakten Abstiegs übereinstimmt.")
    rows = []
    for blk in nz["smooth"]:
        for r in blk["rows"]:
            rows.append({"ε": f"{blk['eps']:.0e}".replace("e-0", "e-"), "Schwelle": "0" if r["thr"] == 0 else f"{r['thr']:.0e}".replace("e-0", "e-"), "gleiche Menge": f"{r['same']} von {r['of']}", "Nichtnullen im Mittel": de(r["nonzeros"], 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.markdown("---")

# --- Experiment 4: frische Stichproben ------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Welches von beiden? Auswahl über frische Stichproben")
st.caption("Jede Stichprobe wird neu gezogen und mit festem α angepasst; gezählt wird, wie oft ein Merkmal gewählt wird (Zähleinheit: frische Stichprobe). Szenario, ρ, n, σ, α und ℓ stehen in der Seitenleiste.")
fresh_n = st.select_slider("Zahl der frischen Stichproben", options=list(C.FRESH_OPTIONS), key="fresh_select", help="Mehr Stichproben verkleinern den Monte-Carlo-Fehler (bei 500 etwa 2 Prozentpunkte).")
f4 = fresh_of(s, n_fresh=int(fresh_n))
st.plotly_chart(build_selection(f4), width="stretch", key="selection_chart")
a1, a2, a3 = st.columns(3)
a1.metric("Anteil x₁ bei genau einem", C.pct(f4["share1"]), help=f"Unter den Stichproben, in denen genau eines der beiden gewählt wird ({f4['n_one']} von {f4['n_fresh']}).")
a2.metric("Nichtnullen im Mittel", de(f4["nz"], 2))
a3.metric("Attrappen x₄–x₁₀ im Mittel", C.pct(f4["att"]), help="Ein α nahe dem Kreuzvalidierungswert zielt auf Vorhersage, nicht auf Auswahl: es wählt viele Attrappen.")
kind, text = ev.verdict_q2(f4, s.scen, s.n, s.rho, s.sigma, s.alpha, s.l1)
getattr(st, kind)(text)
with st.expander("🧭 Wann ist die richtige Auswahl überhaupt möglich? Die Bedingung von Zhao/Yu"):
    st.markdown(
        "Eine Attrappe x₀ hat Korrelation a zu jedem wahren Merkmal (β = 1, untereinander unabhängig). Die Größe **q = |C₂₁ C₁₁⁻¹ sign(β₁)|** (C Korrelationsmatrix, Index 1 wahre Merkmale, 2 Attrappe) "
        "entscheidet nach Zhao/Yu (2006), ob das Lasso die wahre Menge finden kann: **q < 1 ist fast notwendig und hinreichend**, nicht notwendig und hinreichend. "
        "In der **Bevölkerung** (keine Daten, kein Rauschen, α = 0,05) wählt das Lasso die Attrappe in diesen sieben Entwürfen genau dann, wenn q > 1; q = 1 ist der Rand und wird nicht gezählt. "
        "Mit **Daten** (α = 0,1, σ = 2, 500 Stichproben je Zelle, vorgerechnet) fällt P(Attrappe) mit n für q < 1 und steigt für q > 1."
    )
    cond_live, cond_fz = _condition(), fz["condition"]
    st.dataframe(pd.DataFrame([{"Entwurf": r["label"], "q": de(r["q"], 2), "Attrappe in der Bevölkerung": "Rand (q = 1)" if abs(r["q"] - 1.0) < 1e-9 else "gewählt" if r["chosen"] else "nicht gewählt",
                                "P(Attrappe) n = 100": C.pct(z["rates"]["100"]), "n = 1000": C.pct(z["rates"]["1000"]), "n = 10 000": C.pct(z["rates"]["10000"])} for r, z in zip(cond_live, cond_fz)]),
                 hide_index=True, width="stretch")

st.markdown("---")

# --- Experiment 5: Bootstrap ----------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Die Falle – der Bootstrap derselben Stichprobe")
st.caption(
    "Je **Originalstichprobe** werden B = 100 Bootstrap-Stichproben (Größe n, mit Zurücklegen) bzw. Teilstichproben (Größe n/2, ohne Zurücklegen) gezogen und mit demselben α angepasst; "
    "ein Punkt im Bild ist eine Originalstichprobe (Zähleinheit). Das Kreuz sind die Häufigkeiten über frische Stichproben."
)
b1c, b2c, b3c = st.columns(3)
with b1c:
    scheme = st.selectbox("Wiederholungsverfahren", list(C.SCHEME_OPTIONS), key="scheme_select")
with b2c:
    pi = st.select_slider("Schwelle π („gilt als stabil“)", options=list(C.PI_OPTIONS), key="pi_select", format_func=lambda v: de(v, 1), help="Ein Merkmal gilt als stabil, wenn es in mindestens dem Anteil π der Wiederholungen gewählt wird.")
with b3c:
    m_orig = st.select_slider("Zahl der Originalstichproben m", options=list(C.M_ORIG_OPTIONS), key="m_orig_select", help="Je Originalstichprobe 100 Anpassungen; mehr Originale verkleinern den Monte-Carlo-Fehler der Zählungen.")
sb = replace(s, scheme=scheme, pi=float(pi), m_orig=int(m_orig))
bo = _boot(sb)
f5 = fresh_of(s, n_fresh=C.FRESH_OPTIONS[1])
st.plotly_chart(build_originals(bo["po"], f5), width="stretch", key="originals_chart")
z1, z2, z3 = st.columns(3)
z1.metric("|P(x₁) − P(x₂)| je Original (Mittel)", de(bo["summary"]["absdiff"], 3), help=f"Über frische Stichproben: {de(abs(f5['p1'] - f5['p2']), 3)}.")
z2.metric("Anteil x₁ über 0,75 oder unter 0,25 („klar“)", C.pct(bo["summary"]["clear"]), help=f"Gezählt über die {bo['summary']['n_share']} Originale, in denen überhaupt eine Wiederholung genau eines der beiden wählt.")
z3.metric("Regel ≥ π: mindestens eine Attrappe stabil", C.pct(bo["summary"]["att_st"]), help="Anteil der Originalstichproben, in denen mindestens eine der sieben Attrappen die Schwelle π erreicht.")
kind, text = ev.verdict_q3(f5, bo["summary"], s.scen, scheme, sb.pi)
getattr(st, kind)(text)
po0 = {k: bo["po"][k][0] for k in ("p1", "p2", "share")}
st.caption(f"Eine einzelne Originalstichprobe (die erste) für sich: x₁ in {C.pct(po0['p1'])}, x₂ in {C.pct(po0['p2'])} der Wiederholungen; über frische Stichproben x₁ {C.pct(f5['p1'])}, x₂ {C.pct(f5['p2'])}. Eine einzelne Stichprobe ist ein Punkt, kein Befund.")
with st.expander("📊 Vorgerechnet: 200 Originalstichproben je Zelle (große Zählungen)"):
    st.markdown("n = 100, ρ = 0,95, σ = 2, p = 10, α fest (Lasso 0,09, Elastic Net 0,13 mit ℓ = 0,5), je Zelle 2000 frische Stichproben und 200 Originalstichproben mit je 100 Bootstrap- und 100 Teilstichproben (Zahl der Zeilen siehe Zähleinheit). "
                "Welt C: Merkmale mit den Verteilungen des Containergenerators; G: alle Merkmale normal; E: Export-Zweig des Generators. „klar“: Anteil von x₁ bei genau einem über 0,75 oder unter 0,25.")
    names = {"C_S3_L": "C, S3, Lasso", "C_S3_L_seed2": "C, S3, Lasso (zweiter Zufalls-Seed)", "C_S1_L": "C, S1, Lasso", "C_S2_L": "C, S2, Lasso", "C_S3_E": "C, S3, Elastic Net", "C_S2_E": "C, S2, Elastic Net",
             "C_S3_L_rho099": "C, S3, Lasso, ρ = 0,99", "G_S3_L": "G, S3, Lasso", "G_S1_L": "G, S1, Lasso", "G_S2_L": "G, S2, Lasso", "E_S2E_L": "E, S2E, Lasso", "E_S2E_E": "E, S2E, Elastic Net"}
    rows = []
    for key, cell in fz["selection"].items():
        f, bt, sb_ = cell["fresh"], cell["boot"], cell["sub"]
        rows.append({"Zelle": names[key], "frisch x₁": C.pct(f["p1"]), "frisch x₂": C.pct(f["p2"]), "frisch beide": C.pct(f["both"]), "Anteil x₁ bei genau einem (frisch)": C.pct(f["share1"]),
                     "|ΔP| frisch": de(abs(f["p1"] - f["p2"]), 3), "|ΔP| Bootstrap": de(bt["absdiff"], 3), "klar Bootstrap": C.pct(bt["clear"]), "klar Teilstichproben": C.pct(sb_["clear"]),
                     "Regel ≥ 0,8: genau x₁ / genau x₂ / beide (Bootstrap)": f"{C.pct(bt['only1_st'])} / {C.pct(bt['only2_st'])} / {C.pct(bt['both_st'])}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown("Rechenbeispiel der Schranke der Stabilitätsauswahl nach Meinshausen/Bühlmann (2010), die **Austauschbarkeit der Nichtprädiktoren** voraussetzt (hier nicht erfüllt): "
                f"q = 7 gewählte von p = 10 und π = 0,8 geben q²/((2π − 1)p) = {ev.frac_text(ev.E.stability_bound(7, 10, ev.F(4, 5)))} ≈ 8,17, mehr als die 7 vorhandenen Attrappen: die Schranke sagt hier nichts.")

st.markdown("---")

# --- Experiment 6: Elastic Net --------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 6: Elastic Net – der Gruppeneffekt und sein Preis")
st.caption("Lasso und Elastic Net auf denselben frischen Stichproben mit demselben α (Zähleinheit: frische Stichprobe). Szenario, ρ, n, σ und α stehen in der Seitenleiste.")
en_l1 = st.select_slider("Mischung ℓ des Elastic Net", options=list(C.EN_L1_OPTIONS), key="en_l1_select", format_func=lambda v: de(v, 2), help="ℓ = 1 wäre das Lasso; kleiner nimmt mehr vom quadratischen Anteil.")
fl6 = fresh_of(s, l1=1.0, n_fresh=C.FRESH_OPTIONS[1])
fe6 = fresh_of(s, l1=float(en_l1), n_fresh=C.FRESH_OPTIONS[1])
st.plotly_chart(build_compare(fl6, fe6, float(en_l1)), width="stretch", key="compare_chart")
kind, text = ev.verdict_elastic(fl6, fe6, s.scen, float(en_l1))
getattr(st, kind)(text)
hl = ev.E.unique_solution(*ev.E.hand_data(False), ev.F(2))
he = ev.E.unique_solution(*ev.E.hand_data(False), ev.F(2), ev.F(1, 2))
st.caption(f"Das Handbeispiel (Experiment 1) bei α = 2: Lasso ({fr(hl[0])}; {fr(hl[1])}), Elastic Net mit ℓ = 1/2 ({fr(he[0])}; {fr(he[1])}): der Gruppeneffekt in der kleinsten Form. Ridge (immer alle Merkmale, nie eine Null) und die Wahl von λ: "
           f"{L('tikhonov-demo')}, {L('parameterwahl-demo')}.")

st.markdown("---")

# --- Grenzen --------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wo es weitergeht |
|---|---|---|
| **α ist fest.** Die Auswahlzahlen gelten für ein vorgegebenes α, nicht für ein je Stichprobe per Kreuzvalidierung gewähltes. | Die Wahl von α ist selbst verrauscht; hier nicht gemessen. Ein α nahe dem Kreuzvalidierungswert zielt auf Vorhersage und wählt viele Attrappen. | {L('parameterwahl-demo')} (λ-Wahl) |
| **Die vorgerechneten Zählungen gelten für n = 100, p = 10, ρ = 0,95, σ = 2.** | Andere Reglerstellungen rechnet die App live mit kleinerer Wiederholungszahl; die README nennt nur die vorgerechneten Zellen. | – |
| **Ein klar stärkeres Merkmal (Szenario S2).** | Dort stimmen Bootstrap und frische Stichproben überein: die Falle gilt, wenn zwei Merkmale gleich stark sind. Die Attrappe x₂ wird auch dann in etwa der Hälfte der Stichproben mitgewählt. | – |
| **Nur kontinuierliche Merkmale.** | Redundante Dummy-Spalten (Rang kleiner als p) sind strukturelle Kollinearität. | {L('faktoren-codierung-demo')} |
| **Tests und Intervalle nach der Auswahl.** | Ohne Korrektur sind sie nach einer Lasso-Auswahl nicht gültig (nach den gelesenen Quellen, hier **nicht** gemessen). | {L('modellwahl-demo')} |
| **Gaußsche Kleinste Quadrate.** | Gestraftes Logit, Zähldaten und andere Verteilungen sind nicht Teil dieser Demo. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $n$ Container mit zentrierter Antwort $y \in \mathbb{R}^n$ (Verweilzeit in Tagen) und $p$ Merkmalen (Spalten von $X$); gesucht sind Koeffizienten $b$, ihre Unsicherheit und, beim Lasso, welche Koeffizienten null sind.

| Symbol | Bedeutung |
|---|---|
| $n$, $p$ | Zahl der Container; Zahl der Merkmale **ohne** Achsenabschnitt |
| $x_j$, $b_j$ | $j$-tes Merkmal (Spalte von $X$) und sein Koeffizient |
| $\rho$, $\hat r$ | Korrelation von $x_1$ und $x_2$ in der Bevölkerung; in der Stichprobe (Stichprobenkorrelation) |
| $\sigma$ | Standardabweichung des Fehlers |
| $G = X'X/n$, $c = X'y/n$ | Gram-Matrix und Kovarianz der Merkmale mit $y$ (Merkmale standardisiert) |
| $\alpha$, $\ell$ | Stärke der Strafe; Mischung ($\ell = 1$ Lasso, $\ell = 0$ Ridge) |
| $S(z,\gamma)$ | weiche Schwelle $\operatorname{sign}(z)\,(\lvert z\rvert-\gamma)_+$ |
| $q$ | Größe der Bedingung von Zhao/Yu |

**Varianzaufblähung.** Mit $V = (X'X)^{-1}$ und zwei zentrierten Spalten gleicher Länge gilt für die Kleinste-Quadrate-Koeffizienten (Fehler $\mathrm{N}(0,\sigma^2)$):

$$\operatorname{Var}(b_1) = \frac{\sigma^2}{n\,(1-\hat r^2)}, \qquad \operatorname{Corr}(b_1, b_2) = -\hat r .$$

$$\operatorname{Var}(b_1 + b_2) = \frac{2\sigma^2}{n\,(1+\hat r)}, \qquad \operatorname{Var}(b_1 - b_2) = \frac{2\sigma^2}{n\,(1-\hat r)} .$$

Der **VIF** ist $\mathrm{VIF}_j = 1/(1-R_j^2)$ mit $R_j^2$ der Hilfsregression von $x_j$ auf die übrigen Merkmale; bei zwei Merkmalen ist $R_j^2 = \hat r^2$. VIF und Kondition von $X$ sind verschiedene Größen (der VIF hängt nicht von der Skala der Spalten ab).

**Tests.** Der Einzeltest für $b_j$ ist ein nichtzentrales $t$ mit Nichtzentralität $\beta_j/\mathrm{SE}_j$ und $n-3$ Freiheitsgraden (mit Achsenabschnitt); der gemeinsame Test von $\beta = 0$ ein nichtzentrales $F(2, n-3)$ mit Nichtzentralität $\beta' X'X \beta/\sigma^2$.

**Lasso und Elastic Net.** Konvention von scikit-learn und glmnet (Faktor $\tfrac{1}{2n}$, $\alpha$ = glmnet-$\lambda$):

$$\min_b \; \frac{1}{2n}\,\lVert y - Xb\rVert_2^2 + \alpha\Bigl(\ell\,\lVert b\rVert_1 + \frac{1-\ell}{2}\,\lVert b\rVert_2^2\Bigr).$$

Koordinatenabstieg auf $(G, c)$:

$$b_j \leftarrow \frac{S\bigl(c_j - \sum_{k\ne j} G_{jk} b_k,\; \alpha\ell\bigr)}{G_{jj} + \alpha(1-\ell)} .$$

Optimalität: $g = c - Gb$ erfüllt $g_j = \alpha\ell\,\operatorname{sign}(b_j) + \alpha(1-\ell)\,b_j$ für $b_j \ne 0$ und $|g_j| \le \alpha\ell$ sonst. Im orthonormalen Fall ($G = I$) ist die Lösung die weiche Schwelle $S(c_j,\alpha\ell)/(1+\alpha(1-\ell))$; $\alpha_{\max} = \max_j |c_j|/\ell$.

**Bedingung von Zhao/Yu.** Mit der Korrelationsmatrix $C$ (Index 1 wahre Merkmale, 2 Nichtprädiktoren) und $\beta_1$ der wahren Koeffizienten:

$$q = \bigl\lvert C_{21}\, C_{11}^{-1}\, \operatorname{sign}(\beta_1) \bigr\rvert_\infty < 1 .$$

Das ist nach Zhao/Yu (2006) „fast notwendig und hinreichend“ für Auswahlkonsistenz des Lasso. **Stabilitätsauswahl** (Meinshausen/Bühlmann 2010) zählt Häufigkeiten über Teilstichproben der Größe $n/2$; ihre Schranke für die erwartete Zahl falsch gewählter Merkmale $E(V) \le q^2/((2\pi-1)p)$ setzt Austauschbarkeit voraus.

**Konventionen im Vergleich.** Konventionstabelle (die Form der Strafe ist je Quelle verschieden; ESL schreibt den Faktor $\tfrac12$ in die Lagrange-Form, Tibshirani 1996 die Nebenbedingungsform mit standardisierten Merkmalen):

| Größe | diese Demo | Hinweis |
|---|---|---|
| Faktor der Quadratsumme | $1/(2n)$ | wie scikit-learn und glmnet (glmnet-$\lambda$ = $\alpha$ hier) |
| Standardisierung | je Anpassung, Teiler $n$ | ohne sie ändert Skalieren einer Spalte das Lasso |
| $p$ | Merkmale ohne Achsenabschnitt | in Demos zu Kleinste Quadrate mit Achsenabschnitt zählt $p$ eine Spalte mehr |
| Korrelation | Stichprobe bei Rechnungen an einer Stichprobe, Bevölkerung bei Formeln im Mittel | der VIF der Stichprobe streut um die Formel mit der Bevölkerungskorrelation |

**Literatur** (nach Quellenberichten zitiert): James u. a., *An Introduction to Statistical Learning*, 2. Aufl., Abschn. 3.3.3, S. 102 (VIF). Hastie/Tibshirani/Friedman, *The Elements of Statistical Learning*, Abschn. 3.2 (S. 54 f.), 3.4.2 und Kapitel 18 (S. 662).
Tibshirani 1996, *Regression Shrinkage and Selection via the Lasso*, JRSS B 58. Friedman/Hastie/Tibshirani 2010, *Regularization Paths for Generalized Linear Models via Coordinate Descent*, J. Stat. Softw. 33(1), Gl. 5 und 6. Zhao/Yu 2006, *On Model Selection Consistency of Lasso*, JMLR 7, 2541–2563.
Meinshausen/Bühlmann 2010, *Stability Selection*, JRSS B 72.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)

sync_query_params({k: st.session_state[k] for k in SETTING_SPECS})
