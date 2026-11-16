"""Konstanten: Regler-Grenzen, feste Annahmen, Szenarien, Farben und Presets."""

import math

GITHUB = "https://github.com/sebastian-hanisch/"   # Nachbar-Demos: GitHub-Links werden aus dem Namen gebaut


def repo_link(name):
    """Markdown-Link auf ein Demo-Repo."""
    return f"[{name}]({GITHUB}{name})"


P_FEATURES = 10                 # Merkmale x1 bis x10 (ohne Achsenabschnitt)
RHO_OPTIONS = (0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999)
N_OPTIONS = (30, 50, 100, 200, 400)
SIGMA_OPTIONS = (1.0, 2.0, 3.0, 4.0)
ALPHA_OPTIONS = (0.03, 0.045, 0.06, 0.09, 0.12, 0.18, 0.25, 0.35, 0.5)
L1_OPTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)          # ℓ der Seitenleiste: 1 = Lasso
EN_L1_OPTIONS = (0.1, 0.25, 0.5, 0.75, 0.9)       # ℓ des Vergleichs in Experiment 6
SCENARIOS = ("S3", "S1", "S2", "S2E")
WORLD_OF = {"S1": "C", "S3": "C", "S2": "C", "S2E": "E"}   # Messwelt je Szenario (C Container-Welt, E Export-Zweig; die Gauß-Welt G dient nur den Tests)
SCENARIO_LABELS = {
    "S3": "S3: x1 und x2 wirken beide schwach (β = 0,5; 0,5; x3: 1)",
    "S1": "S1: x1 und x2 wirken beide stark (β = 1; 1; x3: 1)",
    "S2": "S2: nur x1 wirkt (β = 2), x2 ist eine Attrappe",
    "S2E": "S2 im Export-Zweig des Generators (Verweilzeit lognormal)",
}
BETAS = {
    "S1": (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "S3": (0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "S2": (2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
}
THRESHOLD_OPTIONS = (0.0, 1e-8, 1e-4, 1e-2)       # „Nichtnull“-Schwelle in Experiment 3
ALPHA_REL_OPTIONS = (1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001)   # α/α_max auf dem Pfad
HAND_ALPHA_OPTIONS = ("0", "1/4", "1/2", "1", "3/2", "2", "5/2", "3", "7/2")                  # α des Handbeispiels (Brüche)
VIF_B_OPTIONS = (0.5, 1.0)                        # Signal b: β = (b, b) in Experiment 2
VIF_REPS_MIN, VIF_REPS_MAX, VIF_REPS_STEP = 500, 5000, 500   # Wiederholungen der Simulation in Experiment 2
FRESH_OPTIONS = (200, 500, 1000)                  # frische Stichproben in Experiment 4
SCHEME_OPTIONS = ("Bootstrap (Größe n)", "Teilstichproben (Größe n/2)")
PI_OPTIONS = (0.6, 0.7, 0.8, 0.9)                 # Schwelle „gilt als stabil“
M_ORIG_OPTIONS = (10, 20, 40)                     # Originalstichproben in Experiment 5
B_RESAMPLES = 100                                 # Bootstrap- bzw. Teilstichproben je Originalstichprobe
SEED_MAX, DEFAULT_SEED = 999999, 1
NZ_THRESHOLD = 1e-8                               # Auswahl = Koeffizient mit Betrag über dieser Schwelle (der Abstieg liefert exakte Nullen)
CLEAR_HI, CLEAR_LO = 0.75, 0.25                   # „klar“: Anteil von x1 bei genau einem über 0,75 oder unter 0,25
RHO_ONE = 1.0

DEFAULTS = dict(
    scen="S3", rho=0.95, n=100, sigma=2.0, alpha=0.09, l1=1.0, seed=DEFAULT_SEED,
    hand_swap=False, hand_alpha="2",
    vif_b=0.5, vif_reps=2000,
    path_rel=0.1, path_thr=1e-8,
    fresh=500,
    scheme=SCHEME_OPTIONS[0], pi=0.8, m_orig=20,
    en_l1=0.5,
)

COLOR_X1 = "#2F6B65"
COLOR_X2 = "#D2691E"
COLOR_DUMMY = "rgba(130,130,130,0.8)"
COLOR_LINE = "#4c78a8"
COLOR_TRUE = "#b2182b"
CHART_HEIGHT = 380

PRESETS = {     # jedes Preset trägt seinen eigenen Zufalls-Seed: so sind die Presets verschieden, auch wenn zwei dasselbe Szenario laden
    "Von Hand (vier Container)": {"seed": 1},
    "Zwei Angaben, gleich schwach (S3)": {"scen": "S3", "seed": 31},
    "Zwei Angaben, gleich stark (S1)": {"scen": "S1", "seed": 32},
    "Nur der Vorlauf wirkt (S2, echter Generator)": {"scen": "S2E", "seed": 33},
    "Elastic Net: Preis des Gruppeneffekts": {"scen": "S2E", "l1": 0.5, "seed": 34},
    "Starke Kollinearität (ρ = 0,99)": {"scen": "S3", "rho": 0.99, "seed": 35},
}
PRESET_HELP = {
    "Von Hand (vier Container)": "Das Handbeispiel mit vier erfundenen Containern, Brüche statt Messwerte; die übrigen Einstellungen auf Standard (S3, ρ = 0,95, n = 100, σ = 2, α = 0,09, Lasso).",
    "Zwei Angaben, gleich schwach (S3)": "Zwei Merkmale mit Korrelation 0,95, beide mit β = 0,5 und ein drittes mit β = 1: hier wählt das Lasso je Stichprobe eines von beiden.",
    "Zwei Angaben, gleich stark (S1)": "Beide Zwillinge mit β = 1: das Lasso wählt meist beide, die Auswahl der Bootstrap-Häufigkeiten bleibt trotzdem je Stichprobe entschieden.",
    "Nur der Vorlauf wirkt (S2, echter Generator)": "Antwort ist die lognormale Verweilzeit der Export-Container; der Vorlauf wirkt, die zweite Vorlaufangabe ist eine erfundene Kopie mit Korrelation 0,95.",
    "Elastic Net: Preis des Gruppeneffekts": "Wie S2E, aber mit ℓ = 0,5: das Elastic Net nimmt die Kopie des wirkenden Merkmals mit.",
    "Starke Kollinearität (ρ = 0,99)": "Wie S3, aber ρ = 0,99: die Einzelkoeffizienten sind noch unsicherer, die Summe bleibt genau.",
}


def pretty(x, digits=3):
    """Deutsche Zahl mit Dezimalkomma."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x, digits=1):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100.0 * x:.{digits}f} %".replace(".", ",")
