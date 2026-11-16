"""Plotly-Figuren (alle Achsen fest, `fixedrange`, damit Touch-Scrollen nicht am Diagramm hängen bleibt)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import kol_constants as C

LAYOUT = dict(height=C.CHART_HEIGHT, margin=dict(l=50, r=20, t=40, b=50), legend=dict(orientation="h", y=-0.22))
LAYOUT_NOLEGEND = dict(height=C.CHART_HEIGHT, margin=dict(l=50, r=20, t=40, b=50), showlegend=False)


def _fix(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _feature(j):
    return f"x{j + 1}"


def build_sample(smp):
    """Eine Stichprobe: wahres β, Kleinste Quadrate (mit ±1 Standardfehler) und Lasso je Merkmal (standardisierte Skala; β gilt für die Szenario-Merkmale, im Export-Zweig nur als Bezug)."""
    p = len(smp["beta"])
    xs = [_feature(j) for j in range(p)]
    fig = go.Figure()
    fig.add_bar(x=xs, y=smp["beta"], name="wahres β", marker_color="rgba(150,150,150,0.55)")
    if smp["ols"] is not None:
        fig.add_scatter(x=xs, y=smp["ols"], mode="markers", name="Kleinste Quadrate (± 1 Standardfehler)", marker=dict(color=C.COLOR_X2, size=9),
                        error_y=dict(type="data", array=smp["ols_se"], visible=True))
    fig.add_scatter(x=xs, y=smp["lasso"], mode="markers", name="Lasso", marker=dict(color=C.COLOR_X1, size=11, symbol="diamond"))
    fig.update_layout(title="Eine Stichprobe: wahres β, Kleinste Quadrate und Lasso", yaxis_title="Koeffizient (standardisierte Merkmale)", **LAYOUT)
    return _fix(fig)


def build_hand_path(h):
    """Pfad der Koeffizienten des Handbeispiels über α (Brüche, als Zahlen gezeichnet); eine Spalte je y."""
    grid = [float(a) for a in h["grid"]]
    path = np.array([[float(v) for v in b] for b in h["path"]])
    fig = go.Figure()
    fig.add_scatter(x=grid, y=path[:, 0], name="b₁", line=dict(color=C.COLOR_X1, width=3))
    fig.add_scatter(x=grid, y=path[:, 1], name="b₂", line=dict(color=C.COLOR_X2, width=3))
    fig.add_vline(x=float(h["alpha"]), line_dash="dash", line_color="grey")
    fig.add_vline(x=float(h["first"]), line_dash="dot", line_color=C.COLOR_X1, annotation_text="erstes Merkmal tritt ein", annotation_position="top left")
    fig.add_vline(x=float(h["second"]), line_dash="dot", line_color=C.COLOR_X2, annotation_text="zweites tritt ein", annotation_position="top left")
    fig.update_layout(xaxis_title="α", yaxis_title="Koeffizient", title="Lasso-Pfad der vier Container", **LAYOUT)
    fig.update_xaxes(autorange="reversed")
    return _fix(fig)


def build_hand_plane(h, ex, ey):
    """Die Koeffizientenebene: Höhenlinie der Quadratsumme (lang entlang der Gegendiagonalen), Kleinste Quadrate und Lasso-Pfad."""
    path = np.array([[float(v) for v in b] for b in h["path"]])
    b = [float(v) for v in h["rep"]["b"]]
    fig = go.Figure()
    fig.add_scatter(x=ex, y=ey, mode="lines", name="Höhenlinie der Quadratsumme (ein s² über dem Minimum)", line=dict(color=C.COLOR_LINE))
    fig.add_scatter(x=path[:, 0], y=path[:, 1], mode="lines", name="Lasso-Pfad", line=dict(color=C.COLOR_X1, width=3))
    fig.add_scatter(x=[b[0]], y=[b[1]], mode="markers", name="Kleinste Quadrate", marker=dict(color=C.COLOR_X2, size=12))
    lx, ly = float(h["lasso"][0]), float(h["lasso"][1])
    fig.add_scatter(x=[lx], y=[ly], mode="markers", name="Lasso bei diesem α", marker=dict(color=C.COLOR_X1, size=12, symbol="diamond"))
    fig.update_layout(xaxis_title="b₁", yaxis_title="b₂", title="Koeffizientenebene", **LAYOUT)
    return _fix(fig)


def build_rates(v):
    """Anteile der Wiederholungen, in denen verworfen wird (Simulation als Balken, exakte nichtzentrale Rechnung als Raute)."""
    labels = ["Einzeltest x₁", "beide Einzeltests", "gemeinsamer F-Test", "F-Test ohne jeden Einzeltest"]
    sim = [v["p_single"], v["p_both_single"], v["p_F"], v["p_F_no_single"]]
    fig = go.Figure()
    fig.add_bar(x=labels, y=[100 * x for x in sim], name="Simulation", marker_color=C.COLOR_X1, text=[C.pct(x) for x in sim], textposition="outside")
    fig.add_scatter(x=[labels[0], labels[2]], y=[100 * v["p_single_exact"], 100 * v["p_F_exact"]], mode="markers", name="exakt (nichtzentrale Verteilung)",
                    marker=dict(color=C.COLOR_X2, size=13, symbol="diamond"))
    fig.update_layout(yaxis_title="verworfen in % der Wiederholungen", yaxis_range=[0, 112], title=f"Tests bei ρ = {C.pretty(v['rho'])}, β = ({C.pretty(v['b'], 1)}; {C.pretty(v['b'], 1)})", **LAYOUT)
    return _fix(fig)


def build_vif_hist(vifs, formula):
    fig = go.Figure()
    fig.add_histogram(x=vifs, nbinsx=40, marker_color=C.COLOR_LINE, name="VIF der Stichprobe")
    fig.add_vline(x=formula, line_dash="dash", line_color=C.COLOR_X2, annotation_text="Formel 1/(1 − ρ²)", annotation_position="top right")
    fig.update_layout(xaxis_title="VIF der Stichprobe", yaxis_title="Zahl der Designs", title="Der VIF der Stichprobe streut um die Formel mit der Bevölkerungskorrelation", **LAYOUT_NOLEGEND)
    return _fix(fig)


def build_path(p, s_path_thr):
    """Koeffizientenpfad über α/α_max (logarithmisch); x₁ und x₂ farbig, x₃ dunkel, die Attrappen grau."""
    rel = p["alphas"] / p["alpha_max"]
    fig = go.Figure()
    for j in range(p["b"].shape[1]):
        color = C.COLOR_X1 if j == 0 else C.COLOR_X2 if j == 1 else C.COLOR_TRUE if j == 2 else C.COLOR_DUMMY
        fig.add_scatter(x=rel, y=p["b"][:, j], mode="lines", name=_feature(j), line=dict(color=color, width=3 if j < 3 else 1.5), showlegend=j < 3 or j == 3)
    fig.add_vline(x=p["alpha_sel"] / p["alpha_max"], line_dash="dash", line_color="grey")
    fig.update_xaxes(type="log", autorange="reversed", title="α / α_max")
    fig.update_layout(yaxis_title="Koeffizient (standardisiert)", title="Koeffizientenpfad", **LAYOUT)
    return _fix(fig)


def build_selection(f):
    labels = ["x₁", "x₂", "beide", "nur x₁", "nur x₂", "Attrappen x₄–x₁₀ (Mittel)"]
    vals = [f["p1"], f["p2"], f["both"], f["only1"], f["only2"], f["att"]]
    colors = [C.COLOR_X1, C.COLOR_X2, C.COLOR_LINE, C.COLOR_X1, C.COLOR_X2, C.COLOR_DUMMY]
    fig = go.Figure()
    fig.add_bar(x=labels, y=[100 * v for v in vals], marker_color=colors, text=[C.pct(v) for v in vals], textposition="outside")
    fig.update_layout(yaxis_title="gewählt in % der frischen Stichproben", yaxis_range=[0, 112], title="Auswahlhäufigkeit über frische Stichproben", **LAYOUT_NOLEGEND)
    return _fix(fig)


def build_originals(po, fresh):
    """Ein Punkt je Originalstichprobe: Bootstrap-Häufigkeit von x₁ gegen die von x₂; das Kreuz markiert die Häufigkeiten über frische Stichproben."""
    fig = go.Figure()
    fig.add_scatter(x=po["p1"], y=po["p2"], mode="markers", name="eine Originalstichprobe", marker=dict(color=C.COLOR_LINE, size=9, opacity=0.75))
    fig.add_scatter(x=[fresh["p1"]], y=[fresh["p2"]], mode="markers", name="frische Stichproben", marker=dict(color=C.COLOR_TRUE, size=16, symbol="x"))
    fig.update_layout(xaxis_title="Häufigkeit von x₁ im Bootstrap dieser Stichprobe", yaxis_title="Häufigkeit von x₂", xaxis_range=[-0.02, 1.02], yaxis_range=[-0.02, 1.02],
                      title="Jede Originalstichprobe ein Punkt", **LAYOUT)
    return _fix(fig)


def build_compare(fl, fe, l1_en):
    labels = ["x₁", "x₂", "beide", "Attrappen x₄–x₁₀ (Mittel)"]
    keys = ["p1", "p2", "both", "att"]
    fig = go.Figure()
    fig.add_bar(x=labels, y=[100 * fl[k] for k in keys], name="Lasso (ℓ = 1)", marker_color=C.COLOR_X1, text=[C.pct(fl[k]) for k in keys], textposition="outside")
    fig.add_bar(x=labels, y=[100 * fe[k] for k in keys], name=f"Elastic Net (ℓ = {C.pretty(l1_en, 2)})", marker_color=C.COLOR_X2, text=[C.pct(fe[k]) for k in keys], textposition="outside")
    fig.update_layout(barmode="group", yaxis_range=[0, 115], yaxis_title="gewählt in % der frischen Stichproben", title="Lasso gegen Elastic Net", **LAYOUT)
    return _fix(fig)
