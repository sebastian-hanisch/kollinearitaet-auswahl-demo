"""Auszug aus dem Containergenerator der `verweilzeit-demo` (`vwz_generator.log_median_and_s`, `vwz_constants`): nur der Export-Zweig, ohne Imports aus dem anderen Repo.

Herkunft: `verweilzeit-demo/vwz_generator.py` (Stand 2026-11-16). Dort ist die Verweilzeit eines Containers lognormal, `log D ~ N(μ, s²)`. Für Export-Container (Art 1) gilt
    Median = 0,6 + 0,85·v      (v = Vorlauf in Tagen bis zur geplanten Schiffsabfahrt, gleichverteilt 1 bis 10),
    μ = ln(Median) + u_k·0,3   (u_k = Effekt des Kunden k aus zwölf Kunden, auf ein Drittel gedämpft),
    s = 0,25.
Die zwölf Kundeneffekte u_k sind dort `numpy.random.default_rng(12345).normal(0, 0,30, 12)`; hier als Zahlen eingefroren (sie hängen sonst am numpy-Zufallsstrom).
Der Mittelwert von D ist exakt linear in v, die Standardabweichung wächst mit dem Mittel (Verhältnis 0,279). Alles im Generator ist erfunden (kein Echtdatenbezug).
Die Exponentialfunktion kommt aus `kol_stream.exp` (nur Grundrechenarten), nicht aus der libm."""

import numpy as np

import kol_stream as S

KUNDEN_EFFEKTE = (-0.42714751093638936, 0.3791185374387331, -0.2611985213877257, -0.07775197048031927, -0.02260299210315629, -0.2222653956256827,
                  -0.410337810534883, 0.19466784065791196, 0.10831743391646849, -0.5858589189036569, 0.7042228963136555, 0.2905490717255771)
K_KUNDEN = len(KUNDEN_EFFEKTE)
U_SCALE_EXPORT = 0.3      # Kundeneffekt bei Export: ein Drittel
S_BASE_EXPORT = 0.25      # Streuung der log-Verweilzeit bei Export
VORLAUF_LO, VORLAUF_HI = 1.0, 10.0


def export_median(v):
    """Median der Verweilzeit eines Export-Containers in Tagen."""
    return 0.6 + 0.85 * np.asarray(v, dtype=float)


def export_dwell(v, kunde, z, noise_scale=1.0):
    """Verweilzeit D = Median · exp(u_k·0,3 + s·z) mit z ~ N(0,1) (vektorisiert); `noise_scale` skaliert s wie im Generator."""
    u = np.take(np.asarray(KUNDEN_EFFEKTE), kunde) * U_SCALE_EXPORT
    return export_median(v) * S.exp(u + S_BASE_EXPORT * noise_scale * z)


def export_mean_factor(noise_scale=1.0):
    """E[D] / Median, gemittelt über die zwölf Kunden: exp(s²/2)·Mittel_k exp(u_k·0,3) (der Mittelwert ist linear im Vorlauf)."""
    s = S_BASE_EXPORT * noise_scale
    u = np.asarray(KUNDEN_EFFEKTE) * U_SCALE_EXPORT
    return float(np.exp(s * s / 2.0) * np.mean(np.exp(u)))
