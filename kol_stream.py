"""Gestapelter Zufall ohne C-Bibliothek: viele SplitMix64-Folgen auf einmal, Normalzahlen nur mit den vier Grundrechenarten und Wurzel.

Warum es dieses Modul neben `kol_rng.py` gibt: `kol_rng.SplitMix64.normal()` ist auf jeder Plattform bitgleich, kostet aber 78 bis 143 µs je Zahl (Dezimalrechnung);
die Messreihen dieser Demo brauchen Millionen Normalzahlen. Hier laufen die Zähler vieler Folgen gleichzeitig in numpy (vorzeichenlose 64-Bit-Ganzzahlen, Überlauf
gewollt) und die Normalzahl kommt aus dem Umkehrverfahren AS 241 (Wichura 1988, PPND16, Genauigkeit etwa 1e-16) mit eigenem Logarithmus aus einer atanh-Reihe.
`+`, `-`, `*`, `/` und `sqrt` sind in IEEE 754 exakt gerundet, `frexp`/`ldexp` sind exakt: das Ergebnis hängt weder von der libm noch von SIMD-Breiten ab.
Die Ganzzahlfolge einer Folge ist **dieselbe** wie die von `kol_rng.SplitMix64` (Test); die Normalzahlen sind es nicht (anderes Verfahren: Umkehrung statt Box-Muller).

Eine Messzelle bekommt aus ihrem Zufalls-Seed und einer Marke (`tag`) einen Hauptstrom; dessen k-te Ausgabe ist der Zufalls-Seed der k-ten Stichprobe. Jede
Stichprobe zieht ihre Zahlen aus ihrer eigenen Folge: das Ergebnis hängt nicht davon ab, in welchen Stapeln gerechnet wird."""

import numpy as np

from kol_rng import MASK, SplitMix64

_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_C1 = np.uint64(0xBF58476D1CE4E5B9)
_C2 = np.uint64(0x94D049BB133111EB)
_S30, _S27, _S31, _S11, _S12 = np.uint64(30), np.uint64(27), np.uint64(31), np.uint64(11), np.uint64(12)
TWO52 = float(1 << 52)

LN2_HI = 0.693147180369123816490
LN2_LO = 1.90821492927058770002e-10
SQRT_HALF = 0.7071067811865476


def mix64(state):
    """Ausgabefunktion von SplitMix64 auf einen (Feld von) Zählerstand."""
    z = np.asarray(state, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = (z ^ (z >> _S30)) * _C1
        z = (z ^ (z >> _S27)) * _C2
    return z ^ (z >> _S31)


def u64_block(seeds, count):
    """Für jeden Zufalls-Seed die ersten `count` Ausgaben von SplitMix64: Feld (K, count) vom Typ uint64. Gleich `SplitMix64(seed).next_u64()` der Reihe nach."""
    seeds = np.atleast_1d(np.asarray(seeds, dtype=np.uint64))
    steps = np.arange(1, count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        state = seeds[:, None] + steps[None, :] * _GAMMA
    return mix64(state)


def child_seeds(seed: int, tag: int, count: int):
    """`count` Zufalls-Seeds (uint64) aus dem Hauptstrom von (seed, tag)."""
    master = SplitMix64(((int(seed) & MASK) * 0x9E3779B97F4A7C15 + (int(tag) & MASK) * 0xD1B54A32D192ED03 + 0x632BE59BD9B4E019) & MASK).next_u64()
    return u64_block(np.array([master], dtype=np.uint64), count)[0]


def uniform_open(u64):
    """Gleichverteilt in (0, 1), nie 0 oder 1: (52-Bit-Ganzzahl + ½) / 2^52 (mit 53 Bit wäre 1 - 2^-54 nicht darstellbar und der Wert 1,0 möglich); u und 1 - u sind exakt."""
    return ((u64 >> _S12).astype(np.float64) + 0.5) / TWO52


def below(u64, n: int):
    """Ganzzahlen 0..n-1 aus 53 Bit (Verzerrung höchstens n / 2^53); n ≤ 2^10 reicht für die Stichproben dieser Demo."""
    return (((u64 >> _S11) * np.uint64(n)) >> np.uint64(53)).astype(np.int64)


def ln(x):
    """Natürlicher Logarithmus für x > 0 nur mit Grundrechenarten (Reihe in s = (m-1)/(m+1)), Fehler unter 2e-16 relativ."""
    x = np.asarray(x, dtype=np.float64)
    m, e = np.frexp(x)
    small = m < SQRT_HALF
    m = np.where(small, m * 2.0, m)
    e = e.astype(np.float64) - small
    s = (m - 1.0) / (m + 1.0)
    s2 = s * s
    poly = np.full_like(s2, 1.0 / 25.0)
    for k in range(11, -1, -1):
        poly = poly * s2 + 1.0 / (2 * k + 1)
    # poly = 1 + s²/3 + s⁴/5 + ... + s²⁴/25 (13 Glieder; Rest unter 1e-19)
    return e * LN2_HI + (e * LN2_LO + 2.0 * s * poly)


def exp(x):
    """Exponentialfunktion für |x| bis etwa 700 nur mit Grundrechenarten: x = k ln2 + r, Reihe für e^r, dann ldexp (exakt)."""
    x = np.asarray(x, dtype=np.float64)
    k = np.rint(x / 0.6931471805599453)
    r = (x - k * LN2_HI) - k * LN2_LO
    term = np.ones_like(r)
    total = np.ones_like(r)
    for i in range(1, 19):
        term = term * r / i
        total = total + term
    return np.ldexp(total, k.astype(np.int64))


_A = (3.3871328727963666080, 1.3314166789178437745e2, 1.9715909503065514427e3, 1.3731693765509461125e4, 4.5921953931549871457e4,
      6.7265770927008700853e4, 3.3430575583588128105e4, 2.5090809287301226727e3)
_B = (1.0, 4.2313330701600911252e1, 6.8718700749205790830e2, 5.3941960214247511077e3, 2.1213794301586595867e4, 3.9307895800092710610e4,
      2.8729085735721942674e4, 5.2264952788528545610e3)
_C = (1.42343711074968357734, 4.63033784615654529590, 5.76949722146069140550, 3.64784832476320460504, 1.27045825245236838258,
      2.41780725177450611770e-1, 2.27238449892691845833e-2, 7.74545014278341407640e-4)
_D = (1.0, 2.05319162663775882187, 1.67638483018380384940, 6.89767334985100004550e-1, 1.48103976427480074590e-1,
      1.51986665636164571966e-2, 5.47593808499534494600e-4, 1.05075007164441684324e-9)
_E = (6.65790464350110377720, 5.46378491116411436990, 1.78482653991729133580, 2.96560571828504891230e-1, 2.65321895265761230930e-2,
      1.24266094738807843860e-3, 2.71155556874348757815e-5, 2.01033439929228813265e-7)
_F = (1.0, 5.99832206555887937690e-1, 1.36929880922735805310e-1, 1.48753612908506148525e-2, 7.86869131145613259100e-4,
      1.84631831751005468180e-5, 1.42151175831644588870e-7, 2.04426310338993978564e-15)


def _horner(coef_low_first, r):
    acc = np.full_like(r, coef_low_first[-1])
    for c in reversed(coef_low_first[:-1]):
        acc = acc * r + c
    return acc


def norm_ppf(p):
    """Quantilfunktion der Standardnormalverteilung (AS 241, PPND16) für p in (0, 1) ohne libm."""
    p = np.asarray(p, dtype=np.float64)
    q = p - 0.5
    # Mitte
    r1 = 0.180625 - q * q
    mid = q * _horner(_A, r1) / _horner(_B, r1)
    # Ränder
    pt = np.where(q < 0.0, p, 1.0 - p)
    r = np.sqrt(-ln(pt))
    near = r <= 5.0
    rn = np.where(near, r - 1.6, 0.0)
    rf = np.where(near, 0.0, r - 5.0)
    tail = np.where(near, _horner(_C, rn) / _horner(_D, rn), _horner(_E, rf) / _horner(_F, rf))
    tail = np.where(q < 0.0, -tail, tail)
    return np.where(np.abs(q) <= 0.425, mid, tail)


def normals(seeds, count):
    """Standardnormalzahlen: Feld (K, count), Zeile k aus der Folge des Zufalls-Seeds `seeds[k]`."""
    return norm_ppf(uniform_open(u64_block(seeds, count)))


def stream_normals(seed: int, tag: int, shape):
    """Eine einzige Folge (Zufalls-Seed, Marke) als Feld der Form `shape`."""
    size = int(np.prod(shape))
    return normals(child_seeds(seed, tag, 1), size).reshape(shape)
