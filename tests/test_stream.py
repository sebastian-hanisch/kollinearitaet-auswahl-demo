"""Zufall: die gestapelten SplitMix64-Folgen und die Normalzahlen ohne C-Bibliothek (`kol_stream`).

Plattform: die Ganzzahlfolge ist exakt dieselbe wie die von `kol_rng.SplitMix64`; die Normalzahlen und die Hilfsfunktionen `ln`/`exp` benutzen nur +, -, *, /, sqrt, frexp und ldexp
(IEEE 754 exakt gerundet), darum gelten die Bitwerte unten auf Windows und Linux gleich. Die Genauigkeit wird gegen die libm und gegen `statistics.NormalDist` geprüft (Band)."""

import math
from statistics import NormalDist

import numpy as np
import pytest
from scipy import stats

import kol_stream as S
from kol_rng import MASK, SplitMix64


@pytest.mark.parametrize("seed", [0, 1, 12345, 2 ** 63 + 5, MASK])
def test_integer_stream_equals_the_reference_class(seed):
    sm = SplitMix64(seed)
    ints = [sm.next_u64() for _ in range(25)]
    assert [int(v) for v in S.u64_block([seed], 25)[0]] == ints


def test_bit_exact_reference_values():
    """Bitwerte (hex) von Normalzahlen, ln und exp: auf jeder Plattform gleich, weil nur exakt gerundete Grundoperationen vorkommen."""
    z = S.normals(np.array([12345], dtype=np.uint64), 4)[0]
    assert [float(v).hex() for v in z] == ["-0x1.1ca8cd7d56d4ap+0", "-0x1.a629ffa35a35fp-1", "-0x1.2d61fe05cfc9dp+0", "-0x1.dc4b423422d21p-1"]
    assert [float(v).hex() for v in S.ln(np.array([0.3, 7.0, 1e-5]))] == ["-0x1.34378fcbda721p+0", "0x1.f2272ae325a57p+0", "-0x1.7069e2aa2aa5bp+3"]
    assert [float(v).hex() for v in S.exp(np.array([0.3, -2.5, 11.0]))] == ["0x1.599058c8c1a95p+0", "0x1.50385c094f425p-4", "0x1.d3c4488ee4f7ep+15"]
    assert int(S.child_seeds(7, 3, 2)[1]) == 9639466228269566187


def test_ln_and_exp_agree_with_the_math_library_to_the_last_digits():
    x = np.array([1e-300, 1e-10, 0.1, 0.5, 0.7071067811865476, 1.0, 1.5, 2.0, 10.0, 12345.678, 1e100])
    got, want = S.ln(x), np.log(x)
    assert np.all(np.abs(got - want) <= 4e-16 * np.maximum(np.abs(want), 1.0))
    y = np.array([-700.0, -50.0, -1.0, -0.3, 0.0, 0.3, 1.0, 2.5, 10.0, 50.0, 700.0])
    assert np.all(np.abs(S.exp(y) - np.exp(y)) <= 1e-15 * np.exp(y))


def test_normal_quantile_matches_the_reference_inverse_cdf():
    p = np.concatenate([np.linspace(1e-12, 1 - 1e-12, 4001), [2.0 ** -54, 1 - 2.0 ** -53, 0.5, 0.075, 0.925]])
    nd = NormalDist()
    want = np.array([nd.inv_cdf(float(v)) for v in p])
    assert np.max(np.abs(S.norm_ppf(p) - want)) < 5e-15


def test_normals_have_the_right_moments_and_distribution():
    z = S.normals(S.child_seeds(1, 1, 200), 1000).ravel()          # 200 000 Zahlen
    assert abs(z.mean()) < 0.012 and abs(z.std() - 1.0) < 0.01
    assert abs(float(stats.skew(z))) < 0.03 and abs(float(stats.kurtosis(z))) < 0.06
    assert stats.kstest(z, "norm").pvalue > 0.001


def test_a_row_does_not_depend_on_the_batch():
    seeds = S.child_seeds(5, 9, 12)
    assert np.array_equal(S.normals(seeds[:5], 30), S.normals(seeds, 30)[:5])


def test_child_seeds_are_a_prefix_stream():
    assert np.array_equal(S.child_seeds(1, 2, 5), S.child_seeds(1, 2, 12)[:5])
    assert not np.array_equal(S.child_seeds(1, 2, 5), S.child_seeds(1, 3, 5))
    assert not np.array_equal(S.child_seeds(1, 2, 5), S.child_seeds(2, 2, 5))


def test_open_uniform_never_hits_zero_or_one_and_below_is_in_range_and_uniform():
    u64 = S.u64_block(S.child_seeds(3, 3, 50), 400)
    u = S.uniform_open(u64)
    assert 0.0 < u.min() and u.max() < 1.0
    edge = S.uniform_open(np.array([0, MASK], dtype=np.uint64))
    assert 0.0 < edge[0] and edge[1] < 1.0 and np.all(np.isfinite(S.norm_ppf(edge)))     # auch die äußersten Ganzzahlen geben endliche Normalzahlen
    idx = S.below(u64, 7)
    assert idx.min() == 0 and idx.max() == 6
    counts = np.bincount(idx.ravel(), minlength=7)
    expected = idx.size / 7
    assert np.all(np.abs(counts - expected) < 5 * math.sqrt(expected))


def test_the_reference_class_normal_has_the_same_distribution_as_the_stacked_normal():
    """Zwei verschiedene Verfahren (Box-Muller mit Dezimalrechnung, Umkehrung AS 241): beide sind standardnormal (Mittel, Streuung, Kolmogorov-Smirnov)."""
    sm = SplitMix64(99)
    ref = np.array([sm.normal() for _ in range(6000)])
    new = S.normals(S.child_seeds(99, 1, 1), 6000)[0]
    assert abs(ref.mean()) < 0.06 and abs(new.mean()) < 0.06 and abs(ref.std() - new.std()) < 0.05
    assert stats.kstest(ref, "norm").pvalue > 0.001 and stats.kstest(new, "norm").pvalue > 0.001
