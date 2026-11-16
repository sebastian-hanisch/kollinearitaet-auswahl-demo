"""Erzeugt data/kol_frozen.json: die vorgerechnete Messreihe der App (Kollinearität, Summe und Differenz, Einzel- gegen gemeinsamen Test, VIF-Streuung,
Auswahl über frische Stichproben, Bootstrap und Teilstichproben, Nichtnullen, Bedingung von Zhao/Yu).

Aufruf (aus dem Repo-Ordner):  python tools/kol_make_frozen.py            schreibt die Datei (Laufzeit einige Minuten)
                               python tools/kol_make_frozen.py --check    rechnet die Probe jeder Auswahlzelle neu und vergleicht mit der Datei (Band), Laufzeit etwa 10 s
                               python tools/kol_make_frozen.py --full     rechnet alles neu und vergleicht mit der Datei (Band; Laufzeit wie das Schreiben)"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import kol_frozen as FZ  # noqa: E402


def compare(a, b, path="", tol=1e-9, out=None):
    """Vergleicht zwei verschachtelte Strukturen: Gleitkommawerte mit relativer Toleranz `tol` (und absolut 1e-12), alles andere exakt. Gibt die Abweichungen zurück."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a.keys() | b.keys():
            if k not in a or k not in b:
                out.append(f"{path}/{k}: fehlt")
            else:
                compare(a[k], b[k], f"{path}/{k}", tol, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: Länge {len(a)} gegen {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                compare(x, y, f"{path}[{i}]", tol, out)
    elif isinstance(a, float) or isinstance(b, float):
        x, y = float(a), float(b)
        if not (x == y or (x != x and y != y) or abs(x - y) <= max(tol * max(abs(x), abs(y)), 1e-12)):
            out.append(f"{path}: {x!r} gegen {y!r}")
    elif a != b:
        out.append(f"{path}: {a!r} gegen {b!r}")
    return out


def main(argv):
    t0 = time.time()
    if "--check" in argv:
        old = FZ.load()["selection"]
        bad = []
        for key in FZ.CELLS:
            probe = FZ.run_cell(key, FZ.PROBE_M, FZ.PROBE_FRESH)
            bad += compare({k: v for k, v in probe.items()}, old[key]["probe"], key)
        print(f"Probe von {len(FZ.CELLS)} Zellen: {'in Ordnung' if not bad else bad[:10]} ({time.time() - t0:.0f} s)")
        return 1 if bad else 0
    data = FZ.build(progress=lambda k: print(f"  Zelle {k} fertig ({time.time() - t0:.0f} s)", flush=True))
    if "--full" in argv:
        bad = compare(data, FZ.load())
        print(f"ganze Messreihe: {'in Ordnung' if not bad else bad[:10]} ({time.time() - t0:.0f} s)")
        return 1 if bad else 0
    FZ.write(data)
    print(f"{FZ.FROZEN_PATH} geschrieben ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
