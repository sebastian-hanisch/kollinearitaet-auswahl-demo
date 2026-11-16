# Zwei Angaben, die fast dasselbe messen: Kollinearität und Lasso – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-kollinearitaet-auswahl-demo.streamlit.app/)** -->

Diese Demo gehört zu den Demos zu **Regression und verallgemeinerten linearen Modellen** (Leitfrage: *Was darf man aus einer angepassten Gleichung schließen, und wann trügt sie?*) für die Website „Sebastian Hanisch – Operations Research und Machine Learning“.
Sie hebt eine stille Annahme der [hebel-diagnose-demo](https://github.com/sebastian-hanisch/hebel-diagnose-demo) auf: Dort prüft man **Zeilen** (entscheidet ein Punkt über die Gerade?) und nimmt dabei an, dass jede **Spalte** der Designmatrix eigene Information enthält, die Koeffizienten sich also einzeln lesen lassen.
Hier messen **zwei Spalten fast dasselbe**, und die Demo fragt, was dann aus den Koeffizienten, den Tests und der Merkmalsauswahl wird. Vehikel ist ein **erfundener Containerstrom** (Export-Zweig der [verweilzeit-demo](https://github.com/sebastian-hanisch/verweilzeit-demo)): der Vorlauf laut Reederei und der „Vorlauf bis Ladeschluss“ (eine erfundene zweite Angabe) mit Korrelation ρ.
Nichts davon ist wissenschaftlich neu (VIF, Lasso, Bootstrap sind Lehrbuchstoff); neu ist die **Vorführbarkeit mit Zähleinheiten und Gegenproben**.

Nachbarn: Kondition und Stellenverlust der Normalgleichung stehen in der [ols-projektion-demo](https://github.com/sebastian-hanisch/ols-projektion-demo), redundante Dummy-Spalten in der [faktoren-codierung-demo](https://github.com/sebastian-hanisch/faktoren-codierung-demo), Modellwahl und Inferenz nach der Auswahl in der [modellwahl-demo](https://github.com/sebastian-hanisch/modellwahl-demo).
Ridge-Regression und die Wahl von λ (Tikhonov, GCV, L-Kurve) stehen in der [tikhonov-demo](https://github.com/sebastian-hanisch/tikhonov-demo) und der [parameterwahl-demo](https://github.com/sebastian-hanisch/parameterwahl-demo) und werden hier **nicht** erneut gemessen; dass die Koeffizienten bei Kollinearität wandern und die Wichtigkeit im Wald sich aufteilt, zeigt die [korrelierte-merkmale-demo](https://github.com/sebastian-hanisch/korrelierte-merkmale-demo) und wird hier nicht als neu gesagt. <!-- links-ok: Repo existiert auf GitHub, lokal fehlt der Ordner -->
Den Bootstrap als Verfahren erklärt die [bootstrap-demo](https://github.com/sebastian-hanisch/bootstrap-demo); hier ist er nur das Werkzeug einer Falle. <!-- links-ok: Repo existiert auf GitHub, lokal fehlt der Ordner -->

## Warum dieses Problem

Fast gleiche Merkmale sind der Normalfall (zwei Angaben zur selben Sache, ein Maß in zwei Einheiten). Drei Dinge passieren dann, die man in einer einzelnen Anpassung nicht sieht: (1) jeder **Einzelkoeffizient** wird unsicher, die **Summe** nicht, sodass alle Einzeltests „kein Effekt“ sagen können, obwohl beide Merkmale wirken; (2) das **Lasso** wählt je Stichprobe eines von beiden; (3) die **Häufigkeit**, mit der Bootstrap oder Teilstichproben *einer* Stichprobe ein Merkmal wählen, wirkt wie eine Wahrscheinlichkeit und ist doch je Stichprobe entschieden.
Kernfragen der Demo: *Wie viel erfahren die Daten über zwei fast gleiche Merkmale? Welches von beiden wählt das Lasso, und wovon hängt das ab? Was zeigt der Bootstrap derselben Stichprobe, was frische Stichproben nicht zeigen?*

## Modell

**Daten:** n Container, Antwort y (Verweilzeit in Tagen), p = 10 Merkmale ohne Achsenabschnitt, je Anpassung standardisiert (Mittel 0, Standardabweichung 1, Teiler n), y zentriert. **Lasso/Elastic Net** (Konvention von scikit-learn und glmnet, Faktor 1/(2n); α entspricht dem glmnet-λ): minimiere (1/(2n))·‖y − Xb‖² + α·(ℓ·‖b‖₁ + (1 − ℓ)/2·‖b‖₂²); ℓ = 1 ist das Lasso, ℓ = 0 Ridge. **VIF:** 1/(1 − R²ⱼ) der Hilfsregression von xⱼ auf die übrigen Merkmale; bei zwei Merkmalen 1/(1 − r̂²) mit der **Stichproben**korrelation r̂ (die Formel mit der Bevölkerungskorrelation ρ gilt für die Stichprobe nur im Mittel).
**Messwelten** (x₁ und x₂ mit Korrelation ρ, x₃ bis x₁₀ unabhängig, σ = 2): **C** (Standard der App) Container-Welt: x₁ = Vorlauf, gleichverteilt 1 bis 10; x₂ = erfundene zweite Vorlaufangabe; x₃ Bernoulli(2/7) (Wochenende), x₄ Normal (Gewicht, das Rauschmerkmal des Generators), x₅ und x₆ Bernoulli(0,3), x₇ bis x₁₀ Normal; y = Xβ + σε (additiv, konstantes σ). **G** Gauß-Welt (nur in den Tests und in der vorgerechneten Tabelle). **E** Export-Zweig des Generators: die Antwort ist die lognormale Verweilzeit selbst (`kol_generator.py`, Auszug aus der verweilzeit-demo, Herkunft im Modulkopf).
**Szenarien:** S1 β = (1; 1; 1; 0 …), S3 β = (0,5; 0,5; 1; 0 …), S2 β = (2; 0; 0 …) (x₂ ist eine Attrappe), S2E wie S2 in Welt E. Die Wirkung beider Merkmale in S1 und S3 ist eine **Annahme der Simulation** (alles im Generator ist erfunden); der Generator hat von Haus aus genau ein wirkendes Vorlaufmerkmal.
Formeln, Symboltabelle und Konventionstabelle stehen im 📐-Expander der App.

## Methodik

- **Koordinatenabstieg** auf (G, c) = (Z'Z/n, Z'y/n) mit weicher Schwelle, **gestapelt** über viele Anpassungen in numpy (je Anpassung O(p²) je Durchlauf, Abbruch bei größter Koeffizientenänderung unter 1e-12); exakte Nullen sind exakt 0,0. scikit-learn dient nur als Orakel im Test (`requirements-dev.txt`).
- **Zufall** nur aus SplitMix64: `kol_rng.py` ist die geprüfte Vorlage des Portfolios (Ganzzahlfolge und Normalzahlen auf jeder Plattform bitgleich, aber 78 bis 143 µs je Normalzahl). Die Messreihen brauchen Millionen Normalzahlen; darum zählt `kol_stream.py` die **Ganzzahlfolgen vieler Zufalls-Seeds gleichzeitig** (dieselbe Folge wie `kol_rng.SplitMix64`, Test) und erzeugt Normalzahlen mit dem Umkehrverfahren AS 241 (Wichura 1988) und eigenem Logarithmus und Exponential **nur aus Grundrechenarten**, ohne libm: das Ergebnis hängt nicht von der Plattform ab (Bitwerte im Test). Jede Stichprobe zieht aus ihrer eigenen Folge; das Ergebnis hängt nicht von der Stapelgröße ab (Test).
- **Zähleinheiten.** Experiment 1: eine der 256 Bootstrap-Stichproben der vier Container. Experiment 2: die **Wiederholung** (neue Fehler, festes Design) bzw. ein **Design** (Zufallsdesign). Experimente 3, 4 und 6: die **frische Stichprobe**. Experiment 5: die **Originalstichprobe**, je 100 Bootstrap-Stichproben (Größe n, mit Zurücklegen) bzw. Teilstichproben (Größe n/2, ohne Zurücklegen), mit **demselben α**; in jeder Wiederholung wird neu standardisiert. „Auswahl“ heißt: Betrag des Koeffizienten über 1e-8.
- **Vorgerechnet** (`data/kol_frozen.json`, Skript `tools/kol_make_frozen.py`, „vorgerechnet“ gekennzeichnet in App und README): die großen Zählungen (je Auswahlzelle 2000 frische und 200 Originalstichproben zu je 100 + 100 Wiederholungen; 100 000 bzw. 20 000 Wiederholungen der Tests). `tests/test_frozen.py` rechnet die Probe jeder Zelle neu und vergleicht (Band), mit `KOL_FULL=1` die ganze Messreihe. **Live** rechnet die App kleinere Fassungen derselben Funktionen (z. B. 500 frische Stichproben, 20 Originalstichproben).
- **Verdikte** sind reine Funktionen (`kol_evaluation.verdict_*`) und nennen nur Zahlen der aktuellen Rechnung.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen gelten **für das Modell oben bei den genannten Parametern** (soweit nicht anders angegeben: n = 100, p = 10, ρ = 0,95, σ = 2, α fest 0,09 beim Lasso und 0,13 beim Elastic Net mit ℓ = 0,5, Welt C, 2000 frische Stichproben, 200 Originalstichproben je 100 Wiederholungen); Quelle ist `data/kol_frozen.json`, Zufalls-Seeds stehen darin. Die Tabelle „Zahl → Test“ unten nennt je Befund den prüfenden Test.

| Nr. | Befund (Bedingung, Zähleinheit) |
|---|---|
| B1 | **Aufblähung, Summe, Differenz** (festes Design, n = 100, Fehler N(0, 1), 100 000 Wiederholungen je ρ; n·Var in Einheiten σ², Formel in Klammern): bei ρ = 0,95 n·Var(b₁) = 10,29 (10,26), n·Var(b₁ + b₂) = 1,035 (1,026), n·Var(b₁ − b₂) = 40,1 (40); bei ρ = 0,99: 50,4 (50,25), 1,014 (1,005), 200,7 (200); bei ρ = 0,999: 502,1 (500,25), 1,010 (1,0005), 2007,7 (2000). Die Korrelation von b₁ und b₂ ist −ρ auf drei Stellen. Die Daten kennen die Summe, nicht die Aufteilung. |
| B2 | **Einzeltest gegen gemeinsamen Test** (festes Design, n = 100, σ = 2, β = (b, b), Niveau 5 %, 20 000 Wiederholungen je Zeile; exakte nichtzentrale Verteilung in Klammern): **ρ = 0,95, b = 0,5:** Standardfehler von b₁ 0,641 (bei ρ = 0 0,200); ein Einzeltest verwirft in 12,3 % (12,1 %), beide in 0,8 %, der gemeinsame F-Test in 99,4 % (99,5 %); in **75,9 %** der Wiederholungen verwirft der F-Test, ohne dass ein Einzeltest verwirft. b = 1,0: 33,6 % (34,0 %), F ohne Einzeltest 32,5 %. ρ = 0,99, b = 0,5: 6,5 % (6,4 %), F ohne Einzeltest 88,7 %. ρ = 0, b = 0,5: 69,7 % (69,7 %), beide 48,9 %, F 88,6 % (88,6 %). Größte Abweichung Simulation gegen exakt: unter 0,015. |
| B3 | **Intervalle bleiben gültig** (β = (0,5; 0,5), 20 000 Wiederholungen): Überdeckung des 95-%-Intervalls für b₁ 95,2 / 94,8 / 94,8 / 94,8 % bei ρ = 0 / 0,9 / 0,99 / 0,999; mittlere Breite × 2,29 / × 7,09 / × 22,37 gegen ρ = 0. Kollinearität verletzt keine Annahme des Modells, sie macht die Antwort breit. |
| B4 | **Der VIF der Stichprobe streut** (Zufallsdesign, Bevölkerungskorrelation ρ, 4000 Designs je Zeile; Formel mit ρ): ρ = 0,9, n = 30: 5-bis-95-%-Bereich −40 % bis +84 % um 5,26, bei n = 100 −24 % bis +38 % (3,99 bis 7,25), bei n = 1000 −9 % bis +10 %; ρ = 0,99, n = 100: −27 % bis +42 %. Bei n = 30 liegt der VIF in 3,1 % (ρ = 0,9) bzw. 4,2 % (ρ = 0,99) der Designs über dem Doppelten der Formel, bei n ≥ 100 in keinem der 4000. |
| B5 | **Der Generator selbst** (Export-Zweig: Vorlauf, Wochenende, Gewicht, elf Kunden-Dummies, ein Design mit 60 000 Containern): größte VIF 1,87; Vorlauf, Wochenende und Gewicht je unter 1,001. Hohe Kollinearität kommt erst mit einem zweiten, fast gleichen Merkmal. |
| B6 | **Welches von beiden?** (Lasso, 2000 frische Stichproben je Zelle): **S3:** x₁ in 76,1 %, x₂ in 76,6 %, beide in 52,7 %; wenn genau eines gewählt wird (in 945 von 2000), ist es x₁ in 49,5 %: wie ein Münzwurf (zweiter Zufalls-Seed: 75,3 / 76,2 / 51,5 %, 49,0 %). **S1:** beide in 85,6 %, genau eines nur in 289 von 2000 Stichproben, Anteil x₁ dann 52,2 %. **S2:** x₁ in 99,9 %, die Attrappe x₂ in 46,8 %, Anteil x₁ bei genau einem 99,8 %. **S2E:** x₁ 100,0 %, x₂ 48,0 %. **S3 mit ρ = 0,99:** x₁ 61,3 %, x₂ 62,8 %, beide 24,1 %, genau eines in 1517 von 2000, Anteil x₁ 49,0 %. Die sieben Attrappen x₄ bis x₁₀ werden im Mittel in 64,0 % (S3) gewählt: ein α nahe dem Kreuzvalidierungswert zielt auf Vorhersage, nicht auf Auswahl. |
| B7 | **Die Falle des Bootstraps** (Zähleinheit: Originalstichprobe, m = 200, je 100 Wiederholungen): **S3, Lasso:** über frische Stichproben unterscheiden sich x₁ und x₂ um 0,005 (\|P(x₁) − P(x₂)\|), je Originalstichprobe im Bootstrap im Mittel um **0,392** (Teilstichproben 0,374). In **65,0 %** der 200 Originale liegt der Anteil von x₁ bei genau einem über 0,75 oder unter 0,25 (Teilstichproben 58,0 %; zweiter Zufalls-Seed 59,0 % und 56,5 %; Monte-Carlo-Fehler etwa ± 3,4 Punkte je Wert); der Mittelwert des Anteils ist nicht schief (0,477 im Bootstrap gegen 0,495 frisch), die Streuung über die Originale beträgt 0,343: **je Stichprobe entschiedener, als die Welt ist**. Die Regel „Häufigkeit ≥ 0,8 gilt als stabil“ meldet im Bootstrap genau x₁ in 39,5 %, genau x₂ in 45,5 %, beide in 5,0 %, keines in 10,0 % der Originale und in 85,0 % **mindestens eine Attrappe**. Der Bootstrap **unterschätzt** die Einschlusshäufigkeit der wahren Zwillinge (68,9 % und 71,7 % gegen 76,1 % und 76,6 % frisch; beide 40,7 % gegen 52,7 %) und **überschätzt** die der Attrappen (71,7 % gegen 64,0 %; Teilstichproben 72,4 %). S1: 0,259, „klar“ in 76,0 % (Teilstichproben 71,5 %); ρ = 0,99 in S3: 0,508, 59,5 % (60,0 %); Gauß-Welt G, S3: 0,439, 71,0 % (64,0 %). |
| B8 | **Gegenprobe: ein klar stärkeres Merkmal (S2, S2E)**: Bootstrap und frische Stichproben unterscheiden x₁ und x₂ gleich stark (Welt C: 0,493 gegen 0,531; Welt G: 0,488 gegen 0,520; Export-Zweig: 0,490 gegen 0,519; alle Beträge um höchstens 0,05 verschieden). Anteil x₁ bei genau einem: Bootstrap 0,912 gegen 0,998 frisch (Welt C), Export-Zweig 0,989. **Die Falle gilt, wenn zwei Merkmale gleich stark sind, nicht, wenn eines klar stärker ist.** In 7 von 12 Zellen (alle mit zwei gleich starken Zwillingen) übersteigt der Unterschied je Original den über frische Stichproben um mindestens 0,15, in den 5 übrigen (S2, S2E) nicht. |
| B9 | **Elastic Net** (ℓ = 1/2, α = 0,13, 2000 frische Stichproben): **S3:** beide Zwillinge in 87,9 % (Lasso 52,7 %), x₁ 93,8 %, x₂ 94,1 %. **S2: die Kopie x₂ in 95,2 %** (Lasso 46,8 %); im Export-Zweig S2E 99,2 % (Lasso 48,0 %). Die sieben Attrappen werden im Mittel in 73,5 % (S3), 73,9 % (S2) und 67,8 % (S2E) gewählt (Lasso 64,0 %, 64,0 %, 57,0 %). Der Gruppeneffekt hat einen Preis: er nimmt die Kopie mit. Der Bootstrap zeigt für die Kopie x₂ in S2 90,6 % gegen 95,2 % über frische Stichproben. |
| B10 | **Nichtnullen und Rauschgrenze** (S3, Lasso, α = 0,09, Zufalls-Seed 1230, 2000 frische Stichproben): Auswahlhäufigkeit der Attrappen und mittlere Zahl der Nichtnullen bei Schwelle 0 und 1e-8: 63,5 % und 6,94; bei 1e-4: 63,5 % und 6,94; bei 1e-3: 63,2 % und 6,91; 1e-2: 60,0 % und 6,69; 5e-2: 47,1 % und 5,75; 0,1: 34,1 % und 4,78. In 22,7 % der Stichproben liegt der kleinste gewählte Betrag unter 1e-2, in 0,25 % unter 1e-4 und in keiner unter 1e-8. Ridge wählt immer alle 10. **Eine Zahl der Nichtnullen ohne Schwellenangabe ist keine Messung.** |
| B11 | **Glattes Verfahren** (L-BFGS-B auf √(b² + ε), 200 Stichproben, gleiche gewählte Menge wie der exakte Abstieg): ε = 1e-12: Schwelle 0: 6 von 200, 1e-8: 7, 1e-6: 89, 1e-4: 199, 1e-3: 195, 1e-2: 150; ε = 1e-8: Schwelle 1e-4: 89, 1e-3: 196. Bei einer Schwelle gleich √ε stimmen 89 von 200 überein; gemessen nur an diesen zwei Werten von ε; bei 1e-2 weichen für beide ε schon 50 von 200 ab, weil echte kleine Koeffizienten verschwinden. |
| B12 | **Bedingung von Zhao/Yu** (eine Attrappe mit Korrelation a zu jedem wahren Merkmal; q = n_wahr·a): in der **Bevölkerung** (Lasso auf der Gram-Matrix, α = 0,05, kein Rauschen) wird die Attrappe in den sieben Entwürfen genau dann gewählt, wenn q > 1 (q = 0,5; 0,6; 0,8; 0,9; 0,95: nein; q = 1,2: ja mit Betrag 0,036; q = 1,0 ist der Rand und wird nicht gezählt). Mit **Daten** (α = 0,1, σ = 2, 500 Stichproben je Zelle) fällt P(Attrappe) für q < 1 mit n (q = 0,5: 57,6 / 19,6 / 0,0 % bei n = 100 / 1000 / 10 000; q = 0,6: 59,0 / 22,6 / 1,2 %; q = 0,95 nur langsam: 44,8 / 40,2 / 21,2 %), steigt für q = 1,2 (59,6 / 72,6 / 97,6 %) und bleibt bei q = 1,0 um 50 % (58,4 / 47,4 / 49,4 %). **Nur sieben Entwürfe, kein Beweis.** |
| B13 | **Handbeispiel** (vier erfundene Container, Brüche): X'X = [[20, 16], [16, 20]], ρ = 4/5, VIF = 25/9; Var(b₁ + b₂) = 1/18·σ² (orthogonal 1/10), Var(b₁ − b₂) = 1/2·σ²; b = (11/18; 1/9), SSE = 1/9; mit vertauschten Zeilen von y tauschen die Koeffizienten, die Summe 13/18 bleibt. Lasso: x₁ tritt bei α = 7/2 ein, x₂ bei α = 1; bei α = 2 Lasso (3/10; 0), Elastic Net mit ℓ = 1/2 (7/20; 1/10). Bootstrap, alle 256 Stichproben bei α = 2: nur x₁ in 108, nur x₂ in 36, beide in 0, keines in 24, nicht eindeutig (höchstens zwei verschiedene Zeilen) in 88; mit vertauschten Zeilen genau das Spiegelbild. Schranke der Stabilitätsauswahl (Meinshausen/Bühlmann 2010, setzt Austauschbarkeit voraus, hier nicht erfüllt): q = 7, p = 10, π = 0,8 geben 49/6 ≈ 8,17, mehr als die 7 vorhandenen Attrappen. |
| B14 | **Orakel** (Messung dieser README, 500 Stichproben der Welt C, S3, n = 100; nicht im Test, der Test prüft 60 Instanzen auf unter 1e-7): eigener Abstieg gegen scikit-learn (tol 1e-12): größte Koeffizientenabweichung 7,0·10⁻¹² (Lasso, α = 0,09) und 2,5·10⁻¹² (Elastic Net, α = 0,13, ℓ = 0,5); gleiche gewählte Menge in 500 von 500; größte Verletzung der Optimalitätsbedingungen 9,3·10⁻¹³. |

### Zahl → Test

| Zahl | Test (Datei `tests/`) | Art |
|---|---|---|
| B1 | `test_claims.py::test_b1_*`; Formeln gegen die Matrix: `test_inference.py::test_variance_formulas_equal_the_matrix` | gedruckte Zahl gegen die Datei (halbe Einheit der letzten Stelle); Neu-Rechnung mit anderem Zufalls-Seed als Band ± 4 % |
| B2 | `test_claims.py::test_b2_*`; Orakel: nichtzentrales t und F sowie Vollsimulation mit der Referenzfolge (`test_inference.py`) | gedruckt gegen die Datei; Simulation gegen exakt Band 0,015; Neu-Rechnung als Band |
| B3, B4, B5 | `test_claims.py::test_b3_*`, `test_b4_*`, `test_b5_*` | gedruckt gegen die Datei |
| B6, B7 | `test_claims.py::test_b6_*`, `test_b7_*`; je Preset die Verdikt-Art in `test_verdicts.py` | gedruckt gegen die Datei; Neu-Rechnung mit anderem Zufalls-Seed als Band |
| B8 | `test_claims.py::test_b8_*`, `test_count_over_all_cells_in_which_the_trap_applies` | gedruckt gegen die Datei; Zählung über alle 12 Zellen |
| B9, B10, B11 | `test_claims.py::test_b9_*`, `test_b10_*`, `test_b11_*` | gedruckt gegen die Datei |
| B12 | `test_claims.py::test_b12_*`; Orakel: Aufzählung in Brüchen (`test_oracle_kollinearitaet_auswahl.py`) | gedruckt gegen die Datei; Entscheidung der Bevölkerung exakt |
| B13 | `test_exact.py`, `test_claims.py::test_b13_*` | exakt (Brüche) |
| B14 | `test_oracle_kollinearitaet_auswahl.py` (60 Instanzen, Band 1e-7); die Zahlen der Zeile B14 selbst sind eine Messung dieser README | Band; Messung, nicht im Test |

Alle übrigen Zahlen der App (Verdikte, Metriken) kommen aus den Funktionen, die diese Tests prüfen; die Verdikt-Texte stehen in `tests/test_verdicts.py` an handgesetzten Zeilen.

## Befunde und Korrekturen gegenüber dem Plan

Die Planung vor dem Bau stützte sich auf eine erste Rechnung mit anderen Zufallsströmen (numpy). Im Bau mit dem eigenen Zufall neu gemessen, gelten diese Zahlen; was sich verschob oder nicht bestätigte:
- **Bootstrap, S3:** der Anteil „klar“ ist 65,0 % (Bootstrap) und 58,0 % (Teilstichproben), die erste Rechnung ergab 61,0 % und 55,0 %; ein zweiter Zufalls-Seed gibt 59,0 % und 56,5 %: die Zählung schwankt um 6 Punkte, der Befund (je Stichprobe entschieden, im Mittel nicht schief) hält. Die Zahl „0,888 gegen 0,521“ einer früheren Messung gilt für **eine** Originalstichprobe und steht hier nicht; über 200 Originale ist der Mittelwert des Anteils 0,477 gegen 0,495 frisch.
- **Neu standardisiert:** in jeder Bootstrap- und Teilstichprobe werden die Merkmale hier neu standardisiert (die erste Rechnung zentrierte nur y); \|P(x₁) − P(x₂)\| je Original ist 0,392 statt 0,405.
- **Elastic Net, S2:** die Attrappe x₂ wird in Welt C zu 95,2 % gewählt (erste Rechnung, Gauß-Welt: 96,6 %); Export-Zweig 99,2 % statt 99,4 %.
- **Export-Zweig:** α ist hier fest 0,09 (die erste Rechnung nahm 0,045·max|Kov| = 0,102): x₂ in 48,0 % (Lasso) statt 47,0 %.
- **Stichproben-VIF:** der 5-bis-95-%-Bereich bei n = 30, ρ = 0,9 ist −40 % bis +84 % (erste Rechnung −41 % bis +85 %); Unterschiede im Rahmen des Monte-Carlo-Fehlers.
- **„Wie ein Münzwurf“** gilt für S3 (Anteil x₁ 49,5 %); bei S1 wird genau eines nur in 289 von 2000 Stichproben gewählt, die Aussage „Lasso wählt eines von beiden“ gilt dort nicht.
- **Zwei Fehler beim Bau gefunden:** (a) `scipy.stats.ncf` mit Nichtzentralität 0 liefert in der benutzten Version einen negativen Wert (−0,95 statt 0,05); bei β = 0 wird die zentrale F-Verteilung benutzt (Test). (b) Mit 53 Bit lässt sich die gleichverteilte Zahl 1 − 2⁻⁵⁴ nicht darstellen, sie wurde zu 1,0 und hätte eine unendliche Normalzahl ergeben; die Demo benutzt 52 Bit (Test an den äußersten Ganzzahlen).
- **Rand q = 1** (Bedingung von Zhao/Yu): der Koeffizient der Attrappe in der Bevölkerung ist dort 4·10⁻¹⁷, also numerisch null, und wird als Rand nicht gezählt.

## Ehrliche Grenzen

- **α ist fest** (0,09 bzw. 0,13), nie je Stichprobe per Kreuzvalidierung gewählt; „die Wahl von α ist selbst verrauscht“ ist hier **nicht** gemessen.
- Die **vorgerechneten Zählungen** gelten für n = 100, p = 10, ρ = 0,95 (eine Zelle mit ρ = 0,99), σ = 2; die Regler der App rechnen andere Einstellungen live mit kleinerer Zahl von Wiederholungen. Elastic Net vorgerechnet nur mit ℓ = 0,5. Teilstichproben mit **demselben** α wie die volle Stichprobe (nicht geprüft, ob ein anderes α die Zahlen ändert).
- **Zahlen mit Monte-Carlo-Fehler:** die Anteile aus 200 Originalstichproben haben bei 65 % etwa ± 3,4 Punkte, die aus 2000 frischen Stichproben etwa ± 1 Punkt; Unterschiede von wenigen Punkten zwischen Zellen sind keine Befunde.
- **Szenarien S1 und S3 im Export-Zweig** (beide Merkmale wirken) lassen sich mit dem Generator nicht ohne Zusatzannahme bauen: sie laufen in Welt C (additiv, konstantes σ), nicht mit der lognormalen Verweilzeit.
- Die **Bedingung von Zhao/Yu** ist nur an sieben Entwürfen mit ein bis zwei wahren Merkmalen und einer Attrappe geprüft; sie ist „fast notwendig und hinreichend“ (Zhao/Yu 2006), nicht „notwendig und hinreichend“. Die **Stabilitätsauswahl** selbst (Schwelle π über Teilstichproben, randomisiertes Lasso) ist **nicht** gemessen; nur das Rechenbeispiel der Schranke steht oben.
- Die **Deutung** „der Bootstrap übernimmt den Zufall der einen Stichprobe“ ist eine Deutung; gemessen ist die Streuung über Originalstichproben.
- Die Zahl der **Nichtnullen** ist nur für ein Szenario (S3) und einen Löser-Satz gemessen; ob Merkmale auf dem Pfad wieder austreten können, ist nicht gemessen.
- Aussagen zu **Tests und Intervallen nach einer Lasso-Auswahl** (ohne Korrektur nicht gültig) stehen nach den gelesenen Quellen im Expander, sind hier **nicht** gemessen.
- Quellen sind nur so zitiert, wie sie gelesen sind; nicht gelesene Arbeiten (u. a. zur Überschätzung der Einschlusshäufigkeit beim Bootstrap) stehen nicht im Text.
- **Windows und Linux:** Zählungen und Anteile aus Neu-Rechnung werden als Band geprüft; die Rechnung benutzt nur exakt gerundete Grundoperationen, die Probe der vorgerechneten Zellen ist aber auf jeder Plattform durch den Linux-Lauf der CI abgesichert.

## Verifikation

- **Kern:** Koordinatenabstieg gegen die Brüche des Handbeispiels (Aufzählung aller Aktivmengen und Vorzeichen), gegen scikit-learn (`Lasso`, `ElasticNet`, `lasso_path`), gegen die geschlossene weiche Schwelle im orthonormalen Fall und gegen die Optimalitätsbedingungen am eigenen Ergebnis; VIF auf drei Wegen; Summe und Differenz geschlossen gegen die Matrix; Einzel- und F-Test-Raten gegen nichtzentrales t und F, dazu eine unabhängige Vollsimulation mit der Referenzfolge; die Bedingung der Auswahl gegen die Aufzählung in Brüchen.
- **Eigenschaften:** α ≥ α_max gibt nur Nullen; α → 0 Kleinste Quadrate; Vertauschen zweier Spalten vertauscht ihre Koeffizienten; Standardisieren macht das Lasso skalenunabhängig (ohne es nicht); y mit c skaliert das Lasso bei α·c (nur Lasso); ℓ = 0 ist Ridge ohne exakte Nullen; bei zwei gleichen Spalten teilt das Elastic Net gleich, das Lasso legt nur die Summe fest; die L1-Norm wächst entlang des Pfads.
- **Randfälle:** ρ = 1, n kleiner als p, konstante Spalte, konstantes y, α = 0 und über α_max, ℓ = 0, Bootstrap mit höchstens zwei verschiedenen Zeilen, eine Originalstichprobe, Zufalls-Seed 0.
- **Zufall:** Ganzzahlfolge gleich `SplitMix64`, Bitwerte von Normalzahl, ln und exp, Genauigkeit gegen libm und `statistics.NormalDist`, Verteilungstest, Unabhängigkeit von der Stapelgröße.
- **Claims, Verdikte, vorgerechnete Messreihe:** jede README-Zahl in `tests/test_claims.py`, jedes Verdikt an handgesetzten Zeilen, je Preset die erwartete Verdikt-Art; Probe jeder Auswahlzelle neu gerechnet.
- **Presets und Oberfläche:** vollständig, Werte innerhalb der Reglergrenzen, alle verschieden; AppTest-Rauchtest (jeder Button, jeder Regler an beiden Rändern, jedes Szenario), Permalink, Layout (Schnellstart oben in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben, jede Änderung eines Reglers hebt die Markierung auf), Footer-Test.
- Die Suite umfasst 267 Tests (`pytest --collect-only -q tests`), davon einer nur mit `KOL_FULL=1`.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore`, `.streamlit/config.toml` | Caches und Secrets; Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (drei Kernfragen, sechs Experimente, Grenzen, Formeln) |
| `kol_algorithm.py` | Kern: Standardisieren, Gram-Matrix, gestapelter Koordinatenabstieg, Pfad, Optimalitätsbedingungen, VIF |
| `kol_exact.py` | Brüche: Handbeispiel, Aufzählung aller Aktivmengen, Bootstrap-Aufzählung, Schranke |
| `kol_sampling.py` | Messwelten, frische Stichproben, Bootstrap- und Teilstichproben, Kennzahlen je Originalstichprobe |
| `kol_inference.py` | Experiment 2: festes Design, Tests, nichtzentrale Rechnung, Streuung des VIF |
| `kol_nonzero.py`, `kol_condition.py` | Nichtnullen und glattes Verfahren; Bedingung von Zhao/Yu |
| `kol_generator.py` | Auszug aus dem Containergenerator (Export-Zweig) |
| `kol_rng.py`, `kol_stream.py` | geprüfte SplitMix64-Vorlage; gestapelte Folgen und Normalzahlen ohne libm |
| `kol_frozen.py`, `tools/kol_make_frozen.py`, `data/kol_frozen.json` | vorgerechnete Messreihe: Aufbau, Skript (`--check`, `--full`), Ergebnis |
| `kol_constants.py`, `kol_presets.py`, `kol_layout.py` | Regler-Grenzen und Presets; Permalink und aktives Preset; fester Kopf |
| `kol_evaluation.py`, `kol_visualization.py` | Rechnungen hinter den Experimenten und Verdikte (reine Funktionen); Plotly-Abbildungen |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit (streamlit, numpy, scipy, pandas, plotly); dazu pytest und scikit-learn für die Gegenproben |
| `tests/` | Brüche, Kern, Orakel, Eigenschaften, Randfälle, Zufall, Messwelten, Inferenz, Claims, Verdikte, vorgerechnete Messreihe, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Ridge und die Wahl von λ**: nur als Verweis (tikhonov-demo, parameterwahl-demo); Ridge kommt nur als Grenzfall ℓ = 0 in den Tests vor.
- **α je Stichprobe per Kreuzvalidierung** und die Stabilitätsauswahl selbst (Teilstichproben mit Schwelle, randomisiertes Lasso): nicht gebaut.
- **Inferenz nach der Auswahl** (Tests und Intervalle nach dem Lasso), kategoriale Merkmale und Codierungsfallen, Cook-Abstand und Hebel: gehören zu anderen Demos.
- **Gestrafte Logit-Modelle und Zähldaten**: nur gaußsche Kleinste Quadrate.
- **PDF-Export** und eine Seitenteilung: nicht gebaut (die Seite hat sechs Experimente und liegt an der Grenze der Länge).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Mit `KOL_FULL=1` rechnet `tests/test_frozen.py` die ganze vorgerechnete Messreihe neu (einige Minuten); die Messreihe selbst entsteht mit `python tools/kol_make_frozen.py`.

Gebaut mit Streamlit, Plotly, NumPy, SciPy und pandas; scikit-learn nur für die Gegenproben im Test.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
