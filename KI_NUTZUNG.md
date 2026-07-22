# Dokumentation der KI-Nutzung

## Grundsatz

Generative KI wurde als Entwicklungs- und Reflexionswerkzeug eingesetzt. Sie
ist keine fachliche Quelle. Daten, Definitionen, statistische Ergebnisse und
Modellmetriken stammen aus den angegebenen amtlichen Quellen beziehungsweise
aus der reproduzierbaren Notebook-Ausführung. Die Verantwortung für die
Abgabe und die Interpretation liegt beim Verfasser.

## Einsatzprotokoll

| Datum | System | Einsatz | Ergebnis und menschliche Kontrolle |
|---|---|---|---|
| 21.07.2026 | OpenAI Codex | Umstellung des bisherigen Projekts auf Hitze und Luftqualität | Forschungsfrage, überprüfbare These und Folgetagsziel in die Projektstruktur übertragen. |
| 21.07.2026 | OpenAI Codex | Recherche der amtlichen Schnittstellen | Zunächst Potsdam-Stationen ausgewählt; diese Vorstufe wurde am 22.07.2026 aufgrund der geänderten Ortsvorgabe vollständig durch Frankfurt ersetzt. |
| 21.07.2026 | OpenAI Codex | Codeentwurf für Download, Zeitzonenbereinigung und Join | URLs und Prüfsummen werden im Manifest festgehalten; Zeitstempel und Stationsdistanz werden im Notebook sichtbar kontrolliert. |
| 21.07.2026 | OpenAI Codex | Operationalisierung des dominierenden Schadstoffs | Aktuelle UBA-LQI-Grenzen als gemeinsame gesundheitliche Skala verwendet; Definition und rückwirkende Re-Klassifikation werden offengelegt. |
| 21.07.2026 | OpenAI Codex | Hypothesentest und Visualisierungen | Einseitige Tests und Bootstrap-Intervalle implementiert; Ergebnis nicht an die Ausgangsthese angepasst. Der NO₂-Teil wird als nicht bestätigt ausgewiesen. |
| 21.07.2026 | OpenAI Codex | Vergleich von Prognosemodellen | Baseline, logistische Regression und Random Forest auf den zurückgehaltenen Jahren 2024–2025 verglichen; Auswahl nach gemessenem Macro-F1. |
| 21.07.2026 | OpenAI Codex | Slider-Oberfläche und Präsentationsaufbau | Szenarioeingaben mit denselben Modellmerkmalen verbunden; Aussagen der Folien aus exportierten Notebook-Ergebnissen übernommen. |
| 22.07.2026 | OpenAI Codex | Umstellung der Fallstudie auf Frankfurt am Main | HLNUG-Station DEHE005 Frankfurt-Höchst und DWD-Station 01420 Frankfurt/Main ausgewählt; Distanz und Messumfang kontrolliert. |
| 22.07.2026 | OpenAI Codex | Reparatur und Vollständigkeitskontrolle des Datenabrufs | Nach Ausfall der UBA-Route den offiziellen HLNUG-Zugang verwendet; die 1.100-Zeilen-Grenze erkannt und durch dokumentierte Monatsabfragen behoben. |
| 22.07.2026 | OpenAI Codex | Neuberechnung und Ergebnisprüfung | Notebook vollständig ausgeführt; unerwarteten positiven NO₂-Unterschied beibehalten und These als nicht vollständig unterstützt ausgewiesen. |

## Beispiel des zentralen KI-Arbeitsauftrags

> Passe das Projekt auf den Einfluss hoher Temperaturen auf die städtische
> Luftqualität an, führe UBA- und DWD-Daten über Datum, Uhrzeit und Standort
> zusammen, überprüfe die These und prognostiziere den am Folgetag
> dominierenden Luftschadstoff. Erstelle ein kurzes Notebook, eine Präsentation
> und eine anpassbare Slider-Oberfläche.

## Kontrollmaßnahmen

- KI-Aussagen werden nicht als Quelle zitiert; maßgeblich sind HLNUG, UBA-LQI-Methodik und DWD.
- Jede Rohdatei besitzt eine URL und SHA-256-Prüfsumme im Manifest.
- Die Distanz und die unterschiedlichen Zeitzonen der Messstationen sind im
  Notebook explizit dokumentiert.
- Das Notebook wurde vollständig automatisiert ausgeführt; Fehler würden die
  Ausführung abbrechen.
- Die Hypothese wird anhand vorab definierter Richtungstests bewertet. Das
  unerwartete NO₂-Ergebnis bleibt erhalten und wird als Teilbefund diskutiert.
- Die Prognosegüte wird ausschließlich auf zeitlich späteren Daten bewertet.
- Eine Mehrheitsklassen-Baseline verhindert, dass scheinbar hohe Accuracy ohne
  Lerngewinn als Erfolg interpretiert wird.
- Grenzen wie Einzelstandort, fehlende Verkehrs- und Vorläuferdaten sowie
  perfekte historische Wetterkenntnis werden offengelegt.

## Menschliche Eigenleistung vor der Abgabe

Der Verfasser sollte Notebook und Folien vollständig lesen, die Slider selbst
testen und in eigenen Worten erklären können:

1. warum Schadstoffkonzentrationen erst über den LQI vergleichbar gemacht werden,
2. warum eine zeitliche Testtrennung nötig ist,
3. weshalb Ozon die These stützt, NO₂ sie aber nicht vollständig bestätigt,
4. warum die Modellwahrscheinlichkeit keine amtliche Warnung darstellt.

Eigene Änderungen an Code, Grafikauswahl und Interpretation sollten hier vor
der finalen Abgabe ergänzt werden.
