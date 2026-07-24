# Dokumentation der KI-Nutzung

Generative KI wurde als Entwicklungswerkzeug für die Umstellung des
Repositories auf die Analyse meteorologischer Einflüsse auf die normierte
Photovoltaikerzeugung eingesetzt.

## Unterstützte Arbeitsschritte

- Strukturierung von Forschungsfrage, These und Zielvariable
- Entwurf der SMARD-/DWD-Importpipeline
- Implementierung von Feature Engineering und zeitlicher Modellvalidierung
- Entwicklung und Gestaltung der Streamlit-App
- Erstellung des reproduzierbaren Notebook-Generators
- Formulierung von Dokumentation und automatisierten Tests

## Menschlich beziehungsweise fachlich zu prüfen

- Auswahl geeigneter DWD-Stationen in allen Bundesländern
- Vollständigkeit, Einheiten und Zeitbezug der realen Daten
- jährliche Werte der installierten PV-Leistung
- Plausibilität regionaler PV-Gewichte
- fachliche Interpretation des thermischen Effekts
- Modellgüte im realen zeitlichen Test

## Bewusste methodische Entscheidungen

- Synthetische Daten sind in App und Notebook ausdrücklich als Demo markiert.
- PV-Erzeugung wird durch installierte Leistung und eine Stunde normiert.
- Installierte Leistung ist kein Wetterprädiktor, sondern nur Nenner der
  Zielvariable.
- Ein zeitlicher Split verhindert eine zufällige Vermischung zukünftiger und
  vergangener Beobachtungen.
- Die Modultemperatur wird transparent über eine einfache NOCT-artige Formel
  angenähert; sie ist keine Messung.
- Unsicherheitsintervalle beruhen auf Testresiduen und sind keine vollständig
  kalibrierte probabilistische Prognose.

Alle Ergebnisse müssen vor einer Abgabe mit realen Daten reproduziert,
fachlich geprüft und korrekt zitiert werden.
