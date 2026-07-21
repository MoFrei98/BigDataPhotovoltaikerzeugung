# Dokumentation der KI-Nutzung

## Grundsatz

Generative KI wurde als Entwicklungs- und Reflexionswerkzeug eingesetzt. Die
inhaltliche Verantwortung, Quellenprüfung und finale Bewertung der Ergebnisse
liegen beim Verfasser der Abgabe.

## Einsatzprotokoll

| Datum | System | Einsatz | Ergebnis und menschliche Prüfung |
|---|---|---|---|
| 21.07.2026 | OpenAI Codex | Prüfung des alten Iris-/ML-Projektgerüsts | Nicht passende Klassifikationsarchitektur erkannt; Altprojekt wurde unverändert nach `archive/` verschoben. |
| 21.07.2026 | OpenAI Codex | Präzisierung von Forschungsfrage und Operationalisierung | Definition der absoluten Entkopplung wurde in überprüfbare Bedingungen für reales Gesamt-BIP und territoriale CO₂-Emissionen übersetzt. |
| 21.07.2026 | OpenAI Codex | Erstellung von Download-, Analyse- und Visualisierungscode | Code wird durch vollständige Notebook-Ausführung, Plausibilitätsprüfungen und Vergleich mit OWID-Metadaten kontrolliert. |
| 21.07.2026 | OpenAI Codex | Auswahl transparenter Prognose-Baselines | Modelle werden nicht aufgrund einer KI-Empfehlung, sondern anhand zeitlich geordneter Out-of-Sample-Fehler ausgewählt. |
| 21.07.2026 | OpenAI Codex | Struktur und Gestaltung der Präsentation | Folienaussagen werden ausschließlich aus den reproduzierten Notebook-Ergebnissen übernommen. |

## Beispiel für den zentralen Arbeitsauftrag

> Prüfe, ob das alte Klassifikationsprojekt für eine Analyse der Entkopplung von
> Wirtschaftswachstum und CO₂-Emissionen geeignet ist. Erstelle ein
> reproduzierbares Jupyter-Notebook, teste die These für EU-27 und prognostiziere
> Deutschlands territoriale CO₂-Emissionen bis 2030.

## Grenzen und Kontrollmaßnahmen

- KI-Ausgaben gelten nicht als Quelle. Zitiert werden die ursprünglichen Datenanbieter.
- Datenbeschreibungen werden mit den OWID-Metadaten abgeglichen.
- Berechnete Länderanteile werden aus einer exportierten Ergebnistabelle nachvollzogen.
- Die Prognose verwendet ausschließlich zeitlich frühere Daten für die Validierung.
- Unsicherheitsintervalle und methodische Grenzen werden sichtbar ausgewiesen.
- Vor Abgabe sind Notebook, Folien und Quellenangaben vollständig manuell zu lesen.

## Eigenleistung

Der Verfasser sollte vor Abgabe ergänzen, welche Codeabschnitte angepasst,
welche Grafiken ausgewählt und welche Interpretationen eigenständig formuliert
oder verändert wurden. Auch verworfene KI-Vorschläge können hier dokumentiert
werden, wenn sie die methodische Entscheidung beeinflusst haben.
