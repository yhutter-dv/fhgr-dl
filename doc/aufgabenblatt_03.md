---
titlepage: true
titlepage-color: "ffffff"
titlepage-rule-color: "ffffff"
titlepage-text-color: "000000"
toc-own-page: true
colorlinks: false
title: Aufgabenblatt 03 
author:
- Yannick Hutter 
lang: de
mainfont: SF Pro Text 
sansfont: SF Pro Text 
monofont: Fira Code 
header-left: "\\small \\thetitle"
header-center: "\\small \\leftmark"
header-right: "\\small \\theauthor"
footer-left: "\\leftmark"
footer-center: ""
footer-right: "\\small Seite \\thepage"
---

# Aufgabe 1: Wiederholung

Die Optimierung kann auf 3 Ebenen vorgenommen werden:

- Deep Learning Network
- Back Propagation
- Overfitting Management

## Deep Learning Network

Das Deep Learning Network kann mithilfe der folgenden Parameter optimiert werden:

- Anzahl der Epochen - Kann jedoch mit der Regularisationsmethode **Early Stopping** gut gehandhabt werden
- Grösse der Batches - Wichtig
- Anzahl der Hidden Layers - Standard verwenden bspw. 2
- Anzahl der Hidden Nodes - Wichtig
- Aktivierungsfunktionen - Standard verwenden
- Initialisierungsgewichte - Wichtig

## Back Propagation

Bei der Back Propagation gibt es folgende Parameter:

- Optimierer - In den meisten Fällen kann der Adam Optimierer (adaptives Modell) verwendet werden
- Lernrate

Grosse Lernrate lassen das Modell schnell lernen, jedoch den Gradienten explodieren, d.h. das Modell wird nie aufhören und zu keiner Lösung kommen. Wenn die Lernrate zu klein ist, wird nicht das optimale globale Minimum innerhalb der Verlustfunktion gefunden.

## Overfitting Management

> Achtung: Overfitting Management sollte auch nur dann betrieben werden, wenn man ein Modell hat, welches einigermassen gute Ergebnisse liefert.

Die Parameter im Overfitting Management sind die folgenden:

- Regularisierung
- Dropouts
- Batch Normalization (Normalisierung der Gewichte) - Alle Gewichte haben den Mittelwert 0 und die Standardabweichung 1

