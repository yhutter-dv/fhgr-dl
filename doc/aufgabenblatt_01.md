---
titlepage: true
titlepage-color: "ffffff"
titlepage-rule-color: "ffffff"
titlepage-text-color: "000000"
toc-own-page: true
colorlinks: false
title: Aufgabenblatt 01 
author:
- Yannick Hutter 
lang: de
lof: true
lot: true
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

> Welche Theorien haben Sie in diesem Teil des Kurses gelernt? Wie stufen Sie die Wichtigkeit deren ein?

TODO

## Aufgabe 2: Grundelemente
Stellen Sie das Vorgehen und den Aufbau eines "Feed Forward Neural Networks" grafisch dar. Die Grafik soll alle Optimierungsschritte und Ablaufschritte enthalten.

![Feed Forward Network](./images/feed_forward_network.png)

### Ablauf

#### Bestimmung der Input
In einem ersten Schritt müssen zuerst die Inputgewichte `initial bestimmt` werden. Hierzu gibt es verschiedene Verfahren. Ein bekanntest Verfahren ist die `Random Uniform Initialization`. Hierbei werden die Gewichte zufällig um den Mittelwert verteilt. Anschliessend wird die Forward Propagation durchgegangen. Hierbei werden die Inputdaten mit den Gewichten multipliziert und mit einem Bias (kleiner Fehler) durch eine Aktivierungsfunktion (Sigmoid) gerreicht. Anschliessend werden die Gewichte mittels einem Optimizer angepasst. Der Optimizer benutzt eine Fehlerfunktion. Das Ziel hierbei ist es das globale Minmimum der Fehlerfunktion zu finden. Der Optimierungsschritt wird innerhalb der `Back Propagation` gemacht. 




