# Zusammenfassung Deep Learning

## Fragen

> Worin unterscheidet sich "supervised Learning" von "unsupervised Learning"?

Beim supervised Learning kennt man das Ziel (Label), welches beim unsupervised nicht der Fall ist. Alle Deep-Learning Verfahren sind supervised.

> Worin unterscheidet sich eine Regression im Gegensatz zu einer Klassifikation?

Die Regression versucht einen numerischen Wert vorherzusagen, bei der Klassifikation wird ein kategorischer Wert ermittelt. 

> Worin unterscheidet sich "Overfitting" von "Underfitting"?

Bei **Overfitting** ist das Modell zu sehr an die Daten angepasst und für andere, neue Daten unbrauchbar. Bei **Underfitting** ist das Modell nicht in der Lage irgendwelche Dinge vorherzusagen.

> Worin liegt der grosse Unterschied zwischen Maschine- und Deep-Learning?

Bei Machine-Learning ist noch ein Labeling durch Menschen notwendig, wobei beim Deep-Learning dieser Schritt vom Modell selbst erfolgt. Achtung das bedeutet nicht, dass keine Datenaufbereitung mehr notwendig ist. Deep-Learning Modelle haben in der Regela auch mehr Hidden Layers.

> Worin liegt der Unterschied zwischen Linearer- und Logistischer-Regression?

Die Linearen-Regression ist eine Binäre Klassifikation, es macht einen harten Schnitt und ermittelt immer eine Klasse. Die Logistische Regeression gibt Wahrscheinlichkeiten pro Klasse an.

> Warum muss die Aktivierungsfunktion des Output-Layers eine andere sein als die der Hidden-Layers

Die Aktivierungsfunktion des Output-Layers muss die Fragestellung beantworten und ist daher von der Fragestellung selbst abhängig.

## Künstliche Intelligence 

Es gibt verschiedene Verfahren, welche auf unterschiedliche Probleme angewendet werden können:

- **Regelbasierte Verfahren**: Wenn...dann
- **Machine Learning**: Bestimmung Hauptpreis 
- **Deep Learning**: Hauptsächlich für Bildklassifikation und Spracherkennung 

## Perceptron

Das Perceptron ist aus folgenden Elementen aufgebaut:
- Mehrere Inputs
- Mehrere Gewichte (ein Gewicht pro Input)
- Bias

Jeder Input wird mit dem entsprechenden Gewicht multipliziert und aufsummiert. Am Schluss wird noch der Bias dazugenommen. Der **Bias** kann als "kleiner Fehler" interpretiert werden. Dieser Wert geht danach an eine entsprechende Funktion weiter (Lineare, Logistische Regression etc.). Die einzigen Dinge welche das Modell selbst berechnen sind die **Gewichte** und der **Bias**.

## Back Propagation
Die Back Propagation ist ein sehr wichtiger Bestand Teil des Machine Learnings. Zu Beginn gibt es die Forward Propagation. Hierbei werden die Werte einmal durch das Netz gereicht. Anschliessend wird der Fehler anhand einer **Verlustfunktion** berechnet. Die Verlustfunktion misst, wie sich die Vorhersage vom Modell zum reellen Wert unterscheidet.

## Loss Funktion
Beispiele für Loss Funktionen bei der Regression sind:

- Mean Squared Error
- Mean Absolute Error (Wurzel aus Mean Squared Error)

Beispiele für Loss Funktionen bei der Klassifikation sind:

- Binary Cross-Entropy
- Categorical Cross-Entropy

## Deep Learning
Beim Deep Learning ist das Ziel **Dinge vorherzusagen**. Es gibt grundsätzlich zwei Probleme welche wir lösen wollen:

- Klassifikation: Einfache Klassifikation und Multi-Klassifikation
- Regression


## Berechnung der Gewichte
Um am Anfang die Gewichtige zu bestimmen gibt es mehrere Möglichkeiten:

- Random Uniform Initialization: Es werden zufällige Werte herausgesucht welche um den Mittelwert 0 verteilt sind
- Zeroes: Alle Werte werden auf 0 gesetzt
- Ones: Alle Werte werden auf 0 gesetzt

## Bestimmung von Hidden Layers und Nodes

Am Anfang sollte mit einem Hidden Layer gestartet werden. Je mehr Hidden Layers desto komplexer. Die meisten Probleme können mit 2 Hidden Layers gelöst werden.

> Die Anzahl der Hidden Layers sollte zwischen der Anzahl der Input und Output Nodes liegen

## Aktivierungsfunktionen

### Rectified Linear Unit (ReLU)
Ist die populärste Funktion für das Deep Learning. Alle negativen Werte werden auf 0 gesetzt und die positiven Werte bleiben positiv.

### Sigmoid
Die Werte werden in den Bereich zwischen 0 und 1 gemapt.

## Output Layer
Bei den Output Layers werden folgende Funktionen verwendet:

- Sigmoid (Binäre Klassifikation): Wahrscheinlichkeiten zwischen 0 und 1
- Softmax (Multi-Klassifikation): Die Summe über alle Werte ist 1 
- Linear: Wird verwendet um Regressionsprobleme zu lösen

## Konfusions Matrix
Bei der Konfusions Matrix werden die vorhergesagten Werte mit den tatsächlichen Werten verglichen und in einer Matrix mittels True Positive (TP), False Positives (FP), True Negatives (TN) und False Negatives (FN) dargestellt.

## Optimizer
Ein Optimirer ist ein Algorithmus welcher die Parameter des Models so beeinflusst, dass die Verlustfunktion möglichst gering wird und die Gewichte dahingegen optimiert.

- Stochastic Gradient
- Adaptive Moment Estimation

### Gradient Descent
Die Optimierer benutzen im Hintergrund ein Gradientenverfahren. Hierbei werden Minima der Kurve gesucht. Das Minimum ist dort wo die Ableitung Null ist (tiefste Punkte der Kurve). Hierbei kann es mehrere lokale Minima geben. Die optimale Lösung hierbei st jedoch das globale Minima (tiefster Punkt insgesamt in der Kurve).

## Epoche
Ein einzelner Trainingsschritt wird als Epoche bezeichnet. Bei jeder Epoche wird jeweils Forward- und Backwards-Propagation durchgeführt.

## Batches
Die Trainingsdaten werden in Batches organisiert. Zu Beginn werden kleine Batches erstellt (viele kleine Daten). So kann das Modell schnell lernen der Gradient ist jedoch instabieler. Anschliessend werden die Batches vergrössert. Hierdurch steigt die Berechnungszeit, der Gradient wird jedoch stabiler und robuster. Dies wird bspw. mit dem Adabatch erreicht.

## Klassen und Labels
Grundsätzlich wird zwischen, Klassen, Samples und Labels unterschieden:

- Klassen: Entspricht einer Kategorie
- Samples: Entspricht den Datenpunkten
- Label: Ist die Klasse welche mit einem bestimmten Sample assoziiert ist

## Tensoren
Tensoren beschreiben die Dimensionalität der Daten. Tensoren beinhalten immer nummerische Werte:

- 0D Tensoren - Skalare
- 1D Tensoren - Vektoren
- 2D Tensoren - Matrizen
- 3D Tensoren - Sensoren höherer Ordnung (3D = Würfel)

## Achsen in Numpy in Bezug auf Python

- Die erste Achse (Achse 0) entspricht der Anzahl der Samples

## Feed Neural Network
Erlaubt nur das Arbeiten mit 2D Tensoren
