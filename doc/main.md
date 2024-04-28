---
titlepage: true
titlepage-color: "ffffff"
titlepage-rule-color: "ffffff"
titlepage-text-color: "000000"
toc: true
toc-own-page: true
colorlinks: false
title: Zusammenfassung 
subtitle: Deep Learning
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

# Mögliche Prüfungsfragen

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

# Grundlagen


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

# Deep Learning
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

## Gradient Descent
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

# Tensoren
Tensoren beschreiben die Dimensionalität der Daten. Tensoren beinhalten immer nummerische Werte:

- 0D Tensoren - Skalare
- 1D Tensoren - Vektoren
- 2D Tensoren - Matrizen
- 3D Tensoren - Sensoren höherer Ordnung (3D = Würfel)

## Achsen in Numpy in Bezug auf Python

- Die erste Achse (Achse 0) entspricht der Anzahl der Samples

# Feed Neural Network
Erlaubt nur das Arbeiten mit 2D Tensoren

# Regularisierung
Regularisierung ist eine Methode **um Overfitting zu verhindern**. Dies wird mithilfe eines **Bestrafungsgewichtes** erreicht, welches **zur Loss-Funktion hinzugerechnet wird**. Regularisierung hat mehrere Vorteile:

- Verbesserung der generellen Performanz des Models
- Verbesserung der Generalisierung von neuen Daten
- Die Verwendung des Bestrafungsgewichtes ermutigt das Modell kleinere Bias Werte zu verwenden

> Die Regularisierung selbst kann über den Hyperparameter `regularization` gesteuert werden. Dieser definiert, wie fest bestraft werden soll (Gewicht des Bestrafungswertes), d.h. der Lambda Parameter

## L1 Regularisierung

Die Formel für die L1 Regularisierung lautet wie folgt:

$$ Loss = Data Loss + \lambda * |w| $$

- $DataLoss$ ist die Verlustfunktion des neuronalen Netzwerkes
- $w$ ist der Vektor mit den Gewichten (Bias)
- $|w|$ ist die L1 Norm der Gewichte (Bias)
- $\lambda$ ist der Hyperparameter **regularization** 

> Die L1 Norm ist die Summe der absoluten Werte der Gewichte

## L2 Regularisierung

Die Formel für die L2 Regularisierung lautet wie folgt:

$$ Loss = Data Loss + \lambda * ||w||^2 $$

- $DataLoss$ ist die Verlustfunktion des neuronalen Netzwerkes
- $w$ ist der Vektor mit den Gewichten (Bias)
- $||w||$ ist die L2 Norm der Gewichte (Bias)
- $\lambda$ ist der Hyperparameter **regularization** 

> Die L2 Norm ist die Wurzel Summe der quadrierten Gewichte

## Unterschiede L1 und L2 Regularisierung

- Die L1 Regularisierung tendiert dazu **Gewichtsvektoren mit vielen Nullen** zu erzeugen
- Die L1 Regularisierung ist nützlich für die Feature-Auswahl, nicht relevante Features werden durch die Regularisierung entfernt
- Die L2 Regulariiserung verteilt die Gewichte einheitlicher über alle Features

## Dropout

Dropout ist eine Regularisierungsmethode, bei welcher während des Trainings zufälligerweise Teile des Neuronalen Netzes "deaktiviert" werden. Dies führt dazu, dass die anderen Teilnehmer des Netzes dazu gezwungen werden, mehr robuste Features zu lernen.

> Dropout verhindert, dass das Modell sich zu sehr auf einzelne Neuronen verlässt.

Gängige Dropout-Raten sind:

- 0.8 für den Input Layer
- 0.5 für Hidden Layer

Der Dropout wird bei jedem Forward- und Backward Propagation Schritt pro Batch angewendet.

## Early Stopping

Early Stopping verhindert, dass das Modell weiter trainiert, wenn beim Validierungsschritt keine signifikante Verbesserung erfolgt. Early Stopping verhindert ebenfalls das Overfitting der Trainingsdaten.

## Bagging

Die Grundidee des Bagging besteht darin, dass mehrere Modelle gleichzeitig verwendet werden. Als Vorhersagewert wird dabei der Durchschnitt verwendet. Der Ablauf des Baggings ist wie folgt:

- Erstellung von k Datensätzen der ursprünglichen Grösse
- Trainieren von k Modellen
- Test: Nutze Durschnitt der Vorhersage aller Modelle

## Dataset Augmentation

Dataset Augmentation ist eine Technik, bei denen neue Trainingsdaten generiert werden, indem verschiedene Transformationen angewendet werden:

- Farbveränderung
- Skalierung
- Zoom

Mithilfe von Data Augmentation kann das Model robustere Features lernen, zudem verbessert sich die Fähigkeit neue Daten zu verallgemeinern.

> Achtung: Data Augmentation funktioniert nicht für jede Art von Datensätzen, bspw. MNIST

## Batch Normalisierung

Mithilfe der Batch Normalisierung werden die Input Daten von Layern normalisiert:

- Stabilisierung der Verteilung bei der Aktivierungsfunktion
- Verhinderung von Overfitting
- Verbesserung der generellen Performanz des Modells

