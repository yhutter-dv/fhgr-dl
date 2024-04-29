---
titlepage: true
titlepage-color: "ffffff"
titlepage-rule-color: "ffffff"
titlepage-text-color: "000000"
toc-own-page: true
colorlinks: false
title: Aufgabenblatt 02 
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


## Regularisierung
Regularisierung ist eine Methode **um Overfitting zu verhindern**. 

> Die Regularisierung selbst kann über den Hyperparameter `regularization` gesteuert werden. Dieser definiert, wie fest bestraft werden soll (Gewicht des Bestrafungswertes), d.h. der Lambda Parameter

### L1 Regularisierung

Die Formel für die L1 Regularisierung lautet wie folgt:

$$ Loss = Data Loss + \lambda * |w| $$

- $DataLoss$ ist die Verlustfunktion des neuronalen Netzwerkes
- $w$ ist der Vektor mit den Gewichten (Bias)
- $|w|$ ist die L1 Norm der Gewichte (Bias)
- $\lambda$ ist der Hyperparameter **regularization** 

> Die L1 Norm ist die Summe der absoluten Werte der Gewichte

### L2 Regularisierung

Die Formel für die L2 Regularisierung lautet wie folgt:

$$ Loss = Data Loss + \lambda * ||w||^2 $$

- $DataLoss$ ist die Verlustfunktion des neuronalen Netzwerkes
- $w$ ist der Vektor mit den Gewichten (Bias)
- $||w||$ ist die L2 Norm der Gewichte (Bias)
- $\lambda$ ist der Hyperparameter **regularization** 

> Die L2 Norm ist die Wurzel Summe der quadrierten Gewichte

### Unterschiede L1 und L2 Regularisierung

- Die L1 Regularisierung tendiert dazu **Gewichtsvektoren mit vielen Nullen** zu erzeugen
- Die L1 Regularisierung ist nützlich für die Feature-Auswahl, nicht relevante Features werden durch die Regularisierung entfernt
- Die L2 Regulariiserung verteilt die Gewichte einheitlicher über alle Features

### Dropout

Dropout ist eine Regularisierungsmethode, bei welcher während des Trainings zufälligerweise Teile des Neuronalen Netzes "deaktiviert" werden. Dies führt dazu, dass die anderen Teilnehmer des Netzes dazu gezwungen werden, mehr robuste Features zu lernen.

> Dropout verhindert, dass das Modell sich zu sehr auf einzelne Neuronen verlässt.

Gängige Dropout-Raten sind:

- 0.8 für den Input Layer
- 0.5 für Hidden Layer

Der Dropout wird bei jedem Forward- und Backward Propagation Schritt pro Batch angewendet.

### Early Stopping

Early Stopping verhindert, dass das Modell weiter trainiert, wenn beim Validierungsschritt keine signifikante Verbesserung erfolgt. Early Stopping verhindert ebenfalls das Overfitting der Trainingsdaten.

### Bagging

Die Grundidee des Bagging besteht darin, dass mehrere Modelle gleichzeitig verwendet werden. Als Vorhersagewert wird dabei der Durchschnitt verwendet. Der Ablauf des Baggings ist wie folgt:

- Erstellung von k Datensätzen der ursprünglichen Grösse
- Trainieren von k Modellen
- Test: Nutze Durschnitt der Vorhersage aller Modelle

### Dataset Augmentation

Dataset Augmentation ist eine Technik, bei denen neue Trainingsdaten generiert werden, indem verschiedene Transformationen angewendet werden:

- Farbveränderung
- Skalierung
- Zoom

Mithilfe von Data Augmentation kann das Model robustere Features lernen, zudem verbessert sich die Fähigkeit neue Daten zu verallgemeinern.

> Achtung: Data Augmentation funktioniert nicht für jede Art von Datensätzen, bspw. MNIST

### Batch Normalisierung

Mithilfe der Batch Normalisierung werden die Input Daten von Layern normalisiert:

- Stabilisierung der Verteilung bei der Aktivierungsfunktion
- Verhinderung von Overfitting
- Verbesserung der generellen Performanz des Modells

