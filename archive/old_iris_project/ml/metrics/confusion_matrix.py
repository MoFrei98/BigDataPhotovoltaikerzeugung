from sklearn.metrics import confusion_matrix
from ml.metrics.metric import Metric
import pandas as pd

class ConfusionMatrix(Metric):
    def __init__(self):
        super().__init__(name="Confusion Matrix")

    def calculate(self, y_true, y_pred):
        labels = sorted(y_true.unique())
        cm = confusion_matrix(y_true, y_pred)

        summary = []
        for i, label in enumerate(labels):
            # Korrekte sind auf der Diagonale (i == j)
            correct = cm[i][i]
            # Fehler sind alle in der Zeile, außer dem korrekten Wert
            total_actual = sum(cm[i])
            errors = total_actual - correct

            line = f"{label:12} -> Corret: {correct:2} | Incorrect: {errors:2}"
            summary.append(line)

        return "\n".join(summary)