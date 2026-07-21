from sklearn.metrics import accuracy_score
from ml.metrics.metric import Metric

class Accuracy(Metric):
    def __init__(self, metric_function=accuracy_score):
        super().__init__(name="Accuracy")
        # save function as an object
        self.metric_function = metric_function

    def calculate(self, y_true, y_pred):
        # call saved object as a function
        return self.metric_function(y_true, y_pred)