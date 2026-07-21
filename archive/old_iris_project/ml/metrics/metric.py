from abc import ABC, abstractmethod

class Metric(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def calculate(self, y_true, y_pred):
        pass