from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, input_shape, output_shape):
        self.model = None
        self.input_shape = input_shape
        self.output_shape = output_shape

    @abstractmethod
    def build(self, **params):
        pass

    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass