from sklearn.tree import DecisionTreeClassifier
from ml.models.model import Model

class TreeModel(Model):
    def __init__(self, input_shape, output_shape):
        super().__init__(input_shape, output_shape)
        self.is_trained = False

    def build(self, max_depth=None, **params):
        self.model = DecisionTreeClassifier(max_depth=max_depth, **params)

    def fit(self, x, y):
        self.model.fit(x, y)
        self.is_trained = True

    def predict(self, x):
        if not self.is_trained:
            raise ValueError("Model must be trained using 'fit()' before prediction.")
        return self.model.predict(x)