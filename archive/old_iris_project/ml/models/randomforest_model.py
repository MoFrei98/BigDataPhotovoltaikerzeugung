from sklearn.ensemble import RandomForestClassifier
from ml.models.model import Model

class RandomForestModel(Model):
    def __init__(self, input_shape, output_shape):
        super().__init__(input_shape, output_shape)
        self.is_trained = False

    def build(self, n_estimators=100, **params):
        self.model = RandomForestClassifier(n_estimators=n_estimators, **params)

    def fit(self, x, y):
        self.model.fit(x, y)
        self.is_trained = True

    def predict(self, x):
        if not self.is_trained:
            raise ValueError("Model must be trained using 'fit()' before prediction.")
        return self.model.predict(x)