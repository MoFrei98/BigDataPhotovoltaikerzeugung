from sklearn.neighbors import KNeighborsClassifier
from ml.models.model import Model

class KNNModel(Model):
    def __init__(self, input_shape, output_shape):
        super().__init__(input_shape, output_shape)
        self.is_trained = False

    def build(self, n_neighbors=5, **params):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, **params)

    def fit(self, x, y):
        self.model.fit(x, y)
        self.is_trained = True

    def predict(self, x):
        if not self.is_trained:
            raise ValueError("Model must be trained using 'fit()' before prediction.")
        return self.model.predict(x)