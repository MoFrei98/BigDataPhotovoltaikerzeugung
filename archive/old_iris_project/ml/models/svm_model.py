from sklearn.svm import SVC
from ml.models.model import Model
from ml.kernel.kernel import Kernel

class SVMModel(Model):
    def __init__(self, input_shape, output_shape):
        super().__init__(input_shape, output_shape)
        self.model = None
        self.is_trained = False

    def build(self, kernel=None, **params):
        """
        Build SVM model with kernel support.
        
        Args:
            kernel: Either a Kernel object or a string ('rbf' default)
            **params: Additional parameters to pass to SVC
        """
        if kernel is None or isinstance(kernel, str):
            # Fallback for string kernels
            self.model = SVC(kernel=kernel or 'rbf', **params)
        elif isinstance(kernel, Kernel):
            # Use Kernel object and merge its params with additional params
            kernel_params = kernel.get_params()
            kernel_params.update(params)
            self.model = SVC(**kernel_params)
        else:
            raise TypeError("kernel must be a Kernel object or a string")

    def fit(self, x, y):
        self.model.fit(x, y)
        self.is_trained = True

    def predict(self, x):
        if not self.is_trained:
            raise ValueError("Model must be trained using 'fit()' before prediction.")
        return self.model.predict(x)