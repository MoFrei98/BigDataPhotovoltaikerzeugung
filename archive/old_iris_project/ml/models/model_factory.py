from ml.models.svm_model import SVMModel
from ml.models.tree_model import TreeModel
from ml.models.knn_model import KNNModel
from ml.models.randomforest_model import RandomForestModel
from ml.models.lda_model import LDAModel
from ml.kernel.linear_kernel import LinearKernel
from ml.kernel.rbf_kernel import RBFKernel
from ml.kernel.poly_kernel import PolyKernel
from ml.kernel.sigmoid_kernel import SigmoidKernel

class ModelFactory:
    _registry = {
        # --- Strategy 1-4: SVM Variants with dedicated Kernel classes ---
        "svm_linear":    (SVMModel,         {"kernel": LinearKernel()}),
        "svm_rbf":       (SVMModel,         {"kernel": RBFKernel()}),
        "svm_poly":      (SVMModel,         {"kernel": PolyKernel(degree=3)}),
        "svm_sigmoid":   (SVMModel,         {"kernel": SigmoidKernel()}),
        # --- Strategy 5: Decision Tree ---
        "decision_tree": (TreeModel,        {"max_depth": 5}),
        # --- Strategy 6: K-Nearest Neighbors ---
        "knn":           (KNNModel,         {"n_neighbors": 5}),
        # --- Strategy 7: Random Forest ---
        "random_forest": (RandomForestModel,{"n_estimators": 100}),
        # --- Strategy 8: Linear Discriminant Analysis ---
        "lda":           (LDAModel,         {}),
    }

    @staticmethod
    def get_model(algo_type, input_shape, output_shape):
        if algo_type not in ModelFactory._registry:
            raise ValueError(f"Unknown algorithm type: {algo_type}")

        model_class, build_params = ModelFactory._registry[algo_type]
        model = model_class(input_shape, output_shape)
        model.build(**build_params)
        return model