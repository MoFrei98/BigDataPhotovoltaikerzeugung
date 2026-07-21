from ml.kernel.kernel import Kernel

class RBFKernel(Kernel):
    def __init__(self, gamma: str | float = "scale"):
        self.gamma = gamma

    @property
    def name(self) -> str:
        return "rbf"

    def get_params(self) -> dict:
        return {"kernel": "rbf", "gamma": self.gamma}