from ml.kernel.kernel import Kernel

class SigmoidKernel(Kernel):
    def __init__(self, gamma: str | float = "scale"):
        self.gamma = gamma

    @property
    def name(self) -> str:
        return "sigmoid"

    def get_params(self) -> dict:
        return {"kernel": "sigmoid", "gamma": self.gamma}