from ml.kernel.kernel import Kernel

class PolyKernel(Kernel):
    def __init__(self, degree: int = 3, gamma: str | float = "scale"):
        self.degree = degree
        self.gamma = gamma

    @property
    def name(self) -> str:
        return "poly"

    def get_params(self) -> dict:
        return {"kernel": "poly", "degree": self.degree, "gamma": self.gamma}