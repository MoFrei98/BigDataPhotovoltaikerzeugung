from ml.kernel.kernel import Kernel

class LinearKernel(Kernel):
    @property
    def name(self) -> str:
        return "linear"

    def get_params(self) -> dict:
        return {"kernel": "linear"}