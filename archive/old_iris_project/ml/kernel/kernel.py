from abc import ABC, abstractmethod

class Kernel(ABC):
    @abstractmethod
    def get_params(self) -> dict:
        """Gibt die Kernel-Parameter für sklearn zurück."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass