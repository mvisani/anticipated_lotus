from .abstract_model import AbstractModel
import numpy as np
from typing import Optional


class ModelDummy(AbstractModel):
    def from_params(params):
        return ModelDummy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.random.rand(X.shape[0])

    def fit(self, X: np.ndarray, y: np.ndarray, weight: Optional[np.ndarray]) -> None:
        pass

    def dump_model(self, path):
        pass

    def load_model(path):
        pass
