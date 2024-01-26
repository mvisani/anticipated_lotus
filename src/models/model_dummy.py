from hyperopt import hp
from .abstract_model import AbstractModel
import numpy as np
from typing import Optional


class ModelDummy(AbstractModel):
    @staticmethod
    def from_params(params):
        return ModelDummy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.random.rand(X.shape[0])

    def fit(
        self, X: np.ndarray, y: np.ndarray, weight: Optional[np.ndarray] = None
    ) -> None:
        pass

    def dump_model(self, path):
        pass

    def load_model(path):
        pass

    @staticmethod
    def search_space():
        # define a search space
        space = {
            "x": hp.uniform("x", 0, 1),
        }
        return space

    def consistent_hash(self) -> str:
        return "dummy"
