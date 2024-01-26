import numpy as np
from typing import Optional
from dict_hash import Hashable


class AbstractModel(Hashable):
    def fit(
        self, X: np.ndarray, y: np.ndarray, weight: Optional[np.ndarray] = None
    ) -> None:
        raise NotImplementedError(
            "This method needs to be implemented, please implement it."
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the labels of"""
        raise NotImplementedError(
            "This method needs to be implemented, please implement it."
        )

    @staticmethod
    def from_params(params) -> "Self":
        """Returns a new model with provided parameters."""
        raise NotImplementedError(
            "This method needs to be implemented, please implement it."
        )

    def dump_model(self, path):
        """Saves model to specifer path."""
        raise NotImplementedError(
            "This method needs to be implemented, please implement it."
        )

    def load_model(path):
        """Loads model from specified path"""
        raise NotImplementedError(
            "This method needs to be implemented, please implement it."
        )

    @staticmethod
    def search_space():
        """Returns the search space for the model."""
        raise NotImplementedError(
            "This method needs to be implemented, please implement it."
        )
