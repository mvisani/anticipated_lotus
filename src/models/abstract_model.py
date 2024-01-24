import numpy as np
from typing import Optional


class AbstractModel:
    def fit(self, X: np.ndarray, y: np.ndarray, weight: Optional[np.ndarray]) -> None:
        raise NotImplementedError(
            "This method needs to be implemented, please implement it."
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the labels of"""
        raise NotImplementedError(
            "This method needs to be implemented, please implement it."
        )

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
