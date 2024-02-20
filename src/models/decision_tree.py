from hyperopt import hp
from .abstract_model import AbstractModel
import numpy as np
from typing import Optional
from sklearn.tree import DecisionTreeClassifier
from dict_hash import sha256
import compress_pickle


class DecisionTree(AbstractModel):
    def __init__(
        self,
        criterion: str,
        splitter: str,
        max_depth: int,
        min_samples_split: float,
        min_samples_leaf: float,
        max_features,
    ) -> None:
        super().__init__()
        params = dict(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=5764,
            class_weight="balanced",
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
        )
        self.model = DecisionTreeClassifier(
            **params,
        )
        self.params = params

    def search_space():
        return {
            "criterion": hp.choice("criterion", ["gini", "entropy", "log_loss"]),
            "splitter": hp.choice("splitter", ["best", "random"]),
            "max_depth": hp.uniformint("max_depth", 1, 10),
            "min_samples_split": hp.uniform("min_samples_split", 0, 1),
            "min_samples_leaf": hp.uniform("min_samples_leaf", 0, 1),
            "max_features": hp.choice("max_features", ["sqrt", "log2"]),
        }

    @staticmethod
    def from_params(params):
        assert isinstance(params, dict)
        assert isinstance(params["criterion"], str), (
            f"Criterion must be a string, but is {type(params['criterion'])}. "
            f"The provided params are {params}."
        )
        assert isinstance(params["splitter"], str)
        assert isinstance(params["max_depth"], int)
        assert isinstance(params["min_samples_split"], float)
        assert isinstance(params["min_samples_leaf"], float)
        assert isinstance(params["max_features"], str)
        assert params["criterion"] in ["gini", "entropy", "log_loss"]
        assert params["splitter"] in ["best", "random"]
        assert params["max_depth"] >= 1
        assert params["min_samples_split"] >= 0
        assert params["min_samples_leaf"] >= 0
        assert params["max_features"] in ["sqrt", "log2"]

        return DecisionTree(
            criterion=params["criterion"],
            splitter=params["splitter"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
        )

    def fit(
        self, X: np.ndarray, y: np.ndarray, weight: Optional[np.ndarray] = None
    ) -> None:
        self.model.fit(X, y, sample_weight=weight)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def dump_model(self, path: str) -> None:
        compress_pickle.dump(self, path)

    @staticmethod
    def load_model(path: str):
        return compress_pickle.load(path)

    def consistent_hash(self) -> str:
        return sha256(
            {
                **self.params,
                "model": "DecisionTree",
            }
        )
