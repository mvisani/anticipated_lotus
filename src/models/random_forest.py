from hyperopt import hp
from .abstract_model import AbstractModel
import numpy as np
from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from dict_hash import sha256


class RandomForest(AbstractModel):
    def __init__(
        self,
        n_estimators: int,
        criterion: str,
        min_samples_split: float,
        min_samples_leaf: float,
        max_features,
        oob_score,
    ) -> None:
        super().__init__()
        params = dict(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=None,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            oob_score=oob_score,
            random_state=658749,
            n_jobs=-1,
        )
        self.model = RandomForestClassifier(
            **params,
        )

    def search_space():
        return {
            "n_estimators": hp.choice("n_estimators", [50, 100, 200, 300, 1000]),
            "criterion": hp.choice("criterion", ["gini", "entropy", "log_loss"]),
            "min_samples_split": hp.uniform("min_samples_split", 0, 1),
            "min_samples_leaf": hp.uniform("min_samples_leaf", 0, 1),
            "max_features": hp.choice("max_features", ["sqrt", "log2"]),
            "oob_score": hp.choice("oob_score", [True, False]),
        }

    @staticmethod
    def from_params(params):
        return RandomForest(
            n_estimators=params["n_estimators"],
            criterion=params["criterion"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            oob_score=params["oob_score"],
        )

    def fit(
        self, X: np.ndarray, y: np.ndarray, weight: Optional[np.ndarray] = None
    ) -> None:
        self.model.fit(X, y, sample_weight=weight)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def consistent_hash(self) -> str:
        return sha256(
            {
                **self.params,
                "model": "RandomForest",
            }
        )
