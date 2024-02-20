from hyperopt import hp
from .abstract_model import AbstractModel
import numpy as np
from typing import Optional
from xgboost import XGBClassifier
from dict_hash import sha256
import compress_pickle


class XGBoost(AbstractModel):
    def __init__(
        self,
        n_estimators: int,
        max_depth: int,
        max_leaves: int,
        grow_policy: int,
        booster: str,
        tree_method: str,
    ) -> None:
        super().__init__()
        params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_leaves=max_leaves,
            grow_policy=grow_policy,
            booster=booster,
            tree_method=tree_method,
            n_jobs=-1,
            random_state=9876,
            device="cpu",
        )
        self.model = XGBClassifier(
            **params,
        )
        self.params = params

    def search_space():
        return {
            "n_estimators": hp.choice("n_estimators", [10, 20, 50, 100, 200, 300]),
            "max_depth": hp.uniformint("max_depth", 1, 20),
            "max_leaves": hp.uniformint("max_leaves", 0, 50),
            "grow_policy": hp.choice("grow_policy", ["depthwise", "lossguide"]),
            "booster": hp.choice("booster", ["gbtree", "gblinear", "dart"]),
            "tree_method": hp.choice(
                "tree_method", ["auto", "exact", "approx", "hist"]
            ),
        }

    @staticmethod
    def from_params(params):
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            max_leaves=params["max_leaves"],
            grow_policy=params["grow_policy"],
            booster=params["booster"],
            tree_method=params["tree_method"],
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
                "model": "XGBoost",
            }
        )
