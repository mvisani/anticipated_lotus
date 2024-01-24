import numpy as np
from typing import Dict
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(
    label: np.ndarray,
    prediction: np.ndarray,
) -> Dict[str, float]:
    assert isinstance(label, np.ndarray)
    assert isinstance(prediction, np.ndarray)
    assert label.shape == prediction.shape
    return {
        "accuracy": accuracy_score(label, prediction),
        "balanced_accuracy": balanced_accuracy_score(label, prediction),
        "f1": f1_score(label, prediction),
        "recall": recall_score(label, prediction),
        "roc_auc": roc_auc_score(label, prediction),
        "mcc": matthews_corrcoef(label, prediction),
    }
