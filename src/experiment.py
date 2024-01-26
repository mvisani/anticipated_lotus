import numpy as np
from typing import Dict, Optional, Type
from .models.abstract_model import AbstractModel
from .utils import calculate_metrics


def experiments(
    features: Dict[str, np.ndarray],
    params,
    model_class: Type[AbstractModel],
    random_state: int,
) -> Dict[str, Dict[str, float]]:
    """
    Run experiments using the given features, model parameters, and random state.

    Args:
        features (Dict[str, np.ndarray]): A dictionary containing the features for training and testing.
        params: The parameters for the model.
        model_class (Type[AbstractModel]): The class of the model to be used.
        random_state (int): The random state for shuffling the training data.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary containing the evaluation scores for the training and testing sets.
    """
    # create the labels based on the features
    label_train_pos = np.ones(features["train_positive"].shape[0])
    label_train_neg = np.zeros(features["train_negative"].shape[0])
    label_test_pos = np.ones(features["test_positive"].shape[0])
    label_test_neg = np.zeros(features["test_negative"].shape[0])

    # concatenate positive and negative labels and training
    label_train = np.concatenate([label_train_pos, label_train_neg])
    label_test = np.concatenate([label_test_pos, label_test_neg])
    X_train = np.concatenate([features["train_positive"], features["train_negative"]])
    X_test = np.concatenate([features["test_positive"], features["test_negative"]])

    # randomize the order of the training data
    indices = np.arange(X_train.shape[0])
    rnd = np.random.RandomState(random_state)
    rnd.shuffle(indices)
    X_train = X_train[indices]
    label_train = label_train[indices]

    # fit the model
    model = model_class.from_params(params)
    model.fit(X_train, label_train)

    # predict training labels
    predictions_train = model.predict(X_train)
    scores_train = calculate_metrics(label_train, predictions_train)

    # predict the labels
    predictions_test = model.predict(X_test)
    scores_test = calculate_metrics(label_test, predictions_test)

    return {
        "train": scores_train,
        "test": scores_test,
    }
