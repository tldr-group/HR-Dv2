import numpy as np
from typing import Literal

from sklearn.preprocessing import scale
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
# from xgboost import XGBClassifier

AllowedClassifiers = Literal["linear", "logistic", "rf", "xgboost", "mlp"]
Classifiers = RidgeClassifier | LogisticRegression | RandomForestClassifier | MLPClassifier

MAX_ITERS = 3000


def get_classifier(
    classifier: AllowedClassifiers,
) -> Classifiers:
    if classifier == "linear":
        return RidgeClassifier(
            max_iter=MAX_ITERS,
            class_weight="balanced",
        )
    elif classifier == "logistic":
        return LogisticRegression(
            "l2",
            n_jobs=12,
            max_iter=MAX_ITERS,
            warm_start=False,
            class_weight="balanced",
        )
    elif classifier == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            max_features=2,
            max_depth=None,
            n_jobs=12,
            warm_start=False,
            class_weight="balanced",
        )
    elif classifier == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(100, 100, 100),
            activation="relu",
            solver="adam",
            max_iter=MAX_ITERS,
            warm_start=False,
        )
    else:
        raise Exception("classifier not supported")


def flatten_mask_training_data(feature_stack: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given $feature_stack and $labels, flatten both and reshape accordingly. Add a class offset if using XGB gpu."""
    h, w, feat = feature_stack.shape
    flat_labels = labels.reshape((h * w))
    flat_features = feature_stack.reshape((h * w, feat))
    labelled_mask = np.nonzero(flat_labels)

    fit_data = flat_features[labelled_mask[0], :]
    target_data = flat_labels[labelled_mask[0]]
    return fit_data, target_data


def get_training_data(
    features: list[np.ndarray], labels: list[np.ndarray], std_scale: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    init = False
    for label, feat in zip(labels, features):
        if init is False:
            fit_data, target_data = flatten_mask_training_data(feat, label)
            all_fit_data = fit_data
            all_target_data = target_data
            init = True
        else:
            fit_data, target_data = flatten_mask_training_data(feat, label)
            all_fit_data = np.concatenate((all_fit_data, fit_data), axis=0)
            all_target_data = np.concatenate((all_target_data, target_data), axis=0)
    # if std_scale:
    #     all_fit_data = scale(all_fit_data, axis=0)
    return all_fit_data, all_target_data


def train_model(
    model: Classifiers,
    features: list[np.ndarray],
    labels: list[np.ndarray],
) -> None:
    """Train the model using the features and labels."""
    fit_data, target_data = get_training_data(features, labels)
    model.fit(fit_data, target_data)
    return model


def apply_model(model: Classifiers, features: np.ndarray) -> np.ndarray:
    """Apply the model to the features and return the predictions."""
    h, w, feat = features.shape
    flat_features = features.reshape((h * w, feat))
    predictions = model.predict(flat_features)
    return predictions.reshape((h, w))
