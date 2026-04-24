import numpy as np

from pipelines.dev.evaluate import build_evaluation_dict, compute_accuracy
from pipelines.dev.train import train_digit_classifier


def test_pipelines_importable():
    import pipelines  # noqa: F401


def test_train_digit_classifier_fits():
    rng = np.random.default_rng(0)
    X = 0.1 * rng.standard_normal((400, 32)).astype(np.float64)
    y = rng.integers(0, 10, size=400)
    model = train_digit_classifier(X, y, max_iter=200, random_state=0)
    assert hasattr(model, "predict")
    assert model.score(X, y) >= 0.0


def test_compute_accuracy_perfect():
    rng = np.random.default_rng(1)
    X = 0.1 * rng.standard_normal((50, 8)).astype(np.float64)
    y = rng.integers(0, 3, size=50)
    model = train_digit_classifier(X, y, max_iter=300, random_state=0)
    acc = compute_accuracy(model, X, y)
    assert 0.0 <= acc <= 1.0


def test_build_evaluation_dict_shape():
    d = build_evaluation_dict(0.9123)
    assert d["classification_metrics"]["accuracy"]["value"] == 0.9123
