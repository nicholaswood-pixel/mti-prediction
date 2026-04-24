"""Evaluate digit classifier from training artifact; write SageMaker-style evaluation.json."""
import json
import logging
import pathlib
import tarfile
import tempfile

import joblib
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def compute_accuracy(model, X_test, y_test) -> float:
    """Return multiclass accuracy in [0, 1]."""
    pred = model.predict(X_test)
    return float(np.mean(pred == y_test))


def build_evaluation_dict(accuracy: float) -> dict:
    """Shape compatible with pipeline JsonGet on classification_metrics.accuracy.value."""
    return {
        "classification_metrics": {
            "accuracy": {
                "value": float(accuracy),
            },
        },
    }


if __name__ == "__main__":
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(model_path, "r:gz") as tar:
            tar.extractall(path=tmp)

        root = pathlib.Path(tmp)
        model_file = next(root.rglob("model.joblib"), None)
        test_file = next(root.rglob("test_data.joblib"), None)
        if model_file is None:
            raise FileNotFoundError("model.joblib not found inside model.tar.gz.")
        if test_file is None:
            raise FileNotFoundError("test_data.joblib not found inside model.tar.gz.")

        model = joblib.load(model_file)
        test_bundle = joblib.load(test_file)
        X_test, y_test = test_bundle["X_test"], test_bundle["y_test"]

    acc = compute_accuracy(model, X_test, y_test)
    logger.info("Test accuracy: %.4f", acc)

    report = build_evaluation_dict(acc)
    out_dir = pathlib.Path("/opt/ml/processing/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = out_dir / "evaluation.json"
    eval_path.write_text(json.dumps(report))
    logger.info("Wrote %s", eval_path)
