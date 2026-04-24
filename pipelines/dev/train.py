"""Train a lightweight digit classifier on prepared MNIST arrays."""
import argparse
import logging
import os
from contextlib import nullcontext

import joblib
from sklearn.linear_model import SGDClassifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# SGD defaults (kept in sync with train_digit_classifier for MLflow param logging)
_DEFAULT_RANDOM_STATE = 42
_DEFAULT_LOSS = "log_loss"
_DEFAULT_TOL = 1e-3
_DEFAULT_ALPHA = 1e-4


def train_digit_classifier(X_train, y_train, max_iter: int = 100, random_state: int = 42):
    """Fit an SGD logistic-style multiclass model (template default; replace with your model)."""
    model = SGDClassifier(
        loss=_DEFAULT_LOSS,
        max_iter=max_iter,
        random_state=random_state,
        tol=_DEFAULT_TOL,
        alpha=_DEFAULT_ALPHA,
    )
    model.fit(X_train, y_train)
    return model


def _mlflow_run_context(mlflow_tracking_uri: str, experiment_name: str):
    if not (mlflow_tracking_uri or "").strip():
        return nullcontext()

    import mlflow

    mlflow.set_tracking_uri(mlflow_tracking_uri.strip())
    mlflow.set_experiment(experiment_name.strip() or "dev-digit-training")
    run_name = os.environ.get("SM_TRAINING_JOB_NAME") or None
    return mlflow.start_run(run_name=run_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=os.environ.get("MLFLOW_TRACKING_URI", ""),
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "dev-digit-training"),
    )
    args = parser.parse_args()

    bundle_path = os.path.join(args.train, "dev.joblib")
    logger.info("Loading prepared data from %s", bundle_path)
    bundle = joblib.load(bundle_path)
    X_train, y_train = bundle["X_train"], bundle["y_train"]
    X_test, y_test = bundle["X_test"], bundle["y_test"]

    use_mlflow = bool((args.mlflow_tracking_uri or "").strip())
    run_ctx = _mlflow_run_context(args.mlflow_tracking_uri, args.mlflow_experiment_name)
    if use_mlflow:
        logger.info(
            "MLflow tracking enabled (experiment=%s)",
            (args.mlflow_experiment_name or "").strip() or "dev-digit-training",
        )

    with run_ctx:
        if use_mlflow:
            import mlflow

            mlflow.log_params(
                {
                    "max_iter": args.max_iter,
                    "random_state": _DEFAULT_RANDOM_STATE,
                    "loss": _DEFAULT_LOSS,
                    "tol": _DEFAULT_TOL,
                    "alpha": _DEFAULT_ALPHA,
                    "n_train_samples": int(X_train.shape[0]),
                    "n_test_samples": int(X_test.shape[0]),
                    "n_features": int(X_train.shape[1]) if X_train.ndim > 1 else 1,
                }
            )
            if job_name := os.environ.get("SM_TRAINING_JOB_NAME"):
                mlflow.set_tag("sagemaker_training_job_name", job_name)

        model = train_digit_classifier(X_train, y_train, max_iter=args.max_iter)
        train_acc = float(model.score(X_train, y_train))
        test_acc = float(model.score(X_test, y_test))
        logger.info("Training accuracy (in-sample): %.4f", train_acc)
        logger.info("Test accuracy: %.4f", test_acc)

        if use_mlflow:
            import mlflow

            mlflow.log_metrics({"train_accuracy": train_acc, "test_accuracy": test_acc})
            mlflow.sklearn.log_model(model, artifact_path="model")

    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    joblib.dump(
        {"X_test": X_test, "y_test": y_test},
        os.path.join(args.model_dir, "test_data.joblib"),
    )
    logger.info("Wrote model.joblib and test_data.joblib to %s", args.model_dir)
