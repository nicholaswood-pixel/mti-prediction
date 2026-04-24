"""Download and split MNIST (784-d OpenML) for downstream training."""
import argparse
import logging
import os

import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def load_and_split_digits(max_train_samples: int, max_test_samples: int, random_state: int = 42):
    """Fetch OpenML MNIST 784, stratified train/test split, then subsample for speed if limits are set."""
    logger.info("Fetching OpenML mnist_784 (requires outbound network on the processing job).")
    mnist = fetch_openml(name="mnist_784", version=1, as_frame=False)
    X, y = mnist.data, np.asarray(mnist.target, dtype=np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    if max_train_samples > 0 and X_train.shape[0] > max_train_samples:
        X_train, y_train = X_train[:max_train_samples], y_train[:max_train_samples]
    if max_test_samples > 0 and X_test.shape[0] > max_test_samples:
        X_test, y_test = X_test[:max_test_samples], y_test[:max_test_samples]
    logger.info(
        "Prepared arrays: train=%s test=%s feature_dim=%s",
        X_train.shape[0],
        X_test.shape[0],
        X_train.shape[1],
    )
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/opt/ml/processing/prepared",
        help="SageMaker processing output directory for prepared artifacts.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=12_000,
        help="Cap training rows after split (0 = use full split).",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=3_000,
        help="Cap test rows after split (0 = use full split).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    X_train, y_train, X_test, y_test = load_and_split_digits(
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        random_state=args.random_state,
    )
    out_path = os.path.join(args.output_dir, "dev.joblib")
    joblib.dump(
        {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test},
        out_path,
    )
    logger.info("Wrote %s", out_path)
