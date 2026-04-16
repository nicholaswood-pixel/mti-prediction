"""Evaluation script for MTI forecast predictions."""
import json
import logging
import pathlib
import tarfile

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())



if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    pred_df = pd.read_csv("predictions.csv")
    valid = pred_df.dropna(subset=["actual_mti", "predicted_mti"]).copy()

    if valid.empty:
        mse = float("nan")
        std = float("nan")
        mae = float("nan")
        rmse = float("nan")
        r2 = float("nan")
        horizon_scores = []
    else:
        mse = mean_squared_error(valid["actual_mti"], valid["predicted_mti"])
        std = np.std(valid["actual_mti"] - valid["predicted_mti"])
        mae = mean_absolute_error(valid["actual_mti"], valid["predicted_mti"])
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(valid["actual_mti"], valid["predicted_mti"]))
        horizon_scores = []
        for horizon, horizon_df in valid.groupby("horizon"):
            horizon_mse = mean_squared_error(horizon_df["actual_mti"], horizon_df["predicted_mti"])
            horizon_scores.append(
                {
                    "horizon": horizon,
                    "rows": int(len(horizon_df)),
                    "mse": float(horizon_mse),
                }
            )

    report_dict = {
        "regression_metrics": {
            "mse": {
                "value": float(mse),
                "standard_deviation": float(std),
            },
            # For regression, "accuracy" is commonly represented using R^2.
            # Keep the original MSE structure unchanged for existing thresholds.
            "mae": float(mae),
            "rmse": float(rmse),
            "accuracy": float(r2),
            "horizons": horizon_scores,
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report with mse: %f", mse)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
