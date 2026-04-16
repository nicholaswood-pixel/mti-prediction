"""Training entry point for MTI LightGBM multi-horizon forecasting."""
import argparse
import json
import os
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = [
    "mti_lag_1",
    "mti_lag_2",
    "mti_lag_3",
    "mti_roll_3",
    "vessel_score_lag_1",
    "vessel_score_lag_3",
    "voyages_score_lag_1",
    "voyages_score_lag_3",
    "reporting_score_lag_1",
    "reporting_score_lag_3",
    "month_sin",
    "month_cos",
]
HORIZONS = [1, 2, 3]


def train_predict_horizon(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Train one horizon model and return month-10 anchored predictions."""
    working_df = df.copy()
    target_col = f"target_h{horizon}"
    pred_col = f"predicted_mti_h{horizon}"
    working_df[target_col] = working_df.groupby("imo_number")["mti_score"].shift(-horizon)

    train_df = working_df[(working_df["month"] <= 10) & (working_df[target_col].notna())].copy()
    test_df = working_df[working_df["month"] == 10].copy()
    test_df = test_df.dropna(subset=FEATURES)

    if train_df.empty or test_df.empty:
        return pd.DataFrame(columns=["imo_number", "predicted_month", "predicted_mti", "horizon"])

    model = lgb.LGBMRegressor(
        objective="regression",
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=500,
        min_data_in_leaf=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        random_state=42,
    )
    model.fit(train_df[FEATURES], train_df[target_col])

    test_df[pred_col] = model.predict(test_df[FEATURES])
    return pd.DataFrame(
        {
            "imo_number": test_df["imo_number"],
            "predicted_month": test_df["month"] + horizon,
            "predicted_mti": test_df[pred_col],
            "horizon": f"t+{horizon}",
        }
    )


def get_horizon_metrics(pred_with_actual: pd.DataFrame) -> List[Dict[str, float]]:
    """Compute RMSE/MSE/MAE for each horizon where actuals exist."""
    metrics = []
    valid = pred_with_actual.dropna(subset=["actual_mti", "predicted_mti"])
    for horizon, hdf in valid.groupby("horizon"):
        mse = mean_squared_error(hdf["actual_mti"], hdf["predicted_mti"])
        metrics.append(
            {
                "horizon": horizon,
                "count": int(len(hdf)),
                "mse": float(mse),
                "rmse": float(np.sqrt(mse)),
                "mae": float(mean_absolute_error(hdf["actual_mti"], hdf["predicted_mti"])),
            }
        )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    args = parser.parse_args()

    prepared_path = os.path.join(args.train, "prepared.csv")
    df_model = pd.read_csv(prepared_path)

    preds = [train_predict_horizon(df_model, horizon=h) for h in HORIZONS]
    predictions = pd.concat(preds, ignore_index=True)

    actuals = df_model[["imo_number", "month", "mti_score"]].rename(
        columns={"month": "predicted_month", "mti_score": "actual_mti"}
    )
    final_output = predictions.merge(actuals, on=["imo_number", "predicted_month"], how="left")
    final_output = final_output.sort_values(["imo_number", "predicted_month"]).reset_index(drop=True)

    horizon_metrics = get_horizon_metrics(final_output)
    valid = final_output.dropna(subset=["actual_mti", "predicted_mti"])
    overall_mse = (
        float(mean_squared_error(valid["actual_mti"], valid["predicted_mti"])) if not valid.empty else None
    )

    os.makedirs(args.model_dir, exist_ok=True)
    final_output.to_csv(os.path.join(args.model_dir, "predictions.csv"), index=False)
    with open(os.path.join(args.model_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall_mse": overall_mse,
                "evaluated_rows": int(len(valid)),
                "horizon_metrics": horizon_metrics,
            },
            f,
            indent=2,
        )
