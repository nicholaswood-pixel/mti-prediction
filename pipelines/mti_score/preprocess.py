"""Feature engineering for MTI score forecasting."""
import argparse
import logging
import os
import pathlib

import boto3
import numpy as np
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

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


def parse_s3_uri(s3_uri):
    """Split S3 URI into bucket and key."""
    bucket = s3_uri.split("/")[2]
    key = "/".join(s3_uri.split("/")[3:])
    return bucket, key


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)

    bucket, key = parse_s3_uri(args.input_data)
    input_file = f"{base_dir}/data/mti_scores_history.csv"

    logger.info("Downloading data from bucket=%s key=%s", bucket, key)
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, input_file)

    logger.info("Loading MTI data")
    df = pd.read_csv(input_file, low_memory=False)
    os.unlink(input_file)

    # Aligns with notebook: drop emissions score if missing/not present.
    df = df.drop(columns=["emissions_score"], errors="ignore")
    df = df.sort_values(["imo_number", "year", "month"]).reset_index(drop=True)

    numeric_cols = [
        "mti_score",
        "vessel_score",
        "voyages_score",
        "reporting_score",
        "month",
        "year",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    for lag in [1, 2, 3]:
        df[f"mti_lag_{lag}"] = df.groupby("imo_number")["mti_score"].shift(lag)

    for col in ["vessel_score", "voyages_score", "reporting_score"]:
        df[f"{col}_lag_1"] = df.groupby("imo_number")[col].shift(1)
        df[f"{col}_lag_3"] = df.groupby("imo_number")[col].shift(3)

    df["mti_roll_3"] = df.groupby("imo_number")["mti_score"].transform(
        lambda s: s.shift(1).rolling(3).mean()
    )

    model_df = df.dropna(subset=FEATURES).reset_index(drop=True)
    model_df = model_df[["imo_number", "year", "month", "mti_score"] + FEATURES]

    pathlib.Path(f"{base_dir}/prepared").mkdir(parents=True, exist_ok=True)
    output_file = f"{base_dir}/prepared/prepared.csv"
    logger.info("Writing prepared dataset to %s", output_file)
    model_df.to_csv(output_file, index=False)
