"""Feature engineering for MTI score forecasting."""
import argparse
import logging
import os
import pathlib
import time

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


def load_from_athena(database: str, table: str, workgroup: str, output_s3: str) -> pd.DataFrame:
    """Export Athena table to S3 via UNLOAD, then read as pandas DataFrame."""
    athena = boto3.client("athena")
    s3 = boto3.client("s3")

    if not output_s3.endswith("/"):
        output_s3 = output_s3 + "/"

    query = (
        f"UNLOAD (SELECT * FROM {database}.{table}) "
        f"TO '{output_s3}' "
        "WITH (format='TEXTFILE', field_delimiter=',', compression='GZIP', include_header=true)"
    )

    logger.info("Starting Athena UNLOAD for %s.%s", database, table)
    resp = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": output_s3},
        WorkGroup=workgroup,
    )
    qid = resp["QueryExecutionId"]

    while True:
        qe = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]
        state = qe["Status"]["State"]
        if state in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            break
        time.sleep(5)

    if state != "SUCCEEDED":
        reason = qe["Status"].get("StateChangeReason", "unknown")
        raise RuntimeError(f"Athena query failed ({state}): {reason}")

    output_location = qe["ResultConfiguration"]["OutputLocation"]
    bucket, prefix = parse_s3_uri(output_location)
    logger.info("Athena UNLOAD output prefix: s3://%s/%s", bucket, prefix)

    # UNLOAD writes one or more gz files under the prefix.
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".gz"):
                keys.append(key)
    if not keys:
        raise RuntimeError("Athena UNLOAD produced no .gz files.")

    local_dir = "/opt/ml/processing/data/athena"
    pathlib.Path(local_dir).mkdir(parents=True, exist_ok=True)
    frames = []
    for i, key in enumerate(sorted(keys)):
        local_path = os.path.join(local_dir, f"part-{i}.csv.gz")
        s3.download_file(bucket, key, local_path)
        frames.append(pd.read_csv(local_path, compression="gzip", low_memory=False))
        os.unlink(local_path)

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, default="")
    parser.add_argument("--athena-database", type=str, default=None)
    parser.add_argument("--athena-table", type=str, default=None)
    parser.add_argument("--athena-workgroup", type=str, default="primary")
    parser.add_argument("--athena-output-s3", type=str, default=None)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)

    if args.input_data:
        bucket, key = parse_s3_uri(args.input_data)
        input_file = f"{base_dir}/data/mti_scores_history.csv"

        logger.info("Downloading data from bucket=%s key=%s", bucket, key)
        s3 = boto3.resource("s3")
        s3.Bucket(bucket).download_file(key, input_file)

        logger.info("Loading MTI data from S3 CSV")
        df = pd.read_csv(input_file, low_memory=False)
        os.unlink(input_file)
    else:
        if not (args.athena_database and args.athena_table and args.athena_output_s3):
            raise ValueError(
                "Either provide --input-data (s3://...) or provide Athena args: "
                "--athena-database, --athena-table, --athena-output-s3"
            )
        logger.info(
            "Loading MTI data from Athena table %s.%s", args.athena_database, args.athena_table
        )
        df = load_from_athena(
            database=args.athena_database,
            table=args.athena_table,
            workgroup=args.athena_workgroup,
            output_s3=args.athena_output_s3,
        )

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
