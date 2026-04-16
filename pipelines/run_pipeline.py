# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""A CLI to create or update and run pipelines."""
from __future__ import absolute_import

import argparse
import json
import os
import sys
import traceback

from pipelines._utils import get_pipeline_driver, convert_struct, get_pipeline_custom_tags


def _maybe_log_mlflow_summary(execution, kwargs: dict) -> None:
    """Log a lightweight summary run to SageMaker managed MLflow (Studio) if configured."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return

    try:
        import mlflow  # type: ignore
    except ImportError:
        print("MLFLOW_TRACKING_URI is set but mlflow is not installed; skipping MLflow logging.")
        return

    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME") or kwargs.get("pipeline_name") or "default"
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name="sagemaker-pipeline-execution"):
            mlflow.log_param("pipeline_execution_arn", execution.arn)
            mlflow.log_param("region", kwargs.get("region", ""))
            mlflow.log_param("pipeline_name", kwargs.get("pipeline_name", ""))
            mlflow.log_param("model_package_group_name", kwargs.get("model_package_group_name", ""))
        print(f"MLflow summary logged to experiment '{experiment}'.")
    except Exception as err:  # pragma: no cover
        print(f"MLflow logging skipped due to error: {err}")


def main():  # pragma: no cover
    """The main harness that creates or updates and runs the pipeline.

    Creates or updates the pipeline and runs it.
    """
    parser = argparse.ArgumentParser(
        "Creates or updates and runs the pipeline for the pipeline script."
    )

    parser.add_argument(
        "-n",
        "--module-name",
        dest="module_name",
        type=str,
        help="The module name of the pipeline to import.",
    )
    parser.add_argument(
        "-kwargs",
        "--kwargs",
        dest="kwargs",
        default=None,
        help="Dict string of keyword arguments for the pipeline generation (if supported)",
    )
    parser.add_argument(
        "-role-arn",
        "--role-arn",
        dest="role_arn",
        type=str,
        help="The role arn for the pipeline service execution role.",
    )
    parser.add_argument(
        "-description",
        "--description",
        dest="description",
        type=str,
        default=None,
        help="The description of the pipeline.",
    )
    parser.add_argument(
        "-tags",
        "--tags",
        dest="tags",
        default=None,
        help="""List of dict strings of '[{"Key": "string", "Value": "string"}, ..]'""",
    )
    args = parser.parse_args()

    if args.module_name is None or args.role_arn is None:
        parser.print_help()
        sys.exit(2)
    tags = convert_struct(args.tags)
    if not isinstance(tags, list):
        tags = []

    try:
        pipeline = get_pipeline_driver(args.module_name, args.kwargs)
        print("###### Creating/updating a SageMaker Pipeline with the following definition:")
        parsed = json.loads(pipeline.definition())
        print(json.dumps(parsed, indent=2, sort_keys=True))

        all_tags = get_pipeline_custom_tags(args.module_name, args.kwargs, tags)

        upsert_response = pipeline.upsert(
            role_arn=args.role_arn, description=args.description, tags=all_tags
        )
        print("\n###### Created/Updated SageMaker Pipeline: Response received:")
        print(upsert_response)

        execution = pipeline.start()
        print(f"\n###### Execution started with PipelineExecutionArn: {execution.arn}")

        print("Waiting for the execution to finish...")

        # Setting the attempts and delay (in seconds) will modify the overall time the pipeline waits. 
        # If the execution is taking a longer time, update these parameters to a larger value.
        # Eg: The total wait time is calculated as 60 * 120 = 7200 seconds (2 hours)
        execution.wait(max_attempts=120, delay=60)
        
        print("\n#####Execution completed. Execution step details:")

        print(execution.list_steps())
        kwargs_dict = convert_struct(args.kwargs)
        _maybe_log_mlflow_summary(execution, kwargs_dict)
        # Todo print the status?
    except Exception as e:  # pylint: disable=W0703
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
