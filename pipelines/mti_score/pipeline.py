"""Workflow pipeline for MTI score forecasting with LightGBM."""
import os

import boto3
import sagemaker
import sagemaker.session
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_sagemaker_client(region):
    """Get SageMaker client for the selected region."""
    boto_session = boto3.Session(region_name=region)
    return boto_session.client("sagemaker")


def get_session(region, default_bucket):
    """Get SageMaker session based on region."""
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline_session(region, default_bucket):
    """Get SageMaker Pipeline session based on region."""
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )


def get_pipeline_custom_tags(new_tags, region, sagemaker_project_name=None):
    """Copy SageMaker project tags to the pipeline run."""
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.describe_project(ProjectName=sagemaker_project_name)
        project_arn = response["ProjectArn"]
        response = sm_client.list_tags(ResourceArn=project_arn)
        for project_tag in response["Tags"]:
            new_tags.append(project_tag)
    except Exception as err:  # pragma: no cover
        print(f"Error getting project tags: {err}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_name=None,
    role=None,
    default_bucket=None,
    model_package_group_name="MTIScorePredictionPackageGroup",
    pipeline_name="MTIScorePredictionPipeline",
    base_job_prefix="mti-score-prediction",
    processing_instance_type="ml.m5.12xlarge",
    training_instance_type="ml.m5.12xlarge",
):
    """Build and return the MTI score prediction pipeline."""
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
    pipeline_session = get_pipeline_session(region, default_bucket)

    # instance_type must be a plain string here, not a Pipeline ParameterString.
    # SKLearnProcessor/SKLearn call image_uris.retrieve() at definition time, which
    # cannot use pipeline variables for instance_type.
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    mse_threshold = ParameterFloat(name="MseThreshold", default_value=0.05)
    # Legacy S3 CSV input. SageMaker rejects empty container arguments; use a sentinel to mean
    # "skip S3 and load from Athena" (see preprocess.py).
    input_data = ParameterString(name="InputDataUrl", default_value="__USE_ATHENA__")

    athena_database = ParameterString(
        name="AthenaDatabase", default_value="ps-prod-maritime_transparency_index"
    )
    athena_table = ParameterString(name="AthenaTable", default_value="mti_scores")
    athena_workgroup = ParameterString(name="AthenaWorkGroup", default_value="primary")
    athena_output_s3 = ParameterString(
        name="AthenaOutputS3",
        default_value=f"s3://elz-s3-ai-dev-maritime-transparency-index/",
    )

    sklearn_processor = SKLearnProcessor(
        framework_version="1.0-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/preprocess",
        sagemaker_session=pipeline_session,
        role=role,
    )
    preprocess_step_args = sklearn_processor.run(
        outputs=[
            ProcessingOutput(output_name="prepared", source="/opt/ml/processing/prepared"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        arguments=[
            "--input-data",
            input_data,
            "--athena-database",
            athena_database,
            "--athena-table",
            athena_table,
            "--athena-workgroup",
            athena_workgroup,
            "--athena-output-s3",
            athena_output_s3,
        ],
    )
    step_process = ProcessingStep(name="PreprocessMTIData", step_args=preprocess_step_args)

    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/train"
    trainer = SKLearn(
        framework_version="1.0-1",
        py_version="py3",
        entry_point="train.py",
        source_dir=BASE_DIR,
        # requirements.txt lives in source_dir; do not also list it under dependencies —
        # that duplicates the file in the tarball (same basename) and can break setup.
        instance_type=training_instance_type,
        # SKLearn estimator in this SDK doesn't support distributed training.
        # Keep a fixed single instance to avoid pipeline-definition errors.
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/train",
        sagemaker_session=pipeline_session,
        role=role,
    )
    train_step_args = trainer.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["prepared"].S3Output.S3Uri,
                content_type="text/csv;header=present",
            )
        }
    )
    step_train = TrainingStep(name="TrainMTIScoreModel", step_args=train_step_args)

    eval_image_uri = sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type=processing_instance_type,
    )
    evaluator = ScriptProcessor(
        image_uri=eval_image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/evaluate",
        sagemaker_session=pipeline_session,
        role=role,
    )
    eval_step_args = evaluator.run(
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
        ],
        outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
        code=os.path.join(BASE_DIR, "evaluate.py"),
    )
    evaluation_report = PropertyFile(
        name="MTIEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="EvaluateMTIScoreModel",
        step_args=eval_step_args,
        property_files=[evaluation_report],
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=(
                f"{step_eval.arguments['ProcessingOutputConfig']['Outputs'][0]['S3Output']['S3Uri']}"
                "/evaluation.json"
            ),
            content_type="application/json",
        )
    )
    model = Model(
        image_uri=eval_image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
    )
    register_step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
        # Model package versions cannot be tagged; use customer metadata instead.
        customer_metadata_properties={
            "eval_mse": Join(
                on="",
                values=[
                    JsonGet(
                        step_name=step_eval.name,
                        property_file=evaluation_report,
                        json_path="regression_metrics.mse.value",
                    )
                ],
            ),
            "eval_mse_std": Join(
                on="",
                values=[
                    JsonGet(
                        step_name=step_eval.name,
                        property_file=evaluation_report,
                        json_path="regression_metrics.mse.standard_deviation",
                    )
                ],
            ),
            "eval_mae": Join(
                on="",
                values=[
                    JsonGet(
                        step_name=step_eval.name,
                        property_file=evaluation_report,
                        json_path="regression_metrics.mae",
                    )
                ],
            ),
            "eval_rmse": Join(
                on="",
                values=[
                    JsonGet(
                        step_name=step_eval.name,
                        property_file=evaluation_report,
                        json_path="regression_metrics.rmse",
                    )
                ],
            ),
            "eval_accuracy_r2": Join(
                on="",
                values=[
                    JsonGet(
                        step_name=step_eval.name,
                        property_file=evaluation_report,
                        json_path="regression_metrics.accuracy",
                    )
                ],
            ),
        },
    )
    step_register = ModelStep(name="RegisterMTIScoreModel", step_args=register_step_args)

    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.mse.value",
        ),
        right=mse_threshold,
    )
    step_cond = ConditionStep(
        name="CheckMSEMTIEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )

    return Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count,
            model_approval_status,
            mse_threshold,
            input_data,
            athena_database,
            athena_table,
            athena_workgroup,
            athena_output_s3,
        ],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=pipeline_session,
    )
