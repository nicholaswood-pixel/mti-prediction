"""SageMaker Pipeline template: digit classification example in `dev` package (sklearn)."""
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
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
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
    """Copy SageMaker project tags to the pipeline run (optional Studio project integration)."""
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
    model_package_group_name="DevDigitTemplatePackageGroup",
    pipeline_name="DevDigitTemplatePipeline",
    base_job_prefix="dev-digit-template",
    processing_instance_type="ml.m5.large",
    training_instance_type="ml.m5.large",
):
    """Build and return the dev example pipeline (preprocess → train → evaluate → conditional register)."""
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
    pipeline_session = get_pipeline_session(region, default_bucket)

    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    accuracy_threshold = ParameterFloat(
        name="AccuracyThreshold",
        default_value=0.85,
    )
    # String parameters: ProcessingJob container "arguments" must be strings; Integer refs fail CreatePipeline.
    max_train_samples = ParameterString(
        name="MaxTrainSamples",
        default_value="12000",
    )
    max_test_samples = ParameterString(
        name="MaxTestSamples",
        default_value="3000",
    )
    mlflow_tracking_uri = ParameterString(
        name="MlflowTrackingUri",
        default_value=os.environ.get("MLFLOW_TRACKING_URI", ""),
    )
    mlflow_experiment_name = ParameterString(
        name="MlflowExperimentName",
        default_value=os.environ.get("MLFLOW_EXPERIMENT_NAME", "dev-digit-training"),
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
            "--output-dir",
            "/opt/ml/processing/prepared",
            "--max-train-samples",
            max_train_samples,
            "--max-test-samples",
            max_test_samples,
        ],
    )
    step_process = ProcessingStep(name="PreprocessDevData", step_args=preprocess_step_args)

    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/train"
    trainer = SKLearn(
        framework_version="1.0-1",
        py_version="py3",
        entry_point="train.py",
        source_dir=BASE_DIR,
        dependencies=[os.path.join(BASE_DIR, "requirements.txt")],
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/train",
        sagemaker_session=pipeline_session,
        role=role,
        hyperparameters={
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "mlflow_experiment_name": mlflow_experiment_name,
        },
    )
    train_step_args = trainer.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["prepared"].S3Output.S3Uri,
            )
        }
    )
    step_train = TrainingStep(name="TrainDevModel", step_args=train_step_args)

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
        name="DevEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="EvaluateDevModel",
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
        customer_metadata_properties={
            "eval_accuracy": Join(
                on="",
                values=[
                    JsonGet(
                        step_name=step_eval.name,
                        property_file=evaluation_report,
                        json_path="classification_metrics.accuracy.value",
                    )
                ],
            ),
        },
    )
    step_register = ModelStep(name="RegisterDevModel", step_args=register_step_args)

    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="classification_metrics.accuracy.value",
        ),
        right=accuracy_threshold,
    )
    step_cond = ConditionStep(
        name="CheckDevAccuracy",
        conditions=[cond_gte],
        if_steps=[step_register],
        else_steps=[],
    )

    return Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count,
            model_approval_status,
            accuracy_threshold,
            max_train_samples,
            max_test_samples,
            mlflow_tracking_uri,
            mlflow_experiment_name,
        ],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=pipeline_session,
    )
