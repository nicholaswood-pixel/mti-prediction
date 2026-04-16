from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_codebuild as codebuild,
    aws_iam as iam,
    aws_s3 as s3,
)
from constructs import Construct


class MlopsInfraStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        project_name = self.node.try_get_context("project_name") or "mti-score"
        source_bucket_name = self.node.try_get_context("source_bucket_name")
        source_bucket_arn = self.node.try_get_context("source_bucket_arn")
        source_object_key = self.node.try_get_context("source_object_key") or "source.zip"

        artifact_bucket = s3.Bucket(
            self,
            "ArtifactBucket",
            bucket_name=f"{project_name}-artifacts-{self.account}-{self.region}",
            encryption=s3.BucketEncryption.S3_MANAGED,
            versioned=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
            auto_delete_objects=False,
        )

        pipeline_role = iam.Role(
            self,
            "SageMakerPipelineExecutionRole",
            role_name=f"{project_name}-sagemaker-pipeline-role",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchLogsFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryReadOnly"),
            ],
        )

        codebuild_role = iam.Role(
            self,
            "CodeBuildExecutionRole",
            role_name=f"{project_name}-codebuild-role",
            assumed_by=iam.ServicePrincipal("codebuild.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AWSCodeBuildDeveloperAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchLogsFullAccess"),
            ],
        )

        source = (
            codebuild.Source.s3(
                bucket=s3.Bucket.from_bucket_attributes(
                    self,
                    "SourceBucket",
                    bucket_name=source_bucket_name,
                    bucket_arn=source_bucket_arn,
                ),
                path=source_object_key,
            )
            if source_bucket_name and source_bucket_arn
            else codebuild.Source.git_hub(
                owner=self.node.try_get_context("github_owner") or "your-org",
                repo=self.node.try_get_context("github_repo") or "mlops-template",
                branch_or_ref=self.node.try_get_context("github_branch") or "main",
                clone_depth=1,
            )
        )

        build_project = codebuild.Project(
            self,
            "PipelineBuildProject",
            project_name=f"{project_name}-pipeline-build",
            role=codebuild_role,
            source=source,
            timeout=Duration.minutes(60),
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                compute_type=codebuild.ComputeType.SMALL,
                privileged=False,
            ),
            build_spec=codebuild.BuildSpec.from_source_filename("codebuild-buildspec.yml"),
            environment_variables={
                "ARTIFACT_BUCKET": codebuild.BuildEnvironmentVariable(value=artifact_bucket.bucket_name),
                "SAGEMAKER_PROJECT_NAME": codebuild.BuildEnvironmentVariable(value=project_name),
                "SAGEMAKER_PROJECT_ID": codebuild.BuildEnvironmentVariable(value=f"{project_name}-001"),
                "SAGEMAKER_PIPELINE_ROLE_ARN": codebuild.BuildEnvironmentVariable(
                    value=pipeline_role.role_arn
                ),
            },
        )

        CfnOutput(self, "ArtifactBucketName", value=artifact_bucket.bucket_name)
        CfnOutput(self, "SageMakerPipelineRoleArn", value=pipeline_role.role_arn)
        CfnOutput(self, "CodeBuildProjectName", value=build_project.project_name)
