#!/usr/bin/env python3
"""Provision MLOps template infrastructure using boto3 only."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import boto3
from botocore.exceptions import ClientError


@dataclass
class Config:
    region: str
    project_name: str
    github_owner: str
    github_repo: str
    github_branch: str
    github_connection_arn: str
    sagemaker_pipeline_role_name: str
    codebuild_role_name: str
    codepipeline_role_name: str
    mlflow_tracking_uri: Optional[str]
    mlflow_python_client_version: str
    mlflow_experiment_name: str
    target_git_repo_dir: Optional[str]
    push_template_to_remote: bool
    source_bucket_name: Optional[str]
    source_object_key: str

    @property
    def account_id(self) -> str:
        sts = boto3.client("sts", region_name=self.region)
        return sts.get_caller_identity()["Account"]

    @property
    def artifact_bucket_name(self) -> str:
        return f"{self.project_name}-artifacts-{self.account_id}-{self.region}"

    @property
    def codebuild_project_name(self) -> str:
        return f"{self.project_name}-pipeline-build"

    @property
    def codepipeline_name(self) -> str:
        return f"{self.project_name}-codepipeline"


def ensure_bucket(cfg: Config) -> None:
    s3 = boto3.client("s3", region_name=cfg.region)
    try:
        s3.head_bucket(Bucket=cfg.artifact_bucket_name)
        print(f"Bucket exists: {cfg.artifact_bucket_name}")
    except ClientError:
        kwargs = {"Bucket": cfg.artifact_bucket_name}
        if cfg.region != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": cfg.region}
        s3.create_bucket(**kwargs)
        print(f"Created bucket: {cfg.artifact_bucket_name}")

    s3.put_bucket_versioning(
        Bucket=cfg.artifact_bucket_name, VersioningConfiguration={"Status": "Enabled"}
    )
    s3.put_public_access_block(
        Bucket=cfg.artifact_bucket_name,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )


def ensure_existing_role(region: str, role_name: str) -> str:
    iam = boto3.client("iam", region_name=region)
    try:
        role_arn = iam.get_role(RoleName=role_name)["Role"]["Arn"]
        print(f"Using existing role: {role_name}")
        return role_arn
    except iam.exceptions.NoSuchEntityException as err:
        raise ValueError(
            f"Required role '{role_name}' does not exist. "
            "Create it first and rerun."
        ) from err


def get_role_arn(region: str, role_name: str) -> str:
    iam = boto3.client("iam", region_name=region)
    return iam.get_role(RoleName=role_name)["Role"]["Arn"]


def ensure_codebuild_project(
    cfg: Config,
    codebuild_role_arn: str,
    pipeline_role_arn: str,
) -> None:
    cb = boto3.client("codebuild", region_name=cfg.region)

    env_vars = [
        {"name": "ARTIFACT_BUCKET", "value": cfg.artifact_bucket_name, "type": "PLAINTEXT"},
        {"name": "PROJECT_NAME", "value": cfg.project_name, "type": "PLAINTEXT"},
        {"name": "SAGEMAKER_PIPELINE_ROLE_ARN", "value": pipeline_role_arn, "type": "PLAINTEXT"},
        {"name": "MLFLOW_EXPERIMENT_NAME", "value": cfg.mlflow_experiment_name, "type": "PLAINTEXT"},
    ]
    if cfg.mlflow_tracking_uri:
        env_vars.extend(
            [
                {
                    "name": "MLFLOW_TRACKING_URI",
                    "value": cfg.mlflow_tracking_uri,
                    "type": "PLAINTEXT",
                },
                {
                    "name": "MLFLOW_PYTHON_CLIENT_VERSION",
                    "value": cfg.mlflow_python_client_version,
                    "type": "PLAINTEXT",
                },
            ]
        )
        print("CodeBuild will use existing MLflow tracking URI from MLFLOW_TRACKING_URI.")
    else:
        print("MLFLOW_TRACKING_URI not set; CodeBuild runs will not log to MLflow.")

    project_def = {
        "name": cfg.codebuild_project_name,
        "description": f"Build and run SageMaker pipeline for {cfg.project_name}",
        "serviceRole": codebuild_role_arn,
        "source": {"type": "CODEPIPELINE", "buildspec": "codebuild-buildspec.yml"},
        "artifacts": {"type": "CODEPIPELINE"},
        "environment": {
            "type": "LINUX_CONTAINER",
            "image": "aws/codebuild/standard:7.0",
            "computeType": "BUILD_GENERAL1_SMALL",
            "privilegedMode": False,
            "environmentVariables": env_vars,
        },
        "timeoutInMinutes": 60,
    }

    for attempt in range(1, 7):
        try:
            projects = cb.batch_get_projects(names=[cfg.codebuild_project_name])["projects"]
            if projects:
                cb.update_project(**project_def)
                print(f"Updated CodeBuild project: {cfg.codebuild_project_name}")
            else:
                cb.create_project(**project_def)
                print(f"Created CodeBuild project: {cfg.codebuild_project_name}")
            return
        except cb.exceptions.InvalidInputException as err:
            error_text = str(err)
            # IAM propagation can lag briefly after role creation/updates.
            if "sts:AssumeRole on service role" in error_text and attempt < 7:
                wait_seconds = attempt * 5
                print(
                    "CodeBuild service role not ready yet; retrying in "
                    f"{wait_seconds}s (attempt {attempt}/6)"
                )
                time.sleep(wait_seconds)
                codebuild_role_arn = get_role_arn(cfg.region, cfg.codebuild_role_name)
                project_def["serviceRole"] = codebuild_role_arn
                continue
            raise


def ensure_codepipeline(
    cfg: Config, codepipeline_role_arn: str, codebuild_project_name: str
) -> None:
    cp = boto3.client("codepipeline", region_name=cfg.region)

    source_artifact = "SourceOutput"
    pipeline_def = {
        "pipeline": {
            "name": cfg.codepipeline_name,
            "roleArn": codepipeline_role_arn,
            "artifactStore": {"type": "S3", "location": cfg.artifact_bucket_name},
            "stages": [
                {
                    "name": "Source",
                    "actions": [
                        {
                            "name": "Source",
                            "actionTypeId": {
                                "category": "Source",
                                "owner": "AWS",
                                "provider": "CodeStarSourceConnection",
                                "version": "1",
                            },
                            "runOrder": 1,
                            "configuration": {
                                "ConnectionArn": cfg.github_connection_arn,
                                "FullRepositoryId": f"{cfg.github_owner}/{cfg.github_repo}",
                                "BranchName": cfg.github_branch,
                                "DetectChanges": "true",
                                "OutputArtifactFormat": "CODE_ZIP",
                            },
                            "outputArtifacts": [{"name": source_artifact}],
                            "inputArtifacts": [],
                        }
                    ],
                },
                {
                    "name": "Build",
                    "actions": [
                        {
                            "name": "BuildAndRunSageMakerPipeline",
                            "actionTypeId": {
                                "category": "Build",
                                "owner": "AWS",
                                "provider": "CodeBuild",
                                "version": "1",
                            },
                            "runOrder": 1,
                            "configuration": {"ProjectName": codebuild_project_name},
                            "outputArtifacts": [],
                            "inputArtifacts": [{"name": source_artifact}],
                        }
                    ],
                },
            ],
            "version": 1,
        }
    }

    try:
        cp.get_pipeline(name=cfg.codepipeline_name)
        cp.update_pipeline(**pipeline_def)
        print(f"Updated CodePipeline: {cfg.codepipeline_name}")
    except cp.exceptions.PipelineNotFoundException:
        cp.create_pipeline(**pipeline_def)
        print(f"Created CodePipeline: {cfg.codepipeline_name}")
    except ClientError as err:
        raise RuntimeError(
            "Failed to create/update CodePipeline. Check IAM permissions for "
            "CodePipeline/CodeBuild/CodeConnections and confirm the connection ARN is valid."
        ) from err


def ensure_connection_available(cfg: Config) -> None:
    """Validate GitHub connection ARN exists and is available."""
    # AWS renamed CodeStar Connections to CodeConnections.
    client_names = ["codeconnections", "codestar-connections"]
    last_error: Optional[Exception] = None
    for client_name in client_names:
        try:
            conn_client = boto3.client(client_name, region_name=cfg.region)
            response = conn_client.get_connection(ConnectionArn=cfg.github_connection_arn)
            status = response.get("Connection", {}).get("ConnectionStatus", "UNKNOWN")
            print(f"Connection status: {status}")
            if status != "AVAILABLE":
                raise ValueError(
                    "GITHUB_CONNECTION_ARN is not AVAILABLE. Open AWS Console -> "
                    "Developer Tools -> Connections and complete/authorize the connection."
                )
            return
        except Exception as err:  # pragma: no cover
            last_error = err
            continue
    raise RuntimeError(
        "Unable to validate GITHUB_CONNECTION_ARN via codeconnections API."
    ) from last_error


def load_dotenv_file() -> None:
    """Load key=value pairs from a .env file into process env.

    Precedence: existing environment variables are not overwritten.
    Search order:
    1) Current working directory `.env`
    2) Repository root `.env` (two levels up from this script)
    """
    candidate_paths = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]

    dotenv_path = next((path for path in candidate_paths if path.exists()), None)
    if not dotenv_path:
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def load_config_from_env() -> Config:
    load_dotenv_file()
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if not region:
        raise ValueError("Set AWS_REGION (or AWS_DEFAULT_REGION) before running.")

    project_name = os.getenv("PROJECT_NAME", "mti-score")
    sagemaker_pipeline_role_name = os.getenv(
        "SAGEMAKER_PIPELINE_ROLE_NAME", "AmazonSageMakerServiceCatalogProductsExecutionRole"
    )
    return Config(
        region=region,
        project_name=project_name,
        github_owner=os.getenv("GITHUB_OWNER", "your-org"),
        github_repo=os.getenv("GITHUB_REPO", "mlops-template"),
        github_branch=os.getenv("GITHUB_BRANCH", "main"),
        github_connection_arn=os.getenv("GITHUB_CONNECTION_ARN", ""),
        sagemaker_pipeline_role_name=sagemaker_pipeline_role_name,
        codebuild_role_name=os.getenv("CODEBUILD_ROLE_NAME", "AmazonSageMakerServiceCatalogProductsCodeBuildRole"),
        codepipeline_role_name=os.getenv(
            "CODEPIPELINE_ROLE_NAME", "AmazonSageMakerServiceCatalogProductsCodePipelineRole"
        ),
        mlflow_tracking_uri=(
            (os.getenv("MLFLOW_TRACKING_URI") or "").strip() or None
        ),
        mlflow_python_client_version=os.getenv("MLFLOW_PYTHON_CLIENT_VERSION", "2.16.2").strip()
        or "2.16.2",
        mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", f"{project_name}-experiments"),
        target_git_repo_dir=os.getenv("TARGET_GIT_REPO_DIR"),
        push_template_to_remote=os.getenv("PUSH_TEMPLATE_TO_REMOTE", "true").lower() == "true",
        source_bucket_name=os.getenv("SOURCE_BUCKET_NAME"),
        source_object_key=os.getenv("SOURCE_OBJECT_KEY", "source.zip"),
    )


def sync_template_to_git_repo(cfg: Config) -> None:
    if not cfg.target_git_repo_dir:
        print("TARGET_GIT_REPO_DIR not set; skipping template sync.")
        return

    source_root = Path(__file__).resolve().parents[2]
    target_root = Path(cfg.target_git_repo_dir).resolve()

    if not (target_root / ".git").exists():
        raise ValueError(
            f"TARGET_GIT_REPO_DIR is not a git repository: {target_root}"
        )

    ignore_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}
    ignore_files = {".env"}

    for item in source_root.iterdir():
        if item.name in ignore_dirs or item.name in ignore_files:
            continue
        dest = target_root / item.name
        if item.is_dir():
            shutil.copytree(
                item,
                dest,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    ".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"
                ),
            )
        else:
            shutil.copy2(item, dest)

    print(f"Template code synced to git repo: {target_root}")


def _copy_template_files(source_root: Path, target_root: Path) -> None:
    ignore_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}
    ignore_files = {".env"}

    for item in source_root.iterdir():
        if item.name in ignore_dirs or item.name in ignore_files:
            continue
        dest = target_root / item.name
        if item.is_dir():
            shutil.copytree(
                item,
                dest,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    ".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"
                ),
            )
        else:
            shutil.copy2(item, dest)


def push_template_to_remote_repo(cfg: Config) -> None:
    if not cfg.push_template_to_remote:
        print("PUSH_TEMPLATE_TO_REMOTE is false; skipping remote push.")
        return

    source_root = Path(__file__).resolve().parents[2]
    remote_url = f"https://github.com/{cfg.github_owner}/{cfg.github_repo}.git"
    branch = cfg.github_branch or "main"

    with tempfile.TemporaryDirectory(prefix="mlops-template-push-") as tmp_dir:
        repo_dir = Path(tmp_dir)
        _copy_template_files(source_root, repo_dir)

        def run_git(*args: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                ["git", *args],
                cwd=repo_dir,
                check=True,
                capture_output=True,
                text=True,
            )

        try:
            run_git("init")
            run_git("checkout", "-b", branch)
            run_git("add", ".")
            status = run_git("status", "--porcelain")
            if not status.stdout.strip():
                print("No template files to push.")
                return

            author_name = os.getenv("GIT_AUTHOR_NAME", "Nicholas Wood")
            author_email = os.getenv("GIT_AUTHOR_EMAIL", "nicholas.wood@polestarglobal.com")
            run_git("config", "user.name", author_name)
            run_git("config", "user.email", author_email)
            run_git("commit", "-m", "Initialize repository with MLOps template")
            run_git("remote", "add", "origin", remote_url)
            run_git("push", "-u", "origin", branch)
            print(f"Template pushed to {remote_url} on branch '{branch}'.")
        except subprocess.CalledProcessError as err:
            stderr = (err.stderr or "").strip()
            raise RuntimeError(
                "Failed to push template to remote repository. "
                "Check git credentials and repo permissions."
                + (f" Git error: {stderr}" if stderr else "")
            ) from err


def main() -> None:
    cfg = load_config_from_env()
    if not cfg.github_connection_arn:
        raise ValueError("Set GITHUB_CONNECTION_ARN in environment or .env.")

    ensure_connection_available(cfg)
    sync_template_to_git_repo(cfg)
    push_template_to_remote_repo(cfg)

    ensure_bucket(cfg)
    pipeline_role_arn = ensure_existing_role(cfg.region, cfg.sagemaker_pipeline_role_name)
    codebuild_role_arn = ensure_existing_role(cfg.region, cfg.codebuild_role_name)
    codepipeline_role_arn = ensure_existing_role(cfg.region, cfg.codepipeline_role_name)
    # Resolve role fresh from IAM so CodeBuild uses the exact current role ARN.
    codebuild_role_arn = get_role_arn(cfg.region, cfg.codebuild_role_name)
    ensure_codebuild_project(cfg, codebuild_role_arn, pipeline_role_arn)
    ensure_codepipeline(cfg, codepipeline_role_arn, cfg.codebuild_project_name)
    print("Provisioning complete.")
    print(f"artifact_bucket={cfg.artifact_bucket_name}")
    print(f"pipeline_role_arn={pipeline_role_arn}")
    print(f"codebuild_project={cfg.codebuild_project_name}")
    print(f"codepipeline_name={cfg.codepipeline_name}")


if __name__ == "__main__":
    main()
