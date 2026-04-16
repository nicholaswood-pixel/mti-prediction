# boto3 Infra Provisioning

This folder replaces CDK for infrastructure provisioning by using direct `boto3` client calls.

## What it provisions

- Artifact S3 bucket (versioning + public access block)
- SageMaker pipeline execution role
- CodeBuild execution role
- CodeBuild project configured for CodePipeline input/output
- CodePipeline with:
  - Source stage: GitHub via CodeStar connection
  - Build stage: CodeBuild (runs `codebuild-buildspec.yml`)
- CodeBuild environment variables for an **existing** MLflow server (optional), so `run-pipeline` can log runs to that server

## Run

Create a `.env` file in the repo root (or current working directory):

```bash
AWS_REGION=us-east-1
PROJECT_NAME=mti-score
GITHUB_OWNER=<GITHUB_OWNER>
GITHUB_REPO=mlops-template
GITHUB_BRANCH=main
GITHUB_CONNECTION_ARN=arn:aws:codestar-connections:us-east-1:<ACCOUNT_ID>:connection/<CONNECTION_ID>
TARGET_GIT_REPO_DIR=/path/to/your/target-git-repo
PUSH_TEMPLATE_TO_REMOTE=true

# Optional: wire CodeBuild to an existing MLflow tracking server (no server is created here)
# MLFLOW_TRACKING_URI=arn:aws:sagemaker:us-east-1:123456789012:mlflow-tracking-server/my-server
MLFLOW_PYTHON_CLIENT_VERSION=2.16.2
MLFLOW_EXPERIMENT_NAME=<PROJECT_NAME>-experiments
```

Then run:

```bash
python infra/boto3/provision_infra.py
```

## Notes

- The script is idempotent for create/update flows.
- Your IAM principal must allow IAM/S3/CodeBuild/SageMaker API actions.
- Required env var: `AWS_REGION` (or `AWS_DEFAULT_REGION`).
- Required env var: `GITHUB_CONNECTION_ARN`.
- Order: the script validates the CodeConnections/CodeStar connection, then syncs/pushes template code to GitHub (if enabled), then provisions AWS resources.
- Optional env var: `TARGET_GIT_REPO_DIR` (if set, template files are copied there after connection validation and before infra provisioning).
- Optional env var: `PUSH_TEMPLATE_TO_REMOTE` (default `true`; pushes first commit to `https://github.com/<GITHUB_OWNER>/<GITHUB_REPO>.git` after connection validation and before infra provisioning).
- Optional env vars for commit identity: `GIT_AUTHOR_NAME`, `GIT_AUTHOR_EMAIL`.
- MLflow: set `MLFLOW_TRACKING_URI` to your **existing** server (for SageMaker managed MLflow this is commonly the tracking server **ARN**). Provisioning does not create a server.
- MLflow: set `MLFLOW_PYTHON_CLIENT_VERSION` to match the MLflow version on that server so CodeBuild installs a compatible client (`mlflow` + `sagemaker-mlflow`). See [Integrate MLflow with your environment](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-track-experiments.html).
- Existing process environment variables take precedence over values in `.env`.
