# Python CDK Infrastructure for `mlops-template`

This CDK app provisions base AWS infrastructure that matches this repository's SageMaker pipeline workflow.

## What it creates

- S3 artifact bucket for pipeline artifacts and intermediate data.
- SageMaker pipeline execution IAM role.
- CodeBuild project that runs the existing `codebuild-buildspec.yml`.
- CodeBuild IAM role with permissions needed to create/update/start SageMaker pipelines and pass the pipeline role.

## Prerequisites

- Python 3.10+
- AWS CDK v2 CLI installed (`npm i -g aws-cdk`)
- AWS credentials configured for target account/region

## Deploy

From `infra/cdk`:

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
cdk deploy \
  -c account=<ACCOUNT_ID> \
  -c region=<REGION> \
  -c project_name=mti-score \
  -c github_owner=<GITHUB_OWNER> \
  -c github_repo=mlops-template \
  -c github_branch=main
```

## Optional S3 source instead of GitHub

If you prefer to source build input from an S3 zip:

```bash
cdk deploy \
  -c account=<ACCOUNT_ID> \
  -c region=<REGION> \
  -c source_bucket_name=<BUCKET_NAME> \
  -c source_bucket_arn=<BUCKET_ARN> \
  -c source_object_key=source.zip
```

When both `source_bucket_name` and `source_bucket_arn` are provided, the stack uses S3 source; otherwise it uses GitHub source.

## Why no bootstrap step

This app uses `BootstraplessSynthesizer` so it can deploy in AWS Organizations environments where SCPs block IAM actions required by `cdk bootstrap` (for example `iam:AttachRolePolicy` on `CDKToolkit` roles).
