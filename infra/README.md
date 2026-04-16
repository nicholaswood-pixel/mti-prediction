# Custom infrastructure template

This folder holds **IaC you own**, so you are not limited to the SageMaker Studio seed project’s auto-created CodeBuild/CodePipeline. You can deploy the stack below directly, or wrap it in **AWS Service Catalog** and register it as a **custom SageMaker project template** (admin setup in Studio).

## What `cloudformation/mti-sagemaker-ci.yaml` creates

| Resource | Purpose |
|----------|---------|
| **S3 bucket** | Artifact and pipeline-related storage (`ARTIFACT_BUCKET` for `run-pipeline`). |
| **IAM role (SageMaker)** | Trust `sagemaker.amazonaws.com`; used for **both** pipeline upsert and training/processing jobs in the sample env wiring. |
| **IAM role (CodeBuild)** | Clone from Git via **AWS CodeConnections**, write logs, call SageMaker APIs, `PassRole` to the SageMaker role. |
| **CodeBuild project** | Clones your repo (`codebuild-buildspec.yml` at repo root) and runs `pip install .` + `run-pipeline`. |

**Note:** The SageMaker role attaches `AmazonSageMakerFullAccess` for a working baseline. Tighten IAM for production (scoped S3 ARNs, no `*` SageMaker where possible).

## Prerequisites

1. **AWS CodeConnections** (formerly *CodeStar Connections*) — in the console under **Developer tools** / **Settings** / **Connections** (wording may vary). Create a connection to your Git provider and wait until it is **Available** (complete the OAuth / app install flow).
2. Copy the connection **ARN**. New connections typically look like `arn:aws:codeconnections:REGION:ACCOUNT:connection/...`; older ones may still show `arn:aws:codestar-connections:...`. Either form works with this template’s IAM policy.
3. **Repository URL** (HTTPS) that matches that connection, e.g. `https://github.com/org/repo`.

## Deploy (CLI example)

```bash
aws cloudformation deploy \
  --stack-name mti-score-prediction-ci \
  --template-file infra/cloudformation/mti-sagemaker-ci.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    ProjectName=mti-score-prediction \
    RepositoryProvider=GITHUB \
    RepositoryUrl=https://github.com/YOUR_ORG/mti-score-prediction \
    BranchName=main \
    CodeConnectionArn=arn:aws:codeconnections:REGION:ACCOUNT:connection/UUID
```

Use `RepositoryProvider=GITLAB`, `BITBUCKET`, or `GITHUB_ENTERPRISE` when your clone URL matches that host.

Then start a build:

```bash
aws codebuild start-build --project-name mti-score-prediction-pipeline-build
```

(Replace the project name if you changed `ProjectName`.)

**Stack updates:** If you deployed an older revision that used the parameter name `CodeStarConnectionArn`, redeploy with **`CodeConnectionArn`** (same ARN value).

## Service Catalog / Studio custom project

1. Upload this template (or a product template that **nested-stacks** it) to an S3 bucket your catalog can read.
2. Create a **Service Catalog product** whose provisioning artifact is that template.
3. Add the product to the portfolio SageMaker Studio uses for **custom templates**.

Exact Studio admin steps change with AWS releases; search the latest **“SageMaker custom project templates”** / **Service Catalog** documentation for your console.

## Relationship to `codebuild-buildspec.yml`

The root buildspec expects **environment variables** on the CodeBuild project:

- `SAGEMAKER_PIPELINE_ROLE_ARN` (required)
- `SAGEMAKER_EXEC_PIPELINE_ROLE_ARN` (optional; defaults to the pipeline role)
- `ARTIFACT_BUCKET` (required)
- `SAGEMAKER_PROJECT_NAME` / `SAGEMAKER_PROJECT_ID` (optional; buildspec defaults apply if unset)

The CloudFormation template sets all of these on the project it creates. If you still use a **Studio-generated** CodeBuild project, add the same variables there (or align names with what your org’s template exports).
