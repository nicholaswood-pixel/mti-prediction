# SageMaker Pipelines project template (MNIST)

This repository is a **clone-and-customize template** for data scientists: change the code under `pipelines/dev/` (preprocess, train, evaluate) and keep the same SageMaker Pipeline layout, CI entry point, and packaging conventions.

The **default example** is **MNIST digit classification** (784 features from OpenML `mnist_784`, `scikit-learn` SGD classifier). It demonstrates:

1. **Preprocess** — download/split data, write `dev.joblib` to the processing output prefix.
2. **Train** — fit a classifier, write `model.joblib` and `test_data.joblib` into the training model artifact.
3. **Evaluate** — score test accuracy, write `evaluation.json` for the Model Registry.
4. **Condition** — register the model package only if **test accuracy ≥ `AccuracyThreshold`** (default `0.85`).

## Where to edit

| Path | Role |
|------|------|
| `pipelines/dev/preprocess.py` | Data loading, splitting, feature prep |
| `pipelines/dev/train.py` | Model training and artifacts in `SM_MODEL_DIR` |
| `pipelines/dev/evaluate.py` | Metrics JSON; keep `classification_metrics.accuracy.value` **or** update `pipeline.py` JsonGet paths and the condition step together |
| `pipelines/dev/pipeline.py` | Step graph, parameters, instance types, registration |
| `pipelines/dev/requirements.txt` | Extra pip packages for the **training** container (`source_dir`) |

Shared CLI: `get-pipeline-definition`, `run-pipeline` (see `setup.py` entry points).

## Pipeline flow

Defined in `pipelines/dev/pipeline.py`:

1. `PreprocessDevData` — `SKLearnProcessor` runs `preprocess.py`.
2. `TrainDevModel` — `SKLearn` script mode runs `train.py` with `source_dir` including `requirements.txt`.
3. `EvaluateDevModel` — `ScriptProcessor` runs `evaluate.py` on the training artifact.
4. `CheckDevAccuracy` — if accuracy passes, `RegisterDevModel` registers to the Model Registry.

**Note:** Preprocessing calls `fetch_openml`; the processing job needs **outbound internet** (or host the prepared data on S3 and change `preprocess.py` to read from `ProcessingInput` instead).

## Parameters (defaults are template-friendly)

- `AccuracyThreshold` — minimum test accuracy to allow registration (default `0.85`).
- `MaxTrainSamples` / `MaxTestSamples` — subsample after split for faster/cheaper runs (defaults `12000` / `3000`; set large values or adjust `preprocess.py` to use full MNIST).

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install .
pip install .[test]
python -m pytest
```

## Render the pipeline definition JSON

```bash
get-pipeline-definition \
  --module-name pipelines.dev.pipeline \
  --kwargs '{"region":"us-east-1","role":"arn:aws:iam::123456789012:role/SageMakerExecutionRole","default_bucket":"my-artifact-bucket","pipeline_name":"dev-digit-pipeline","model_package_group_name":"dev-digit-pipeline","base_job_prefix":"dev-digit-pipeline"}'
```

## Create/update and run the pipeline

```bash
run-pipeline \
  --module-name pipelines.dev.pipeline \
  --role-arn arn:aws:iam::123456789012:role/SageMakerPipelineServiceRole \
  --tags '[]' \
  --kwargs '{"region":"us-east-1","role":"arn:aws:iam::123456789012:role/SageMakerExecutionRole","default_bucket":"my-artifact-bucket","pipeline_name":"dev-digit-pipeline","model_package_group_name":"dev-digit-pipeline","base_job_prefix":"dev-digit-pipeline"}'
```

- `--role-arn` — role used to **create/update** the pipeline.
- `role` inside `--kwargs` — execution role for processing, training, and registration.

## CI/CD

`codebuild-buildspec.yml` installs the package and runs `run-pipeline` for `pipelines.dev.pipeline`.

Optional infrastructure examples live under `infra/` (update names and ARNs for your account).

## Repository layout

```text
|-- codebuild-buildspec.yml
|-- pipelines
|   |-- dev
|   |   |-- evaluate.py
|   |   |-- pipeline.py
|   |   |-- preprocess.py
|   |   |-- train.py
|   |   `-- requirements.txt
|   |-- get_pipeline_definition.py
|   |-- run_pipeline.py
|   `-- _utils.py
|-- tests
|   `-- test_pipelines.py
|-- setup.py
`-- README.md
```

## Renaming the template

After cloning, you can rename the Python package folder (e.g. `pipelines/dev` → `pipelines/your_project`), then update `setup.py` `package_data`, `codebuild-buildspec.yml` `--module-name`, and any docs or notebooks accordingly.
