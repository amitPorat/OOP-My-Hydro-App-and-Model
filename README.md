# OOP My Hydro App and Model

Operational flood forecasting project with a Streamlit interface, preprocessing pipeline, two-stage LSTM training flow, and end-to-end inference testing.

## Scope

This repository currently includes:

- Data preparation entry point: `prepare_data.py`
- Training orchestration entry point: `train.py`
- System entry point (UI/headless mode switch): `main.py`
- End-to-end local test runner: `test_end_to_end.py`
- Core implementation modules under `src/`

## Project Structure

```text
.
|-- configs/
|   `-- system_config.yaml
|-- src/
|   |-- preprocess.py
|   |-- inference.py
|   |-- trainer.py
|   `-- ...
|-- ui/
|   `-- Home.py
|-- main.py
|-- prepare_data.py
|-- train.py
|-- test_end_to_end.py
`-- requirements.txt
```

## Environment Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

System-level configuration is read from:

- `configs/system_config.yaml`

Review and adjust all local data paths before running heavy preprocessing, training, or inference.

## Usage

### 1) Launch Streamlit UI

```bash
python main.py --mode ui
```

### 2) Run preprocessing pipeline

```bash
python prepare_data.py
```

This runs UGRID preprocessing and attempts ICON preprocessing via `IconPreprocessor`.

### 3) Run training pipeline

Use the training entry point from Python so you can pass config and overrides explicitly:

```bash
python -c "from train import run_training; print(run_training('configs/system_config.yaml'))"
```

### 4) Run end-to-end test

```bash
python test_end_to_end.py
```

## Notes

- `main.py --mode headless` is currently a placeholder and does not execute the full pipeline yet.
- `prepare_data.py` documents that parts of ICON preprocessing may still require completing the kriging migration in `IconPreprocessor.run()`.

## GitHub Workflow

Remote repository:

- <https://github.com/amitPorat/OOP-My-Hydro-App-and-Model.git>

Push current branch:

```bash
git push
```

If first push requires upstream tracking:

```bash
git push -u origin master
```
