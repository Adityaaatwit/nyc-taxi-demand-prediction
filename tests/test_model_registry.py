import mlflow
import json
from pathlib import Path

import pytest

from src.models.mlflow_config import configure_mlflow

configure_mlflow()


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info

run_information_path = Path("run_information.json")
if not run_information_path.exists():
    pytest.skip(
        "Run the DVC evaluate stage before testing the local model registry.",
        allow_module_level=True,
    )

model_path = load_model_information(run_information_path)["model_uri"]
model = mlflow.sklearn.load_model(model_path)


def test_load_model_from_registry():
    assert model is not None, "Failed to load model from registry"
