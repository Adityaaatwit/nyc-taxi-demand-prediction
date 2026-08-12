"""Shared local MLflow configuration."""

import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv


DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def configure_mlflow():
    root_path = Path(__file__).resolve().parents[2]
    load_dotenv(root_path / ".env")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri
