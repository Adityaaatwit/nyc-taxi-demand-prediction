import pytest
import mlflow
import json
from pathlib import Path
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn import set_config
from src.models.mlflow_config import configure_mlflow


set_config(transform_output="pandas")

configure_mlflow()


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info

required_paths = [
    Path("run_information.json"),
    Path("data/processed/train.csv"),
    Path("data/processed/test.csv"),
    Path("models/encoder.joblib"),
]
if any(not path.exists() for path in required_paths):
    pytest.skip(
        "Restore or build the DVC artifacts before running performance tests.",
        allow_module_level=True,
    )

model_path = load_model_information("run_information.json")["model_uri"]
model = mlflow.sklearn.load_model(model_path)

# current path
current_path = Path(__file__)
# set the root path
root_path = current_path.parent.parent
# data_path
train_data_path = root_path / "data/processed/train.csv"
test_data_path = root_path / "data/processed/test.csv"

# path for the encoder
encoder_path = root_path / "models/encoder.joblib"
encoder = joblib.load(encoder_path)

# build the model pipeline
model_pipe = Pipeline(steps=[
    ("encoder",encoder),
    ("regressor",model)
])

# test function
@pytest.mark.parametrize(argnames="data_path,threshold",
                         argvalues=[(train_data_path,0.1),
                                    (test_data_path,0.1)])
def test_performance(data_path, threshold):
    # load the data from path
    data = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime")
    # make X and y
    X = data.drop(columns=["total_pickups"])
    y = data["total_pickups"]
    # do predictions
    y_pred = model_pipe.predict(X)
    # calculate the loss
    loss = mean_absolute_percentage_error(y, y_pred)
    # check the performance
    assert loss <= threshold,  f"The model does not pass the performance threshold of {threshold}"
