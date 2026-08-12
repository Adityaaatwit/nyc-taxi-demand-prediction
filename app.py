import datetime as dt
import os
from pathlib import Path
from time import sleep

import joblib
import pandas as pd
import pymysql
import streamlit as st
from dotenv import load_dotenv
from pymysql.cursors import DictCursor
from sklearn import set_config
from sklearn.pipeline import Pipeline


ROOT_PATH = Path(__file__).parent
load_dotenv(ROOT_PATH / ".env")
set_config(transform_output="pandas")

LOCATION_TABLE = "app_locations"
DEMAND_TABLE = "demand_features"


def database_config():
    keys = [
        "MYSQL_HOST",
        "MYSQL_PORT",
        "MYSQL_DATABASE",
        "MYSQL_USER",
        "MYSQL_PASSWORD",
    ]
    missing = [key for key in keys if not os.getenv(key)]
    if missing:
        raise RuntimeError(
            "Missing MySQL settings: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill in local credentials."
        )

    try:
        port = int(os.environ["MYSQL_PORT"])
    except ValueError as exc:
        raise RuntimeError("MYSQL_PORT must be an integer.") from exc

    return {
        "host": os.environ["MYSQL_HOST"],
        "port": port,
        "database": os.environ["MYSQL_DATABASE"],
        "user": os.environ["MYSQL_USER"],
        "password": os.environ["MYSQL_PASSWORD"],
        "cursorclass": DictCursor,
        "connect_timeout": 5,
    }


def database_connection():
    return pymysql.connect(**database_config())


@st.cache_data
def load_locations():
    query = f"""
        SELECT pickup_longitude, pickup_latitude, region
        FROM {LOCATION_TABLE}
        ORDER BY region
    """
    with database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return pd.DataFrame(cursor.fetchall())


@st.cache_data
def load_date_range():
    query = f"""
        SELECT
            MIN(tpep_pickup_datetime) AS min_datetime,
            MAX(tpep_pickup_datetime) AS max_datetime
        FROM {DEMAND_TABLE}
    """
    with database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            bounds = cursor.fetchone()

    if not bounds or bounds["min_datetime"] is None:
        raise RuntimeError(
            "The demand_features table is empty. Run scripts/load_mysql_data.py."
        )
    return bounds["min_datetime"], bounds["max_datetime"]


@st.cache_data
def load_features(timestamp):
    query = f"""
        SELECT
            tpep_pickup_datetime,
            lag_1,
            lag_2,
            lag_3,
            lag_4,
            region,
            total_pickups,
            avg_pickups,
            day_of_week
        FROM {DEMAND_TABLE}
        WHERE tpep_pickup_datetime = %s
        ORDER BY region
    """
    with database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (timestamp.to_pydatetime(),))
            rows = cursor.fetchall()
    return pd.DataFrame(rows)


def load_model_artifacts():
    artifact_paths = {
        "scaler": ROOT_PATH / "models/scaler.joblib",
        "encoder": ROOT_PATH / "models/encoder.joblib",
        "model": ROOT_PATH / "models/model.joblib",
        "kmeans": ROOT_PATH / "models/mb_kmeans.joblib",
    }
    missing = [str(path) for path in artifact_paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(
            "Missing model artifacts. Restore or build them with DVC: "
            + ", ".join(missing)
        )
    return {name: joblib.load(path) for name, path in artifact_paths.items()}


try:
    df_plot = load_locations()
    min_datetime, max_datetime = load_date_range()
    artifacts = load_model_artifacts()
except (RuntimeError, pymysql.MySQLError) as exc:
    st.error(str(exc))
    st.stop()

if df_plot.empty:
    st.error("The app_locations table is empty. Run scripts/load_mysql_data.py.")
    st.stop()

scaler = artifacts["scaler"]
encoder = artifacts["encoder"]
model = artifacts["model"]
kmeans = artifacts["kmeans"]

st.title("Uber Demand in New York City 🚕🌆")

st.sidebar.title("Options")
map_type = st.sidebar.radio(
    label="Select the type of Map",
    options=["Complete NYC Map", "Only for Neighborhood Regions"],
    index=1,
)

st.subheader("Date")
date = st.date_input(
    "Select the date",
    value=None,
    min_value=min_datetime.date(),
    max_value=max_datetime.date(),
)
st.write("**Date:**", date)

st.subheader("Time")
time = st.time_input("Select the time", value=None, step=dt.timedelta(minutes=15))
st.write("**Current Time:**", time)

if date and time:
    next_interval = dt.datetime.combine(date, time) + dt.timedelta(minutes=15)
    index = pd.Timestamp(next_interval)
    st.write("Demand for Time:", next_interval.time())
    st.write("**Date & Time:**", index)

    input_data = load_features(index)
    if input_data.empty:
        st.warning(
            "No feature rows exist for that interval. Choose an earlier time "
            "within the available date range."
        )
        st.stop()
    input_data = input_data.set_index("tpep_pickup_datetime")

    st.subheader("Location")
    sample_loc = df_plot.sample(1, random_state=None).reset_index(drop=True)
    latitude = sample_loc["pickup_latitude"].item()
    longitude = sample_loc["pickup_longitude"].item()
    current_region = int(sample_loc["region"].item())
    st.write("**Your Current Location**")
    st.write(f"Lat: {latitude}")
    st.write(f"Long: {longitude}")

    with st.spinner("Fetching your Current Region"):
        sleep(1)

    st.write("Region ID:", current_region)
    scaled_coordinates = scaler.transform(
        sample_loc[["pickup_longitude", "pickup_latitude"]]
    )

    st.subheader("MAP")

    colors = [
        "#FF0000", "#FF4500", "#FF8C00", "#FFD700", "#ADFF2F",
        "#32CD32", "#008000", "#006400", "#00FF00", "#7CFC00",
        "#00FA9A", "#00FFFF", "#40E0D0", "#4682B4", "#1E90FF",
        "#0000FF", "#0000CD", "#8A2BE2", "#9932CC", "#BA55D3",
        "#FF00FF", "#FF1493", "#C71585", "#FF4500", "#FF6347",
        "#FFA07A", "#FFDAB9", "#FFE4B5", "#F5DEB3", "#EEE8AA",
    ]
    region_colors = {
        region: colors[int(region) % len(colors)]
        for region in df_plot["region"].unique()
    }
    df_plot = df_plot.assign(color=df_plot["region"].map(region_colors))

    prediction_pipeline = Pipeline(
        [
            ("encoder", encoder),
            ("regressor", model),
        ]
    )

    if map_type == "Complete NYC Map":
        displayed_regions = sorted(input_data["region"].astype(int).unique())
        map_data = df_plot
    else:
        cluster_coordinates = getattr(
            scaled_coordinates, "values", scaled_coordinates
        )
        distances = kmeans.transform(cluster_coordinates)
        distances = getattr(distances, "values", distances).ravel()
        displayed_regions = sorted(distances.argsort()[:9].tolist())
        map_data = df_plot[df_plot["region"].isin(displayed_regions)]

    with st.spinner("Preparing the demand map"):
        sleep(1)
        st.map(
            data=map_data,
            latitude="pickup_latitude",
            longitude="pickup_longitude",
            size=0.01,
            color="color",
        )

    prediction_data = (
        input_data[input_data["region"].isin(displayed_regions)]
        .sort_values("region")
    )
    predictions = prediction_pipeline.predict(
        prediction_data.drop(columns=["total_pickups"])
    )

    st.markdown("### Map Legend")
    for region_id, demand in zip(
        prediction_data["region"].astype(int).tolist(), predictions
    ):
        label = (
            f"{region_id} (Current region)"
            if region_id == current_region
            else str(region_id)
        )
        color = colors[region_id % len(colors)]
        st.markdown(
            f'<div style="display: flex; align-items: center;">'
            f'<div style="background-color:{color}; width: 20px; height: 10px; '
            f'margin-right: 10px;"></div>'
            f'Region ID: {label}<br>Demand: {int(demand)}<br><br>',
            unsafe_allow_html=True,
        )
