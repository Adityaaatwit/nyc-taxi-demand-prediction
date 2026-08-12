"""Load the Streamlit app's DVC-managed CSV artifacts into local MySQL."""

import os
from pathlib import Path

import pandas as pd
import pymysql
from dotenv import load_dotenv


ROOT_PATH = Path(__file__).resolve().parent.parent
LOCATION_PATH = ROOT_PATH / "data/external/plot_data.csv"
DEMAND_PATH = ROOT_PATH / "data/processed/test.csv"

LOCATION_COLUMNS = ["pickup_longitude", "pickup_latitude", "region"]
DEMAND_COLUMNS = [
    "tpep_pickup_datetime",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "region",
    "total_pickups",
    "avg_pickups",
    "day_of_week",
]

CREATE_LOCATIONS = """
CREATE TABLE IF NOT EXISTS app_locations (
    id INT UNSIGNED NOT NULL,
    pickup_longitude DOUBLE NOT NULL,
    pickup_latitude DOUBLE NOT NULL,
    region SMALLINT UNSIGNED NOT NULL,
    PRIMARY KEY (id),
    INDEX idx_app_locations_region (region)
) ENGINE=InnoDB
"""

CREATE_DEMAND = """
CREATE TABLE IF NOT EXISTS demand_features (
    tpep_pickup_datetime DATETIME NOT NULL,
    lag_1 DOUBLE NOT NULL,
    lag_2 DOUBLE NOT NULL,
    lag_3 DOUBLE NOT NULL,
    lag_4 DOUBLE NOT NULL,
    region SMALLINT UNSIGNED NOT NULL,
    total_pickups DOUBLE NOT NULL,
    avg_pickups DOUBLE NOT NULL,
    day_of_week TINYINT UNSIGNED NOT NULL,
    PRIMARY KEY (tpep_pickup_datetime, region)
) ENGINE=InnoDB
"""

INSERT_LOCATION = """
INSERT INTO app_locations (
    id, pickup_longitude, pickup_latitude, region
) VALUES (%s, %s, %s, %s)
"""

INSERT_DEMAND = """
INSERT INTO demand_features (
    tpep_pickup_datetime, lag_1, lag_2, lag_3, lag_4,
    region, total_pickups, avg_pickups, day_of_week
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def database_config():
    load_dotenv(ROOT_PATH / ".env")
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
    return {
        "host": os.environ["MYSQL_HOST"],
        "port": int(os.environ["MYSQL_PORT"]),
        "database": os.environ["MYSQL_DATABASE"],
        "user": os.environ["MYSQL_USER"],
        "password": os.environ["MYSQL_PASSWORD"],
        "connect_timeout": 5,
    }


def require_columns(frame, expected, path):
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")


def batched_rows(frame, columns, batch_size=2_000):
    values = frame.loc[:, columns].values.tolist()
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def main():
    missing_files = [
        str(path) for path in (LOCATION_PATH, DEMAND_PATH) if not path.exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Missing DVC data artifacts: "
            + ", ".join(missing_files)
            + ". Restore or generate them before loading MySQL."
        )

    locations = pd.read_csv(LOCATION_PATH)
    demand = pd.read_csv(DEMAND_PATH, parse_dates=["tpep_pickup_datetime"])
    require_columns(locations, LOCATION_COLUMNS, LOCATION_PATH)
    require_columns(demand, DEMAND_COLUMNS, DEMAND_PATH)

    locations = locations.loc[:, LOCATION_COLUMNS].reset_index(drop=True)
    locations.insert(0, "id", locations.index + 1)

    connection = pymysql.connect(**database_config())
    try:
        with connection.cursor() as cursor:
            cursor.execute(CREATE_LOCATIONS)
            cursor.execute(CREATE_DEMAND)
            cursor.execute("DELETE FROM app_locations")
            cursor.execute("DELETE FROM demand_features")

            for rows in batched_rows(
                locations,
                ["id", *LOCATION_COLUMNS],
            ):
                cursor.executemany(INSERT_LOCATION, rows)

            for rows in batched_rows(demand, DEMAND_COLUMNS):
                cursor.executemany(INSERT_DEMAND, rows)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    print(
        f"Loaded {len(locations):,} locations and {len(demand):,} demand rows "
        f"into {os.environ['MYSQL_DATABASE']}."
    )


if __name__ == "__main__":
    main()
