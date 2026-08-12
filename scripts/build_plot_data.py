"""Build a small, deterministic map sample from the cleaned taxi coordinates."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT_PATH = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT_PATH / "data/interim/df_without_outliers.csv"
OUTPUT_PATH = ROOT_PATH / "data/external/plot_data.csv"
SCALER_PATH = ROOT_PATH / "models/scaler.joblib"
KMEANS_PATH = ROOT_PATH / "models/mb_kmeans.joblib"

COORDINATE_COLUMNS = ["pickup_longitude", "pickup_latitude"]
ROWS_PER_REGION = 500
CHUNK_SIZE = 1_000_000


def smallest_keys_per_region(frame):
    """Retain the globally smallest random keys for each region."""
    retained = []
    for _, region_frame in frame.groupby("region", sort=False):
        retained.append(region_frame.nsmallest(ROWS_PER_REGION, "_sample_key"))
    return pd.concat(retained, ignore_index=True)


def main():
    required = [INPUT_PATH, SCALER_PATH, KMEANS_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing reproduced artifacts: " + ", ".join(missing)
        )

    scaler = joblib.load(SCALER_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    random = np.random.default_rng(42)
    sample = None

    total_rows = 0
    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_PATH,
            usecols=COORDINATE_COLUMNS,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        scaled_coordinates = scaler.transform(chunk[COORDINATE_COLUMNS])
        chunk["region"] = kmeans.predict(scaled_coordinates)
        chunk["_sample_key"] = random.random(len(chunk))
        candidates = chunk if sample is None else pd.concat([sample, chunk])
        sample = smallest_keys_per_region(candidates)
        total_rows += len(chunk)
        print(f"Processed {total_rows:,} coordinates (chunk {chunk_number}).")

    sample = (
        sample.sort_values(["region", "_sample_key"])
        .drop(columns="_sample_key")
        .reset_index(drop=True)
    )
    sample["region"] = sample["region"].astype(int)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT_PATH, index=False)
    print(
        f"Saved {len(sample):,} map points across "
        f"{sample['region'].nunique()} regions to {OUTPUT_PATH}."
    )


if __name__ == "__main__":
    main()
