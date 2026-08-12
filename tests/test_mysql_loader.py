import pandas as pd
import pytest

from scripts.load_mysql_data import batched_rows, require_columns


def test_batched_rows_preserves_order_and_batch_size():
    frame = pd.DataFrame({"id": [1, 2, 3], "region": [10, 20, 30]})

    batches = list(batched_rows(frame, ["id", "region"], batch_size=2))

    assert batches == [[[1, 10], [2, 20]], [[3, 30]]]


def test_require_columns_reports_missing_fields():
    frame = pd.DataFrame({"region": [1]})

    with pytest.raises(ValueError, match="pickup_latitude"):
        require_columns(
            frame,
            ["region", "pickup_latitude"],
            "locations.csv",
        )
