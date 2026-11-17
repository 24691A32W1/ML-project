"""Quick sanity tests.  Run:  python -m pytest tests   (or: python tests/test_pipeline.py)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import FEATURES, TARGET
from src.generate_dataset import generate
from src.preprocessing import build_preprocessor, clean_dataframe, get_split
from src.utils import categorize_marks


def test_dataset_shape_and_no_duplicates():
    df = generate()
    assert len(df) >= 1000
    assert not df.drop(columns=["Student_ID"]).duplicated().any()
    assert df[TARGET].between(0, 100).all()


def test_target_not_in_features():
    assert TARGET not in FEATURES and "Student_ID" not in FEATURES


def test_split_ratio_and_reproducibility():
    df = clean_dataframe(generate())
    a = get_split(df)
    b = get_split(df)
    assert abs(len(a[1]) / len(df) - 0.2) < 0.01
    assert a[1].index.equals(b[1].index)


def test_preprocessor_handles_missing_and_unseen_category():
    df = clean_dataframe(generate())
    X_train, X_test, _, _ = get_split(df)
    pre = build_preprocessor().fit(X_train)
    X = X_test.copy()
    X.iloc[0, X.columns.get_loc("Gender")] = "Other"
    X.iloc[1, X.columns.get_loc("Sleep_Hours")] = np.nan
    assert not np.isnan(pre.transform(X)).any()


def test_category_boundaries():
    assert categorize_marks(85) == "Excellent"
    assert categorize_marks(84.99) == "Very Good"
    assert categorize_marks(70) == "Very Good"
    assert categorize_marks(55) == "Good"
    assert categorize_marks(54.99) == "Average"
    assert categorize_marks(40) == "Average"
    assert categorize_marks(39.99) == "Needs Improvement"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
