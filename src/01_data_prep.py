from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import config

# def _load_raw_data_from_hf_or_local() -> pd.DataFrame:
#     """
#     Try to load the raw dataset from the Hugging Face dataset repo.
#     If that fails (e.g., no token or repo yet), fall back to the local CSV.
#     """
#     # Preferred: load from HF dataset repo if token and repo are configured
#     if config.HF_TOKEN and config.HF_DATASET_REPO:
#         try:
#             remote_path = download_dataset_file(
#                 filename="data/engine_data.csv",
#                 repo_id=config.HF_DATASET_REPO,
#                 token=config.HF_TOKEN,
#                 local_dir=config.DATA_DIR,
#             )
#             return pd.read_csv(remote_path)
#         except Exception:
#             # Fall back to local file
#             pass

#     # Local fallback
#     if not config.RAW_DATA_FILE.exists():
#         raise FileNotFoundError(
#             f"Raw data file not found at {config.RAW_DATA_FILE}. "
#             "Ensure engine_data.csv exists or upload it to the HF dataset repo."
#         )

#     return pd.read_csv(config.RAW_DATA_FILE)

def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning and feature engineering.
    """
    # Standardize column names
    df = df.rename(columns=config.RAW_COLUMN_RENAME_MAP)

    # Keep only the expected columns (drop any extras, if present)
    expected_cols = set(config.FEATURE_COLUMNS + [config.TARGET_COLUMN])
    df = df[[col for col in df.columns if col in expected_cols]]

    # Drop duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Handle missing values: for this numeric dataset, fill with median
    if df.isna().any().any():
        df = df.fillna(df.median(numeric_only=True))

    # Ensure target is integer/binary
    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].astype(int)

    return df


def _train_test_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the cleaned dataframe into train and test sets.
    """
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    train_df = X_train.copy()
    train_df[config.TARGET_COLUMN] = y_train

    test_df = X_test.copy()
    test_df[config.TARGET_COLUMN] = y_test

    return train_df, test_df