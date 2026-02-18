from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from agents.base_agent import BaseAgent


class DataCleaningAgent(BaseAgent):
    def __init__(self):
        super().__init__("DataCleaningAgent")

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        dataset_path = state.get("dataset_path")
        if not dataset_path:
            return {
                "data_cleaning": {
                    "status": "failed",
                    "error": "dataset_path missing in state",
                }
            }

        path = Path(dataset_path)
        if not path.exists():
            return {
                "data_cleaning": {
                    "status": "failed",
                    "error": f"dataset file not found: {dataset_path}",
                }
            }

        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")

        rows_before, cols_before = df.shape
        missing_before = int(df.isna().sum().sum())
        duplicates_removed = int(df.duplicated().sum())

        # remove duplicates
        df = df.drop_duplicates()

        # fill missing numeric with median
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        # fill missing categorical with mode
        cat_cols = df.select_dtypes(exclude=["number"]).columns
        for col in cat_cols:
            if df[col].isna().any():
                mode = df[col].mode(dropna=True)
                fill_value = mode.iloc[0] if not mode.empty else "unknown"
                df[col] = df[col].fillna(fill_value)

        # strip extra spaces from object cols
        for col in cat_cols:
            df[col] = df[col].astype(str).str.strip()

        rows_after, cols_after = df.shape
        missing_after = int(df.isna().sum().sum())

        cleaned_path = Path("data") / "processed" / f"cleaned_{path.name}"
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cleaned_path, index=False)

        return {
            "data_cleaning": {
                "status": "success",
                "input_dataset_path": str(path),
                "cleaned_dataset_path": str(cleaned_path),
                "rows_before": int(rows_before),
                "rows_after": int(rows_after),
                "columns_before": int(cols_before),
                "columns_after": int(cols_after),
                "missing_before": missing_before,
                "missing_after": missing_after,
                "duplicates_removed": duplicates_removed,
            }
        }
