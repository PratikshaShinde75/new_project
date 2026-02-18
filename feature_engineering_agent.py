from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from agents.base_agent import BaseAgent


class FeatureEngineeringAgent(BaseAgent):
    def __init__(self):
        super().__init__("FeatureEngineeringAgent")

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        dataset_path = state.get("cleaned_dataset_path") or state.get("dataset_path")
        if not dataset_path:
            return {"feature_engineering": {"status": "failed", "error": "No dataset path available"}}

        path = Path(dataset_path)
        if not path.exists():
            return {"feature_engineering": {"status": "failed", "error": f"Dataset not found: {dataset_path}"}}

        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")

        target_column = state.get("target_column")
        original_columns = list(df.columns)

        categorical_columns = [
            col for col in df.select_dtypes(exclude=["number"]).columns
            if col != target_column
        ]

        # Keep one-hot size controlled for practical runtime in local machine.
        encodable_columns = [col for col in categorical_columns if df[col].nunique(dropna=True) <= 20]
        skipped_columns = [col for col in categorical_columns if col not in encodable_columns]

        if encodable_columns:
            df = pd.get_dummies(df, columns=encodable_columns, drop_first=True)

        engineered_path = Path("data") / "processed" / f"features_{path.name}"
        engineered_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(engineered_path, index=False)

        return {
            "feature_engineering": {
                "status": "success",
                "input_dataset_path": str(path),
                "engineered_dataset_path": str(engineered_path),
                "features_before": int(len(original_columns)),
                "features_after": int(len(df.columns)),
                "encoded_columns": encodable_columns,
                "skipped_high_cardinality_columns": skipped_columns,
            }
        }
