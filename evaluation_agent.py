from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from agents.base_agent import BaseAgent


class EvaluationAgent(BaseAgent):
    def __init__(self):
        super().__init__("EvaluationAgent")

    @staticmethod
    def _prepare_feature_frame(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        X = df.select_dtypes(include=["number", "bool"]).copy()
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        return X[feature_columns]

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        model_training = state.get("model_training", {})
        model_path = model_training.get("model_path")
        if not model_path:
            return {"evaluation": {"status": "failed", "error": "Model path not available from model training stage"}}

        model_file = Path(model_path)
        if not model_file.exists():
            return {"evaluation": {"status": "failed", "error": f"Model file not found: {model_path}"}}

        dataset_path = (
            state.get("engineered_dataset_path")
            or state.get("cleaned_dataset_path")
            or state.get("dataset_path")
        )
        if not dataset_path:
            return {"evaluation": {"status": "failed", "error": "No dataset path available for evaluation"}}

        data_file = Path(dataset_path)
        if not data_file.exists():
            return {"evaluation": {"status": "failed", "error": f"Dataset not found: {dataset_path}"}}

        try:
            df = pd.read_csv(data_file)
        except UnicodeDecodeError:
            df = pd.read_csv(data_file, encoding="latin-1")

        with model_file.open("rb") as f:
            payload = pickle.load(f)

        model = payload.get("model")
        task_type = payload.get("task_type", "classification")
        target_column = payload.get("target_column")
        feature_columns = payload.get("feature_columns", [])

        if not target_column or target_column not in df.columns:
            return {"evaluation": {"status": "failed", "error": "Target column missing in evaluation dataset"}}

        y = df[target_column]
        X = self._prepare_feature_frame(df, feature_columns)
        if X.empty:
            return {"evaluation": {"status": "failed", "error": "No usable feature columns for evaluation"}}

        if task_type == "classification":
            y = y.astype(str)
            stratify = y if y.nunique(dropna=True) > 1 else None
        else:
            y = pd.to_numeric(y, errors="coerce")
            valid_mask = ~y.isna()
            X = X[valid_mask]
            y = y[valid_mask]
            stratify = None

        if len(X) < 10:
            return {"evaluation": {"status": "failed", "error": "Not enough rows for evaluation"}}

        try:
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify
            )
        except ValueError:
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=None
            )

        predictions = model.predict(X_test)

        if task_type == "classification":
            metrics = {
                "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
                "precision_weighted": round(float(precision_score(y_test, predictions, average="weighted", zero_division=0)), 4),
                "recall_weighted": round(float(recall_score(y_test, predictions, average="weighted", zero_division=0)), 4),
                "f1_weighted": round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 4),
            }
        else:
            rmse = float(mean_squared_error(y_test, predictions) ** 0.5)
            metrics = {
                "r2": round(float(r2_score(y_test, predictions)), 4),
                "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
                "rmse": round(rmse, 4),
            }

        return {
            "evaluation": {
                "status": "success",
                "task_type": task_type,
                "target_column": target_column,
                "model_path": str(model_file),
                "test_rows": int(len(X_test)),
                "metrics": metrics,
            }
        }
