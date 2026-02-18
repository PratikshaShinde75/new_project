from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import pickle
import re
import importlib.util

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR

from agents.base_agent import BaseAgent

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None


def _load_settings():
    config_path = Path(__file__).resolve().parents[1] / "app" / "config.py"
    spec = importlib.util.spec_from_file_location("project_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load app/config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.settings


settings = _load_settings()


class ModelTrainingAgent(BaseAgent):
    def __init__(self):
        super().__init__("ModelTrainingAgent")

    @staticmethod
    def _slug_from_state(state: dict[str, Any]) -> str:
        problem = (state.get("problem_statement") or "").strip().lower()
        problem_understanding = state.get("problem_understanding", {}) or {}
        target_variable = str(problem_understanding.get("target_variable", "")).strip().lower()
        keywords = [str(k).strip().lower() for k in (problem_understanding.get("keywords", []) or []) if str(k).strip()]
        stop_words = {
            "the", "and", "for", "with", "from", "using", "into", "that", "this", "what",
            "predict", "prediction", "predictions", "forecast", "estimate", "classify",
            "model", "data", "dataset", "analysis", "project", "problem", "statement",
        }

        ordered_tokens: list[str] = []
        seen: set[str] = set()

        def _add_tokens(text: str) -> None:
            for token in re.findall(r"[a-zA-Z0-9]+", text.lower()):
                if len(token) <= 2 or token in stop_words:
                    continue
                if token not in seen:
                    seen.add(token)
                    ordered_tokens.append(token)

        if target_variable:
            _add_tokens(target_variable)
        for keyword in keywords:
            _add_tokens(keyword)
        if not ordered_tokens and problem:
            _add_tokens(problem)

        selected = ordered_tokens[:4]
        if not selected:
            selected = ["ml"]

        clean = re.sub(r"[^a-zA-Z0-9]+", "_", "_".join(selected)).strip("_")
        return clean[:40] if clean else "ml"

    @staticmethod
    def _pick_target_column(df: pd.DataFrame, state: dict[str, Any]) -> str:
        target_column = state.get("target_column")
        if target_column and target_column in df.columns:
            return target_column

        target_variable = (state.get("problem_understanding", {}) or {}).get("target_variable", "")
        if target_variable:
            for col in df.columns:
                if str(target_variable).lower() in str(col).lower():
                    return str(col)

        return str(df.columns[-1])

    @staticmethod
    def _resolve_model(model_name: str, task_type: str):
        if task_type == "classification":
            if model_name in {"LogisticRegression", "Logistic Regression"}:
                return LogisticRegression(max_iter=2000)
            if model_name in {"SVM", "SVC"}:
                return SVC()
            if model_name in {"XGBoost", "XGBoostClassifier"}:
                if XGBClassifier is not None:
                    return XGBClassifier(random_state=42, eval_metric="logloss")
            return RandomForestClassifier(random_state=42)

        if model_name in {"RandomForestRegressor", "Random Forest Regressor"}:
            return RandomForestRegressor(random_state=42)
        if model_name in {"SVR"}:
            return SVR()
        if model_name in {"XGBoost Regressor", "XGBoostRegressor"}:
            if XGBRegressor is not None:
                return XGBRegressor(random_state=42)
            return RandomForestRegressor(random_state=42)
        return LinearRegression()

    @staticmethod
    def _is_classification(task_type: str, y: pd.Series) -> bool:
        if task_type == "classification":
            return True
        if task_type in {"regression", "forecasting"}:
            return False
        return (not pd.api.types.is_numeric_dtype(y)) or y.nunique(dropna=True) <= 12

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        dataset_path = (
            state.get("engineered_dataset_path")
            or state.get("cleaned_dataset_path")
            or state.get("dataset_path")
        )
        if not dataset_path:
            return {"model_training": {"status": "failed", "error": "No dataset path available"}}

        path = Path(dataset_path)
        if not path.exists():
            return {"model_training": {"status": "failed", "error": f"Dataset not found: {dataset_path}"}}

        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")

        if df.shape[1] < 2:
            return {"model_training": {"status": "failed", "error": "Dataset needs at least 2 columns"}}

        target_column = self._pick_target_column(df, state)
        if target_column not in df.columns:
            return {"model_training": {"status": "failed", "error": f"Target column not found: {target_column}"}}

        y = df[target_column]
        X = df.drop(columns=[target_column])
        X = X.select_dtypes(include=["number", "bool"]).copy()
        if X.empty:
            return {"model_training": {"status": "failed", "error": "No numeric features available for training"}}

        task_type_from_state = (
            state.get("problem_understanding", {}).get("task_type")
            or state.get("task_type")
            or "unknown"
        )
        is_classification = self._is_classification(task_type_from_state, y)
        task_type = "classification" if is_classification else "regression"

        if is_classification:
            y = y.astype(str)
        else:
            y = pd.to_numeric(y, errors="coerce")
            valid_mask = ~y.isna()
            X = X[valid_mask]
            y = y[valid_mask]

        if len(X) < 10:
            return {"model_training": {"status": "failed", "error": "Not enough rows after preprocessing"}}

        model_name = state.get("model_selection", {}).get("recommended_model", "")
        model = self._resolve_model(model_name, task_type)

        stratify = y if is_classification and y.nunique(dropna=True) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )

        xgb_label_map: list[str] | None = None
        if is_classification and model_name in {"XGBoost", "XGBoostClassifier"}:
            all_labels = pd.Index(y_train).append(pd.Index(y_test)).astype(str).unique()
            xgb_label_map = [str(v) for v in all_labels]
            y_train_fit = pd.Categorical(y_train.astype(str), categories=all_labels).codes
            y_test_eval = pd.Categorical(y_test.astype(str), categories=all_labels).codes
        else:
            y_train_fit = y_train
            y_test_eval = y_test

        model.fit(X_train, y_train_fit)
        predictions = model.predict(X_test)

        metric_name = "accuracy" if is_classification else "r2"
        metric_value = (
            float(accuracy_score(y_test_eval, predictions))
            if is_classification
            else float(r2_score(y_test, predictions))
        )

        statement = state.get("problem_statement", "")
        date_text = datetime.now().strftime("%Y_%m_%d")
        slug = self._slug_from_state(state)
        model_file_name = f"{slug}_model_{date_text}.pkl"
        model_path = settings.models_dir / model_file_name
        model_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": model,
            "task_type": task_type,
            "target_column": target_column,
            "feature_columns": list(X.columns),
            "problem_statement": statement,
            "saved_date": date_text,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "xgb_label_map": xgb_label_map,
        }
        with model_path.open("wb") as f:
            pickle.dump(payload, f)

        return {
            "model_training": {
                "status": "success",
                "task_type": task_type,
                "target_column": target_column,
                "model_file_name": model_file_name,
                "model_path": str(model_path),
                "metric_name": metric_name,
                "metric_value": round(metric_value, 4),
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
            }
        }
