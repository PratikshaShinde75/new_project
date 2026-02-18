from __future__ import annotations

from pathlib import Path
from typing import Any

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


class ModelSelectionAgent(BaseAgent):
    def __init__(self):
        super().__init__("ModelSelectionAgent")

    @staticmethod
    def _candidate_models(task_type: str) -> list[str]:
        if task_type == "classification":
            return ["Logistic Regression", "Random Forest", "SVM", "XGBoost"]
        if task_type in {"regression", "forecasting"}:
            return ["Linear Regression", "Random Forest Regressor", "SVR", "XGBoost Regressor"]
        return []

    @staticmethod
    def _pick_target_column(df: pd.DataFrame, state: dict[str, Any]) -> str:
        target_column = state.get("target_column")
        if target_column and target_column in df.columns:
            return str(target_column)

        target_variable = (state.get("problem_understanding", {}) or {}).get("target_variable", "")
        if target_variable:
            for col in df.columns:
                if str(target_variable).lower() in str(col).lower():
                    return str(col)

        return str(df.columns[-1])

    @staticmethod
    def _is_classification(task_type: str, y: pd.Series) -> bool:
        if task_type == "classification":
            return True
        if task_type in {"regression", "forecasting"}:
            return False
        return (not pd.api.types.is_numeric_dtype(y)) or y.nunique(dropna=True) <= 12

    @staticmethod
    def _resolve_model(model_name: str, task_type: str):
        if task_type == "classification":
            if model_name == "Logistic Regression":
                return LogisticRegression(max_iter=2000)
            if model_name == "SVM":
                return SVC()
            if model_name == "XGBoost":
                if XGBClassifier is None:
                    return None
                return XGBClassifier(random_state=42, eval_metric="logloss")
            return RandomForestClassifier(random_state=42)

        if model_name == "Linear Regression":
            return LinearRegression()
        if model_name == "SVR":
            return SVR()
        if model_name == "XGBoost Regressor":
            if XGBRegressor is None:
                return None
            return XGBRegressor(random_state=42)
        return RandomForestRegressor(random_state=42)

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        dataset_path = (
            state.get("engineered_dataset_path")
            or state.get("cleaned_dataset_path")
            or state.get("dataset_path")
        )
        if not dataset_path:
            return {"model_selection": {"status": "failed", "error": "No dataset path available"}}

        path = Path(dataset_path)
        if not path.exists():
            return {"model_selection": {"status": "failed", "error": f"Dataset not found: {dataset_path}"}}

        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")

        task_type = (
            state.get("problem_understanding", {}).get("task_type")
            or state.get("task_type")
            or "unknown"
        )
        rows, columns = df.shape
        if columns < 2:
            return {"model_selection": {"status": "failed", "error": "Dataset needs at least 2 columns"}}

        target_column = self._pick_target_column(df, state)
        if target_column not in df.columns:
            return {"model_selection": {"status": "failed", "error": f"Target column not found: {target_column}"}}

        y = df[target_column]
        X = df.drop(columns=[target_column]).select_dtypes(include=["number", "bool"]).copy()
        if X.empty:
            return {"model_selection": {"status": "failed", "error": "No numeric features available for model selection"}}

        X = X.fillna(X.median(numeric_only=True))

        is_classification = self._is_classification(task_type, y)
        run_task_type = "classification" if is_classification else "regression"
        if is_classification:
            y = y.astype(str)
            if y.nunique(dropna=True) < 2:
                return {"model_selection": {"status": "failed", "error": "Classification target needs at least 2 classes"}}
        else:
            y = pd.to_numeric(y, errors="coerce")
            valid_mask = ~y.isna()
            X = X[valid_mask]
            y = y[valid_mask]
            if len(y) < 10:
                return {"model_selection": {"status": "failed", "error": "Not enough valid rows for regression model selection"}}

        stratify = y if is_classification and y.nunique(dropna=True) > 1 else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify
            )
        except ValueError:
            # Fallback when stratified split is not possible for rare classes.
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        candidates = self._candidate_models(run_task_type)
        compared_models: list[dict[str, Any]] = []
        metric_label = "Accuracy" if run_task_type == "classification" else "R2 Score"

        for model_name in candidates:
            model = self._resolve_model(model_name, run_task_type)
            if model is None:
                compared_models.append(
                    {
                        "model": model_name,
                        "metric": metric_label,
                        "score": None,
                        "note": "Not available (install xgboost)",
                    }
                )
                continue

            try:
                if run_task_type == "classification" and model_name == "XGBoost":
                    all_labels = pd.Index(y_train).append(pd.Index(y_test)).astype(str).unique()
                    y_train_fit = pd.Categorical(y_train.astype(str), categories=all_labels).codes
                    y_test_eval = pd.Categorical(y_test.astype(str), categories=all_labels).codes
                    model.fit(X_train, y_train_fit)
                    predictions = model.predict(X_test)
                    score_value = float(accuracy_score(y_test_eval, predictions))
                else:
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    score_value = (
                        float(accuracy_score(y_test, predictions))
                        if run_task_type == "classification"
                        else float(r2_score(y_test, predictions))
                    )
                compared_models.append(
                    {
                        "model": model_name,
                        "metric": metric_label,
                        "score": round(score_value * 100, 2),
                    }
                )
            except Exception as ex:
                compared_models.append(
                    {
                        "model": model_name,
                        "metric": metric_label,
                        "score": None,
                        "note": f"Skipped: {type(ex).__name__}",
                    }
                )

        available = [item for item in compared_models if item.get("score") is not None]
        available.sort(key=lambda item: float(item["score"]), reverse=True)
        recommended = available[0]["model"] if available else "No model selected"

        return {
            "model_selection": {
                "status": "success",
                "task_type": run_task_type,
                "dataset_path": str(path),
                "rows": int(rows),
                "columns": int(columns),
                "target_column": target_column,
                "candidate_models": candidates,
                "models_compared": compared_models,
                "recommended_model": recommended,
            }
        }
