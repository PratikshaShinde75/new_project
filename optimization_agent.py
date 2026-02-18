from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import importlib.util
import json
import pickle
import re

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from agents.base_agent import BaseAgent


def _load_settings():
    config_path = Path(__file__).resolve().parents[1] / "app" / "config.py"
    spec = importlib.util.spec_from_file_location("project_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load app/config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.settings


settings = _load_settings()


class OptimizationAgent(BaseAgent):
    def __init__(self):
        super().__init__("OptimizationAgent")

    @staticmethod
    def _slug(text: str) -> str:
        clean = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower()).strip("_")
        return clean[:40] if clean else "analysis"

    @staticmethod
    def _prepare_data(df: pd.DataFrame, feature_columns: list[str], target_column: str, task_type: str):
        X = df.select_dtypes(include=["number", "bool"]).copy()
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_columns]
        y = df[target_column]

        if task_type == "classification":
            y = y.astype(str)
            stratify = y if y.nunique(dropna=True) > 1 else None
            return X, y, stratify

        y = pd.to_numeric(y, errors="coerce")
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        return X, y, None

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        evaluation = state.get("evaluation", {})
        model_training = state.get("model_training", {})
        if evaluation.get("status") != "success" or model_training.get("status") != "success":
            return {
                "optimization": {
                    "status": "failed",
                    "error": "Evaluation or model training stage not completed",
                }
            }

        dataset_path = (
            state.get("engineered_dataset_path")
            or state.get("cleaned_dataset_path")
            or state.get("dataset_path")
        )
        model_path = model_training.get("model_path")
        if not dataset_path or not model_path:
            return {"optimization": {"status": "failed", "error": "Dataset or model path missing"}}

        dataset_file = Path(dataset_path)
        model_file = Path(model_path)
        if not dataset_file.exists() or not model_file.exists():
            return {"optimization": {"status": "failed", "error": "Dataset or model file not found"}}

        try:
            df = pd.read_csv(dataset_file)
        except UnicodeDecodeError:
            df = pd.read_csv(dataset_file, encoding="latin-1")

        with model_file.open("rb") as f:
            payload = pickle.load(f)

        task_type = payload.get("task_type", "classification")
        target_column = payload.get("target_column")
        feature_columns = payload.get("feature_columns", [])
        if not target_column or target_column not in df.columns or not feature_columns:
            return {"optimization": {"status": "failed", "error": "Invalid model payload for optimization"}}

        X, y, stratify = self._prepare_data(df, feature_columns, target_column, task_type)
        if len(X) < 10:
            return {"optimization": {"status": "failed", "error": "Not enough rows for optimization"}}

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=None
            )

        if task_type == "classification":
            baseline_metric = float(evaluation.get("metrics", {}).get("accuracy", 0.0))
            param_grid = [
                {"n_estimators": 100, "max_depth": None},
                {"n_estimators": 200, "max_depth": None},
                {"n_estimators": 300, "max_depth": 10},
            ]
            best_score = baseline_metric
            best_params = {}
            best_model = None
            for params in param_grid:
                candidate = RandomForestClassifier(random_state=42, **params)
                candidate.fit(X_train, y_train)
                preds = candidate.predict(X_test)
                score = float(accuracy_score(y_test, preds))
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_model = candidate
        else:
            baseline_metric = float(evaluation.get("metrics", {}).get("r2", 0.0))
            param_grid = [
                {"n_estimators": 100, "max_depth": None},
                {"n_estimators": 200, "max_depth": None},
                {"n_estimators": 300, "max_depth": 10},
            ]
            best_score = baseline_metric
            best_params = {}
            best_model = None
            for params in param_grid:
                candidate = RandomForestRegressor(random_state=42, **params)
                candidate.fit(X_train, y_train)
                preds = candidate.predict(X_test)
                score = float(r2_score(y_test, preds))
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_model = candidate
        metric_name = "accuracy" if task_type == "classification" else "r2"

        improved = bool(best_model is not None and best_score > baseline_metric + 1e-6)
        optimized_model_path = str(model_file)
        if improved:
            tuned_payload = dict(payload)
            tuned_payload["model"] = best_model
            tuned_payload["optimized"] = True
            tuned_payload["optimization_metric"] = metric_name
            tuned_payload["optimization_score"] = best_score
            tuned_payload["best_params"] = best_params
            tuned_name = f"tuned_{model_file.name}"
            tuned_file = model_file.parent / tuned_name
            with tuned_file.open("wb") as f:
                pickle.dump(tuned_payload, f)
            optimized_model_path = str(tuned_file)

        # Save optimization artifacts every run.
        date_text = datetime.now().strftime("%Y_%m_%d")
        slug = self._slug(state.get("problem_statement", ""))
        out_dir = settings.outputs_dir / "optimization"
        out_dir.mkdir(parents=True, exist_ok=True)
        optimization_json_path = out_dir / f"optimization_{slug}_{date_text}.json"
        optimization_md_path = out_dir / f"optimization_{slug}_{date_text}.md"

        artifact = {
            "task_type": task_type,
            "initial_accuracy": round(baseline_metric, 4),
            "after_optimization": round(best_score, 4),
            "baseline_metric": round(baseline_metric, 4),
            "optimized_metric": round(best_score, 4),
            "improved": improved,
            "best_params": best_params,
            "dataset_path": str(dataset_file),
            "baseline_model_path": str(model_file),
            "optimized_model_path": optimized_model_path,
        }
        optimization_json_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

        md_lines = [
            "# Optimization Summary",
            "",
            f"- Task Type: {task_type}",
            f"- Initial Accuracy: {round(baseline_metric, 4)}",
            f"- After Optimization: {round(best_score, 4)}",
            f"- Improved: {improved}",
            f"- Baseline Model Path: {str(model_file)}",
            f"- Optimized Model Path: {optimized_model_path}",
        ]
        if best_params:
            md_lines.append("- Best Parameters:")
            for key, value in best_params.items():
                md_lines.append(f"  - {key} = {value}")
        else:
            md_lines.append("- Best Parameters: N/A")
        optimization_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

        return {
            "optimization": {
                "status": "success",
                "initial_accuracy": round(baseline_metric, 4),
                "after_optimization": round(best_score, 4),
                "baseline_metric": round(baseline_metric, 4),
                "optimized_metric": round(best_score, 4),
                "improved": improved,
                "best_params": best_params,
                "optimized_model_path": optimized_model_path,
                "optimization_json_path": str(optimization_json_path),
                "optimization_md_path": str(optimization_md_path),
            }
        }
