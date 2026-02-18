from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle
import re

import pandas as pd

from agents.base_agent import BaseAgent


class InsightGenerationAgent(BaseAgent):
    def __init__(self):
        super().__init__("InsightGenerationAgent")

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")

    @staticmethod
    def _find_column(df: pd.DataFrame, tokens: list[str]) -> str | None:
        cols = [str(c) for c in df.columns]
        lowered = {c: c.lower() for c in cols}
        tokenized = {c: [p for p in re.split(r"[^a-zA-Z0-9]+", lowered[c]) if p] for c in cols}

        for token in tokens:
            t = token.lower()
            for col in cols:
                if t in tokenized[col]:
                    return col
        for token in tokens:
            t = token.lower()
            for col in cols:
                col_text = lowered[col]
                if col_text == t or col_text.startswith(f"{t}_") or col_text.endswith(f"_{t}") or f"_{t}_" in col_text:
                    return col
        return None

    @staticmethod
    def _resolve_target_column(state: dict[str, Any], df: pd.DataFrame) -> str:
        evaluation = state.get("evaluation", {}) or {}
        model_training = state.get("model_training", {}) or {}
        problem_understanding = state.get("problem_understanding", {}) or {}
        for candidate in [
            evaluation.get("target_column"),
            model_training.get("target_column"),
            problem_understanding.get("target_variable"),
            state.get("target_column"),
        ]:
            if not candidate:
                continue
            for col in df.columns:
                if str(candidate).lower() in str(col).lower() or str(col).lower() in str(candidate).lower():
                    return str(col)
        return str(df.columns[-1])

    @staticmethod
    def _is_health_context(problem_statement: str, target_variable: str) -> bool:
        text = f"{problem_statement} {target_variable}".lower()
        keys = ["mortality", "death", "survival", "patient", "heart", "hospital", "clinical", "medical"]
        return any(k in text for k in keys)

    @staticmethod
    def _positive_mask(series: pd.Series) -> pd.Series | None:
        s = series.astype(str).str.strip().str.lower()
        unique_values = set(s.unique())
        if len(unique_values) < 2:
            return None
        if unique_values.issubset({"0", "1"}):
            return s == "1"
        if unique_values.issubset({"true", "false"}):
            return s == "true"
        if unique_values.issubset({"yes", "no"}):
            return s == "yes"
        return None

    @staticmethod
    def _most_important_features(state: dict[str, Any], df: pd.DataFrame, target_col: str) -> str | None:
        model_path = (
            (state.get("optimization", {}) or {}).get("optimized_model_path")
            or (state.get("model_training", {}) or {}).get("model_path")
        )
        if not model_path:
            return None
        path = Path(model_path)
        if not path.exists():
            return None
        try:
            with path.open("rb") as f:
                payload = pickle.load(f)
        except Exception:
            return None

        model = payload.get("model")
        feature_cols = [c for c in payload.get("feature_columns", []) if c in df.columns and c != target_col]
        if not feature_cols:
            feature_cols = [c for c in df.select_dtypes(include=["number", "bool"]).columns if c != target_col]
        if not feature_cols:
            return None

        pairs: list[tuple[str, float]] = []
        if hasattr(model, "feature_importances_"):
            values = list(model.feature_importances_)
            pairs = list(zip(feature_cols[: len(values)], values))
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if hasattr(coef, "ndim") and coef.ndim > 1:
                vals = [float(abs(v)) for v in coef.mean(axis=0)]
            else:
                vals = [float(abs(v)) for v in coef]
            pairs = list(zip(feature_cols[: len(vals)], vals))

        if not pairs:
            return None
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:3]
        text = ", ".join([f"{name} ({score:.3f})" for name, score in pairs])
        return f"Most Important Features: {text}"

    @staticmethod
    def _metric_value_line(metrics: dict[str, Any]) -> str | None:
        if not metrics:
            return None
        parts: list[str] = []
        for key in ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "r2", "rmse", "mae"]:
            val = metrics.get(key)
            if isinstance(val, (int, float)):
                parts.append(f"{key}={val:.4f}")
            elif val is not None:
                parts.append(f"{key}={val}")
        if not parts:
            return None
        return "Metric Value: " + ", ".join(parts)

    @staticmethod
    def _performance_line(task_type: str, metrics: dict[str, Any]) -> str | None:
        if task_type == "classification":
            acc = metrics.get("accuracy")
            f1 = metrics.get("f1_weighted")
            if isinstance(acc, (int, float)) and isinstance(f1, (int, float)):
                level = "strong" if acc >= 0.8 and f1 >= 0.8 else "moderate" if acc >= 0.7 else "weak"
                return f"Performance Insight: Classification performance is {level} (accuracy={acc:.4f}, f1_weighted={f1:.4f})."
        if task_type in {"regression", "forecasting"}:
            r2 = metrics.get("r2")
            if isinstance(r2, (int, float)):
                level = "strong" if r2 >= 0.8 else "moderate" if r2 >= 0.6 else "weak"
                return f"Performance Insight: Regression performance is {level} (r2={r2:.4f})."
        return None

    @staticmethod
    def _data_quality_line(state: dict[str, Any]) -> str | None:
        dc = state.get("data_cleaning", {}) or {}
        rows_before = dc.get("rows_before")
        rows_after = dc.get("rows_after")
        missing_before = dc.get("missing_before")
        missing_after = dc.get("missing_after")
        if all(v is not None for v in [rows_before, rows_after, missing_before, missing_after]):
            rb = max(int(rows_before), 1)
            ra = int(rows_after)
            mb = max(int(missing_before), 0)
            ma = max(int(missing_after), 0)

            removed_pct = max(0.0, ((rb - ra) / rb) * 100.0)
            missing_reduction_pct = 100.0 if mb == 0 and ma == 0 else (0.0 if mb == 0 else ((mb - ma) / mb) * 100.0)

            if ma == 0 and removed_pct <= 5:
                quality = "high"
            elif ma <= max(1, int(mb * 0.2)) and removed_pct <= 20:
                quality = "moderate"
            else:
                quality = "low"

            return (
                "Data Quality Insight: "
                f"{quality} (rows_removed={removed_pct:.1f}%, missing_reduction={missing_reduction_pct:.1f}%)."
            )
        return None

    @staticmethod
    def _risk_line(state: dict[str, Any], positive_mask: pd.Series | None) -> str | None:
        eda = state.get("eda", {}) or {}
        class_dist = eda.get("class_distribution")
        if positive_mask is not None:
            rate = float(positive_mask.astype(int).mean()) * 100
            if class_dist:
                return f"Risk Insight: positive target rate is {rate:.1f}% and class distribution is {class_dist}."
            return f"Risk Insight: positive target rate is {rate:.1f}%."
        if class_dist:
            return f"Risk Insight: class distribution is {class_dist}."
        return None

    @staticmethod
    def _hidden_pattern_line(state: dict[str, Any]) -> str | None:
        eda = state.get("eda", {}) or {}
        top_corr = eda.get("top_correlated_features", []) or []
        if top_corr:
            return f"Hidden Pattern Insight: strongest correlation signals are {', '.join(top_corr[:3])}."
        return None

    @staticmethod
    def _clinical_impact_line(health_context: bool, positive_mask: pd.Series | None, target_variable: str) -> str | None:
        if not health_context or positive_mask is None:
            return None
        rate = float(positive_mask.astype(int).mean()) * 100
        return f"Clinical Impact Insight: estimated {target_variable} event rate is {rate:.1f}% in current data."

    @staticmethod
    def _business_impact_line(task_type: str, metrics: dict[str, Any], target_variable: str) -> str | None:
        if task_type == "classification":
            acc = metrics.get("accuracy")
            if isinstance(acc, (int, float)):
                if acc >= 0.8:
                    msg = "can support proactive intervention and reduce avoidable loss."
                elif acc >= 0.7:
                    msg = "is useful for assisted decision-making but needs monitoring."
                else:
                    msg = "is currently low for high-stakes automation."
                return f"Business Impact: model for {target_variable} {msg}"
        if task_type in {"regression", "forecasting"}:
            r2 = metrics.get("r2")
            if isinstance(r2, (int, float)):
                return f"Business Impact: r2={r2:.4f} suggests forecast/value planning impact."
        return None

    @staticmethod
    def _deployment_readiness_line(state: dict[str, Any]) -> str | None:
        deployment = state.get("deployment_suggestion", {}) or {}
        final_decision = state.get("final_decision", {}) or {}
        recommendation = deployment.get("recommendation") or final_decision.get("deployment_recommendation")
        quality = final_decision.get("quality")
        if recommendation or quality:
            return f"Deployment Readiness Insight: recommendation={recommendation or 'N/A'}, quality={quality or 'N/A'}."
        return None

    @staticmethod
    def _recommendation_line(state: dict[str, Any]) -> str:
        evaluation = state.get("evaluation", {}) or {}
        optimization = state.get("optimization", {}) or {}
        metrics = evaluation.get("metrics", {}) or {}
        acc = metrics.get("accuracy")
        f1 = metrics.get("f1_weighted")

        if isinstance(acc, (int, float)) and isinstance(f1, (int, float)):
            if acc < 0.7 or f1 < 0.7:
                rec = "improve class balance, engineer stronger features, and retune thresholds."
            elif acc < 0.8:
                rec = "deploy to staging, monitor segment errors, and iterate one more tuning round."
            else:
                rec = "proceed to controlled production with drift and performance monitoring."
        else:
            rec = "collect stronger signal-rich features and revalidate model robustness."

        if optimization.get("status") == "success":
            improved = optimization.get("improved", False)
            baseline = optimization.get("baseline_metric", "N/A")
            optimized = optimization.get("optimized_metric", "N/A")
            rec += f" Optimization improved={improved} ({baseline} -> {optimized})."
        return "Recommendation: " + rec

    @staticmethod
    def _strip_prefix(line: str) -> str:
        text = str(line or "").strip()
        if ":" in text:
            return text.split(":", 1)[1].strip()
        return text

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        evaluation = state.get("evaluation", {})
        if evaluation.get("status") != "success":
            return {"insight_generation": {"status": "failed", "error": "Evaluation results unavailable"}}

        dataset_path = (
            state.get("engineered_dataset_path")
            or state.get("cleaned_dataset_path")
            or state.get("dataset_path")
        )
        if not dataset_path:
            return {"insight_generation": {"status": "failed", "error": "No dataset path available"}}

        path = Path(dataset_path)
        if not path.exists():
            return {"insight_generation": {"status": "failed", "error": f"Dataset not found: {dataset_path}"}}

        df = self._read_csv(path)
        target_col = self._resolve_target_column(state, df)
        if target_col not in df.columns:
            target_col = str(df.columns[-1])

        problem_understanding = state.get("problem_understanding", {}) or {}
        problem_statement = str(state.get("problem_statement", ""))
        target_variable = str(problem_understanding.get("target_variable", target_col))
        task_type = str(evaluation.get("task_type", problem_understanding.get("task_type", "unknown")))
        metrics = evaluation.get("metrics", {}) or {}
        health_context = self._is_health_context(problem_statement, target_variable)
        positive_mask = self._positive_mask(df[target_col])

        most_important_line = self._most_important_features(state, df, target_col)
        metric_value_line = self._metric_value_line(metrics)
        performance_line = self._performance_line(task_type, metrics)
        risk_line = self._risk_line(state, positive_mask)
        data_quality_line = self._data_quality_line(state)
        business_line = self._business_impact_line(task_type, metrics, target_variable)
        if not business_line:
            clinical_impact_line = self._clinical_impact_line(health_context, positive_mask, target_variable)
            deployment_readiness_line = self._deployment_readiness_line(state)
            fallback = clinical_impact_line or deployment_readiness_line
            if fallback:
                business_line = f"Business Impact: {self._strip_prefix(fallback)}"
            else:
                business_line = "Business Impact: N/A"
        recommendation_line = self._recommendation_line(state)

        ordered_sections: list[tuple[str, str | None]] = [
            ("Most Important Features", most_important_line),
            ("Metric Value", metric_value_line),
            ("Performance Insight", performance_line),
            ("Risk Insight", risk_line),
            ("Data Quality Insight", data_quality_line),
            ("Business Impact", business_line),
            ("Recommendation", recommendation_line),
        ]

        insights: list[str] = []
        for label, line in ordered_sections:
            if line:
                if line.startswith(f"{label}:"):
                    insights.append(line)
                else:
                    insights.append(f"{label}: {self._strip_prefix(line)}")
            else:
                insights.append(f"{label}: N/A")

        return {
            "insight_generation": {
                "status": "success",
                "insights": insights,
                "insights_json_path": "",
            }
        }
