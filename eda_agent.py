from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from agents.base_agent import BaseAgent


class EDAAgent(BaseAgent):
    def __init__(self):
        super().__init__("EDAAgent")

    @staticmethod
    def _normalize(text: str) -> str:
        return "".join(ch for ch in str(text).lower() if ch.isalnum())

    @classmethod
    def _resolve_target_column(cls, df: pd.DataFrame, state: dict[str, Any]) -> str | None:
        target_column = state.get("target_column")
        if target_column and target_column in df.columns:
            return str(target_column)

        problem_understanding = state.get("problem_understanding", {}) or {}
        target_variable = str(problem_understanding.get("target_variable", "")).strip()
        keywords = [str(k).strip() for k in (problem_understanding.get("keywords", []) or []) if str(k).strip()]
        task_type = str(problem_understanding.get("task_type", "")).strip().lower()

        normalized_columns = {str(col): cls._normalize(col) for col in df.columns}
        if target_variable:
            target_norm = cls._normalize(target_variable)
            for col, col_norm in normalized_columns.items():
                if target_norm and (target_norm == col_norm or target_norm in col_norm or col_norm in target_norm):
                    return col

        priority_terms = [
            "target",
            "label",
            "class",
            "outcome",
            "status",
            "result",
            "y",
            "churn",
            "mortality",
            "death",
            "default",
            "fraud",
            "risk",
            "event",
        ]
        for term in [*keywords, *priority_terms]:
            term_norm = cls._normalize(term)
            if not term_norm:
                continue
            for col, col_norm in normalized_columns.items():
                if term_norm in col_norm:
                    return col

        # If still unknown for classification, prefer a low-cardinality column.
        if task_type == "classification":
            candidate_cols: list[tuple[str, int]] = []
            for col in df.columns:
                nunique = int(df[col].nunique(dropna=True))
                if 1 < nunique <= 5:
                    candidate_cols.append((str(col), nunique))
            if candidate_cols:
                candidate_cols.sort(key=lambda item: item[1])
                return candidate_cols[0][0]

        # Final fallback: keep behavior deterministic.
        if len(df.columns) > 0:
            return str(df.columns[-1])
        return None

    @staticmethod
    def _class_balance_text(df: pd.DataFrame, target_column: str | None) -> str:
        if not target_column or target_column not in df.columns:
            return "Target column not provided"

        series = df[target_column]
        if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 12:
            return "Not classification target"

        counts = series.astype(str).value_counts(normalize=True, dropna=False)
        if counts.empty:
            return "Not available"

        max_ratio = float(counts.iloc[0])
        if max_ratio <= 0.65:
            return "Balanced"
        return "Imbalanced"

    @staticmethod
    def _class_distribution_details(
        df: pd.DataFrame, target_column: str | None, state: dict[str, Any]
    ) -> list[str]:
        if not target_column or target_column not in df.columns:
            return []

        series = df[target_column]
        if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 12:
            return []

        counts = series.astype(str).value_counts(dropna=False)
        if counts.empty:
            return []

        problem_understanding = state.get("problem_understanding", {}) or {}
        context_text = (
            f"{problem_understanding.get('target_variable', '')} "
            f"{state.get('problem_statement', '')}"
        ).lower()
        health_context = any(
            token in context_text
            for token in ["mortality", "death", "died", "survival", "survived", "heart failure"]
        )

        lines: list[str] = []
        for cls_value, count in counts.items():
            cls_str = str(cls_value)
            if health_context and cls_str in {"0", "1"}:
                label = "Survived" if cls_str == "0" else "Died"
                lines.append(f"Class {cls_str} ({label}): {int(count)} patients")
            else:
                lines.append(f"Class {cls_str}: {int(count)} records")
        return lines

    @staticmethod
    def _top_correlated_features(df: pd.DataFrame, target_column: str | None) -> list[str]:
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.shape[1] < 2:
            return []

        if target_column and target_column in numeric_df.columns:
            corr = numeric_df.corr(numeric_only=True)[target_column].drop(labels=[target_column], errors="ignore")
            corr = corr.abs().sort_values(ascending=False).head(3)
            return [str(col) for col in corr.index]

        corr_matrix = numeric_df.corr(numeric_only=True).abs()
        corr_matrix.values[[range(corr_matrix.shape[0])], [range(corr_matrix.shape[0])]] = 0
        top_pairs = (
            corr_matrix.unstack()
            .sort_values(ascending=False)
            .drop_duplicates()
            .head(3)
            .index
        )
        features: list[str] = []
        for left, right in top_pairs:
            if left not in features:
                features.append(str(left))
            if right not in features:
                features.append(str(right))
            if len(features) >= 3:
                break
        return features[:3]

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        dataset_path = state.get("cleaned_dataset_path") or state.get("dataset_path")
        if not dataset_path:
            return {"eda": {"status": "failed", "error": "No dataset path available"}}

        path = Path(dataset_path)
        if not path.exists():
            return {"eda": {"status": "failed", "error": f"Dataset not found: {dataset_path}"}}

        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")

        target_column = self._resolve_target_column(df, state)
        rows, cols = df.shape
        numeric_cols = int(df.select_dtypes(include=["number"]).shape[1])
        categorical_cols = int(df.select_dtypes(exclude=["number"]).shape[1])
        missing_values = int(df.isna().sum().sum())
        class_distribution = self._class_balance_text(df, target_column)
        class_distribution_details = self._class_distribution_details(df, target_column, state)
        top_features = self._top_correlated_features(df, target_column)

        return {
            "eda": {
                "status": "success",
                "dataset_path": str(path),
                "rows": int(rows),
                "columns": int(cols),
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "missing_values": missing_values,
                "target_column": target_column or "",
                "class_distribution": class_distribution,
                "class_distribution_details": class_distribution_details,
                "top_correlated_features": top_features,
            }
        }
