from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import pickle
import re
import importlib.util

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
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


class VisualizationAgent(BaseAgent):
    def __init__(self):
        super().__init__("VisualizationAgent")

    @staticmethod
    def _slug(problem_statement: str) -> str:
        clean = re.sub(r"[^a-zA-Z0-9]+", "-", (problem_statement or "").strip().lower()).strip("-")
        return clean[:40] if clean else "analysis"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        dataset_path = (
            state.get("engineered_dataset_path")
            or state.get("cleaned_dataset_path")
            or state.get("dataset_path")
        )
        if not dataset_path:
            return {"visualization": {"status": "failed", "error": "Dataset path not available"}}

        dataset_file = Path(dataset_path)
        if not dataset_file.exists():
            return {"visualization": {"status": "failed", "error": f"Dataset not found: {dataset_path}"}}

        try:
            df = pd.read_csv(dataset_file)
        except UnicodeDecodeError:
            df = pd.read_csv(dataset_file, encoding="latin-1")

        statement = state.get("problem_statement", "")
        date_text = datetime.now().strftime("%Y_%m_%d")
        slug = self._slug(statement)
        out_dir = settings.outputs_dir / "charts"
        out_dir.mkdir(parents=True, exist_ok=True)
        chart_paths: list[str] = []

        target_column = state.get("model_training", {}).get("target_column") or state.get("target_column")

        model_path = (
            state.get("optimization", {}).get("optimized_model_path")
            or state.get("model_training", {}).get("model_path")
        )
        model = None
        feature_cols: list[str] = []
        model_target = ""
        model_task_type = "unknown"
        xgb_label_map: list[str] | None = None
        if model_path:
            model_file = Path(model_path)
            if model_file.exists():
                try:
                    with model_file.open("rb") as f:
                        payload = pickle.load(f)
                    model = payload.get("model")
                    feature_cols = [str(c) for c in (payload.get("feature_columns", []) or [])]
                    model_target = str(payload.get("target_column") or "")
                    model_task_type = str(payload.get("task_type") or "unknown")
                    xgb_label_map = payload.get("xgb_label_map")
                except Exception:
                    model = None

        def _save_placeholder_chart(file_name: str, title: str, message: str) -> str:
            fig = plt.figure(figsize=(8, 4))
            plt.axis("off")
            plt.title(title)
            plt.text(0.5, 0.5, message, ha="center", va="center")
            out_path = out_dir / file_name
            fig.savefig(out_path)
            plt.close(fig)
            return str(out_path)

        # 1) Target Distribution
        if target_column and target_column in df.columns:
            value_counts = df[target_column].astype(str).value_counts().head(20)
            if not value_counts.empty:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.pie(
                    value_counts.values,
                    labels=[str(x) for x in value_counts.index],
                    autopct="%1.1f%%",
                    startangle=90,
                    wedgeprops={"width": 0.45, "edgecolor": "white"},
                    textprops={"fontsize": 8},
                )
                ax.set_title(f"Target Distribution: {target_column}")
                ax.axis("equal")
                plt.tight_layout()
                out_path = out_dir / f"target_distribution_{slug}_{date_text}.png"
                fig.savefig(out_path)
                plt.close(fig)
                chart_paths.append(str(out_path))
            else:
                chart_paths.append(
                    _save_placeholder_chart(
                        f"target_distribution_{slug}_{date_text}.png",
                        "Target Distribution",
                        "No target values to plot.",
                    )
                )
        else:
            chart_paths.append(
                _save_placeholder_chart(
                    f"target_distribution_{slug}_{date_text}.png",
                    "Target Distribution",
                    "Target column not found.",
                )
            )

        # 2) Correlation Heatmap (single)
        numeric_cols = [str(c) for c in df.select_dtypes(include=["number"]).columns]
        if len(numeric_cols) >= 2:
            corr_cols = numeric_cols[:12]
            corr_df = df[corr_cols].corr()
            fig = plt.figure(figsize=(8, 4))
            plt.imshow(corr_df, cmap="coolwarm", vmin=-1, vmax=1)
            plt.colorbar()
            plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right", fontsize=8)
            plt.yticks(range(len(corr_cols)), corr_cols, fontsize=8)
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            out_path = out_dir / f"correlation_heatmap_{slug}_{date_text}.png"
            fig.savefig(out_path)
            plt.close(fig)
            chart_paths.append(str(out_path))
        else:
            chart_paths.append(
                _save_placeholder_chart(
                    f"correlation_heatmap_{slug}_{date_text}.png",
                    "Correlation Heatmap",
                    "Need at least 2 numeric columns.",
                )
            )

        # 3) Confusion Matrix
        confusion_path: str | None = None
        cm_target = model_target or str(target_column or "")
        if (
            model is not None
            and model_task_type == "classification"
            and cm_target
            and cm_target in df.columns
            and feature_cols
        ):
            try:
                X_eval = df.select_dtypes(include=["number", "bool"]).copy()
                for col in feature_cols:
                    if col not in X_eval.columns:
                        X_eval[col] = 0
                X_eval = X_eval[feature_cols]
                y_eval = df[cm_target].astype(str)

                if len(X_eval) >= 10 and y_eval.nunique(dropna=True) > 1:
                    try:
                        _, X_test, _, y_test = train_test_split(
                            X_eval, y_eval, test_size=0.2, random_state=42, stratify=y_eval
                        )
                    except ValueError:
                        _, X_test, _, y_test = train_test_split(
                            X_eval, y_eval, test_size=0.2, random_state=42, stratify=None
                        )

                    raw_pred = model.predict(X_test)
                    if xgb_label_map and len(xgb_label_map) > 0:
                        try:
                            y_pred = [str(xgb_label_map[int(v)]) for v in raw_pred]
                        except Exception:
                            y_pred = [str(v) for v in raw_pred]
                    else:
                        y_pred = [str(v) for v in raw_pred]

                    y_true = y_test.astype(str).tolist()
                    labels = sorted(list(set(y_true).union(set(y_pred))))
                    cm = confusion_matrix(y_true, y_pred, labels=labels)

                    fig = plt.figure(figsize=(8, 4))
                    plt.imshow(cm, cmap="Blues")
                    plt.colorbar()
                    plt.title("Confusion Matrix")
                    plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
                    plt.yticks(range(len(labels)), labels, fontsize=8)
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    plt.tight_layout()
                    out_path = out_dir / f"confusion_matrix_{slug}_{date_text}.png"
                    fig.savefig(out_path)
                    plt.close(fig)
                    confusion_path = str(out_path)
            except Exception:
                confusion_path = None

        if not confusion_path:
            confusion_path = _save_placeholder_chart(
                f"confusion_matrix_{slug}_{date_text}.png",
                "Confusion Matrix",
                "Confusion matrix available for classification models.",
            )
        chart_paths.append(confusion_path)

        # 4) Feature Importance
        feature_path: str | None = None
        if model is not None and feature_cols:
            try:
                importances = getattr(model, "feature_importances_", None)
                if importances is not None and len(importances) == len(feature_cols):
                    top = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)[:15]
                    if top:
                        labels = [t[0] for t in top]
                        values = [t[1] for t in top]
                        fig = plt.figure(figsize=(8, 4))
                        plt.barh(labels[::-1], values[::-1])
                        plt.title("Top Feature Importances")
                        plt.tight_layout()
                        out_path = out_dir / f"feature_importance_{slug}_{date_text}.png"
                        fig.savefig(out_path)
                        plt.close(fig)
                        feature_path = str(out_path)
                elif hasattr(model, "coef_"):
                    coefs = getattr(model, "coef_", None)
                    if coefs is not None:
                        coef_vals = coefs[0] if hasattr(coefs, "ndim") and coefs.ndim > 1 else coefs
                        if len(coef_vals) == len(feature_cols):
                            top = sorted(
                                zip(feature_cols, [abs(float(v)) for v in coef_vals]),
                                key=lambda x: x[1],
                                reverse=True,
                            )[:15]
                            if top:
                                labels = [t[0] for t in top]
                                values = [t[1] for t in top]
                                fig = plt.figure(figsize=(8, 4))
                                plt.barh(labels[::-1], values[::-1])
                                plt.title("Top Feature Importances")
                                plt.tight_layout()
                                out_path = out_dir / f"feature_importance_{slug}_{date_text}.png"
                                fig.savefig(out_path)
                                plt.close(fig)
                                feature_path = str(out_path)
            except Exception:
                feature_path = None

        if not feature_path:
            feature_path = _save_placeholder_chart(
                f"feature_importance_{slug}_{date_text}.png",
                "Top Feature Importances",
                "Model does not provide feature importance.",
            )
        chart_paths.append(feature_path)

        return {
            "visualization": {
                "status": "success",
                "chart_paths": chart_paths,
                "chart_count": len(chart_paths),
                "charts_dir": str(out_dir),
            }
        }
