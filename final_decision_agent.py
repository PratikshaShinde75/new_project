from __future__ import annotations

from typing import Any
from pathlib import Path

from agents.base_agent import BaseAgent


class FinalDecisionAgent(BaseAgent):
    def __init__(self):
        super().__init__("FinalDecisionAgent")

    @staticmethod
    def _classification_quality(metrics: dict[str, Any]) -> str:
        accuracy = float(metrics.get("accuracy", 0.0))
        f1 = float(metrics.get("f1_weighted", 0.0))
        if accuracy >= 0.85 and f1 >= 0.8:
            return "excellent"
        if accuracy >= 0.75 and f1 >= 0.7:
            return "good"
        if accuracy >= 0.65 and f1 >= 0.6:
            return "moderate"
        return "low"

    @staticmethod
    def _regression_quality(metrics: dict[str, Any]) -> str:
        r2 = float(metrics.get("r2", 0.0))
        if r2 >= 0.85:
            return "excellent"
        if r2 >= 0.7:
            return "good"
        if r2 >= 0.5:
            return "moderate"
        return "low"

    @staticmethod
    def _metric_summary(task_type: str, metrics: dict[str, Any]) -> tuple[str, float]:
        if task_type == "classification":
            return "accuracy", float(metrics.get("accuracy", 0.0))
        return "r2", float(metrics.get("r2", 0.0))

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        data_cleaning = state.get("data_cleaning", {})
        evaluation = state.get("evaluation", {})
        optimization = state.get("optimization", {})
        report_generator = state.get("report_generator", {})
        code_generator = state.get("code_generator", {})
        model_training = state.get("model_training", {})
        model_selection = state.get("model_selection", {})
        deployment = state.get("deployment_suggestion", {})
        problem = state.get("problem_understanding", {})

        if model_training.get("status") != "success":
            return {
                "final_decision": {
                    "status": "failed",
                    "error": "Model training stage is not successful",
                    "deployment_recommendation": "not_recommended",
                }
            }

        if evaluation.get("status") != "success":
            return {
                "final_decision": {
                    "status": "failed",
                    "error": "Evaluation stage is not successful",
                    "deployment_recommendation": "not_recommended",
                }
            }

        task_type = evaluation.get("task_type", "classification")
        metrics = evaluation.get("metrics", {})
        metric_name, metric_value = self._metric_summary(task_type, metrics)
        if task_type == "classification":
            quality = self._classification_quality(metrics)
        else:
            quality = self._regression_quality(metrics)

        deployment_recommendation = deployment.get("recommendation")
        if not deployment_recommendation:
            deployment_recommendation = "recommended" if quality in {"excellent", "good"} else "conditional"
        confidence_level = "high" if quality in {"excellent", "good"} else "medium" if quality == "moderate" else "low"

        best_model = model_selection.get("recommended_model", "unknown")
        rows_before = data_cleaning.get("rows_before")
        rows_after = data_cleaning.get("rows_after")
        missing_after = data_cleaning.get("missing_after")
        cleaned_impact = "N/A"
        if rows_before is not None and rows_after is not None:
            cleaned_impact = f"rows {rows_before}->{rows_after}"
        if missing_after is not None:
            cleaned_impact = f"{cleaned_impact}, missing_after={missing_after}"

        optimization_improved = bool(optimization.get("improved", False))
        optimization_text = "improved" if optimization_improved else "not improved"

        model_path = model_training.get("model_path", "")
        report_pdf_path = report_generator.get("report_pdf_path", "")
        code_path = code_generator.get("predict_script_path", "")
        deployment_md_path = deployment.get("deployment_md_path", "")
        artifact_health = {
            "model_file_ready": bool(model_path and Path(str(model_path)).exists()),
            "report_pdf_ready": bool(report_pdf_path and Path(str(report_pdf_path)).exists()),
            "code_script_ready": bool(code_path and Path(str(code_path)).exists()),
            "deployment_plan_ready": bool(deployment_md_path and Path(str(deployment_md_path)).exists()),
        }

        summary = (
            f"Model '{best_model}' for {problem.get('task_type', task_type)} is {quality} "
            f"({metric_name}: {round(metric_value, 4)}).\n"
            f"Data quality signal: {cleaned_impact}. "
            f"Optimization: {optimization_text}."
        )

        return {
            "final_decision": {
                "status": "success",
                "best_model": best_model,
                "model_path": model_training.get("model_path", ""),
                "model_file_name": model_training.get("model_file_name", ""),
                "deployment_recommendation": deployment_recommendation,
                "deployment_md_path": deployment.get("deployment_md_path", ""),
                "deployment_json_path": deployment.get("deployment_json_path", ""),
                "quality": quality,
                "confidence_level": confidence_level,
                "primary_metric_name": metric_name,
                "primary_metric_value": round(metric_value, 4),
                "data_cleaning_impact": cleaned_impact,
                "optimization_improved": optimization_improved,
                "artifact_health": artifact_health,
                "summary": summary,
            }
        }
