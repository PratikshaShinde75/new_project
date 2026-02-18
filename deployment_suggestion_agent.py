from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json
import re
import importlib.util

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


class DeploymentSuggestionAgent(BaseAgent):
    def __init__(self):
        super().__init__("DeploymentSuggestionAgent")

    @staticmethod
    def _slug(text: str) -> str:
        clean = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").strip().lower()).strip("-")
        return clean[:40] if clean else "analysis"

    @staticmethod
    def _data_quality_level(state: dict[str, Any]) -> str:
        dc = state.get("data_cleaning", {}) or {}
        rows_before = dc.get("rows_before")
        rows_after = dc.get("rows_after")
        missing_before = dc.get("missing_before")
        missing_after = dc.get("missing_after")
        if not all(v is not None for v in [rows_before, rows_after, missing_before, missing_after]):
            return "unknown"

        rb = max(int(rows_before), 1)
        ra = int(rows_after)
        ma = max(int(missing_after), 0)
        removed_pct = max(0.0, ((rb - ra) / rb) * 100.0)
        if ma == 0 and removed_pct <= 5:
            return "strong"
        if ma <= 5 and removed_pct <= 20:
            return "moderate"
        return "low"

    @staticmethod
    def _pick_variant(options: list[str], seed_text: str) -> str:
        if not options:
            return ""
        seed = sum(ord(ch) for ch in (seed_text or ""))
        return options[seed % len(options)]

    @staticmethod
    def _build_sections(
        score: float, task_type: str, data_quality: str, target_variable: str, seed_text: str
    ) -> tuple[str, str, str, str, list[str]]:
        score_pct = round(score * 100, 1)
        metric_name = "accuracy" if task_type == "classification" else "r2"
        target_text = target_variable or "target outcome"

        if score >= 0.8:
            recommendation = "production_api"
            priority = "high"
            deployment_recommendation = DeploymentSuggestionAgent._pick_variant(
                [
                    f"Strong {metric_name} ({score_pct}%) for {target_text} with {data_quality} data quality; suitable for production deployment.",
                    f"{target_text} model is production-ready with {score_pct}% {metric_name} and {data_quality} data quality.",
                    f"High-confidence {target_text} prediction ({metric_name} {score_pct}%) supports direct production deployment.",
                ],
                seed_text,
            )
            risk_assessment = DeploymentSuggestionAgent._pick_variant(
                [
                    "Low prediction uncertainty; automation risk is manageable with monitoring.",
                    "Risk level is controlled; maintain monitoring and rollback coverage.",
                    "Operational risk is low, but continuous drift tracking is still required.",
                ],
                seed_text,
            )
            suggested_actions = [
                DeploymentSuggestionAgent._pick_variant(
                    [
                        "Deploy production API with drift/performance alerts and rollback safeguards.",
                        "Go live with monitored API deployment and automated health checks.",
                        "Release in production with SLA monitoring and rollback policy.",
                    ],
                    seed_text,
                )
            ]
        elif score >= 0.65:
            recommendation = "staging_api_then_monitor"
            priority = "medium"
            deployment_recommendation = DeploymentSuggestionAgent._pick_variant(
                [
                    f"Moderate {metric_name} ({score_pct}%) for {target_text} with {data_quality} data quality; suitable for decision-support under supervision.",
                    f"{target_text} model is usable for assisted decisions ({metric_name} {score_pct}%), but not full automation.",
                    f"Model performance for {target_text} is moderate; deploy as support workflow with human review.",
                ],
                seed_text,
            )
            risk_assessment = DeploymentSuggestionAgent._pick_variant(
                [
                    "Prediction uncertainty is moderate; full automation is not recommended yet.",
                    "Risk remains medium due to moderate score; keep human validation in the loop.",
                    "Error risk is non-trivial, so supervised deployment is safer than autonomous rollout.",
                ],
                seed_text,
            )
            suggested_actions = [
                DeploymentSuggestionAgent._pick_variant(
                    [
                        "Deploy in staging/support mode, monitor outcomes, and tune further before automation.",
                        "Use decision-support deployment and retrain after monitoring real-world errors.",
                        "Run supervised rollout first, then optimize and re-evaluate for automation readiness.",
                    ],
                    seed_text,
                )
            ]
        else:
            recommendation = "improve_model_before_deploy"
            priority = "low"
            deployment_recommendation = DeploymentSuggestionAgent._pick_variant(
                [
                    f"Low {metric_name} ({score_pct}%) for {target_text} with {data_quality} data quality; not ready for deployment beyond offline evaluation.",
                    f"{target_text} model is below deployment threshold ({metric_name} {score_pct}%), so keep it in improvement phase.",
                    f"Current {target_text} performance is weak; deployment should be postponed until model quality improves.",
                ],
                seed_text,
            )
            risk_assessment = DeploymentSuggestionAgent._pick_variant(
                [
                    "Prediction uncertainty is high and business error risk is significant; do not automate.",
                    "Risk is high due to low model confidence, so deployment is not recommended now.",
                    "Automation risk is unacceptable at current score; keep model in offline testing only.",
                ],
                seed_text,
            )
            suggested_actions = [
                DeploymentSuggestionAgent._pick_variant(
                    [
                        "Improve data/feature quality, retrain, and re-evaluate before any deployment decision.",
                        "Focus on feature engineering and imbalance handling, then retrain before rollout.",
                        "Run another training+tuning cycle and validate metrics before deployment planning.",
                    ],
                    seed_text,
                )
            ]

        return recommendation, priority, deployment_recommendation, risk_assessment, suggested_actions

    @staticmethod
    def _platform_recommendation(score: float) -> tuple[str, list[str], str]:
        if score >= 0.8:
            tier = "advanced_enterprise"
            platforms = [
                "Docker",
                "Kubernetes",
                "Amazon Web Services (EKS/SageMaker)",
                "Google Cloud (GKE/Vertex AI)",
                "Azure Machine Learning",
            ]
            note = "Recommended for large-scale production workloads."
        elif score >= 0.65:
            tier = "intermediate_business"
            platforms = [
                "Amazon Web Services (EC2/SageMaker)",
                "Microsoft Azure",
                "Google Cloud",
                "DigitalOcean",
            ]
            note = "Recommended for API deployment with moderate production traffic."
        else:
            tier = "beginner_demo"
            platforms = [
                "Streamlit Cloud",
                "Render",
                "Hugging Face Spaces",
                "Railway",
                "Heroku",
            ]
            note = "Recommended for student/demo deployment and rapid prototyping."
        return tier, platforms, note

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        evaluation = state.get("evaluation", {})
        optimization = state.get("optimization", {})
        model_training = state.get("model_training", {})

        if evaluation.get("status") != "success":
            return {"deployment_suggestion": {"status": "failed", "error": "Evaluation output not available"}}

        task_type = evaluation.get("task_type", "classification")
        metrics = evaluation.get("metrics", {})
        target_variable = str((state.get("problem_understanding", {}) or {}).get("target_variable", "")).strip()

        if task_type == "classification":
            score = float(metrics.get("accuracy", 0.0))
        else:
            score = float(metrics.get("r2", 0.0))

        data_quality = self._data_quality_level(state)
        seed_text = f"{state.get('problem_statement', '')}|{target_variable}|{task_type}|{round(score,4)}"
        recommendation, priority, deployment_recommendation, risk_assessment, suggested_actions = self._build_sections(
            score=score,
            task_type=task_type,
            data_quality=data_quality,
            target_variable=target_variable,
            seed_text=seed_text,
        )
        platform_tier, recommended_platforms, platform_note = self._platform_recommendation(score)

        checklist = [
            "Version and lock model artifact",
            "Validate schema for incoming payload",
            "Add request/response logging",
            "Add drift and performance monitoring",
            "Define rollback strategy",
        ]
        if optimization.get("status") == "success" and optimization.get("improved"):
            checklist.insert(0, "Use optimized model artifact for deployment")
        if not model_training.get("model_path"):
            checklist.insert(0, "Model artifact path missing - regenerate training output")

        date_text = datetime.now().strftime("%Y_%m_%d")
        slug = self._slug(state.get("problem_statement", ""))
        out_dir = settings.outputs_dir / "deployments"
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"deployment_plan_{slug}_{date_text}.json"
        md_path = out_dir / f"deployment_plan_{slug}_{date_text}.md"

        plan = {
            "recommendation": recommendation,
            "priority": priority,
            "task_type": task_type,
            "score": round(score, 4),
            "data_quality": data_quality,
            "deployment_recommendation": deployment_recommendation,
            "risk_assessment": risk_assessment,
            "suggested_actions": suggested_actions,
            "platform_tier": platform_tier,
            "recommended_platforms": recommended_platforms,
            "platform_note": platform_note,
            "checklist": checklist,
            "model_path": model_training.get("model_path", ""),
        }
        json_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

        md_lines = [
            "# Deployment Plan",
            "",
            "## Deployment Recommendation",
            deployment_recommendation,
            "",
            "## Risk Assessment",
            risk_assessment,
            "",
            "## Suggested Action",
        ] + [f"- {item}" for item in suggested_actions] + [
            "",
            "## Recommended Platform",
            f"- Platform Tier: {platform_tier}",
            f"- Note: {platform_note}",
        ] + [f"- {item}" for item in recommended_platforms] + [
            "",
            "## Metadata",
            f"- Recommendation Key: {recommendation}",
            f"- Priority: {priority}",
            f"- Task Type: {task_type}",
            f"- Score: {round(score, 4)}",
            f"- Data Quality: {data_quality}",
            f"- Model Path: {model_training.get('model_path', 'N/A')}",
            "",
            "## Checklist",
        ] + [f"- {item}" for item in checklist]
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

        return {
            "deployment_suggestion": {
                "status": "success",
                "recommendation": recommendation,
                "priority": priority,
                "deployment_recommendation": deployment_recommendation,
                "risk_assessment": risk_assessment,
                "suggested_action": "\n".join(suggested_actions),
                "data_quality": data_quality,
                "platform_tier": platform_tier,
                "recommended_platforms": recommended_platforms,
                "platform_note": platform_note,
                "deployment_json_path": str(json_path),
                "deployment_md_path": str(md_path),
            }
        }
