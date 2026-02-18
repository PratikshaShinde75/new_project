from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json
import math
import re
import importlib.util

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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


class ReportGeneratorAgent(BaseAgent):
    def __init__(self):
        super().__init__("ReportGeneratorAgent")

    @staticmethod
    def _slug(text: str) -> str:
        clean = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").strip().lower()).strip("-")
        return clean[:40] if clean else "analysis"

    @staticmethod
    def _build_markdown(state: dict[str, Any]) -> str:
        pu = state.get("problem_understanding", {})
        dc = state.get("data_cleaning", {})
        ms = state.get("model_selection", {})
        mt = state.get("model_training", {})
        ev = state.get("evaluation", {})
        fd = state.get("final_decision", {})
        insights = state.get("insight_generation", {}).get("insights", [])

        lines = [
            "# Multi-Agent AI Analysis Report",
            "",
            f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Problem",
            state.get("problem_statement", "N/A"),
            "",
            "## Problem Understanding",
            f"- Task Type: {pu.get('task_type', 'N/A')}",
            f"- Target Variable: {pu.get('target_variable', 'N/A')}",
            f"- Confidence: {pu.get('confidence_level', 'N/A')}",
            "",
            "## Data Cleaning",
            f"- Rows: {dc.get('rows_before', 'N/A')} -> {dc.get('rows_after', 'N/A')}",
            f"- Missing Values: {dc.get('missing_before', 'N/A')} -> {dc.get('missing_after', 'N/A')}",
            f"- Cleaned Dataset: {dc.get('cleaned_dataset_path', 'N/A')}",
            "",
            "## Model",
            f"- Recommended Model: {ms.get('recommended_model', 'N/A')}",
            f"- Saved Model File: {mt.get('model_file_name', 'N/A')}",
            f"- Model Path: {mt.get('model_path', 'N/A')}",
            "",
            "## Evaluation",
        ]

        for metric_name, metric_value in (ev.get("metrics", {}) or {}).items():
            lines.append(f"- {metric_name}: {metric_value}")

        lines += [
            "",
            "## Final Decision",
            f"- Quality: {fd.get('quality', 'N/A')}",
            f"- Deployment Recommendation: {fd.get('deployment_recommendation', 'N/A')}",
            f"- Confidence: {fd.get('confidence_level', 'N/A')}",
            f"- Summary: {fd.get('summary', 'N/A')}",
            "",
            "## Insights",
        ]
        if insights:
            lines.extend([f"- {item}" for item in insights])
        else:
            lines.append("- No insights available.")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _write_pdf(pdf_path: Path, report_text: str) -> None:
        lines = report_text.splitlines()
        lines_per_page = 48
        page_count = max(1, math.ceil(len(lines) / lines_per_page))
        with PdfPages(pdf_path) as pdf:
            for page in range(page_count):
                chunk = lines[page * lines_per_page : (page + 1) * lines_per_page]
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.patch.set_facecolor("white")
                y = 0.97
                for line in chunk:
                    fig.text(0.05, y, line[:140], fontsize=8.8, family="monospace")
                    y -= 0.019
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        date_text = datetime.now().strftime("%Y_%m_%d")
        slug = self._slug(state.get("problem_statement", ""))
        out_dir = settings.outputs_dir / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)

        md_path = out_dir / f"report_{slug}_{date_text}.md"
        pdf_path = out_dir / f"report_{slug}_{date_text}.pdf"
        json_path = out_dir / f"report_{slug}_{date_text}.json"

        report_md = self._build_markdown(state)
        md_path.write_text(report_md, encoding="utf-8")

        summary_json = {
            "problem_statement": state.get("problem_statement", ""),
            "task_type": state.get("problem_understanding", {}).get("task_type", "N/A"),
            "model_file": state.get("model_training", {}).get("model_file_name", "N/A"),
            "evaluation_metrics": state.get("evaluation", {}).get("metrics", {}),
            "final_decision": state.get("final_decision", {}),
        }
        json_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

        self._write_pdf(pdf_path, report_md)

        return {
            "report_generator": {
                "status": "success",
                "report_md_path": str(md_path),
                "report_pdf_path": str(pdf_path),
                "report_json_path": str(json_path),
            }
        }
