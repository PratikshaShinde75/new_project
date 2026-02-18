from __future__ import annotations

import re
from typing import Any

from agents.base_agent import BaseAgent


class ProblemUnderstandingAgent(BaseAgent):
    def __init__(self):
        super().__init__("ProblemUnderstandingAgent")

    @staticmethod
    def _detect_task_type(text: str) -> str:
        t = text.lower()
        if any(k in t for k in ["forecast", "time series", "future", "next month", "next quarter"]):
            return "forecasting"
        if any(k in t for k in ["classify", "churn", "fraud", "default", "risk", "mortality", "yes/no"]):
            return "classification"
        if any(k in t for k in ["predict", "estimate", "price", "sales", "cost", "revenue", "amount"]):
            return "regression"
        if any(k in t for k in ["describe", "analyze", "insight", "summary", "pattern"]):
            return "descriptive"
        return "unknown"

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        stop_words = {
            "the", "and", "for", "with", "from", "using", "into", "that", "this", "want", "need",
            "build", "model", "data", "dataset", "analysis", "task", "target", "business",
        }
        keywords = [t for t in tokens if len(t) > 2 and t not in stop_words]
        # preserve order + unique
        seen = set()
        ordered = []
        for k in keywords:
            if k not in seen:
                ordered.append(k)
                seen.add(k)
        return ordered[:12]

    @staticmethod
    def _infer_target_variable(text: str, keywords: list[str]) -> str:
        raw = text.strip().lower()
        patterns = [
            r"(?:predict|forecast|estimate|classify|detect|identify)\s+(?:the\s+)?(.+?)(?:\s+from|\s+using|\s+based|\s+with|$)",
            r"(?:analysis|analyze)\s+(?:of\s+)?(.+?)(?:\s+from|\s+using|\s+based|\s+with|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, raw)
            if match:
                candidate = re.sub(r"\s+", " ", match.group(1)).strip(" .,-_")
                if candidate and len(candidate) >= 3:
                    return candidate

        if keywords:
            preferred = [
                "churn", "mortality", "default", "fraud", "sales", "price", "revenue",
                "maintenance", "risk", "demand", "outcome",
            ]
            for item in preferred:
                if item in keywords:
                    return item
            return keywords[0]
        return ""

    @staticmethod
    def _infer_goal(task_type: str, target_variable: str) -> str:
        target_text = target_variable if target_variable else "target outcome"
        if task_type == "classification":
            return f"Classify records to predict {target_text}."
        if task_type == "regression":
            return f"Predict the numeric value of {target_text}."
        if task_type == "forecasting":
            return f"Forecast future trend/value of {target_text}."
        if task_type == "descriptive":
            return f"Analyze and describe patterns related to {target_text}."
        return f"Understand and model {target_text}."

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        statement = (state.get("problem_statement") or "").strip()
        if not statement:
            return {
                "problem_understanding": {
                    "task_type": "unknown",
                    "keywords": [],
                    "target_variable": "",
                    "goal": "",
                    "confidence_level": "low",
                }
            }

        task_type = self._detect_task_type(statement)
        keywords = self._extract_keywords(statement)
        target_variable = self._infer_target_variable(statement, keywords)
        goal = self._infer_goal(task_type, target_variable)

        confidence_level = "high" if task_type != "unknown" and keywords else "medium" if keywords else "low"

        return {
            "problem_understanding": {
                "task_type": task_type,
                "keywords": keywords,
                "target_variable": target_variable,
                "confidence_level": confidence_level,
                "goal": goal,
            }
        }
