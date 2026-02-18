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


class CodeGeneratorAgent(BaseAgent):
    def __init__(self):
        super().__init__("CodeGeneratorAgent")

    @staticmethod
    def _slug(text: str) -> str:
        clean = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").strip().lower()).strip("-")
        return clean[:40] if clean else "analysis"

    @staticmethod
    def _build_predict_script(model_path: str, target_column: str) -> str:
        return f'''import pickle
import pandas as pd

MODEL_PATH = r"{model_path}"
TARGET_COLUMN = "{target_column}"


def load_model_payload():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_from_csv(csv_path: str):
    payload = load_model_payload()
    model = payload["model"]
    feature_columns = payload["feature_columns"]

    df = pd.read_csv(csv_path)
    X = df.select_dtypes(include=["number", "bool"]).copy()
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]

    preds = model.predict(X)
    out = df.copy()
    out["prediction"] = preds
    return out


if __name__ == "__main__":
    sample_csv = "sample_input.csv"
    result = predict_from_csv(sample_csv)
    result.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")
'''

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        model_training = state.get("model_training", {})
        if model_training.get("status") != "success":
            return {"code_generator": {"status": "failed", "error": "Model training output not available"}}

        model_path = model_training.get("model_path", "")
        target_column = model_training.get("target_column", "target")
        date_text = datetime.now().strftime("%Y_%m_%d")
        slug = self._slug(state.get("problem_statement", ""))

        out_dir = settings.outputs_dir / "code"
        out_dir.mkdir(parents=True, exist_ok=True)
        script_path = out_dir / f"predict_{slug}_{date_text}.py"
        manifest_path = out_dir / f"code_manifest_{slug}_{date_text}.json"

        script_content = self._build_predict_script(model_path, target_column)
        script_path.write_text(script_content, encoding="utf-8")

        manifest = {
            "generated_script": str(script_path),
            "model_path": model_path,
            "target_column": target_column,
            "created_on": date_text,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return {
            "code_generator": {
                "status": "success",
                "predict_script_path": str(script_path),
                "code_manifest_path": str(manifest_path),
            }
        }
