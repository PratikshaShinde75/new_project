from __future__ import annotations

from datetime import datetime
from typing import Any
from pathlib import Path
import importlib.util

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


def _load_settings():
    config_path = Path(__file__).resolve().parents[1] / "app" / "config.py"
    spec = importlib.util.spec_from_file_location("project_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load app/config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.settings


settings = _load_settings()


class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.llm = ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=0.1,
        )

    def log(self, message: str) -> None:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] [{self.name}] {message}")

    def ask(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = self.llm.invoke(messages)
        return str(response.content).strip()

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Each agent must implement run(state).")
