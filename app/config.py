import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[1]

    data_dir: Path = project_root / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"

    memory_dir: Path = project_root / "memory"
    shared_memory_file: Path = memory_dir / "shared_memory.json"
    vector_memory_dir: Path = memory_dir / "vector_memory"

    outputs_dir: Path = project_root / "outputs"
    models_dir: Path = project_root / "models"

    llm_provider: str = "ollama"
    llm_model: str = os.getenv("LLM_MODEL", "mistral")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


settings = Settings()
