from __future__ import annotations

from pathlib import Path
from typing import Any
import importlib.util


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROJECT_ROOT = Path(__file__).resolve().parents[1]
config_module = _load_module("project_config", PROJECT_ROOT / "app" / "config.py")
file_handler_module = _load_module("project_file_handler", PROJECT_ROOT / "utils" / "file_handler.py")
problem_agent_module = _load_module(
    "project_problem_understanding", PROJECT_ROOT / "agents" / "problem_understanding_agent.py"
)
cleaning_agent_module = _load_module(
    "project_data_cleaning", PROJECT_ROOT / "agents" / "data_cleaning_agent.py"
)
eda_agent_module = _load_module("project_eda", PROJECT_ROOT / "agents" / "eda_agent.py")
feature_engineering_agent_module = _load_module(
    "project_feature_engineering", PROJECT_ROOT / "agents" / "feature_engineering_agent.py"
)
model_selection_agent_module = _load_module(
    "project_model_selection", PROJECT_ROOT / "agents" / "model_selection_agent.py"
)
model_training_agent_module = _load_module(
    "project_model_training", PROJECT_ROOT / "agents" / "model_training_agent.py"
)
evaluation_agent_module = _load_module(
    "project_evaluation", PROJECT_ROOT / "agents" / "evaluation_agent.py"
)
final_decision_agent_module = _load_module(
    "project_final_decision", PROJECT_ROOT / "agents" / "final_decision_agent.py"
)
optimization_agent_module = _load_module(
    "project_optimization", PROJECT_ROOT / "agents" / "optimization_agent.py"
)
visualization_agent_module = _load_module(
    "project_visualization", PROJECT_ROOT / "agents" / "visualization_agent.py"
)
insight_generation_agent_module = _load_module(
    "project_insight_generation", PROJECT_ROOT / "agents" / "insight_generation_agent.py"
)
report_generator_agent_module = _load_module(
    "project_report_generator", PROJECT_ROOT / "agents" / "report_generator_agent.py"
)
code_generator_agent_module = _load_module(
    "project_code_generator", PROJECT_ROOT / "agents" / "code_generator_agent.py"
)
deployment_suggestion_agent_module = _load_module(
    "project_deployment_suggestion", PROJECT_ROOT / "agents" / "deployment_suggestion_agent.py"
)

settings = config_module.settings
read_json = file_handler_module.read_json
write_json = file_handler_module.write_json
ProblemUnderstandingAgent = problem_agent_module.ProblemUnderstandingAgent
DataCleaningAgent = cleaning_agent_module.DataCleaningAgent
EDAAgent = eda_agent_module.EDAAgent
FeatureEngineeringAgent = feature_engineering_agent_module.FeatureEngineeringAgent
ModelSelectionAgent = model_selection_agent_module.ModelSelectionAgent
ModelTrainingAgent = model_training_agent_module.ModelTrainingAgent
EvaluationAgent = evaluation_agent_module.EvaluationAgent
FinalDecisionAgent = final_decision_agent_module.FinalDecisionAgent
OptimizationAgent = optimization_agent_module.OptimizationAgent
VisualizationAgent = visualization_agent_module.VisualizationAgent
InsightGenerationAgent = insight_generation_agent_module.InsightGenerationAgent
ReportGeneratorAgent = report_generator_agent_module.ReportGeneratorAgent
CodeGeneratorAgent = code_generator_agent_module.CodeGeneratorAgent
DeploymentSuggestionAgent = deployment_suggestion_agent_module.DeploymentSuggestionAgent


class Controller:
    def __init__(self):
        self.problem_agent = ProblemUnderstandingAgent()
        self.cleaning_agent = DataCleaningAgent()
        self.eda_agent = EDAAgent()
        self.feature_engineering_agent = FeatureEngineeringAgent()
        self.model_selection_agent = ModelSelectionAgent()
        self.model_training_agent = ModelTrainingAgent()
        self.evaluation_agent = EvaluationAgent()
        self.optimization_agent = OptimizationAgent()
        self.visualization_agent = VisualizationAgent()
        self.insight_generation_agent = InsightGenerationAgent()
        self.report_generator_agent = ReportGeneratorAgent()
        self.code_generator_agent = CodeGeneratorAgent()
        self.deployment_suggestion_agent = DeploymentSuggestionAgent()
        self.final_decision_agent = FinalDecisionAgent()
        self.agent_order = [
            "problem_understanding_agent",
            "data_cleaning_agent",
            "eda_agent",
            "feature_engineering_agent",
            "model_selection_agent",
            "model_training_agent",
            "evaluation_agent",
            "optimization_agent",
            "visualization_agent",
            "insight_generation_agent",
            "report_generator_agent",
            "code_generator_agent",
            "deployment_suggestion_agent",
            "final_decision_agent",
        ]

    def _load_memory(self) -> dict[str, Any]:
        return read_json(settings.shared_memory_file, default={})

    def _save_memory(self, payload: dict[str, Any]) -> None:
        write_json(settings.shared_memory_file, payload)

    def run_pipeline(self, state: dict[str, Any]) -> dict[str, Any]:
        # Stage 1
        problem_out = self.problem_agent.run(state)
        pipeline_state = dict(state)
        pipeline_state["problem_understanding"] = problem_out.get("problem_understanding", {})

        # Stage 2
        cleaning_out = self.cleaning_agent.run(pipeline_state)
        pipeline_state["data_cleaning"] = cleaning_out.get("data_cleaning", {})
        pipeline_state["cleaned_dataset_path"] = pipeline_state["data_cleaning"].get("cleaned_dataset_path")

        # Stage 3
        eda_out = self.eda_agent.run(pipeline_state)
        pipeline_state["eda"] = eda_out.get("eda", {})

        # Stage 4
        feature_out = self.feature_engineering_agent.run(pipeline_state)
        pipeline_state["feature_engineering"] = feature_out.get("feature_engineering", {})
        pipeline_state["engineered_dataset_path"] = pipeline_state["feature_engineering"].get("engineered_dataset_path")

        # Stage 5
        model_selection_out = self.model_selection_agent.run(pipeline_state)
        pipeline_state["model_selection"] = model_selection_out.get("model_selection", {})

        # Stage 6
        model_training_out = self.model_training_agent.run(pipeline_state)
        pipeline_state["model_training"] = model_training_out.get("model_training", {})

        # Stage 7
        evaluation_out = self.evaluation_agent.run(pipeline_state)
        pipeline_state["evaluation"] = evaluation_out.get("evaluation", {})

        # Stage 8
        optimization_out = self.optimization_agent.run(pipeline_state)
        pipeline_state["optimization"] = optimization_out.get("optimization", {})

        # Stage 9
        visualization_out = self.visualization_agent.run(pipeline_state)
        pipeline_state["visualization"] = visualization_out.get("visualization", {})

        # Stage 10
        insight_generation_out = self.insight_generation_agent.run(pipeline_state)
        pipeline_state["insight_generation"] = insight_generation_out.get("insight_generation", {})

        # Stage 11
        report_generator_out = self.report_generator_agent.run(pipeline_state)
        pipeline_state["report_generator"] = report_generator_out.get("report_generator", {})

        # Stage 12
        code_generator_out = self.code_generator_agent.run(pipeline_state)
        pipeline_state["code_generator"] = code_generator_out.get("code_generator", {})

        # Stage 13
        deployment_suggestion_out = self.deployment_suggestion_agent.run(pipeline_state)
        pipeline_state["deployment_suggestion"] = deployment_suggestion_out.get("deployment_suggestion", {})

        # Stage 14
        final_decision_out = self.final_decision_agent.run(pipeline_state)
        pipeline_state["final_decision"] = final_decision_out.get("final_decision", {})

        memory = self._load_memory()
        memory["last_input"] = state
        memory["problem_understanding"] = pipeline_state.get("problem_understanding", {})
        memory["data_cleaning"] = pipeline_state.get("data_cleaning", {})
        memory["eda"] = pipeline_state.get("eda", {})
        memory["feature_engineering"] = pipeline_state.get("feature_engineering", {})
        memory["model_selection"] = pipeline_state.get("model_selection", {})
        memory["model_training"] = pipeline_state.get("model_training", {})
        memory["evaluation"] = pipeline_state.get("evaluation", {})
        memory["optimization"] = pipeline_state.get("optimization", {})
        memory["visualization"] = pipeline_state.get("visualization", {})
        memory["insight_generation"] = pipeline_state.get("insight_generation", {})
        memory["report_generator"] = pipeline_state.get("report_generator", {})
        memory["code_generator"] = pipeline_state.get("code_generator", {})
        memory["deployment_suggestion"] = pipeline_state.get("deployment_suggestion", {})
        memory["final_decision"] = pipeline_state.get("final_decision", {})
        memory["pipeline_status"] = "stage_14_completed"
        memory["agent_order"] = self.agent_order
        self._save_memory(memory)

        return {
            "status": "stage_14_completed",
            "message": "All 14 stages completed including report and deployment artifacts.",
            "problem_understanding": pipeline_state.get("problem_understanding", {}),
            "data_cleaning": pipeline_state.get("data_cleaning", {}),
            "eda": pipeline_state.get("eda", {}),
            "feature_engineering": pipeline_state.get("feature_engineering", {}),
            "model_selection": pipeline_state.get("model_selection", {}),
            "model_training": pipeline_state.get("model_training", {}),
            "evaluation": pipeline_state.get("evaluation", {}),
            "optimization": pipeline_state.get("optimization", {}),
            "visualization": pipeline_state.get("visualization", {}),
            "insight_generation": pipeline_state.get("insight_generation", {}),
            "report_generator": pipeline_state.get("report_generator", {}),
            "code_generator": pipeline_state.get("code_generator", {}),
            "deployment_suggestion": pipeline_state.get("deployment_suggestion", {}),
            "final_decision": pipeline_state.get("final_decision", {}),
            "agent_order": self.agent_order,
        }
