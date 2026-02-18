from pathlib import Path
import importlib.util
import base64

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config_module = _load_module("project_config", PROJECT_ROOT / "app" / "config.py")
controller_module = _load_module("project_controller", PROJECT_ROOT / "app" / "controller.py")

settings = config_module.settings
Controller = controller_module.Controller

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Collaborative Multi-Agent AI Analyst",
    page_icon="🧠",
    layout="wide",
)

# -------------------- CUSTOM CSS --------------------
st.markdown(
    """
    <style>
        .main {
            background-color: #0e1117;
        }
        .stMetric {
            background-color: #1c1f26;
            padding: 15px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- SIDEBAR --------------------
st.sidebar.title("⚙️ System Configuration")
model_choice = st.sidebar.selectbox(
    "Select Model Backend",
    ["Mistral (Ollama)", "OpenAI GPT", "Hybrid Mode"],
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3)
developer_mode = st.sidebar.checkbox("Developer Mode (Show Agent Logs)")

st.sidebar.markdown("---")
st.sidebar.info("Total Agents: 14")
st.sidebar.caption(f"Current model from config: {settings.llm_model}")

# -------------------- MAIN TITLE --------------------
st.title("🧠 Collaborative Multi-Agent AI Data Analyst")
st.markdown("An intelligent AI system powered by 14 specialized agents.")

# -------------------- INPUTS --------------------
st.markdown("### 📂  Upload Dataset")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

problem_statement = ""
target_column = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully.")

    st.markdown("### Dataset Preview")
    st.dataframe(df.head())

    st.markdown("### Dataset Info")
    st.write(f"Total Rows: {df.shape[0]}")
    st.write(f"Total Columns: {df.shape[1]}")

    st.markdown("### 📝 Business Problem")
    problem_statement = st.text_area(
        "Define the analysis goal and expected outcome",
        placeholder="Example: Predict house prices",
        height=140,
    )
else:
    st.info("Please upload a dataset to continue.")

run_clicked = st.button("🚀 Run Multi-Agent Analysis", use_container_width=True)


def _section_header(title: str) -> None:
    st.markdown("---")
    st.markdown(f"### {title}")


def _show_blue_path(label: str, value: str) -> None:
    text = str(value or "N/A")
    st.markdown(
        f"{label}: <span style='color:#4DA3FF'>{text}</span>",
        unsafe_allow_html=True,
    )


def _show_agent_completed(message: str) -> None:
    final_text = str(message or "").strip()
    st.markdown(
        f"<span style='color:#22c55e;'>{final_text}</span>",
        unsafe_allow_html=True,
    )


def _show_download_link(label: str, file_path: str) -> None:
    text = str(file_path or "").strip()
    if not text:
        return
    path = Path(text)
    if not path.exists() or not path.is_file():
        return

    suffix = path.suffix.lower()
    mime = "application/octet-stream"
    if suffix == ".py":
        mime = "text/x-python"
    elif suffix == ".json":
        mime = "application/json"
    elif suffix == ".md":
        mime = "text/markdown"

    file_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    st.markdown(
        f'<a href="data:{mime};base64,{file_b64}" download="{path.name}">{label}</a>',
        unsafe_allow_html=True,
    )


def _flip_detail_state(state_key: str) -> None:
    st.session_state[state_key] = not st.session_state.get(state_key, False)

def _toggle_details(label: str, key: str) -> bool:
    # Temporary mode: hide toggle controls and keep all details visible.
    # Keep this function so we can restore ▶ / ▼ behavior quickly.
    _ = (label, key)
    return True


if run_clicked:
    if uploaded_file is None:
        st.warning("Please upload a CSV dataset first.")
    elif not problem_statement.strip():
        st.warning("Please enter a problem statement.")
    else:
        settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = settings.raw_data_dir / uploaded_file.name
        dataset_path.write_bytes(uploaded_file.getbuffer())

        with st.spinner("🤖 AI Agents are collaborating..."):
            controller = Controller()
            st.session_state["pipeline_result"] = controller.run_pipeline(
                {
                    "dataset_name": dataset_path.name,
                    "dataset_path": str(dataset_path),
                    "target_column": target_column,
                    "problem_statement": problem_statement,
                    "ui_model_choice": model_choice,
                    "temperature": temperature,
                }
            )
            st.session_state["saved_dataset_path"] = str(dataset_path)

if "pipeline_result" in st.session_state:
    pipeline_result = st.session_state["pipeline_result"]

    pu = pipeline_result.get("problem_understanding", {})
    dc = pipeline_result.get("data_cleaning", {})
    eda = pipeline_result.get("eda", {})
    fe = pipeline_result.get("feature_engineering", {})
    ms = pipeline_result.get("model_selection", {})
    mt = pipeline_result.get("model_training", {})
    ev = pipeline_result.get("evaluation", {})
    opt = pipeline_result.get("optimization", {})
    viz = pipeline_result.get("visualization", {})
    ins = pipeline_result.get("insight_generation", {})
    rep = pipeline_result.get("report_generator", {})
    codeg = pipeline_result.get("code_generator", {})
    dep = pipeline_result.get("deployment_suggestion", {})
    fd = pipeline_result.get("final_decision", {})

    _section_header("🔍 Problem Understanding Agent")
    if _toggle_details("Problem Understanding Agent Details", "pu"):
        st.write(f"Task Type: {pu.get('task_type', 'unknown') if pu else 'N/A'}")
        st.write(f"Target Variable: {pu.get('target_variable', '') or 'N/A' if pu else 'N/A'}")
        st.write(f"Confidence: {pu.get('confidence_level', pu.get('confidence', 'low')) if pu else 'N/A'}")
        keywords = pu.get("keywords", []) if pu else []
        st.write("Keywords:", ", ".join(keywords) if keywords else "N/A")
        st.write(f"Goal: {pu.get('goal', '') or 'N/A' if pu else 'N/A'}")
        if pu and pu.get("task_type", "unknown") != "unknown":
            _show_agent_completed("✅ Got it! Your problem is officially decoded.")

    _section_header("🧹 Data Cleaning Agent")
    if dc and dc.get("status") == "success":
        if _toggle_details("Data Cleaning Agent Details", "dc"):
            st.write(f"Rows: {dc.get('rows_before', 0)} -> {dc.get('rows_after', 0)}")
            st.write(f"Columns: {dc.get('columns_before', 0)} -> {dc.get('columns_after', 0)}")
            st.write(f"Missing Values: {dc.get('missing_before', 0)} -> {dc.get('missing_after', 0)}")
            st.write(f"Duplicates Removed: {dc.get('duplicates_removed', 0)}")
            _show_blue_path("Cleaned Dataset", dc.get("cleaned_dataset_path", "N/A"))
        _show_agent_completed("✅ Dataset Cleaned and Validated Successfully! ")
    else:
        st.warning("Data Cleaning Agent is not completed.")
        if dc and dc.get("error"):
            st.error(dc.get("error"))

    _section_header("📊 EDA Agent")
    if eda and eda.get("status") == "success":
        if _toggle_details("EDA Agent Details", "eda"):
            st.write(f"Numeric Columns: {eda.get('numeric_columns', 'N/A')}")
            st.write(f"Categorical Columns: {eda.get('categorical_columns', 'N/A')}")
            st.write(f"Class Distribution: {eda.get('class_distribution', 'N/A')}")
            for dist_line in (eda.get("class_distribution_details", []) or []):
                st.write(dist_line)
            st.write(
                "Top Correlated Features:",
                ", ".join(eda.get("top_correlated_features", [])) or "N/A",
            )
        _show_agent_completed("✅ Data Exploration Completed Successfully!")
    else:
        st.warning("EDA Agent is not completed.")
        if eda and eda.get("error"):
            st.error(eda.get("error"))

    _section_header("🛠️ Feature Engineering Agent")
    if fe and fe.get("status") == "success":
        if _toggle_details("Feature Engineering Agent Details", "fe"):
            encoded_columns = fe.get("encoded_columns", []) or []
            st.write(f"Features Before: {fe.get('features_before', 'N/A')}")
            st.write(f"Features After: {fe.get('features_after', 'N/A')}")
            st.write(
                f"Encoded Columns ({len(encoded_columns)}):",
                ", ".join(encoded_columns) or "N/A",
            )
            st.write(
                "Skipped (High Cardinality):",
                ", ".join(fe.get("skipped_high_cardinality_columns", [])) or "N/A",
            )
            _show_blue_path("Engineered Dataset", fe.get("engineered_dataset_path", "N/A"))
        _show_agent_completed("✅ Feature Engineering Completed!")
    else:
        st.warning("Feature Engineering Agent is not completed.")
        if fe and fe.get("error"):
            st.error(fe.get("error"))

    _section_header("🤖 Model Selection Agent")
    if ms and ms.get("status") == "success":
        if _toggle_details("Model Selection Agent Details", "ms"):
            st.write("Models Compared:")
            compared_models = ms.get("models_compared", []) or []
            if compared_models:
                for item in compared_models:
                    model_name = item.get("model", "N/A")
                    metric_name = item.get("metric", "Score")
                    score_value = item.get("score")
                    if score_value is None:
                        st.write(f"- {model_name}: {item.get('note', 'Not available')}")
                    else:
                        st.write(f"- {model_name}: {score_value}% {metric_name}")
            else:
                st.write("N/A")
            st.write(f"Best Model Selected: {ms.get('recommended_model', 'N/A')}")
        _show_agent_completed("✅ AI has shortlisted the best model candidates successfully.")
    else:
        st.warning("Model Selection Agent is not completed.")
        if ms and ms.get("error"):
            st.error(ms.get("error"))


    _section_header("🧪  Model Training Agent")
    if mt and mt.get("status") == "success":
        if _toggle_details("Model Training Agent Details", "mt"):
            _show_blue_path("Saved Model File", mt.get("model_file_name", "N/A"))
            _show_blue_path("Model Path", mt.get("model_path", "N/A"))
            st.write(
                f"Training Metric ({mt.get('metric_name', 'metric')}): {mt.get('metric_value', 'N/A')}"
            )
            st.write(f"Train Rows: {mt.get('train_rows', 'N/A')}")
            st.write(f"Test Rows: {mt.get('test_rows', 'N/A')}")
        _show_agent_completed("✅ Model Training Completed Successfully!")
    else:
        st.warning("Model Training Agent is not completed.")
        if mt and mt.get("error"):
            st.error(mt.get("error"))

    _section_header("📈 Evaluation Agent")
    if ev and ev.get("status") == "success":
        if _toggle_details("Evaluation Agent Details", "eval"):
            st.write(f"Target Column: {ev.get('target_column', 'N/A')}")
            for metric_name, metric_value in (ev.get("metrics", {}) or {}).items():
                st.write(f"{metric_name}: {metric_value}")
        _show_agent_completed("✅ Model Performance Evaluated Successfully!")
    else:
        st.warning("Evaluation Agent is not completed.")
        if ev and ev.get("error"):
            st.error(ev.get("error"))


    _section_header("🧩 Optimization Agent")
    if opt and opt.get("status") == "success":
        if _toggle_details("Optimization Agent Details", "opt"):
            st.write(f"Initial Accuracy: {opt.get('initial_accuracy', opt.get('baseline_metric', 'N/A'))}")
            st.write(f"After Optimization: {opt.get('after_optimization', opt.get('optimized_metric', 'N/A'))}")
            st.write(f"Improved: {opt.get('improved', 'N/A')}")
            st.write(f"Best Params: {opt.get('best_params', {})}")
            _show_blue_path("Optimized Model Path", opt.get("optimized_model_path", "N/A"))
        _show_agent_completed("✅ Model performance optimized successfully.")
    else:
        st.warning("Optimization Agent is not completed.")
        if opt and opt.get("error"):
            st.error(opt.get("error"))

    _section_header("⚡ Visualization Agent")
    if viz and viz.get("status") == "success":
        if _toggle_details("Visualization Agent Details", "viz"):
            st.write(f"Charts Generated: {viz.get('chart_count', 0)}")
            _show_blue_path("Charts Directory", viz.get("charts_dir", "N/A"))
            chart_files: list[Path] = []
            for chart_path in viz.get("chart_paths", [])[:4]:
                chart_file = Path(str(chart_path))
                if chart_file.exists() and chart_file.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    chart_files.append(chart_file)

            if chart_files:
                col1, col2 = st.columns(2)
                with col1:
                    if len(chart_files) > 0:
                        st.image(str(chart_files[0]), caption="Graph 1", width=420)
                with col2:
                    if len(chart_files) > 1:
                        st.image(str(chart_files[1]), caption="Graph 2", width=420)

                col3, col4 = st.columns(2)
                with col3:
                    if len(chart_files) > 2:
                        st.image(str(chart_files[2]), caption="Graph 3", width=420)
                with col4:
                    if len(chart_files) > 3:
                        st.image(str(chart_files[3]), caption="Graph 4", width=420)
        _show_agent_completed("✅ Visual Insights Generated Successfully!")
    else:
        st.warning("Visualization Agent is not completed.")
        if viz and viz.get("error"):
            st.error(viz.get("error"))

    _section_header("💡 Insight Generation Agent")
    if ins and ins.get("status") == "success":
        if _toggle_details("Insight Generation Agent Details", "ins"):
            for insight in ins.get("insights", []):
                st.write(f"- {insight}")
        _show_agent_completed("✅ Actionable Business Insights Generated!")
    else:
        st.warning("Insight Generation Agent is not completed.")
        if ins and ins.get("error"):
            st.error(ins.get("error"))

    _section_header("📄 Report Generator Agent")
    if rep and rep.get("status") == "success":
        if _toggle_details("Report Generator Agent Details", "rep"):
            _show_blue_path("Report Markdown", rep.get("report_md_path", "N/A"))
            _show_blue_path("Report PDF", rep.get("report_pdf_path", "N/A"))
            _show_blue_path("Report JSON", rep.get("report_json_path", "N/A"))
            report_pdf_path = rep.get("report_pdf_path", "")
            if report_pdf_path and Path(report_pdf_path).exists():
                pdf_bytes = Path(report_pdf_path).read_bytes()
                pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
                st.markdown(
                    f'<a href="data:application/pdf;base64,{pdf_b64}" download="{Path(report_pdf_path).name}">📥 Download PDF Report</a>',
                    unsafe_allow_html=True,
                )
        _show_agent_completed("✅ Report Generated successfully.")
    else:
        st.warning("Report Generator Agent is not completed.")
        if rep and rep.get("error"):
            st.error(rep.get("error"))

    _section_header("💻 Code Generator Agent")
    if codeg and codeg.get("status") == "success":
        if _toggle_details("Code Generator Agent Details", "codeg"):
            _show_blue_path("Predict Script", codeg.get("predict_script_path", "N/A"))
            _show_blue_path("Code Manifest", codeg.get("code_manifest_path", "N/A"))
            st.write("Download Files:")
            _show_download_link("📥 Download Predict Script", codeg.get("predict_script_path", ""))
            _show_download_link("📥 Download Code Manifest", codeg.get("code_manifest_path", ""))
        _show_agent_completed("✅ Deployment Code Generated Successfully!")
    else:
        st.warning("Code Generator Agent is not completed.")
        if codeg and codeg.get("error"):
            st.error(codeg.get("error"))

    _section_header("🌐 Deployment Suggestion Agent")
    if dep and dep.get("status") == "success":
        if _toggle_details("Deployment Suggestion Agent Details", "dep"):
            st.write(f"Recommendation Key: {dep.get('recommendation', 'N/A')}")
            st.write(f"Priority: {dep.get('priority', 'N/A')}")
            st.write(f"Deployment Recommendation: {dep.get('deployment_recommendation', 'N/A')}")
            st.write(f"Risk Assessment: {dep.get('risk_assessment', 'N/A')}")
            suggested_action = dep.get("suggested_action", "N/A")
            if isinstance(suggested_action, str) and "\n" in suggested_action:
                suggested_action = " ".join([x.strip() for x in suggested_action.splitlines() if x.strip()])
            st.write(f"Suggested Action: {suggested_action}")
            st.write("")
            st.write("Recommended Platform:")
            platform_note = dep.get("platform_note", "")
            if platform_note:
                st.write(platform_note)
            for platform in (dep.get("recommended_platforms", []) or []):
                st.write(f"- {platform}")
        _show_agent_completed("✅ Deployment Strategy Finalized!")
    else:
        st.warning("Deployment Suggestion Agent is not completed.")
        if dep and dep.get("error"):
            st.error(dep.get("error"))

    _section_header("🎯 Final Decision Agent")
    if fd and fd.get("status") == "success":
        if _toggle_details("Final Decision Agent Details", "fd"):
            st.write(f"Best Model: {fd.get('best_model', 'N/A')}")
            _show_blue_path("Model File", fd.get("model_file_name", "N/A"))
            _show_blue_path("Model Path", fd.get("model_path", "N/A"))
            st.write(f"Deployment: {fd.get('deployment_recommendation', 'N/A')}")
            st.write(f"Quality: {fd.get('quality', 'N/A')}")
            st.write(f"Confidence Level: {fd.get('confidence_level', 'N/A')}")
            summary_text = str(fd.get("summary", "N/A") or "N/A")
            summary_line = summary_text.splitlines()[0].strip() if summary_text else "N/A"
            st.write(f"Summary: {summary_line}")
            data_quality_signal = str(fd.get("data_cleaning_impact", "")).strip()
            if data_quality_signal:
                st.write(f"Data quality signal: {data_quality_signal}")
            opt_improved = fd.get("optimization_improved")
            if isinstance(opt_improved, bool):
                st.write(f"Optimization: {'improved' if opt_improved else 'not improved'}")
        _show_agent_completed("✅ AI Pipeline Completed Successfully!")
    else:
        st.warning("Final Decision Agent is not completed.")
        if fd and fd.get("error"):
            st.error(fd.get("error"))

    st.markdown("---")
    if pipeline_result.get("status") == "stage_14_completed":
        st.markdown("### ✅ All Agents Completed")
        st.markdown(
            "<span style='color:#22c55e;'>All 14 agents finished successfully. Final output is ready.</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("### ⚠️ Pipeline Update")
        st.write(f"Current status: {pipeline_result.get('status', 'unknown')}")

    if developer_mode:
        st.caption("Developer Mode")
        st.code(
            f"model_choice={model_choice}\ntemperature={temperature}\n"
            f"saved_dataset={st.session_state.get('saved_dataset_path', 'N/A')}",
            language="text",
        )
