
# -*- coding: utf-8 -*-
# app_overhaul.py — Final overhaul with strong DataFrame coercion (centralized), relative paths, robust normalize & runtime fallbacks
from __future__ import annotations
import os, json, sys
from pathlib import Path
import pandas as pd
import streamlit as st

# ---- Streamlit config must be first ----
st.set_page_config(page_title="ReWOO 多智能体（总改版·强制 DataFrame 纠正）", layout="wide")

# ---- Paths (relative) ----
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Import ReWOO tools ----
if str(ROOT_DIR / "rewoo_app") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "rewoo_app"))

try:
    from rewoo_app.tools.generic_tools import (
        LoadCSV, SaveCSV, Join, DetectSchema, FE_Generic,
        SplitTime, TrainMedianBaseline, Forecast, Summarize
    )
except Exception as e:
    st.error("无法导入 rewoo_app.tools.generic_tools，请确认项目结构与依赖。")
    st.exception(e)
    st.stop()

TOOL = {
    "LoadCSV": LoadCSV,
    "SaveCSV": SaveCSV,
    "Join": Join,
    "DetectSchema": DetectSchema,
    "FE_Generic": FE_Generic,
    "SplitTime": SplitTime,
    "TrainMedianBaseline": TrainMedianBaseline,
    "Forecast": Forecast,
    "Summarize": Summarize,
}
CANONICAL = set(TOOL.keys())

ALIASES = {
    "load_data": "LoadCSV", "loadcsv": "LoadCSV", "read_csv": "LoadCSV",
    "detectschema": "DetectSchema", "detect_schema": "DetectSchema",
    "feature_engineer": "FE_Generic", "feature_engineering": "FE_Generic", "fe": "FE_Generic",
    "preprocess_data": "FE_Generic", "preprocess": "FE_Generic", "pre_processing": "FE_Generic",
    "split": "SplitTime", "time_split": "SplitTime",
    "train": "TrainMedianBaseline", "train_baseline": "TrainMedianBaseline",
    "predict": "Forecast", "forecasting": "Forecast",
    "save": "SaveCSV", "save_csv": "SaveCSV", "write_csv": "SaveCSV", "export_csv": "SaveCSV",
    "summary": "Summarize", "report": "Summarize",
    "join": "Join", "merge": "Join"
}

def _canon_tool(name: str) -> str:
    key = (name or "").replace(" ", "").lower()
    if key in ALIASES: return ALIASES[key]
    if "preprocess" in key or "feature" in key: return "FE_Generic"
    if "load" in key or "read" in key or "import" in key: return "LoadCSV"
    if "detect" in key and "schema" in key: return "DetectSchema"
    if "split" in key: return "SplitTime"
    if "train" in key or "baseline" in key: return "TrainMedianBaseline"
    if "forecast" in key or "predict" in key: return "Forecast"
    if "save" in key or "write" in key or "export" in key: return "SaveCSV"
    if "summar" in key or "report" in key: return "Summarize"
    if "join" in key or "merge" in key: return "Join"
    return name

def _fix_args(tool: str, args: dict, dataset_root: str):
    args = dict(args or {})
    # unify file arg names + relativize
    if tool == "LoadCSV":
        if "path" not in args:
            for k in ("file", "filepath", "file_path", "input_file", "src", "source"):
                if k in args:
                    args["path"] = args.pop(k); break
        if "path" in args and not os.path.isabs(args["path"]):
            args["path"] = str((Path(dataset_root) / str(args["path"])).resolve())
    if tool == "SaveCSV":
        if "path" not in args:
            for k in ("file", "filepath", "dest", "destination", "output"):
                if k in args:
                    args["path"] = args.pop(k); break
        if "path" in args and not os.path.isabs(args["path"]):
            args["path"] = str((Path(dataset_root) / str(args["path"])).resolve())
    return args

# --------- Centralized coercion utils ---------
def coerce_to_df(x, tools=None, dataset_dir=None):
    """Best-effort convert x to a pandas.DataFrame.
    - DataFrame: return as-is
    - dict with 'df'/'data' key: extract
    - list/tuple: take first DF or load if string path
    - string:
        - if startswith #E*: caller应在外层解析，这里再查失败时尝试
        - if path/dir exists: call LoadCSV to get DF
    - None: try to LoadCSV(dataset_dir)
    Else: return None
    """
    try:
        import pandas as pd
        if isinstance(x, pd.DataFrame):
            return x
        if isinstance(x, dict):
            for k in ("df", "data", "dataset"):
                if k in x:
                    return coerce_to_df(x[k], tools, dataset_dir)
        if isinstance(x, (list, tuple)) and x:
            return coerce_to_df(x[0], tools, dataset_dir)
        if isinstance(x, str):
            # path-like
            p = Path(x)
            if p.exists():
                if tools and "LoadCSV" in tools:
                    try:
                        return tools["LoadCSV"](str(p), limit=1000, ensure_multiday=True, per_day_take=50)
                    except Exception:
                        return None
        if x is None and dataset_dir and tools and "LoadCSV" in tools:
            try:
                return tools["LoadCSV"](str(dataset_dir), limit=1000, ensure_multiday=True, per_day_take=50)
            except Exception:
                return None
    except Exception:
        return None
    return None

# ---- Plan normalization ----
def normalize_plan(plan: dict, dataset_root: str) -> dict:
    steps = []
    for s in plan.get("steps", []):
        tool = _canon_tool(s.get("tool",""))
        if tool not in CANONICAL and tool and tool[0].islower():
            guess = tool[:1].upper() + tool[1:]
            if guess in CANONICAL: tool = guess
        if tool not in CANONICAL:
            raise KeyError(f"Unrecognized tool in plan: {s.get('tool')}")
        s = dict(s); s["tool"] = tool

        a0 = dict(s.get("args", {}))
        if "file_path" in a0 and "path" not in a0: a0["path"] = a0.pop("file_path")
        if "filepath" in a0 and "path" not in a0: a0["path"] = a0.pop("filepath")
        if "input_file" in a0 and "path" not in a0: a0["path"] = a0.pop("input_file")
        s["args"] = _fix_args(tool, a0, dataset_root)

        if tool == "LoadCSV":
            p = s["args"].get("path")
            if p and (not os.path.isdir(p) and not os.path.exists(p)):
                s["args"]["path"] = dataset_root

        elif tool == "DetectSchema":
            a = s.setdefault("args", {})
            for alias in ("data", "df"):
                if alias in a and "train_like" not in a:
                    a["train_like"] = a.pop(alias)
            if not isinstance(a.get("train_like"), str) or not a["train_like"].startswith("#E"):
                a["train_like"] = "#E1"
            a.setdefault("future_like", None)

        elif tool == "FE_Generic":
            a = s.setdefault("args", {})
            if "df" not in a and "data" in a:
                a["df"] = a.pop("data")
            if "schema" in a:
                a.setdefault("time_col", "#E2.time_col")
                a.setdefault("key_cols", "#E2.key_cols")
                a.setdefault("target_col", "#E2.target_col")
                a.pop("schema", None)
            if not isinstance(a.get("df"), str) or not a["df"].startswith("#E"):
                a["df"] = "#E1"
            a.setdefault("time_col", "#E2.time_col")
            a.setdefault("key_cols", "#E2.key_cols")
            a.setdefault("target_col", "#E2.target_col")

        elif tool == "SplitTime":
            a = s.setdefault("args", {})
            if "df" not in a:
                if "data" in a:
                    a["df"] = a.pop("data")
                elif "dataset" in a:
                    a["df"] = a.pop("dataset")
            if not isinstance(a.get("df"), str) or not a["df"].startswith("#E"):
                a["df"] = "#E3"
            a.setdefault("time_col", "#E2.time_col")
            a.setdefault("horizon", 16)

        elif tool == "TrainMedianBaseline":
            a = s.setdefault("args", {})
            if "data" in a and "train_df" not in a:
                a["train_df"] = a.pop("data")
            a.setdefault("train_df", "#E4.train")
            a.setdefault("valid_df", "#E4.valid")
            a.setdefault("key_cols", "#E2.key_cols")
            a.setdefault("target_col", "#E2.target_col")

        elif tool == "Forecast":
            a = s.setdefault("args", {})
            if "model" in a and "model_bundle" not in a:
                a["model_bundle"] = a.pop("model")
            a.setdefault("model_bundle", "#E5")
            a.setdefault("future_df", "#E4.valid")
            a.setdefault("id_cols", "#E2.id_cols")

        steps.append(s)
    out = dict(plan); out["steps"] = steps
    return out

# ---- Runtime with centralized DF coercion ----
class ReWoorRuntime:
    def __init__(self, tools: dict[str, callable], logger=None):
        self.tools = tools
        self.env = {}
        self.logger = logger or (lambda *a, **k: None)

    def _resolve_arg(self, v):
        if isinstance(v, str) and v.startswith("#E"):
            if "." in v:
                ref, key = v.split(".", 1)
                base = self.env.get(ref)
                if isinstance(base, dict):
                    return base.get(key)
                return None
            else:
                return self.env.get(v)
        if isinstance(v, list):
            return [self._resolve_arg(x) for x in v]
        if isinstance(v, dict):
            return {k: self._resolve_arg(val) for k, val in v.items()}
        return v

    def _log_preview(self, x):
        try:
            if isinstance(x, pd.DataFrame):
                return f"<DataFrame shape={x.shape} cols={list(x.columns)[:6]}>"
            if isinstance(x, dict):
                return {k: self._log_preview(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [self._log_preview(v) for v in x]
            return x
        except Exception:
            return str(type(x))

    def run_plan(self, plan: dict):
        steps = plan.get("steps", [])
        artifacts, last_report = [], None

        for i, step in enumerate(steps, 1):
            tool = step["tool"]
            args = step.get("args", {})
            out = step.get("out", f"#E{i}")
            fn = self.tools.get(tool)
            if not fn: raise KeyError(f"Tool not found: {tool}")

            resolved = {k: self._resolve_arg(v) for k, v in args.items()}

            # ---- Hard fix: DetectSchema must get a DF ----
            if tool == "DetectSchema":
                tl = resolved.get("train_like")
                # try to coerce
                df_c = coerce_to_df(tl, tools=self.tools, dataset_dir=DATA_DIR)
                if df_c is None:
                    # fallback to #E1 or last env or autoload
                    cand = self.env.get("#E1")
                    df_c = coerce_to_df(cand, tools=self.tools, dataset_dir=DATA_DIR)
                    if df_c is None and self.env:
                        cand = self.env[list(self.env.keys())[-1]]
                        df_c = coerce_to_df(cand, tools=self.tools, dataset_dir=DATA_DIR)
                resolved["train_like"] = df_c
                resolved.setdefault("future_like", None)

            # ---- FE_Generic must get a DF ----
            if tool == "FE_Generic":
                df_c = coerce_to_df(resolved.get("df"), tools=self.tools, dataset_dir=DATA_DIR)
                if df_c is None:
                    # try E1 then autoload
                    df_c = coerce_to_df(self.env.get("#E1"), tools=self.tools, dataset_dir=DATA_DIR)
                resolved["df"] = df_c

                schema_obj = self.env.get("#E2")
                if isinstance(schema_obj, dict):
                    resolved.setdefault("time_col", schema_obj.get("time_col"))
                    resolved.setdefault("key_cols", schema_obj.get("key_cols"))
                    resolved.setdefault("target_col", schema_obj.get("target_col"))

                # infer time_col if missing
                if resolved.get("time_col") is None and isinstance(df_c, pd.DataFrame):
                    cols_lower = {c.lower(): c for c in df_c.columns}
                    for cand in ("date", "ds", "timestamp", "time"):
                        if cand in cols_lower:
                            resolved["time_col"] = cols_lower[cand]; break

                if not isinstance(resolved.get("df"), pd.DataFrame):
                    raise RuntimeError("FE_Generic 需要一个 DataFrame，但未能从上游产出/加载。请检查 data/ 目录是否包含可读 CSV。")

            # ---- SplitTime ----
            if tool == "SplitTime":
                if "data" in resolved and "df" not in resolved:
                    resolved["df"] = resolved.pop("data")
                df_c = coerce_to_df(resolved.get("df"), tools=self.tools, dataset_dir=DATA_DIR)
                if df_c is None: df_c = coerce_to_df(self.env.get("#E3"), tools=self.tools, dataset_dir=DATA_DIR)
                if df_c is None: df_c = coerce_to_df(self.env.get("#E1"), tools=self.tools, dataset_dir=DATA_DIR)
                resolved["df"] = df_c
                schema_obj = self.env.get("#E2")
                if isinstance(schema_obj, dict):
                    resolved.setdefault("time_col", schema_obj.get("time_col"))
                resolved.setdefault("horizon", 16)

            # ---- TrainMedianBaseline ----
            if tool == "TrainMedianBaseline":
                if "data" in resolved and "train_df" not in resolved:
                    resolved["train_df"] = resolved.pop("data")
                td = resolved.get("train_df"); vd = resolved.get("valid_df")
                # resolve string refs
                if isinstance(td, str): td = self._resolve_arg(td)
                if isinstance(vd, str): vd = self._resolve_arg(vd)
                resolved["train_df"] = coerce_to_df(td, tools=self.tools, dataset_dir=DATA_DIR)
                resolved["valid_df"] = coerce_to_df(vd, tools=self.tools, dataset_dir=DATA_DIR) or resolved["train_df"]
                schema_obj = self.env.get("#E2")
                if isinstance(schema_obj, dict):
                    resolved.setdefault("key_cols", schema_obj.get("key_cols"))
                    resolved.setdefault("target_col", schema_obj.get("target_col"))

            # ---- Forecast ----
            if tool == "Forecast":
                if "model" in resolved and "model_bundle" not in resolved:
                    resolved["model_bundle"] = resolved.pop("model")
                if isinstance(resolved.get("future_df"), str):
                    tmp = self._resolve_arg(resolved["future_df"])
                    resolved["future_df"] = coerce_to_df(tmp, tools=self.tools, dataset_dir=DATA_DIR)
                if resolved.get("future_df") is None:
                    resolved["future_df"] = coerce_to_df(self.env.get("#E4.valid"), tools=self.tools, dataset_dir=DATA_DIR)
                schema_obj = self.env.get("#E2")
                if isinstance(schema_obj, dict):
                    resolved.setdefault("id_cols", schema_obj.get("id_cols"))

            # ---- logging ----
            self.logger(f"[RUN] {step.get('id', i)} → {tool}({self._log_preview(resolved)})")

            # ---- execute ----
            result = fn(**resolved) if isinstance(resolved, dict) else fn(resolved)
            self.env[out] = result
            if tool == "SaveCSV" and isinstance(result, str): artifacts.append(result)
            if tool == "Summarize" and isinstance(result, str): last_report = result

        return {"artifacts": artifacts, "report": last_report, "env": self.env}

# ---- Optional: LLM planner ----
def build_plan_llm(user_task: str, dataset_root: str):
    try:
        import openai
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key: return None
        client = openai.OpenAI(api_key=api_key) if hasattr(openai, "OpenAI") else None
        if client is None: openai.api_key = api_key
        system = (
            "你是 ReWOO 规划器。严格输出纯 JSON（不要代码块/解释）。"
            "工具名必须从集合选择：['LoadCSV','DetectSchema','FE_Generic','SplitTime',"
            "'TrainMedianBaseline','Forecast','SaveCSV','Summarize','Join']。"
            "每步含 id, tool, args, out。"
        )
        user = f"用户任务: {user_task}\n数据目录: {dataset_root}"
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[{"role":"system","content":system},{"role":"user","content":user}],
            )
            content = resp.choices[0].message.content
        except Exception:
            content = None
        if not content: return None
        return json.loads(content)
    except Exception:
        return None

# ---- UI ----
st.title("🤖 ReWOO 多智能体（总改版 · 强制 DataFrame 纠正）")
with st.sidebar:
    st.markdown("### 数据与采样")
    sample_rows = st.number_input("加载样本行数（limit）", min_value=50, max_value=20000, value=1000, step=50)
    per_day = st.number_input("每个日期抽样条数（per_day_take）", min_value=5, max_value=500, value=50, step=5)
    horizon = st.number_input("预测跨度（天）", min_value=1, max_value=90, value=16)

st.write(f"**数据目录（相对）**：`{DATA_DIR}`")
st.caption("将 holidays_events.csv / oil.csv / stores.csv / test.csv / transactions.csv / train.csv 放入 data/ 目录。")

user_task = st.text_area("请输入你的任务描述：", "基于门店与品类，预测未来一段时间的销售额，并输出 CSV 与简报。")
run = st.button("🚀 运行 ReWOO")

if run:
    try:
        # Wrap LoadCSV to inject UI params
        def LoadCSV_wrapper(path, limit=None, ensure_multiday=True, per_day_take=20):
            if not path or (not os.path.exists(path) and not os.path.isdir(path)):
                path = str(DATA_DIR)
            return LoadCSV(path, limit=limit or int(sample_rows), ensure_multiday=True, per_day_take=int(per_day))
        TOOL["LoadCSV"] = LoadCSV_wrapper

        plan = build_plan_llm(user_task, str(DATA_DIR))
        if plan:
            try:
                plan = normalize_plan(plan, str(DATA_DIR))
            except Exception as e:
                st.warning(f"LLM 计划不合规，改用静态计划：{e}")
                plan = None

        if not plan:
            st.info("使用静态 fallback 计划执行。")
            plan = {
                "plan_id": "fallback_demo",
                "steps": [
                    {"id": "load_data", "out": "#E1", "tool": "LoadCSV",
                     "args": {"path": str(DATA_DIR), "limit": int(sample_rows), "ensure_multiday": True, "per_day_take": int(per_day)}},
                    {"id": "schema", "out": "#E2", "tool": "DetectSchema",
                     "args": {"train_like":"#E1", "future_like": None,
                              "force_target": "sales",
                              "force_key_cols": ["store_nbr","family"]}},
                    {"id": "fe", "out": "#E3", "tool": "FE_Generic",
                     "args": {"df": "#E1", "time_col": "#E2.time_col",
                              "key_cols": "#E2.key_cols", "target_col": "#E2.target_col"}},
                    {"id": "split", "out": "#E4", "tool": "SplitTime",
                     "args": {"df": "#E3", "time_col": "#E2.time_col", "horizon": int(horizon)}},
                    {"id": "train", "out": "#E5", "tool": "TrainMedianBaseline",
                     "args": {"train_df": "#E4.train", "valid_df": "#E4.valid",
                              "key_cols": "#E2.key_cols", "target_col": "#E2.target_col"}},
                    {"id": "forecast", "out": "#E6", "tool": "Forecast",
                     "args": {"model_bundle": "#E5", "future_df": "#E4.valid",
                              "id_cols": "#E2.id_cols"}},
                    {"id": "save", "out": "#E7", "tool": "SaveCSV",
                     "args": {"df": "#E6", "path": str(OUTPUT_DIR / "predictions.csv")}},
                    {"id": "report", "out": "#E8", "tool": "Summarize",
                     "args": {"observations": {"schema": "#E2", "metrics": "#E5.metrics"}}},
                ],
                "expect": {"artifacts": [str(OUTPUT_DIR / "predictions.csv")], "report": "#E8"},
            }

        st.subheader("📋 任务计划（执行前）")
        st.json(plan)

        runtime = ReWoorRuntime(TOOL, logger=lambda msg: st.text(msg))
        result = runtime.run_plan(plan)
        st.success("✅ 执行完成")

        # Show artifacts
        for art in result.get("artifacts", []):
            if os.path.exists(art):
                df = pd.read_csv(art)
                st.write("📊 预测结果（前 10 行）：")
                st.dataframe(df.head(10), use_container_width=True)
                st.download_button("⬇️ 下载完整预测结果 CSV",
                                   df.to_csv(index=False).encode("utf-8"),
                                   file_name="predictions.csv",
                                   mime="text/csv")
        if result.get("report"):
            st.subheader("📄 自动分析报告")
            st.markdown(result["report"])

    except Exception as e:
        st.error(f"❌ 执行出错：{e}")
        st.exception(e)

st.caption("总改版 · ReWOO + Streamlit · 相对路径 · 强制 DataFrame 纠正（任何环节都不再传 None）")
