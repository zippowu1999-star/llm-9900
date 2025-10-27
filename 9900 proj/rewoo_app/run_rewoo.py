
from pathlib import Path

from agents.rewoo_planner import build_plan
from executor.rewoo_runtime import ReWoorRuntime
from tools.generic_tools import (
    LoadCSV, SaveCSV, Join, DetectSchema, FE_Generic,
    SplitTime, TrainMedianBaseline, Forecast, Summarize
)

# auto path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
(DATA_DIR / "outputs").mkdir(parents=True, exist_ok=True)

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

def main():
    plan = build_plan(dataset_root=str(DATA_DIR), horizon=16)
    rt = ReWoorRuntime(TOOL, logger=lambda *a, **k: None)
    res = rt.run_plan(plan)
    print("Artifacts:", res["artifacts"])
    print("Metrics:", res["metrics"])
    print("Report:\n", (res["report"] or "")[:600], "...")

if __name__ == "__main__":
    main()
