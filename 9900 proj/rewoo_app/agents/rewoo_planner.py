
"""
Generic ReWOO planner that adds DetectSchema step and passes inferred roles to tools.
"""
import os
from typing import Dict, Any, List

def build_plan(dataset_root: str, horizon: int = 16) -> Dict[str, Any]:
    def p(name): return os.path.join(dataset_root, name)
    files = ["train.csv","transactions.csv","sales.csv","test.csv",
             "oil.csv","holidays_events.csv","stores.csv","sample_submission.csv"]

    steps: List[Dict[str, Any]] = []
    vid = 1
    def v():
        nonlocal vid
        s = f"#E{vid}"; vid += 1; return s

    outs = {}
    for fname in files:
        full = p(fname)
        if os.path.exists(full):
            out = v()
            steps.append({"id": f"load_{fname.split('.')[0]}", "out": out,
                          "tool": "LoadCSV", "args": {"path": full}})
            outs[fname] = out

    # choose a main table for schema detection (prefer train/transactions/sales)
    main = outs.get("train.csv") or outs.get("transactions.csv") or outs.get("sales.csv")
    future_like = outs.get("sample_submission.csv") or outs.get("test.csv")

    if main is None and outs:
        main = list(outs.values())[0]

    if main is None:
        return {
            "plan_id": "generic_rewoo",
            "steps": [],
            "expect": {"artifacts": [], "report": "数据目录为空；请放入 CSV 文件。", "metrics": {}}
        }

    # Detect schema
    e_schema = v()
    steps.append({"id":"detect_schema", "out": e_schema, "tool": "DetectSchema",
                  "args": {"train_like": main, "future_like": future_like}})

    # Optionally join side tables onto main if keys align
    merged = main
    def sjoin(left, right, on, sid):
        if left and right:
            e = v()
            steps.append({"id": sid, "out": e, "tool": "Join", "args": {"left": left, "right": right, "on": on}})
            return e
        return left
    merged = sjoin(merged, outs.get("oil.csv"), ["date"], "join_oil")
    merged = sjoin(merged, outs.get("stores.csv"), ["store_nbr"], "join_stores")
    merged = sjoin(merged, outs.get("holidays_events.csv"), ["date"], "join_holidays")

    # FE
    e_feat = v()
    steps.append({"id":"fe_generic","out":e_feat,"tool":"FE_Generic",
                  "args":{"df": merged, "time_col": f"{e_schema}.time_col",
                          "key_cols": f"{e_schema}.key_cols",
                          "target_col": f"{e_schema}.target_col"}})

    # Split / Train
    e_split = v()
    steps.append({"id":"split","out":e_split,"tool":"SplitTime",
                  "args":{"df": e_feat, "time_col": f"{e_schema}.time_col", "horizon": horizon}})

    e_model = v()
    steps.append({"id":"train","out":e_model,"tool":"TrainMedianBaseline",
                  "args":{"train_df": f"{e_split}.train","valid_df": f"{e_split}.valid",
                          "key_cols": f"{e_schema}.key_cols", "target_col": f"{e_schema}.target_col"}})

    # Forecast on sample_submission/test if available, else on valid as demo
    future_src = future_like or f"{e_split}.valid"
    e_pred = v()
    steps.append({"id":"forecast","out":e_pred,"tool":"Forecast",
                  "args":{"model_bundle": e_model, "future_df": future_src,
                          "id_cols": f"{e_schema}.id_cols"}})

    # Save + Report
    e_save = v()
    steps.append({"id":"save_pred","out":e_save,"tool":"SaveCSV",
                  "args":{"df": e_pred, "path": os.path.join(dataset_root, "outputs", "predictions.csv")}})

    e_report = v()
    steps.append({"id":"report","out":e_report,"tool":"Summarize",
                  "args":{"observations":{"schema": e_schema, "metrics": f"{e_model}.metrics"}}})

    return {
        "plan_id": "generic_rewoo",
        "steps": steps,
        "expect": {
            "artifacts": [os.path.join(dataset_root, "outputs", "predictions.csv")],
            "report": e_report,
            "metrics": f"{e_model}.metrics"
        }
    }
