# rewoo_app/tools/generic_tools.py
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np

# -------------------------- #
#        Utils / Calendar    #
# -------------------------- #

def AddCalendar(df: pd.DataFrame, time_col: str | None) -> pd.DataFrame:
    d = df.copy()
    if time_col and time_col in d.columns:
        if not pd.api.types.is_datetime64_any_dtype(d[time_col]):
            d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
        d["year"] = d[time_col].dt.year
        d["month"] = d[time_col].dt.month
        d["day"] = d[time_col].dt.day
        d["dow"] = d[time_col].dt.dayofweek
        d["week"] = d[time_col].dt.isocalendar().week.astype(int)
        d["is_weekend"] = (d["dow"] >= 5).astype(int)
    return d


# -------------------------- #
#      Dataset Loader        #
# -------------------------- #

def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV {path}: {e}")

def LoadDataset(
    root_path: str,
    limit: int | None = None,
    ensure_multiday: bool = True,
    per_day_take: int = 20,
    prefer_main: list[str] = ("train.csv", "sales", "main"),
) -> pd.DataFrame:
    """
    自动加载并关联目录下的所有 *.csv：
    - 主表自动推断（优先 train.csv，其次文件名含 sales/main）
    - 自动解析 date 列
    - 基于公共列(left join)合并 oil / holidays / stores / transactions / test 等表
    - 可选：跨多天抽样、行数限制
    """
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"root_path not found: {root_path}")
    csv_files = sorted(root.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No csv files under: {root_path}")

    tables: dict[str, pd.DataFrame] = {f.name: _safe_read_csv(str(f)) for f in csv_files}
    print(f"[LoadDataset] Detected files: {list(tables.keys())}")

    # ---- 选择主表
    main_key = None
    # 1) 完全匹配 train.csv
    for name in tables:
        if name.lower() == "train.csv":
            main_key = name
            break
    # 2) 文件名包含 prefer_main 关键词
    if main_key is None:
        for name in tables:
            lower = name.lower()
            if any(k in lower for k in prefer_main):
                main_key = name
                break
    # 3) 回退：行数最多者
    if main_key is None:
        main_key = max(tables.keys(), key=lambda n: len(tables[n]))
    df = tables[main_key].copy()

    # 解析主表日期
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # ---- 合并其他表（按公共列）
    # 优先 key 集：date / store_nbr / family
    PREFERRED_KEYS = ["date", "store_nbr", "family"]
    for name, sub in tables.items():
        if name == main_key:
            continue
        sub_df = sub.copy()
        # 解析日期
        if "date" in sub_df.columns and not pd.api.types.is_datetime64_any_dtype(sub_df["date"]):
            sub_df["date"] = pd.to_datetime(sub_df["date"], errors="coerce")

        common = list(set(df.columns) & set(sub_df.columns))
        if not common:
            # 没有公共列，跳过
            print(f"[LoadDataset] skip {name}: no common columns with main")
            continue

        # 尽量采用优先 keys 子集作为连接键，避免高维重复
        join_keys = [k for k in PREFERRED_KEYS if k in common]
        if join_keys:
            on = join_keys
        else:
            # 使用公共列中数量较小的一部分（最多 2 列），避免连接爆炸
            common_sorted = sorted(common, key=lambda c: sub_df[c].nunique())
            on = common_sorted[: min(2, len(common_sorted))]

        try:
            before = len(df)
            df = df.merge(sub_df, on=on, how="left", suffixes=("", f"_{Path(name).stem}"))
            print(f"[LoadDataset] merged {name} on {on} -> {before}→{len(df)} rows")
        except Exception as e:
            print(f"[LoadDataset] skip {name}: {e}")

    # ---- 跨多天抽样（用于小样本快速跑通）
    if ensure_multiday and "date" in df.columns:
        d = df[df["date"].notna()].sort_values("date")
        parts, total = [], 0
        for day, grp in d.groupby("date"):
            take = min(per_day_take, len(grp))
            parts.append(grp.head(take))
            total += take
            if limit and total >= limit:
                break
        if parts:
            df = pd.concat(parts, ignore_index=True)
            if limit:
                df = df.head(limit)
    elif limit:
        df = df.head(limit)

    print(f"[LoadDataset] final shape: {df.shape}")
    return df


# -------------------------- #
#     CSV Loader (Generic)   #
# -------------------------- #

def LoadCSV(path: str,
            limit: int | None = None,
            ensure_multiday: bool = True,
            per_day_take: int = 20) -> pd.DataFrame:
    """
    统一入口：
    - 若传入的是“目录”，调用 LoadDataset 自动多表合并
    - 若传入的是“文件”，按 CSV 读取（含日期解析、抽样）
    """
    p = Path(path)
    if p.is_dir():
        return LoadDataset(str(p), limit=limit, ensure_multiday=ensure_multiday, per_day_take=per_day_take)

    df = _safe_read_csv(str(p))
    # 解析日期
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # 多天抽样
    if ensure_multiday and "date" in df.columns:
        d = df[df["date"].notna()].sort_values("date")
        parts, total = [], 0
        for day, grp in d.groupby("date"):
            take = min(per_day_take, len(grp))
            parts.append(grp.head(take))
            total += take
            if limit and total >= limit:
                break
        if parts:
            df = pd.concat(parts, ignore_index=True)
    if limit and len(df) > limit:
        df = df.head(limit)
    print(f"[LoadCSV] loaded shape: {df.shape}")
    return df


# -------------------------- #
#           Join             #
# -------------------------- #

def Join(left: pd.DataFrame, right: pd.DataFrame, on: list[str] | None = None, how: str = "left") -> pd.DataFrame:
    if on is None:
        # 自动找公共列
        on = list(set(left.columns) & set(right.columns))
        if not on:
            raise ValueError("Join needs columns overlap or explicit `on`")
    return left.merge(right, on=on, how=how)


# -------------------------- #
#        Detect Schema       #
# -------------------------- #

def DetectSchema(train_like: pd.DataFrame,
                 future_like: pd.DataFrame | None = None,
                 force_target: str | None = None,
                 force_key_cols: list[str] | None = None) -> dict:
    """
    自动/半自动识别：
      - time_col   : 日期列
      - target_col : 目标列（优先 names: sales/target/y/label/revenue/qty/demand）
      - key_cols   : 维度键（基数适中）
      - id_cols    : 用于输出的 id 列
    支持强制覆盖：force_target / force_key_cols
    """
    df = train_like.copy()

    # 时间列
    time_col = None
    for c in df.columns:
        lc = str(c).lower()
        if lc in ("date", "ds", "timestamp") and pd.api.types.is_datetime64_any_dtype(df[c]):
            time_col = c; break
    if time_col is None:
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                time_col = c; break

    # 目标列
    if force_target and force_target in df.columns:
        target_col = force_target
    else:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        target_col = None
        name_hints = ["sales","target","y","label","revenue","qty","demand"]
        for c in num_cols:
            if any(h in str(c).lower() for h in name_hints):
                target_col = c; break
        if target_col is None and num_cols:
            target_col = max(num_cols, key=lambda c: df[c].std(skipna=True))

    # 键列（适中基数）
    if force_key_cols:
        key_cols = [c for c in force_key_cols if c in df.columns]
    else:
        n = max(len(df), 1)
        cands = []
        for c in df.columns:
            if c in (target_col, time_col, "id"): continue
            if pd.api.types.is_datetime64_any_dtype(df[c]): continue
            uniq = df[c].nunique(dropna=True)
            ratio = uniq / n
            if 2 <= uniq and 0.01 <= ratio <= 0.30:
                cands.append((c, uniq))
        cands.sort(key=lambda x: x[1])
        key_cols = [c for c,_ in cands[:2]]  # 最多两个键

    # 输出 id_cols
    id_pool = []
    for name in ["id"] + key_cols + ([time_col] if time_col else []):
        if name and name not in id_pool:
            id_pool.append(name)
    id_cols = []
    base = future_like if isinstance(future_like, pd.DataFrame) else df
    for c in id_pool:
        if c in base.columns and c not in id_cols:
            id_cols.append(c)

    return {"time_col": time_col, "target_col": target_col, "key_cols": key_cols, "id_cols": id_cols}


# -------------------------- #
#       Feature Engine       #
# -------------------------- #

def FE_Generic(df: pd.DataFrame, time_col: str | None, key_cols: list[str], target_col: str | None) -> pd.DataFrame:
    d = AddCalendar(df, time_col)

    # 目标数值化（防止 rolling 报错）
    if target_col and target_col in d.columns:
        d[target_col] = pd.to_numeric(d[target_col], errors="coerce")

        # lags & rolling
        grp = d.groupby(key_cols) if key_cols else d
        for w in [1, 7, 14]:
            d[f"{target_col}_lag{w}"] = grp[target_col].shift(w)
        for w in [7, 14, 28]:
            d[f"{target_col}_rm{w}"] = grp[target_col].shift(1).rolling(w).mean()

    return d


# -------------------------- #
#         Split Time         #
# -------------------------- #

def SplitTime(df: pd.DataFrame, time_col: str | None, horizon: int = 16) -> dict:
    if not time_col or time_col not in df.columns or len(df) == 0:
        return {"train": df, "valid": df.iloc[0:0], "future": df.iloc[0:0]}
    d = df.sort_values(time_col).copy()
    max_dt = d[time_col].max()
    cutoff = max_dt - pd.Timedelta(days=horizon)
    train = d[d[time_col] <= cutoff]
    valid = d[(d[time_col] > cutoff) & (d[time_col] <= max_dt)]
    future = d[d[time_col] > max_dt]

    # 兜底：若训练集为空（例如样本都在同一天），用 80/20 顺序切分
    if train.empty and len(d) > 1:
        k = max(1, int(len(d) * 0.8))
        train, valid = d.iloc[:k], d.iloc[k:]
    return {"train": train, "valid": valid, "future": future}


# -------------------------- #
#       Median Baseline      #
# -------------------------- #

def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    z = np.where(denom == 0, 1.0, denom)
    return np.mean(2.0 * np.abs(y_pred - y_true) / z)

def TrainMedianBaseline(train_df: pd.DataFrame,
                        valid_df: pd.DataFrame,
                        key_cols: list[str],
                        target_col: str) -> dict:
    d = train_df.copy()
    # 可选：忽略全 0 行，避免全零中位数（按需打开）
    # d = d[d[target_col] != 0]

    if key_cols:
        med_table = d.groupby(key_cols)[target_col].median().reset_index().rename(columns={target_col: "med"})
    else:
        med_table = pd.DataFrame({"med": [d[target_col].median()]})
    model = {"type": "median_by_key", "fitted": True, "keys": key_cols, "med_table": med_table, "target": target_col}

    # 简单验证指标
    metrics = {}
    if len(valid_df) > 0:
        preds = Forecast({"model": model, "metrics": {}}, valid_df, id_cols=key_cols + ["date"]).copy()
        y_true = pd.to_numeric(valid_df[target_col], errors="coerce").fillna(0).to_numpy()
        y_pred = pd.to_numeric(preds["prediction"], errors="coerce").fillna(0).to_numpy()
        metrics["smape"] = float(_smape(y_true, y_pred))
    return {"model": model, "metrics": metrics}


# -------------------------- #
#           Forecast         #
# -------------------------- #

def Forecast(model_bundle: dict,
             future_df: pd.DataFrame,
             id_cols: list[str]) -> pd.DataFrame:
    model = model_bundle.get("model", {})
    if model.get("type") != "median_by_key":
        raise ValueError("Only 'median_by_key' model supported in this baseline")

    keys = model.get("keys", [])
    target = model.get("target")
    med = model.get("med_table")

    df = future_df.copy()
    if keys:
        out = df.merge(med, on=keys, how="left")
        # 缺失填充：用全局中位数
        if "med" not in out.columns:
            out["med"] = np.nan
        global_med = med["med"].median() if "med" in med.columns else 0.0
        out["prediction"] = out["med"].fillna(global_med)
    else:
        global_med = float(med["med"].iloc[0]) if isinstance(med, pd.DataFrame) and len(med) else 0.0
        out = df.copy()
        out["prediction"] = global_med

    keep = [c for c in id_cols if c in out.columns] + ["prediction"]
    return out[keep]


# -------------------------- #
#            Save            #
# -------------------------- #

def SaveCSV(df: pd.DataFrame, path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return str(p)


# -------------------------- #
#          Summarize         #
# -------------------------- #

def Summarize(observations: dict) -> str:
    schema = observations.get("schema", {})
    metrics = observations.get("metrics", {})
    lines = [
        "# 自动分析报告",
        "## 数据架构推断",
        f"- 时间列: `{schema.get('time_col')}`",
        f"- 目标列: `{schema.get('target_col')}`",
        f"- 键列: `{', '.join(schema.get('key_cols', []))}`",
        "",
        "## 训练评估",
        f"- SMAPE: {metrics.get('smape', 'N/A')}",
        "",
        "## 说明",
        "本报告由 ReWOO 基线管线自动生成。结果供快速校验与联调使用，后续可替换更强模型。",
    ]
    return "\n".join(lines)

