#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
posthoc_block_analysis_iterations_updated.py

Iteration-aware post-hoc analysis for repeated outer-CV / stacking runs.

What it does
------------
- discovers iteration folders under <output_dir>/<subtype>/<iteration_glob>
- gathers repeated OOF predictions across all iterations for each requested block
  (RNA, CNV, MUT, CLINICAL, ALL, STACK)
- saves analysis-ready tables first, then builds plots from those tables

Per block, it saves:
- raw long-format repeated OOF predictions
- iteration-level ROC/AUC and PR/AP summaries
- raw per-iteration ROC curve points
- interpolated mean/median ROC with percentile bands
- sample-aggregated predictions across iterations (mean / median / sd / min / max)
- aggregated KM group assignments and KM curve coordinates
- optional iteration-level KM stability curve coordinates and summary bands
- aggregated Cox results (patient-level median prediction)
- iteration-level Cox results and summary stats
- forest-ready Cox tables

Main statistical conventions
----------------------------
- ROC/AUC: one AUC per iteration using pooled OOF predictions across that iteration's outer folds
- ROC CI band: percentile band across iteration-specific interpolated ROC curves
- Main KM/Cox: one row per patient using median predicted probability across iterations
- KM iteration band: summarizes per-iteration KM curves (stability band, not a formal inferential CI)
- Iteration-level HR summary: median HR with percentile interval across iterations + counts of significant HRs

Notes
-----
- This script assumes the model family per block is fixed across iterations and only hyperparameters vary.
- If multiple model_name strings are found for a block, the most frequent one is selected automatically and reported.
"""

import os
import re
import glob
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
)

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test
    HAVE_LIFELINES = True
except Exception:
    HAVE_LIFELINES = False


VALID_BLOCKS = ["RNA", "CNV", "MUT", "CLINICAL", "ALL", "STACK"]


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")


def _parse_figsize(s: Optional[str], default=(6, 4)):
    if s is None:
        return default
    try:
        a, b = [float(x.strip()) for x in str(s).split(",")]
        return (a, b)
    except Exception:
        return default


def _parse_color_map(s: Optional[str], default_map: Dict[str, str]) -> Dict[str, str]:
    out = dict(default_map)
    if not s:
        return out
    pairs = [x.strip() for x in str(s).split(",") if x.strip()]
    for p in pairs:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        out[k.strip()] = v.strip()
    return out




def _parse_block_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if not x:
            continue
        out.append(x.upper())
    return out


def _save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def _save_json(payload: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def save_legend_only(handles, labels, outpath, figsize=(2.8, 2.8), ncol=1, fontsize=8):
    if not handles or not labels:
        return
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.legend(handles, labels, loc="center", frameon=False, ncol=ncol, fontsize=fontsize)
    fig.tight_layout()
    fig.savefig(outpath, dpi=210, bbox_inches="tight", transparent=True)
    plt.close(fig)


def save_text_panel(lines, outpath, figsize=(3.2, 1.8), fontsize=8):
    if not lines:
        return
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=fontsize)
    fig.tight_layout()
    fig.savefig(outpath, dpi=210, bbox_inches="tight", transparent=True)
    plt.close(fig)


# -------------------------
# Metadata helpers
# -------------------------
def _norm_subtype_label(x: str) -> str:
    s = str(x).strip().lower()
    s = s.replace("luminal a", "luma")
    s = s.replace("luminal b", "lumb")
    s = s.replace("her2 enriched", "her2")
    s = s.replace("her2-enriched", "her2")
    s = s.replace("basal-like", "basal")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _norm_binary_token(x) -> str:
    s = str(x).strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"[\s\-/]+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s


def _decode_binary_covariate_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    na_tokens = {"", "na", "n_a", "nan", "none", "null", "missing", "unknown", "not_available"}
    one_tokens = {
        "1", "yes", "y", "true", "t", "ever", "treated", "treatment",
        "received", "with_treatment", "chemo", "has_chemo", "hormone",
        "hormone_therapy", "tx", "present", "positive", "pos"
    }
    zero_tokens = {
        "0", "no", "n", "false", "f", "never", "untreated", "no_treatment",
        "without_treatment", "no_chemo", "without_chemo", "no_hormone",
        "no_hormone_therapy", "absent", "negative", "neg"
    }

    s = series.copy()
    s_non_na = s.dropna()
    if s_non_na.empty:
        return pd.to_numeric(s, errors="coerce")

    norm = s_non_na.map(_norm_binary_token)
    uniq = set(norm.unique().tolist())

    if uniq and uniq.issubset(one_tokens | zero_tokens | na_tokens):
        def _map_one(v):
            tok = _norm_binary_token(v)
            if tok in na_tokens:
                return np.nan
            if tok in one_tokens:
                return 1.0
            if tok in zero_tokens:
                return 0.0
            return np.nan
        return s.map(_map_one).astype(float)

    return pd.to_numeric(s, errors="coerce")


def _decode_covariate_columns(meta: pd.DataFrame, covariates: List[str]) -> pd.DataFrame:
    meta = meta.copy()
    for c in covariates:
        if c not in meta.columns:
            continue
        before = meta[c]
        after = _decode_binary_covariate_series(before)
        changed = (not pd.api.types.is_numeric_dtype(before) and after.notna().sum() > 0)
        if changed:
            non_na_before = before.notna().sum()
            non_na_after = after.notna().sum()
            uniq_after = sorted(pd.Series(after.dropna().unique()).astype(float).tolist())
            print(f"[META] Decoded covariate '{c}' to numeric (non-missing: {non_na_before} -> {non_na_after}; values: {uniq_after})")
        meta[c] = after
    return meta


def load_meta(meta_path: str, subtype: str, subtype_col: str = "auto") -> pd.DataFrame:
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Cannot find meta at {meta_path}")

    meta = pd.read_csv(meta_path, index_col=0)
    meta.index = meta.index.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    if subtype_col == "auto":
        candidates = [
            "pam50_subtype", "PAM50_subtype", "pam50", "PAM50",
            "subtype", "Subtype", "molecular_subtype"
        ]
        found = next((c for c in candidates if c in meta.columns), None)
        if found is None:
            raise ValueError(
                f"Could not auto-detect subtype column in metadata. Available columns include: {list(meta.columns)[:25]}"
            )
        subtype_col = found

    if subtype_col not in meta.columns:
        raise ValueError(f"Subtype column '{subtype_col}' not found in metadata.")

    want = _norm_subtype_label(subtype)
    meta_sub = meta[meta[subtype_col].astype(str).map(_norm_subtype_label) == want].copy()
    if meta_sub.empty:
        raise ValueError(
            f"No rows matched subtype '{subtype}' using column '{subtype_col}'. "
            f"Unique values include: {sorted(meta[subtype_col].dropna().astype(str).unique().tolist())[:20]}"
        )
    print(f"[META] Loaded full metadata: {meta.shape}")
    print(f"[META] Subset to subtype={subtype} via {subtype_col}: {meta_sub.shape}")
    return meta_sub


# -------------------------
# Iteration discovery + prediction loading
# -------------------------
def discover_iteration_runs(output_dir: str, subtype: str, iteration_glob: str = "iter_*") -> List[Dict[str, str]]:
    subtype_root = os.path.join(output_dir, subtype)
    if not os.path.exists(subtype_root):
        raise FileNotFoundError(f"Subtype folder not found: {subtype_root}")

    candidates = [p for p in glob.glob(os.path.join(subtype_root, iteration_glob)) if os.path.isdir(p)]
    runs = []

    for root in sorted(candidates):
        models_dir = os.path.join(root, "cv_models")
        if not os.path.exists(models_dir):
            models_dir = os.path.join(root, "models")
        preds_dir = os.path.join(root, "predictions")
        if not os.path.exists(preds_dir):
            preds_dir = models_dir
        runs.append({
            "iteration_id": os.path.basename(root),
            "root_dir": root,
            "models_dir": models_dir,
            "preds_dir": preds_dir,
        })

    if not runs:
        models_dir = os.path.join(subtype_root, "cv_models")
        if not os.path.exists(models_dir):
            models_dir = os.path.join(subtype_root, "models")
        preds_dir = os.path.join(subtype_root, "predictions")
        if os.path.exists(models_dir) or os.path.exists(preds_dir):
            if not os.path.exists(preds_dir):
                preds_dir = models_dir
            runs.append({
                "iteration_id": "single_run",
                "root_dir": subtype_root,
                "models_dir": models_dir,
                "preds_dir": preds_dir,
            })

    if not runs:
        raise RuntimeError(
            f"No iteration folders found under {subtype_root} with glob '{iteration_glob}', and no fallback single-run folder was detected."
        )

    print(f"[DISCOVER] Found {len(runs)} iteration run(s).")
    for r in runs:
        print(f"  - {r['iteration_id']} | models={r['models_dir']} | preds={r['preds_dir']}")
    return runs


def _read_pred_file(path: str, iteration_id: str, block_hint: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sample_id" not in df.columns:
        raise ValueError(f"Prediction file missing sample_id column: {path}")

    df = df.copy()
    df["sample_id"] = df["sample_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    df["iteration_id"] = iteration_id
    if "fold" not in df.columns:
        m = re.search(r"OUTER_FOLD[_-]?(\d+)", os.path.basename(path), flags=re.IGNORECASE)
        df["fold"] = int(m.group(1)) if m else np.nan
    if "block" not in df.columns and block_hint is not None:
        df["block"] = block_hint
    if "model_name" not in df.columns:
        df["model_name"] = np.nan
    return df


def _select_dominant_model(df: pd.DataFrame, block: str) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    if df.empty or "model_name" not in df.columns:
        return df, pd.DataFrame(columns=["block", "model_name", "n_rows"]), None

    work = df.copy()
    work["model_name"] = work["model_name"].astype(str).str.strip()
    counts = (work.groupby("model_name", as_index=False)
                  .size()
                  .rename(columns={"size": "n_rows"})
                  .sort_values(["n_rows", "model_name"], ascending=[False, True]))
    counts.insert(0, "block", block)

    if counts.empty:
        return work, counts, None

    keep_model = counts.iloc[0]["model_name"]
    if counts.shape[0] > 1:
        print(f"[WARN] Block {block}: multiple model_name values found. Keeping most frequent: {keep_model}")
    work = work[work["model_name"].astype(str).str.strip() == str(keep_model)].copy()
    return work, counts, keep_model


def gather_predictions_for_block(runs: List[Dict[str, str]], block: str) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    block = str(block).upper()
    if block not in VALID_BLOCKS:
        raise ValueError(f"Unknown block '{block}'. Valid: {VALID_BLOCKS}")

    dfs = []
    for run in runs:
        preds_dir = run["preds_dir"]
        it = run["iteration_id"]

        if block == "STACK":
            files = sorted(glob.glob(os.path.join(preds_dir, "PRED__OUTER_FOLD_*__STACK_BLOCKS.csv")))
            for f in files:
                try:
                    df = _read_pred_file(f, iteration_id=it, block_hint="STACK")
                    dfs.append(df)
                except Exception as e:
                    print(f"[WARN] Could not read STACK pred file {f}: {e}")
            continue

        if block == "ALL":
            files = sorted(glob.glob(os.path.join(preds_dir, "*__OUTER_FOLD_*__preds.csv")))
            files = [f for f in files if not os.path.basename(f).startswith("PRED__")]
            for f in files:
                try:
                    df = _read_pred_file(f, iteration_id=it, block_hint="ALL")
                    df["block"] = "ALL"
                    dfs.append(df)
                except Exception as e:
                    print(f"[WARN] Could not read ALL pred file {f}: {e}")
            continue

        files = sorted(glob.glob(os.path.join(preds_dir, f"PRED__OUTER_FOLD_*__{block}__*.csv")))
        for f in files:
            try:
                df = _read_pred_file(f, iteration_id=it, block_hint=block)
                df["block"] = block
                dfs.append(df)
            except Exception as e:
                print(f"[WARN] Could not read {block} pred file {f}: {e}")

    if not dfs:
        empty = pd.DataFrame(columns=["sample_id", "y_true", "predicted_proba", "model_name", "block", "fold", "iteration_id"])
        return empty, pd.DataFrame(columns=["block", "model_name", "n_rows"]), None

    out = pd.concat(dfs, axis=0, ignore_index=True)
    out["block"] = block
    out, model_counts, keep_model = _select_dominant_model(out, block=block)
    return out, model_counts, keep_model


def merge_with_meta(pred_df: pd.DataFrame, meta: pd.DataFrame, time_col: str, covariates: List[str]) -> pd.DataFrame:
    pred_df = pred_df.copy()
    pred_df["sample_id"] = pred_df["sample_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    meta = meta.copy()
    meta.index = meta.index.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    meta = _decode_covariate_columns(meta, covariates)

    need_cols = [time_col] + [c for c in covariates if c in meta.columns]
    merged = pred_df.merge(
        meta[need_cols].rename(columns={time_col: "time_to_event"}),
        left_on="sample_id",
        right_index=True,
        how="left",
    )
    return merged


# -------------------------
# ROC / PR analysis
# -------------------------
def compute_iteration_roc_tables(pred_df: pd.DataFrame, fpr_grid: Optional[np.ndarray] = None):
    if fpr_grid is None:
        fpr_grid = np.linspace(0.0, 1.0, 201)

    by_iter_rows = []
    raw_curve_rows = []
    interp_rows = []

    for iteration_id, g in pred_df.groupby("iteration_id"):
        g = g.dropna(subset=["y_true", "predicted_proba"]).copy()
        if g.empty:
            continue

        y = pd.to_numeric(g["y_true"], errors="coerce")
        p = pd.to_numeric(g["predicted_proba"], errors="coerce")
        ok = (~y.isna()) & (~p.isna())
        y = y.loc[ok].astype(int).values
        p = p.loc[ok].astype(float).values
        if len(np.unique(y)) < 2:
            continue

        try:
            auc = float(roc_auc_score(y, p))
            ap = float(average_precision_score(y, p))
            fpr, tpr, thr = roc_curve(y, p, pos_label=1)
        except Exception as e:
            print(f"[WARN] ROC failed for iteration {iteration_id}: {e}")
            continue

        by_iter_rows.append({
            "iteration_id": iteration_id,
            "n_samples": int(len(y)),
            "n_events": int((y == 1).sum()),
            "auc": auc,
            "ap": ap,
        })

        for x, yv, tv in zip(fpr, tpr, thr):
            raw_curve_rows.append({
                "iteration_id": iteration_id,
                "fpr": float(x),
                "tpr": float(yv),
                "threshold": float(tv),
            })

        tpr_interp = np.interp(fpr_grid, fpr, tpr)
        tpr_interp[0] = 0.0
        tpr_interp[-1] = 1.0
        for x, yv in zip(fpr_grid, tpr_interp):
            interp_rows.append({
                "iteration_id": iteration_id,
                "fpr_grid": float(x),
                "tpr_interp": float(yv),
            })

    by_iter = pd.DataFrame(by_iter_rows)
    raw_curve = pd.DataFrame(raw_curve_rows)
    interp_curve = pd.DataFrame(interp_rows)

    if by_iter.empty:
        summary = pd.DataFrame(columns=[
            "n_iterations", "auc_mean", "auc_sd", "auc_median", "auc_p2_5", "auc_p97_5",
            "ap_mean", "ap_sd", "ap_median", "ap_p2_5", "ap_p97_5"
        ])
        curve_summary = pd.DataFrame(columns=["fpr_grid", "tpr_mean", "tpr_median", "tpr_sd", "tpr_p2_5", "tpr_p97_5"])
        return by_iter, summary, raw_curve, interp_curve, curve_summary

    summary = pd.DataFrame([{
        "n_iterations": int(by_iter.shape[0]),
        "auc_mean": float(by_iter["auc"].mean()),
        "auc_sd": float(by_iter["auc"].std(ddof=1)) if by_iter.shape[0] > 1 else 0.0,
        "auc_median": float(by_iter["auc"].median()),
        "auc_p2_5": float(by_iter["auc"].quantile(0.025)),
        "auc_p97_5": float(by_iter["auc"].quantile(0.975)),
        "ap_mean": float(by_iter["ap"].mean()),
        "ap_sd": float(by_iter["ap"].std(ddof=1)) if by_iter.shape[0] > 1 else 0.0,
        "ap_median": float(by_iter["ap"].median()),
        "ap_p2_5": float(by_iter["ap"].quantile(0.025)),
        "ap_p97_5": float(by_iter["ap"].quantile(0.975)),
    }])

    curve_summary = (interp_curve.groupby("fpr_grid", as_index=False)["tpr_interp"]
                     .agg(
                         tpr_mean="mean",
                         tpr_median="median",
                         tpr_sd="std",
                         tpr_p2_5=lambda s: s.quantile(0.025),
                         tpr_p97_5=lambda s: s.quantile(0.975),
                     ))
    curve_summary["tpr_sd"] = curve_summary["tpr_sd"].fillna(0.0)
    return by_iter, summary, raw_curve, interp_curve, curve_summary


def plot_auc_boxplot(by_iter: pd.DataFrame, outpath: str, title: str, figsize=(5.5, 4.5)):
    if by_iter is None or by_iter.empty:
        return
    vals = by_iter["auc"].dropna().values
    if len(vals) == 0:
        return
    plt.figure(figsize=figsize)
    plt.boxplot(vals, vert=True, showfliers=False)
    x = np.random.normal(loc=1.0, scale=0.05, size=len(vals))
    plt.scatter(x, vals, s=22, alpha=0.7)
    plt.xticks([1], ["AUC across iterations"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("AUC")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=210)
    plt.close()


def plot_mean_roc(curve_summary: pd.DataFrame,
                  interp_curve: pd.DataFrame,
                  outpath: str,
                  title: str,
                  auc_summary: Optional[pd.DataFrame] = None,
                  figsize=(6, 6),
                  mean_color="#1f77b4",
                  band_color="#1f77b4",
                  show_iteration_curves: bool = True):
    if curve_summary is None or curve_summary.empty:
        return

    plt.figure(figsize=figsize)

    if show_iteration_curves and interp_curve is not None and not interp_curve.empty:
        for _, g in interp_curve.groupby("iteration_id"):
            plt.plot(g["fpr_grid"], g["tpr_interp"], color="lightgray", linewidth=0.8, alpha=0.5)

    plt.fill_between(
        curve_summary["fpr_grid"].values,
        curve_summary["tpr_p2_5"].values,
        curve_summary["tpr_p97_5"].values,
        color=band_color,
        alpha=0.18,
        label="Iteration band (2.5–97.5%)",
    )
    plt.plot(curve_summary["fpr_grid"], curve_summary["tpr_mean"], color=mean_color, linewidth=2.0, label="Mean ROC")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.0)

    subtitle = ""
    if auc_summary is not None and not auc_summary.empty:
        r = auc_summary.iloc[0]
        subtitle = f"\nMean AUC={r['auc_mean']:.3f} (SD={r['auc_sd']:.3f}; 95% PI {r['auc_p2_5']:.3f}-{r['auc_p97_5']:.3f})"
    plt.title(f"{title}{subtitle}")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(frameon=False, fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=210)
    plt.close()


# -------------------------
# Aggregation + KM
# -------------------------
def aggregate_sample_predictions(pred_df: pd.DataFrame, covariates: List[str]) -> pd.DataFrame:
    if pred_df.empty:
        cols = ["sample_id", "y_true", "time_to_event", "pred_mean", "pred_median", "pred_sd", "pred_min", "pred_max", "n_iterations_present"] + covariates
        return pd.DataFrame(columns=cols)

    work = pred_df.copy()
    work["predicted_proba"] = pd.to_numeric(work["predicted_proba"], errors="coerce")
    work["y_true"] = pd.to_numeric(work["y_true"], errors="coerce")
    work["time_to_event"] = pd.to_numeric(work["time_to_event"], errors="coerce")

    agg_rows = []
    for sample_id, g in work.groupby("sample_id"):
        proba = pd.to_numeric(g["predicted_proba"], errors="coerce").dropna()
        if proba.empty:
            continue
        row = {
            "sample_id": sample_id,
            "y_true": float(pd.to_numeric(g["y_true"], errors="coerce").dropna().iloc[0]) if pd.to_numeric(g["y_true"], errors="coerce").dropna().size else np.nan,
            "time_to_event": float(pd.to_numeric(g["time_to_event"], errors="coerce").dropna().iloc[0]) if pd.to_numeric(g["time_to_event"], errors="coerce").dropna().size else np.nan,
            "pred_mean": float(proba.mean()),
            "pred_median": float(proba.median()),
            "pred_sd": float(proba.std(ddof=1)) if len(proba) > 1 else 0.0,
            "pred_min": float(proba.min()),
            "pred_max": float(proba.max()),
            "n_iterations_present": int(g["iteration_id"].nunique()),
        }
        for c in covariates:
            if c in g.columns:
                nn = pd.to_numeric(g[c], errors="coerce").dropna()
                row[c] = float(nn.iloc[0]) if nn.size else np.nan
        agg_rows.append(row)
    return pd.DataFrame(agg_rows)


def make_km_groups_from_aggregated(agg_df: pd.DataFrame, score_col: str = "pred_median") -> Tuple[pd.DataFrame, float]:
    df = agg_df.copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    cutoff = float(df[score_col].median())
    df["km_score_col"] = score_col
    df["km_cutoff"] = cutoff
    df["km_group"] = np.where(df[score_col] >= cutoff, "High", "Low")
    return df, cutoff


def compute_km_points(group_df: pd.DataFrame,
                      time_col: str = "time_to_event",
                      event_col: str = "y_true",
                      group_col: str = "km_group") -> Tuple[pd.DataFrame, Optional[float]]:
    if not HAVE_LIFELINES:
        return pd.DataFrame(), None

    work = group_df.copy()
    work[time_col] = pd.to_numeric(work[time_col], errors="coerce")
    work[event_col] = pd.to_numeric(work[event_col], errors="coerce")
    work = work.dropna(subset=[time_col, event_col, group_col]).copy()
    if work.empty:
        return pd.DataFrame(), None
    work[event_col] = work[event_col].astype(int)

    rows = []
    kmf = KaplanMeierFitter()
    for grp, g in work.groupby(group_col):
        if g.empty:
            continue
        kmf.fit(g[time_col], event_observed=g[event_col], label=str(grp))
        sf = kmf.survival_function_.reset_index()
        sf.columns = ["time", "survival"]
        ci = kmf.confidence_interval_.reset_index()
        ci.columns = ["time", "ci_lower", "ci_upper"]
        tmp = sf.merge(ci, on="time", how="left")
        tmp.insert(0, "group", str(grp))
        tmp["n_group"] = int(g.shape[0])
        tmp["n_events"] = int(g[event_col].sum())
        rows.append(tmp)

    pval = None
    if set(work[group_col].astype(str).unique()) >= {"Low", "High"}:
        low = work[work[group_col].astype(str) == "Low"]
        high = work[work[group_col].astype(str) == "High"]
        if not low.empty and not high.empty:
            try:
                lr = logrank_test(low[time_col], high[time_col], event_observed_A=low[event_col], event_observed_B=high[event_col])
                pval = float(lr.p_value)
            except Exception:
                pval = None

    out = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()
    return out, pval


def plot_km_aggregated(km_points: pd.DataFrame,
                       outpath: str,
                       title: str,
                       pval: Optional[float] = None,
                       figsize=(6, 4.8),
                       low_color="#4c78a8",
                       high_color="#e45756",
                       show_ci: bool = True):
    if km_points is None or km_points.empty:
        return

    plt.figure(figsize=figsize)
    handles = []
    labels = []
    color_map = {"Low": low_color, "High": high_color}

    for grp in ["Low", "High"]:
        g = km_points[km_points["group"].astype(str) == grp].copy()
        if g.empty:
            continue
        c = color_map.get(grp, None)
        plt.plot(g["time"], g["survival"], color=c, linewidth=1.8)
        if show_ci and {"ci_lower", "ci_upper"}.issubset(g.columns):
            plt.fill_between(g["time"], g["ci_lower"], g["ci_upper"], color=c, alpha=0.15)
        n_group = int(g["n_group"].iloc[0]) if "n_group" in g.columns else np.nan
        n_events = int(g["n_events"].iloc[0]) if "n_events" in g.columns else np.nan
        handles.append(mlines.Line2D([], [], color=c, linewidth=1.8))
        labels.append(f"{grp} (n={n_group}, events={n_events})")

    ttl = title
    if pval is not None and np.isfinite(pval):
        ttl += f"\nLog-rank p={pval:.2e}"
    plt.title(ttl)
    plt.xlabel("Time to event (months)")
    plt.ylabel("Survival probability")
    plt.ylim(0, 1.0)
    plt.legend(handles, labels, frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=210)
    plt.close()


def compute_iteration_km_band(pred_df: pd.DataFrame,
                              time_grid: Optional[np.ndarray] = None,
                              score_col: str = "predicted_proba") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not HAVE_LIFELINES:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    work = pred_df.copy()
    work["time_to_event"] = pd.to_numeric(work["time_to_event"], errors="coerce")
    work["y_true"] = pd.to_numeric(work["y_true"], errors="coerce")
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.dropna(subset=["iteration_id", "time_to_event", "y_true", score_col]).copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    work["y_true"] = work["y_true"].astype(int)

    if time_grid is None:
        max_time = float(work["time_to_event"].max()) if work["time_to_event"].notna().any() else 1.0
        time_grid = np.linspace(0.0, max_time, 200)

    curve_rows = []
    stat_rows = []
    kmf = KaplanMeierFitter()

    for iteration_id, g in work.groupby("iteration_id"):
        cutoff = float(pd.to_numeric(g[score_col], errors="coerce").median())
        g = g.copy()
        g["km_group"] = np.where(g[score_col] >= cutoff, "High", "Low")

        if set(g["km_group"].astype(str).unique()) >= {"Low", "High"}:
            try:
                low = g[g["km_group"] == "Low"]
                high = g[g["km_group"] == "High"]
                lr = logrank_test(low["time_to_event"], high["time_to_event"], event_observed_A=low["y_true"], event_observed_B=high["y_true"])
                stat_rows.append({
                    "iteration_id": iteration_id,
                    "km_cutoff": cutoff,
                    "n_samples": int(g.shape[0]),
                    "n_events": int(g["y_true"].sum()),
                    "logrank_p": float(lr.p_value),
                    "is_sig_p_lt_0_05": bool(float(lr.p_value) < 0.05),
                })
            except Exception:
                stat_rows.append({
                    "iteration_id": iteration_id,
                    "km_cutoff": cutoff,
                    "n_samples": int(g.shape[0]),
                    "n_events": int(g["y_true"].sum()),
                    "logrank_p": np.nan,
                    "is_sig_p_lt_0_05": False,
                })
        else:
            stat_rows.append({
                "iteration_id": iteration_id,
                "km_cutoff": cutoff,
                "n_samples": int(g.shape[0]),
                "n_events": int(g["y_true"].sum()),
                "logrank_p": np.nan,
                "is_sig_p_lt_0_05": False,
            })

        for grp in ["Low", "High"]:
            gg = g[g["km_group"] == grp]
            if gg.empty:
                continue
            try:
                kmf.fit(gg["time_to_event"], event_observed=gg["y_true"])
                surv = kmf.predict(time_grid)
                for t, s in zip(time_grid, surv):
                    curve_rows.append({
                        "iteration_id": iteration_id,
                        "group": grp,
                        "time_grid": float(t),
                        "survival": float(s),
                    })
            except Exception as e:
                print(f"[WARN] Iteration KM failed for {iteration_id} / {grp}: {e}")

    curve_df = pd.DataFrame(curve_rows)
    stat_df = pd.DataFrame(stat_rows)
    if curve_df.empty:
        return curve_df, stat_df, pd.DataFrame()

    summary = (curve_df.groupby(["group", "time_grid"], as_index=False)["survival"]
               .agg(
                   survival_mean="mean",
                   survival_median="median",
                   survival_sd="std",
                   survival_p2_5=lambda s: s.quantile(0.025),
                   survival_p97_5=lambda s: s.quantile(0.975),
               ))
    summary["survival_sd"] = summary["survival_sd"].fillna(0.0)
    return curve_df, stat_df, summary


def plot_km_iteration_band(km_summary: pd.DataFrame,
                           outpath: str,
                           title: str,
                           stats_df: Optional[pd.DataFrame] = None,
                           figsize=(6, 4.8),
                           low_color="#4c78a8",
                           high_color="#e45756"):
    if km_summary is None or km_summary.empty:
        return

    plt.figure(figsize=figsize)
    color_map = {"Low": low_color, "High": high_color}

    for grp in ["Low", "High"]:
        g = km_summary[km_summary["group"].astype(str) == grp].copy()
        if g.empty:
            continue
        c = color_map.get(grp, None)
        plt.fill_between(g["time_grid"], g["survival_p2_5"], g["survival_p97_5"], color=c, alpha=0.16)
        plt.plot(g["time_grid"], g["survival_median"], color=c, linewidth=1.9, label=f"{grp} median across iterations")

    subtitle = ""
    if stats_df is not None and not stats_df.empty and "is_sig_p_lt_0_05" in stats_df.columns:
        n_sig = int(stats_df["is_sig_p_lt_0_05"].fillna(False).sum())
        n_tot = int(stats_df.shape[0])
        subtitle = f"\nIteration KM stability band ({n_sig}/{n_tot} iterations with log-rank p<0.05)"
    plt.title(f"{title}{subtitle}")
    plt.xlabel("Time to event (months)")
    plt.ylabel("Survival probability")
    plt.ylim(0, 1.0)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=210)
    plt.close()


# -------------------------
# Cox analysis
# -------------------------
def _prepare_cox_df(df: pd.DataFrame, score_col: str, covariates: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    work = df.copy()
    work["time_to_event"] = pd.to_numeric(work["time_to_event"], errors="coerce")
    work["event"] = pd.to_numeric(work["y_true"], errors="coerce")
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")

    keep = ["time_to_event", "event", score_col] + [c for c in covariates if c in work.columns]
    work = work[keep].dropna(subset=["time_to_event", "event", score_col]).copy()
    if work.empty:
        return work, []
    work["event"] = work["event"].astype(int)

    usable_covs = []
    for c in covariates:
        if c not in work.columns:
            continue
        work[c] = pd.to_numeric(work[c], errors="coerce")
        if work[c].notna().sum() < 5:
            continue
        if work[c].dropna().nunique() < 2:
            continue
        usable_covs.append(c)

    need = ["time_to_event", "event", score_col] + usable_covs
    work = work[need].dropna().copy()
    return work, usable_covs


def fit_single_cox(df: pd.DataFrame,
                   score_col: str,
                   analysis: str,
                   covariates: List[str],
                   label: str,
                   iteration_id: Optional[str] = None) -> Optional[Dict]:
    if not HAVE_LIFELINES:
        return None

    work, usable_covs = _prepare_cox_df(df, score_col=score_col, covariates=covariates if analysis == "multivariable" else [])
    if work.shape[0] < 10 or work["event"].sum() < 3:
        return None

    cols = ["time_to_event", "event", score_col] + usable_covs
    cph = CoxPHFitter()
    try:
        cph.fit(work[cols], duration_col="time_to_event", event_col="event", robust=True)
    except Exception as e:
        print(f"[WARN] Cox failed for {label} / {analysis} / {iteration_id}: {e}")
        return None

    if score_col not in cph.summary.index:
        return None
    s = cph.summary.loc[score_col]
    p = float(s["p"])
    hr = float(s["exp(coef)"])
    ci_low = float(s["exp(coef) lower 95%"])
    ci_high = float(s["exp(coef) upper 95%"])
    return {
        "label": label,
        "iteration_id": iteration_id if iteration_id is not None else "aggregated",
        "analysis": analysis,
        "score_col": score_col,
        "n_used": int(work.shape[0]),
        "n_events": int(work["event"].sum()),
        "coef": float(s["coef"]),
        "se": float(s["se(coef)"]) if "se(coef)" in s.index else np.nan,
        "z": float(s["z"]) if "z" in s.index else np.nan,
        "hr": hr,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p": p,
        "is_sig_p_lt_0_05": bool(p < 0.05),
        "ci_excludes_1": bool((ci_low > 1.0) or (ci_high < 1.0)),
        "hr_gt1": bool(hr > 1.0),
        "covariates_used": ",".join(usable_covs),
    }


def compute_iteration_cox(pred_df: pd.DataFrame, covariates: List[str]) -> pd.DataFrame:
    rows = []
    for iteration_id, g in pred_df.groupby("iteration_id"):
        uni = fit_single_cox(g, score_col="predicted_proba", analysis="univariable", covariates=[], label=str(g["block"].iloc[0]), iteration_id=iteration_id)
        multi = fit_single_cox(g, score_col="predicted_proba", analysis="multivariable", covariates=covariates, label=str(g["block"].iloc[0]), iteration_id=iteration_id)
        if uni is not None:
            rows.append(uni)
        if multi is not None:
            rows.append(multi)
    return pd.DataFrame(rows)


def summarize_iteration_cox(cox_iter_df: pd.DataFrame) -> pd.DataFrame:
    if cox_iter_df is None or cox_iter_df.empty:
        return pd.DataFrame(columns=[
            "analysis", "n_iterations", "hr_mean", "hr_sd", "hr_median", "hr_p2_5", "hr_p97_5",
            "n_sig_p_lt_0_05", "frac_sig_p_lt_0_05", "n_ci_excludes_1", "frac_ci_excludes_1", "n_hr_gt1", "frac_hr_gt1"
        ])

    rows = []
    for analysis, g in cox_iter_df.groupby("analysis"):
        n = int(g.shape[0])
        rows.append({
            "analysis": analysis,
            "n_iterations": n,
            "hr_mean": float(g["hr"].mean()),
            "hr_sd": float(g["hr"].std(ddof=1)) if n > 1 else 0.0,
            "hr_median": float(g["hr"].median()),
            "hr_p2_5": float(g["hr"].quantile(0.025)),
            "hr_p97_5": float(g["hr"].quantile(0.975)),
            "n_sig_p_lt_0_05": int(g["is_sig_p_lt_0_05"].fillna(False).sum()),
            "frac_sig_p_lt_0_05": float(g["is_sig_p_lt_0_05"].fillna(False).mean()),
            "n_ci_excludes_1": int(g["ci_excludes_1"].fillna(False).sum()),
            "frac_ci_excludes_1": float(g["ci_excludes_1"].fillna(False).mean()),
            "n_hr_gt1": int(g["hr_gt1"].fillna(False).sum()),
            "frac_hr_gt1": float(g["hr_gt1"].fillna(False).mean()),
        })
    return pd.DataFrame(rows)


def plot_cox_aggregated_forest(cox_agg_df: pd.DataFrame,
                               outpath: str,
                               title: str,
                               figsize=(7.0, 3.8),
                               colors: Optional[Dict[str, str]] = None):
    if cox_agg_df is None or cox_agg_df.empty:
        return
    if colors is None:
        colors = {"univariable": "#4c78a8", "multivariable": "#e45756"}

    plot_df = cox_agg_df.copy()
    order = [a for a in ["univariable", "multivariable"] if a in plot_df["analysis"].tolist()]
    if not order:
        order = plot_df["analysis"].tolist()

    plt.figure(figsize=figsize)
    ypos = {analysis: i for i, analysis in enumerate(order[::-1])}
    for _, r in plot_df.iterrows():
        y = ypos[r["analysis"]]
        c = colors.get(r["analysis"], "black")
        plt.plot([r["ci_low"], r["ci_high"]], [y, y], color=c, linewidth=1.6)
        plt.scatter(r["hr"], y, color=c, s=38)
    plt.axvline(1.0, linestyle="--", color="gray", linewidth=1.0)
    plt.xscale("log")
    plt.yticks([ypos[a] for a in order[::-1]], [a.capitalize() for a in order[::-1]])
    plt.xlabel("Hazard ratio (log scale)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=210)
    plt.close()


def make_iteration_forest_ready(cox_iter_df: pd.DataFrame, cox_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if cox_iter_df is not None and not cox_iter_df.empty:
        for analysis, g in cox_iter_df.groupby("analysis"):
            gg = g.copy()
            def _iter_key(x):
                m = re.search(r"(\d+)$", str(x))
                return int(m.group(1)) if m else str(x)
            gg = gg.sort_values(by="iteration_id", key=lambda s: s.map(_iter_key))
            for i, (_, r) in enumerate(gg.iterrows()):
                rows.append({
                    "analysis": analysis,
                    "row_kind": "iteration",
                    "row_label": str(r["iteration_id"]),
                    "row_order": i,
                    "hr": float(r["hr"]),
                    "ci_low": float(r["ci_low"]),
                    "ci_high": float(r["ci_high"]),
                    "p": float(r["p"]),
                    "is_sig_p_lt_0_05": bool(r["is_sig_p_lt_0_05"]),
                })
    if cox_summary_df is not None and not cox_summary_df.empty:
        base_orders = {a: max([-1] + [int(r["row_order"]) for r in rows if r["analysis"] == a]) + 1 for a in cox_summary_df["analysis"]}
        for _, r in cox_summary_df.iterrows():
            rows.append({
                "analysis": r["analysis"],
                "row_kind": "summary_median",
                "row_label": "Median across iterations",
                "row_order": base_orders.get(r["analysis"], 0),
                "hr": float(r["hr_median"]),
                "ci_low": float(r["hr_p2_5"]),
                "ci_high": float(r["hr_p97_5"]),
                "p": np.nan,
                "is_sig_p_lt_0_05": np.nan,
            })
    return pd.DataFrame(rows)


def plot_iteration_hr_forest(forest_df: pd.DataFrame,
                             summary_row: Optional[pd.Series],
                             outpath: str,
                             title: str,
                             figsize=(7.5, 6.5),
                             color="#4c78a8",
                             summary_color="#111111"):
    if forest_df is None or forest_df.empty:
        return

    plot_df = forest_df.copy().sort_values(["row_order", "row_kind"], ascending=[False, True])
    nrows = int(plot_df.shape[0])
    if figsize is None:
        figsize = (7.5, max(4.0, 0.28 * nrows + 1.6))
    else:
        figsize = (figsize[0], max(float(figsize[1]), 0.28 * nrows + 1.6))
    plt.figure(figsize=figsize)

    for y, (_, r) in enumerate(plot_df.iterrows()):
        c = summary_color if r["row_kind"] == "summary_median" else color
        lw = 2.0 if r["row_kind"] == "summary_median" else 1.2
        size = 50 if r["row_kind"] == "summary_median" else 28
        plt.plot([r["ci_low"], r["ci_high"]], [y, y], color=c, linewidth=lw)
        plt.scatter(r["hr"], y, color=c, s=size, zorder=3)

    plt.axvline(1.0, linestyle="--", color="gray", linewidth=1.0)
    plt.xscale("log")
    plt.yticks(range(plot_df.shape[0]), plot_df["row_label"].tolist())

    subtitle = ""
    if summary_row is not None and isinstance(summary_row, pd.Series):
        subtitle = (
            f"\nMedian HR={summary_row['hr_median']:.2f} "
            f"(95% PI {summary_row['hr_p2_5']:.2f}-{summary_row['hr_p97_5']:.2f}); "
            f"sig in {int(summary_row['n_sig_p_lt_0_05'])}/{int(summary_row['n_iterations'])} iterations"
        )
    plt.title(f"{title}{subtitle}")
    plt.xlabel("Hazard ratio (log scale)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=210)
    plt.close()


def plot_iteration_hr_box(cox_iter_df: pd.DataFrame,
                          outpath: str,
                          title: str,
                          figsize=(5.8, 4.3)):
    if cox_iter_df is None or cox_iter_df.empty:
        return

    vals = pd.to_numeric(cox_iter_df["hr"], errors="coerce").dropna().values
    if len(vals) == 0:
        return

    plt.figure(figsize=figsize)
    plt.boxplot(vals, vert=False, showfliers=False)
    y = np.random.normal(loc=1.0, scale=0.04, size=len(vals))
    plt.scatter(vals, y, s=20, alpha=0.7)
    plt.axvline(1.0, linestyle="--", color="gray", linewidth=1.0)
    plt.xscale("log")
    plt.yticks([1], ["HR across iterations"])
    plt.xlabel("Hazard ratio (log scale)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=210)
    plt.close()


def _block_sort_key(block: str) -> Tuple[int, str]:
    block_u = str(block).upper()
    try:
        return (VALID_BLOCKS.index(block_u), block_u)
    except ValueError:
        return (len(VALID_BLOCKS), block_u)


def build_combined_iteration_summary_ready(cox_summary_df: pd.DataFrame, analysis: str) -> pd.DataFrame:
    if cox_summary_df is None or cox_summary_df.empty:
        return pd.DataFrame()

    rows = []
    for order, block in enumerate(sorted(cox_summary_df["block"].dropna().astype(str).unique().tolist(), key=_block_sort_key)):
        sr = cox_summary_df[(cox_summary_df["block"].astype(str) == block) & (cox_summary_df["analysis"] == analysis)].copy()
        if sr.empty:
            continue
        srow = sr.iloc[0]
        rows.append({
            "block": block,
            "analysis": analysis,
            "row_label": block,
            "row_order": order,
            "hr": float(srow["hr_median"]),
            "ci_low": float(srow["hr_p2_5"]),
            "ci_high": float(srow["hr_p97_5"]),
            "n_iterations": int(srow["n_iterations"]),
            "n_sig_p_lt_0_05": int(srow["n_sig_p_lt_0_05"]),
            "frac_sig_p_lt_0_05": float(srow["frac_sig_p_lt_0_05"]),
            "n_ci_excludes_1": int(srow.get("n_ci_excludes_1", 0)),
            "frac_ci_excludes_1": float(srow.get("frac_ci_excludes_1", np.nan)),
            "n_hr_gt1": int(srow.get("n_hr_gt1", 0)),
            "frac_hr_gt1": float(srow.get("frac_hr_gt1", np.nan)),
        })
    return pd.DataFrame(rows)


def plot_combined_iteration_hr_forest(summary_ready_df: pd.DataFrame,
                                      outpath: str,
                                      title: str,
                                      figsize=(8.8, 5.8),
                                      block_colors: Optional[Dict[str, str]] = None,
                                      row_height: float = 0.65,
                                      height_padding: float = 1.5,
                                      min_height: float = 0.0):
    if summary_ready_df is None or summary_ready_df.empty:
        return

    plot_df = summary_ready_df.copy().sort_values("row_order", ascending=False).reset_index(drop=True)
    nrows = int(plot_df.shape[0])
    fig_h = max(float(min_height), float(row_height) * max(nrows, 1) + float(height_padding))
    fig, ax = plt.subplots(figsize=(figsize[0], fig_h))

    if block_colors is None:
        block_colors = {}

    y_positions = np.arange(nrows)[::-1]
    for y, (_, r) in zip(y_positions, plot_df.iterrows()):
        block = str(r["block"])
        color = block_colors.get(block, "#4c78a8")
        ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=color, linewidth=2.0)
        ax.scatter(r["hr"], y, color=color, s=46, zorder=3)
        label = f"sig {int(r['n_sig_p_lt_0_05'])}/{int(r['n_iterations'])} ({float(r['frac_sig_p_lt_0_05']):.0%})"
        ax.text(1.01, y, label, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8)

    ax.axvline(1.0, linestyle="--", color="gray", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["row_label"].tolist())
    ax.set_ylim(-0.5, max(nrows - 1, 0) + 0.5)
    ax.set_xlabel("Hazard ratio (log scale)")
    ax.set_title(title)
    fig.subplots_adjust(right=0.82, left=0.22)
    fig.savefig(outpath, dpi=210)
    plt.close(fig)


def plot_combined_iteration_hr_box(cox_iter_df: pd.DataFrame,
                                   cox_summary_df: pd.DataFrame,
                                   outpath: str,
                                   title: str,
                                   figsize=(8.0, 4.8),
                                   block_colors: Optional[Dict[str, str]] = None,
                                   summary_color="#111111",
                                   row_height: float = 0.65,
                                   height_padding: float = 1.6,
                                   min_height: float = 0.0,
                                   box_width: float = 0.34,
                                   jitter_sd: float = 0.05):
    if cox_iter_df is None or cox_iter_df.empty:
        return

    if block_colors is None:
        block_colors = {}

    blocks = sorted(cox_iter_df["block"].dropna().astype(str).unique().tolist(), key=_block_sort_key)
    if not blocks:
        return

    fig_h = max(float(min_height), float(row_height) * max(len(blocks), 1) + float(height_padding))
    fig, ax = plt.subplots(figsize=(figsize[0], fig_h))
    summary_lookup = {}
    if cox_summary_df is not None and not cox_summary_df.empty:
        for _, r in cox_summary_df.iterrows():
            summary_lookup[(str(r["block"]), str(r["analysis"]))] = r

    positions = np.arange(len(blocks))[::-1]
    data = []
    analysis_name = str(cox_iter_df["analysis"].iloc[0]) if "analysis" in cox_iter_df.columns and not cox_iter_df.empty else ""
    for block in blocks:
        vals = pd.to_numeric(cox_iter_df.loc[cox_iter_df["block"].astype(str) == block, "hr"], errors="coerce").dropna().values
        data.append(vals)

    bp = ax.boxplot(data, vert=False, positions=positions, showfliers=False, patch_artist=True, widths=box_width)
    for patch, block in zip(bp["boxes"], blocks):
        patch.set_facecolor(block_colors.get(block, "#4c78a8"))
        patch.set_alpha(0.35)
        patch.set_edgecolor(block_colors.get(block, "#4c78a8"))
    for med in bp["medians"]:
        med.set_color(summary_color)
        med.set_linewidth(1.6)
    for whisk in bp["whiskers"]:
        whisk.set_color("#888888")
    for cap in bp["caps"]:
        cap.set_color("#888888")

    rng = np.random.default_rng(42)
    for pos, block, vals in zip(positions, blocks, data):
        if len(vals) == 0:
            continue
        y = pos + rng.normal(0.0, jitter_sd, size=len(vals))
        ax.scatter(vals, y, s=16, alpha=0.55, color=block_colors.get(block, "#4c78a8"), zorder=3)
        sr = summary_lookup.get((block, analysis_name))
        if sr is not None:
            ax.plot([float(sr["hr_p2_5"]), float(sr["hr_p97_5"])], [pos, pos], color=summary_color, linewidth=2.0, zorder=4)
            ax.scatter(float(sr["hr_median"]), pos, color=summary_color, s=34, zorder=5)
            label = f"sig {int(sr['n_sig_p_lt_0_05'])}/{int(sr['n_iterations'])} ({float(sr['frac_sig_p_lt_0_05']):.0%})"
            ax.text(1.01, pos, label, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8)

    ax.axvline(1.0, linestyle="--", color="gray", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yticks(positions)
    ax.set_yticklabels(blocks)
    ax.set_ylim(-0.5, max(len(blocks) - 1, 0) + 0.5)
    ax.set_xlabel("Hazard ratio (log scale)")
    ax.set_title(title)
    fig.subplots_adjust(right=0.82, left=0.20)
    fig.savefig(outpath, dpi=210)
    plt.close(fig)


def plot_combined_aggregated_cox_forest(cox_agg_df: pd.DataFrame,
                                        cox_iter_summary_df: Optional[pd.DataFrame],
                                        outpath: str,
                                        title: str,
                                        figsize=(8.4, 5.0),
                                        block_colors: Optional[Dict[str, str]] = None,
                                        row_height: float = 0.65,
                                        height_padding: float = 1.4,
                                        min_height: float = 0.0):
    if cox_agg_df is None or cox_agg_df.empty:
        return

    plot_df = cox_agg_df.copy()
    if "block" not in plot_df.columns:
        return
    plot_df["_sort"] = plot_df["block"].map(lambda x: _block_sort_key(x))
    plot_df = plot_df.sort_values("_sort", ascending=False).reset_index(drop=True)

    if block_colors is None:
        block_colors = {}

    nrows = int(plot_df.shape[0])
    fig_h = max(float(min_height), float(row_height) * max(nrows, 1) + float(height_padding))
    fig, ax = plt.subplots(figsize=(figsize[0], fig_h))

    iter_lookup = {}
    if cox_iter_summary_df is not None and not cox_iter_summary_df.empty:
        for _, r in cox_iter_summary_df.iterrows():
            iter_lookup[(str(r["block"]), str(r["analysis"]))] = r

    y_positions = np.arange(nrows)[::-1]
    labels = []
    for y, (_, r) in zip(y_positions, plot_df.iterrows()):
        block = str(r["block"])
        analysis = str(r["analysis"])
        color = block_colors.get(block, "#4c78a8")
        ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=color, linewidth=2.0)
        ax.scatter(r["hr"], y, color=color, s=46, zorder=3)
        labels.append(f"{block} ({analysis[:5]})")
        sr = iter_lookup.get((block, analysis))
        if sr is not None:
            sig_label = f"sig {int(sr['n_sig_p_lt_0_05'])}/{int(sr['n_iterations'])} ({float(sr['frac_sig_p_lt_0_05']):.0%})"
            ax.text(1.01, y, sig_label, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8)

    ax.axvline(1.0, linestyle="--", color="gray", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.5, max(nrows - 1, 0) + 0.5)
    ax.set_xlabel("Hazard ratio (log scale)")
    ax.set_title(title)
    fig.subplots_adjust(right=0.82, left=0.24)
    fig.savefig(outpath, dpi=210)
    plt.close(fig)


# -------------------------
# Main per-block analysis
# -------------------------
def analyze_block(block: str,
                  pred_df: pd.DataFrame,
                  covariates: List[str],
                  tables_dir: str,
                  plots_dir: str,
                  roc_figsize=(6, 6),
                  km_figsize=(6, 4.8),
                  forest_figsize=(7.5, 6.5),
                  roc_color="#1f77b4",
                  km_low_color="#4c78a8",
                  km_high_color="#e45756",
                  do_iteration_km_band: bool = True) -> Dict[str, pd.DataFrame]:
    out = {}
    block_slug = _slug(block)

    if pred_df.empty:
        return out

    # raw long predictions
    pred_path = os.path.join(tables_dir, f"oof_predictions_long__{block_slug}.csv")
    _save_csv(pred_df, pred_path)
    out["oof_predictions_long"] = pred_df

    # model counts if available
    model_counts = (pred_df.groupby(["block", "model_name"], as_index=False)
                    .size()
                    .rename(columns={"size": "n_rows"})
                    .sort_values(["n_rows", "model_name"], ascending=[False, True]))
    _save_csv(model_counts, os.path.join(tables_dir, f"model_counts__{block_slug}.csv"))
    out["model_counts"] = model_counts

    # ROC
    roc_by_iter, roc_summary, roc_raw, roc_interp, roc_curve_summary = compute_iteration_roc_tables(pred_df)
    roc_by_iter.insert(0, "block", block)
    roc_summary.insert(0, "block", block)
    roc_raw.insert(0, "block", block)
    roc_interp.insert(0, "block", block)
    roc_curve_summary.insert(0, "block", block)

    _save_csv(roc_by_iter, os.path.join(tables_dir, f"roc_by_iteration__{block_slug}.csv"))
    _save_csv(roc_summary, os.path.join(tables_dir, f"roc_summary__{block_slug}.csv"))
    _save_csv(roc_raw, os.path.join(tables_dir, f"roc_curve_points__{block_slug}.csv"))
    _save_csv(roc_interp, os.path.join(tables_dir, f"roc_curve_interpolated__{block_slug}.csv"))
    _save_csv(roc_curve_summary, os.path.join(tables_dir, f"roc_curve_summary__{block_slug}.csv"))

    plot_auc_boxplot(
        roc_by_iter,
        os.path.join(plots_dir, f"auc_boxplot__{block_slug}.png"),
        title=f"{block} — AUC across iterations",
        figsize=(5.6, 4.5),
    )
    plot_mean_roc(
        roc_curve_summary,
        roc_interp,
        os.path.join(plots_dir, f"roc_mean_ci__{block_slug}.png"),
        title=f"{block}",
        auc_summary=roc_summary,
        figsize=roc_figsize,
        mean_color=roc_color,
        band_color=roc_color,
        show_iteration_curves=True,
    )

    out["roc_by_iteration"] = roc_by_iter
    out["roc_summary"] = roc_summary
    out["roc_curve_summary"] = roc_curve_summary

    # sample aggregation
    agg_df = aggregate_sample_predictions(pred_df, covariates=covariates)
    if not agg_df.empty:
        agg_df.insert(0, "block", block)
    _save_csv(agg_df, os.path.join(tables_dir, f"sample_aggregated_predictions__{block_slug}.csv"))
    out["sample_aggregated_predictions"] = agg_df

    # aggregated KM
    km_group_df, km_cutoff = make_km_groups_from_aggregated(agg_df, score_col="pred_median") if not agg_df.empty else (pd.DataFrame(), np.nan)
    if not km_group_df.empty and "block" not in km_group_df.columns:
        km_group_df.insert(0, "block", block)
    _save_csv(km_group_df, os.path.join(tables_dir, f"km_groups__aggregated__{block_slug}.csv"))
    out["km_groups_aggregated"] = km_group_df

    km_points, km_pval = compute_km_points(km_group_df) if not km_group_df.empty else (pd.DataFrame(), None)
    if not km_points.empty:
        km_points.insert(0, "block", block)
    _save_csv(km_points, os.path.join(tables_dir, f"km_curve_points__aggregated__{block_slug}.csv"))
    out["km_curve_points_aggregated"] = km_points

    plot_km_aggregated(
        km_points,
        os.path.join(plots_dir, f"km_aggregated__{block_slug}.png"),
        title=f"{block} — aggregated median prediction",
        pval=km_pval,
        figsize=km_figsize,
        low_color=km_low_color,
        high_color=km_high_color,
        show_ci=True,
    )

    # iteration KM band
    km_iter_curves, km_iter_stats, km_iter_summary = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    if do_iteration_km_band:
        km_iter_curves, km_iter_stats, km_iter_summary = compute_iteration_km_band(pred_df)
        if not km_iter_curves.empty:
            km_iter_curves.insert(0, "block", block)
        if not km_iter_stats.empty:
            km_iter_stats.insert(0, "block", block)
        if not km_iter_summary.empty:
            km_iter_summary.insert(0, "block", block)
        _save_csv(km_iter_curves, os.path.join(tables_dir, f"km_curve_points_by_iteration__{block_slug}.csv"))
        _save_csv(km_iter_stats, os.path.join(tables_dir, f"km_by_iteration_stats__{block_slug}.csv"))
        _save_csv(km_iter_summary, os.path.join(tables_dir, f"km_curve_summary_by_iteration__{block_slug}.csv"))
        plot_km_iteration_band(
            km_iter_summary,
            os.path.join(plots_dir, f"km_iteration_band__{block_slug}.png"),
            title=f"{block}",
            stats_df=km_iter_stats,
            figsize=km_figsize,
            low_color=km_low_color,
            high_color=km_high_color,
        )
    out["km_curve_points_by_iteration"] = km_iter_curves
    out["km_by_iteration_stats"] = km_iter_stats
    out["km_curve_summary_by_iteration"] = km_iter_summary

    # aggregated Cox using patient-level median prediction
    cox_agg_rows = []
    uni_agg = fit_single_cox(agg_df, score_col="pred_median", analysis="univariable", covariates=[], label=block, iteration_id="aggregated") if not agg_df.empty else None
    multi_agg = fit_single_cox(agg_df, score_col="pred_median", analysis="multivariable", covariates=covariates, label=block, iteration_id="aggregated") if not agg_df.empty else None
    if uni_agg is not None:
        cox_agg_rows.append(uni_agg)
    if multi_agg is not None:
        cox_agg_rows.append(multi_agg)
    cox_agg_df = pd.DataFrame(cox_agg_rows)
    if not cox_agg_df.empty:
        cox_agg_df.insert(0, "block", block)
    _save_csv(cox_agg_df, os.path.join(tables_dir, f"cox_aggregated__{block_slug}.csv"))
    out["cox_aggregated"] = cox_agg_df

    if not cox_agg_df.empty:
        plot_cox_aggregated_forest(
            cox_agg_df,
            os.path.join(plots_dir, f"cox_aggregated_forest__{block_slug}.png"),
            title=f"{block} — aggregated patient-level Cox",
            figsize=(7.0, 3.8),
            colors={"univariable": km_low_color, "multivariable": km_high_color},
        )

    # iteration Cox
    cox_iter_df = compute_iteration_cox(pred_df, covariates=covariates)
    if not cox_iter_df.empty:
        cox_iter_df.insert(0, "block", block)
    _save_csv(cox_iter_df, os.path.join(tables_dir, f"cox_by_iteration__{block_slug}.csv"))
    out["cox_by_iteration"] = cox_iter_df

    cox_iter_summary = summarize_iteration_cox(cox_iter_df)
    if not cox_iter_summary.empty:
        cox_iter_summary.insert(0, "block", block)
    _save_csv(cox_iter_summary, os.path.join(tables_dir, f"cox_iteration_summary__{block_slug}.csv"))
    out["cox_iteration_summary"] = cox_iter_summary

    forest_ready = make_iteration_forest_ready(cox_iter_df, cox_iter_summary)
    if not forest_ready.empty:
        forest_ready.insert(0, "block", block)
    _save_csv(forest_ready, os.path.join(tables_dir, f"forest_ready_hr_by_iteration__{block_slug}.csv"))
    out["forest_ready_hr_by_iteration"] = forest_ready

    # Combined cross-model iteration HR plots are generated later from the concatenated all-block tables.

    # manifest for this block
    manifest = pd.DataFrame([{
        "block": block,
        "n_prediction_rows": int(pred_df.shape[0]),
        "n_unique_samples": int(pred_df["sample_id"].nunique()),
        "n_unique_iterations": int(pred_df["iteration_id"].nunique()),
        "model_name_selected": str(pred_df["model_name"].dropna().astype(str).mode().iloc[0]) if pred_df["model_name"].dropna().size else "",
        "km_aggregation_score": "pred_median",
        "cox_aggregation_score": "pred_median",
    }])
    _save_csv(manifest, os.path.join(tables_dir, f"analysis_manifest__{block_slug}.csv"))
    out["analysis_manifest"] = manifest

    return out


# -------------------------
# CLI / main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Iteration-aware post-hoc analysis for repeated CV outputs.")
    ap.add_argument("--output_dir", required=True, help="Root output dir used in training.")
    ap.add_argument("--subtype", required=True, help="Subtype folder under output_dir.")
    ap.add_argument("--meta_path", required=True, help="Path to full metadata CSV; will be subset to the requested subtype.")
    ap.add_argument("--subtype_col", default="auto", help="Metadata subtype column. Default: auto")
    ap.add_argument("--time_col", default="time_to_event", help="Metadata column to use for KM/Cox time.")
    ap.add_argument("--covariates", default="age_dx,chemo_med,hormtxt_evr", help="Comma-separated covariates for multivariable Cox.")
    ap.add_argument("--blocks", default="RNA,CNV,MUT,CLINICAL,ALL,STACK", help="Comma-separated blocks to analyze.")
    ap.add_argument("--iteration_glob", default="iter_*", help="Glob for iteration folders under <output_dir>/<subtype>.")
    ap.add_argument("--outdir", default=None, help="Output directory. Default: <output_dir>/<subtype>/posthoc_iterations")
    ap.add_argument("--skip_iteration_km_band", action="store_true", help="Disable the iteration-level KM stability band.")

    ap.add_argument("--roc_figsize", default="6,6", help="ROC figure size as width,height")
    ap.add_argument("--km_figsize", default="6,4.8", help="KM figure size as width,height")
    ap.add_argument("--forest_figsize", default="7.5,6.5", help="Iteration forest figure size as width,height")

    ap.add_argument("--finalHR", default=None,
                    help="Comma-separated block/model names to include in additional combined HR plots, e.g. RNA,CNV,STACK")
    ap.add_argument("--finalhr_row_height", type=float, default=0.65,
                    help="Fixed vertical space per model row for the additional finalHR forest/box outputs.")
    ap.add_argument("--finalhr_height_padding", type=float, default=1.6,
                    help="Extra figure height padding for the additional finalHR forest/box outputs.")
    ap.add_argument("--finalhr_min_height", type=float, default=0.0,
                    help="Minimum figure height for the additional finalHR forest/box outputs.")
    ap.add_argument("--roc_colors", default=None, help="Comma-separated block:color map for ROC plots")
    ap.add_argument("--km_colors", default="low:#4c78a8,high:#e45756", help="Comma-separated color map for KM groups")

    args = ap.parse_args()

    blocks = [b.strip().upper() for b in args.blocks.split(",") if b.strip()]
    bad = [b for b in blocks if b not in VALID_BLOCKS]
    if bad:
        raise ValueError(f"Invalid blocks: {bad}. Valid: {VALID_BLOCKS}")

    covariates = [c.strip() for c in args.covariates.split(",") if c.strip()]
    final_hr_blocks = _parse_block_list(args.finalHR)
    roc_figsize = _parse_figsize(args.roc_figsize, default=(6, 6))
    km_figsize = _parse_figsize(args.km_figsize, default=(6, 4.8))
    forest_figsize = _parse_figsize(args.forest_figsize, default=(7.5, 6.5))

    roc_color_map = _parse_color_map(
        args.roc_colors,
        default_map={
            "RNA": "#4c78a8",
            "CNV": "#f58518",
            "MUT": "#54a24b",
            "CLINICAL": "#b279a2",
            "ALL": "#8c564b",
            "STACK": "#000000",
        },
    )
    km_color_map = _parse_color_map(
        args.km_colors,
        default_map={"low": "#195EE4", "high": "#D85C2F"},
    )

    outdir = args.outdir if args.outdir else os.path.join(args.output_dir, args.subtype, "posthoc_iterations")
    tables_dir = os.path.join(outdir, "tables")
    plots_dir = os.path.join(outdir, "plots")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    meta = load_meta(args.meta_path, subtype=args.subtype, subtype_col=args.subtype_col)
    runs = discover_iteration_runs(args.output_dir, args.subtype, iteration_glob=args.iteration_glob)

    _save_json({
        "output_dir": args.output_dir,
        "subtype": args.subtype,
        "meta_path": args.meta_path,
        "subtype_col": args.subtype_col,
        "time_col": args.time_col,
        "covariates": covariates,
        "blocks": blocks,
        "finalHR": final_hr_blocks,
        "iteration_glob": args.iteration_glob,
        "n_runs": len(runs),
        "runs": runs,
    }, os.path.join(outdir, "analysis_config.json"))

    all_tables = {
        "oof_predictions_long": [],
        "model_counts": [],
        "roc_by_iteration": [],
        "roc_summary": [],
        "roc_curve_summary": [],
        "sample_aggregated_predictions": [],
        "km_groups_aggregated": [],
        "km_curve_points_aggregated": [],
        "km_curve_points_by_iteration": [],
        "km_by_iteration_stats": [],
        "km_curve_summary_by_iteration": [],
        "cox_aggregated": [],
        "cox_by_iteration": [],
        "cox_iteration_summary": [],
        "forest_ready_hr_by_iteration": [],
        "analysis_manifest": [],
    }

    for block in blocks:
        print(f"[ANALYZE] Block={block}")
        pred_df, model_counts, keep_model = gather_predictions_for_block(runs, block)
        if pred_df.empty:
            print(f"[WARN] No predictions found for block {block}; skipping.")
            continue

        pred_df = merge_with_meta(pred_df, meta, args.time_col, covariates)
        block_out = analyze_block(
            block=block,
            pred_df=pred_df,
            covariates=covariates,
            tables_dir=tables_dir,
            plots_dir=plots_dir,
            roc_figsize=roc_figsize,
            km_figsize=km_figsize,
            forest_figsize=forest_figsize,
            roc_color=roc_color_map.get(block, "#1f77b4"),
            km_low_color=km_color_map.get("low", "#4c78a8"),
            km_high_color=km_color_map.get("high", "#e45756"),
            do_iteration_km_band=(not args.skip_iteration_km_band),
        )

        if not model_counts.empty:
            all_tables["model_counts"].append(model_counts)
        for k in [
            "oof_predictions_long",
            "roc_by_iteration",
            "roc_summary",
            "roc_curve_summary",
            "sample_aggregated_predictions",
            "km_groups_aggregated",
            "km_curve_points_aggregated",
            "km_curve_points_by_iteration",
            "km_by_iteration_stats",
            "km_curve_summary_by_iteration",
            "cox_aggregated",
            "cox_by_iteration",
            "cox_iteration_summary",
            "forest_ready_hr_by_iteration",
            "analysis_manifest",
        ]:
            df = block_out.get(k, None)
            if isinstance(df, pd.DataFrame) and not df.empty:
                all_tables[k].append(df)


    def _subset_final_hr(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or not final_hr_blocks:
            return pd.DataFrame() if df is None else df.copy()
        if "block" not in df.columns:
            return pd.DataFrame()
        return df[df["block"].astype(str).str.upper().isin(final_hr_blocks)].copy()

    # combined all-block tables for easier downstream plotting
    combined_tables = {}
    for key, dfs in all_tables.items():
        if dfs:
            combo = pd.concat(dfs, axis=0, ignore_index=True)
            combined_tables[key] = combo
            _save_csv(combo, os.path.join(tables_dir, f"{key}__ALL_BLOCKS.csv"))

    # combined cross-model Cox plots
    combo_iter = combined_tables.get("cox_by_iteration", pd.DataFrame())
    combo_summary = combined_tables.get("cox_iteration_summary", pd.DataFrame())
    combo_agg = combined_tables.get("cox_aggregated", pd.DataFrame())

    if not combo_agg.empty:
        for analysis in ["univariable", "multivariable"]:
            ca = combo_agg[combo_agg["analysis"] == analysis].copy()
            cs = combo_summary[combo_summary["analysis"] == analysis].copy() if not combo_summary.empty else pd.DataFrame()
            if ca.empty:
                continue
            _save_csv(ca, os.path.join(tables_dir, f"cox_aggregated__ALL_MODELS__{analysis}.csv"))
            plot_combined_aggregated_cox_forest(
                ca,
                cs,
                os.path.join(plots_dir, f"cox_aggregated_forest__ALL_MODELS__{analysis}.png"),
                title=f"All models — aggregated patient-level {analysis} Cox",
                figsize=forest_figsize,
                block_colors=roc_color_map,
            )

    if not combo_iter.empty and not combo_summary.empty:
        for analysis in ["univariable", "multivariable"]:
            ci = combo_iter[combo_iter["analysis"] == analysis].copy()
            cs = combo_summary[combo_summary["analysis"] == analysis].copy()
            if ci.empty or cs.empty:
                continue
            forest_all = build_combined_iteration_summary_ready(cs, analysis=analysis)
            if not forest_all.empty:
                _save_csv(forest_all, os.path.join(tables_dir, f"forest_ready_hr_summary__ALL_MODELS__{analysis}.csv"))
                plot_combined_iteration_hr_forest(
                    forest_all,
                    os.path.join(plots_dir, f"cox_iteration_forest__ALL_MODELS__{analysis}.png"),
                    title=f"All models — {analysis} median HR across iterations",
                    figsize=forest_figsize,
                    block_colors=roc_color_map,
                    row_height=args.finalhr_row_height,
                    height_padding=args.finalhr_height_padding,
                    min_height=args.finalhr_min_height,
                )
                plot_combined_iteration_hr_box(
                    ci,
                    cs,
                    os.path.join(plots_dir, f"cox_iteration_box__ALL_MODELS__{analysis}.png"),
                    title=f"All models — {analysis} HR distribution across iterations",
                    figsize=(8.0, forest_figsize[1]),
                    block_colors=roc_color_map,
                    summary_color="#111111",
                    row_height=args.finalhr_row_height,
                    height_padding=args.finalhr_height_padding,
                    min_height=args.finalhr_min_height,
                )

    if final_hr_blocks:
        combo_iter_final = _subset_final_hr(combo_iter)
        combo_summary_final = _subset_final_hr(combo_summary)
        combo_agg_final = _subset_final_hr(combo_agg)

        if combo_agg_final is not None and not combo_agg_final.empty:
            for analysis in ["univariable", "multivariable"]:
                ca = combo_agg_final[combo_agg_final["analysis"] == analysis].copy()
                cs = combo_summary_final[combo_summary_final["analysis"] == analysis].copy() if combo_summary_final is not None and not combo_summary_final.empty else pd.DataFrame()
                if ca.empty:
                    continue
                _save_csv(ca, os.path.join(tables_dir, f"cox_aggregated__FINALHR__{analysis}.csv"))
                plot_combined_aggregated_cox_forest(
                    ca,
                    cs,
                    os.path.join(plots_dir, f"cox_aggregated_forest__FINALHR__{analysis}.png"),
                    title=f"Final HR models — aggregated patient-level {analysis} Cox",
                    figsize=forest_figsize,
                    block_colors=roc_color_map,
                    row_height=args.finalhr_row_height,
                    height_padding=args.finalhr_height_padding,
                    min_height=args.finalhr_min_height,
                )

        if combo_iter_final is not None and not combo_iter_final.empty and combo_summary_final is not None and not combo_summary_final.empty:
            for analysis in ["univariable", "multivariable"]:
                ci = combo_iter_final[combo_iter_final["analysis"] == analysis].copy()
                cs = combo_summary_final[combo_summary_final["analysis"] == analysis].copy()
                if ci.empty or cs.empty:
                    continue
                forest_final = build_combined_iteration_summary_ready(cs, analysis=analysis)
                if not forest_final.empty:
                    _save_csv(forest_final, os.path.join(tables_dir, f"forest_ready_hr_summary__FINALHR__{analysis}.csv"))
                    plot_combined_iteration_hr_forest(
                        forest_final,
                        os.path.join(plots_dir, f"cox_iteration_forest__FINALHR__{analysis}.png"),
                        title=f"Final HR models — {analysis} median HR across iterations",
                        figsize=forest_figsize,
                        block_colors=roc_color_map,
                        row_height=args.finalhr_row_height,
                        height_padding=args.finalhr_height_padding,
                        min_height=args.finalhr_min_height,
                    )
                    plot_combined_iteration_hr_box(
                        ci,
                        cs,
                        os.path.join(plots_dir, f"cox_iteration_box__FINALHR__{analysis}.png"),
                        title=f"Final HR models — {analysis} HR distribution across iterations",
                        figsize=(8.0, forest_figsize[1]),
                        block_colors=roc_color_map,
                        summary_color="#111111",
                        row_height=args.finalhr_row_height,
                        height_padding=args.finalhr_height_padding,
                        min_height=args.finalhr_min_height,
                    )

    # overall manifest
    manifest_rows = []
    if all_tables["analysis_manifest"]:
        combo = pd.concat(all_tables["analysis_manifest"], axis=0, ignore_index=True)
        for _, r in combo.iterrows():
            manifest_rows.append(r.to_dict())
    _save_csv(pd.DataFrame(manifest_rows), os.path.join(tables_dir, "analysis_manifest__ALL_BLOCKS.csv"))

    if final_hr_blocks:
        print(f"[INFO] Additional finalHR plots generated for: {final_hr_blocks}")
    print(f"✅ Done. Outputs written to: {outdir}")


if __name__ == "__main__":
    main()

