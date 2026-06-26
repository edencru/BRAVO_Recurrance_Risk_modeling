#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Outer cross-validation training and model selection script.


This script runs a leak-free nested cross-validation workflow for subtype-specific 
recurrence-risk modeling using multi-omic feature blocks. 
It is primarily intended to compare feature blocks, evaluate and select candidate model families. 
The script generates fold-level predictions, diagnostic plots, and model summaries 
to support model-family selection. 
More extensive performance assessment 
across repeated/iterated cross-validation runs was performed using the 
separate iterated CV script.


Main workflow:

Load and align subtype-specific input data.
Split samples into outer cross-validation folds.
Within each outer fold, run supervised feature selection using only the training
samples for that fold.
Build fold-specific RNA, CNV, MUT, CLINICAL, and combined feature matrices from
the selected features.
RNA/CNV-style continuous features are scaled.MUT and CLINICAL features are passed through without scaling.
Tune model hyperparameters using inner cross-validation on the training fold.
Evaluate tuned models on the held-out outer validation fold.
Save fold-level models, selected features, preprocessing objects, predictions,
performance summaries, plots, coefficients, and optional SHAP outputs.

Supported feature blocks:

RNAblock__: RNA-derived features, including pathway scores, gene-level features,
immune/TME features, custom gene sets, TIDE, and EcoTyper features when available.
CNVblock__: copy-number-derived features, including gene-, arm-, band-, pathway-,
tumor-fraction-, and percent-altered features when available.
MUTblock__: mutation-derived features, including binary mutated pathway features,
TMB/pathogenic features, and optional pTMB features.
CLINICALblock__: clinical covariates and optional derived clinical features.

Feature selection:
Feature selection is delegated to run_feature_selection() and is performed inside
each outer training fold to avoid data leakage. The selected feature set is then used
to build matching training and validation matrices for that fold.

Modeling:
The script trains and compare models on individual feature blocks and on combined
multi-block matrices. It supports multiple model families, including logistic
regression, SGD logistic models, SVMs, random forests, extra trees, gradient boosting,
histogram gradient boosting, and XGBoost. Recurrent samples are oversampled 
during model fitting using a recurrent-label-aware oversampling strategy.

Stacked ensemble modeling:
Models trained on separate feature blocks are funneled into a meta-model (L2 regression) to combine
their predictions, ie stacked enselble modeling. This allows RNA, CNV, MUT, 
and CLINICAL blocks to contribute separately to a combined recurrence-risk model.

Outputs:
Outputs are written under the subtype-specific output directory and may include:

cv_models/: fitted fold models and model metadata
preprocessing/: feature lists, scalers, column order, and configuration files
matrices/: fold-specific training and validation matrices
predictions/: fold-level and out-of-fold prediction files
plots/: ROC curves, precision-recall curves, prediction histograms, and KM plots
coefficients/: model coefficients or feature importances when available
shap/: SHAP values and summaries when available


Overall, this script is intended to provide a reproducible, leak-free framework for
cross-validated model assessment for selecting model family.
"""

# ====================== Imports ======================

# --- isolate process from user site very early ---
import os, sys, site
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ.pop("PYTHONPATH", None)
try:
    us = site.getusersitepackages()
    if isinstance(us, str):
        sys.path[:] = [p for p in sys.path if not p.startswith(us)]
except Exception:
    pass
home_local = os.path.expanduser("~/.local")
sys.path[:] = [p for p in sys.path if not p.startswith(home_local)]

import os, sys, multiprocessing as mp

# Make joblib/loky spawn fresh interpreters (safer than fork)
os.environ.setdefault("JOBLIB_START_METHOD", "spawn")

# Force the start method in this process
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # Already set in this interpreter
    pass

# Ensure all child processes use THIS exact Python interpreter
try:
    mp.set_executable(sys.executable)
except Exception:
    # Fallback for some environments
    import multiprocessing.spawn as spawn
    spawn.set_executable(sys.executable)




# --- make local modules importable even in isolated mode (-I) ---
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # if your utils live one level up, keep this; else remove

for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import os
import sys
import platform
import glob
import shutil
import re
from typing import Optional
from contextlib import contextmanager

from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

from dataloading_utils import *          # load_all_data
from dataprocessing_utils import *       # process_mutation_pathways, get_gsea
from plot_utils import *                 # (optional helpers)
# Import the updated FS implementation that supports mutation/pTMB import modes.
# Prefer feature_selection_only.py if you replaced it in-place; fall back to the
# older module name used by this CV script.
try:
    from feature_selection_only import run_feature_selection
except ImportError:
    from feature_selection_only import run_feature_selection

# version_guard.py
import json, os, sys, sklearn, platform, time

def _coerce_numeric_float32(df):
    if not isinstance(df, pd.DataFrame):
        return df
    out = df.copy()
    # Try fast path first
    if (out.dtypes == "object").any():
        # Clean common junk then coerce
        for c in out.columns[out.dtypes == "object"]:
            s = out[c].astype(str).str.strip()
            # remove thousand-separators/spaces if present
            s = s.str.replace(r"[,\s]+", "", regex=True)
            out[c] = pd.to_numeric(s, errors="coerce")
    # final fill/cast
    out = out.fillna(0.0).astype(np.float32)
    return out


def dump_with_meta(obj, path, joblib):
    joblib.dump(obj, path)
    meta = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "hostname": platform.node(),
        "cwd": os.getcwd(),
    }
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)

def load_with_strict_version(path, joblib, expected=None):
    meta_path = path + ".meta.json"
    if not os.path.exists(meta_path):
        raise SystemExit(f"FATAL: missing metadata sidecar {meta_path} for {path}. "
                         f"Re-train with the updated script to generate meta files.")
    with open(meta_path) as f:
        meta = json.load(f)
    here = {"python": sys.executable, "sklearn_version": sklearn.__version__}
    print(f"[LOAD] {os.path.basename(path)}  trained={meta['sklearn_version']}@{meta['python']}  now={here['sklearn_version']}@{here['python']}")
    if expected and sklearn.__version__ != expected:
        raise SystemExit(f"FATAL: sklearn {sklearn.__version__} ≠ expected {expected}")
    if sklearn.__version__ != meta["sklearn_version"]:
        raise SystemExit(f"FATAL: pickle trained with sklearn {meta['sklearn_version']}, "
                         f"but loader has {sklearn.__version__}. Use the same venv/python.")
    return joblib.load(path)



# ====================== Directory helper ======================
def build_fold_dirs(output_dir, subtype):
    root = os.path.join(output_dir, subtype)
    d = {
        "root":          root,
        "models":        os.path.join(root, "cv_models"),
        "preprocessing": os.path.join(root, "preprocessing"),
        "shap":          os.path.join(root, "shap"),
        "coefficients":  os.path.join(root, "coefficients"),
        "plots":         os.path.join(root, "plots"),
        "predictions":   os.path.join(root, "predictions"),
        "matrices":      os.path.join(root, "matrices"),
    }
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    return d


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")


# ====================== Dedup & align ======================
def dedup_and_align_data(
    data: dict,
    status_col: str = "Disease_Status",
    prep_dir: str = None,
    raise_on_conflict: bool = True,
):
    """
    Deduplicate META first (hard check for conflicting labels), then deduplicate and align
    every feature block to the deduped meta index.

    Returns:
        cleaned_data, subtype_meta, y_all_enc, all_ids, dup_meta_audit_df
    """
    import numpy as np
    import pandas as pd

    if "meta" not in data or not hasattr(data["meta"], "copy"):
        raise KeyError("data['meta'] must be a pandas DataFrame.")

    cleaned_data = {}
    for k, df in list(data.items()):
        if hasattr(df, "copy"):
            df = df.copy()
            df.index = df.index.astype(str).str.strip()
            df.index = df.index.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
            cleaned_data[k] = df
        else:
            cleaned_data[k] = df

    meta = cleaned_data["meta"].rename(columns={"Disease Status": "Disease_Status"})
    if status_col not in meta.columns:
        raise KeyError(f"Target column {status_col!r} not in meta (columns: {list(meta.columns)})")

    meta.index = meta.index.astype(str).str.strip()
    meta.index = meta.index.str.replace(r"\.0$", "", regex=True)

    vc = meta.index.value_counts()
    dup_ids = vc[vc > 1].index.tolist()

    audit_rows = []
    if dup_ids:
        for sid in dup_ids:
            rows = meta.loc[[sid]] if sid in meta.index else pd.DataFrame(columns=meta.columns)
            labels = rows[status_col].astype(str).str.strip().str.lower().tolist()
            labset = sorted(set(labels))
            audit_rows.append({"sample_id": sid, "n_rows_meta": int(vc.loc[sid]), "labels_found": ",".join(labset)})
            if len(labset) > 1 and raise_on_conflict:
                raise ValueError(
                    f"[DEDUP] Conflicting labels for sample '{sid}' in meta: {labset}. "
                    "Fix or set raise_on_conflict=False to keep first occurrence."
                )
        meta = meta[~meta.index.duplicated(keep="first")].copy()
        print(f"[DEDUP] Collapsed {len(dup_ids)} duplicate sample_id(s) in meta.")

    dup_meta_audit_df = pd.DataFrame(audit_rows)
    if prep_dir:
        os.makedirs(prep_dir, exist_ok=True)
        if not dup_meta_audit_df.empty:
            dup_meta_audit_df.to_csv(os.path.join(prep_dir, "duplicates_meta_audit.csv"), index=False)

    def _dedup_df_by_index(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return df
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == df.shape[1]:
            return df.groupby(level=0).mean()
        agg = {c: ("mean" if pd.api.types.is_numeric_dtype(df[c]) else "first") for c in df.columns}
        return df.groupby(level=0).agg(agg)

    for k, df in list(cleaned_data.items()):
        if k == "meta" or not hasattr(df, "reindex"):
            continue
        before_n = df.shape[0]
        df2 = _dedup_df_by_index(df)
        after_n = df2.shape[0]
        if after_n < before_n:
            print(f"[DEDUP:{k}] Collapsed {before_n - after_n} duplicate row(s) by index (aggregated).")
        df2 = df2.reindex(meta.index).apply(pd.to_numeric, errors="coerce")
        cleaned_data[k] = df2

    y_map = (
        meta[status_col]
        .astype(str).str.strip().str.lower()
        .map({"non-recurrent": 0, "nonrecurrent": 0, "0": 0, "recurrent": 1, "1": 1})
    )
    if y_map.isna().any():
        bad = sorted(set(meta.loc[y_map.isna(), status_col].astype(str).str.strip().str.lower()))
        raise ValueError(f"Unrecognized labels in {status_col}: {bad}")

    y_all_enc = y_map.astype(int)
    all_ids = meta.index

    if prep_dir:
        pd.Index(all_ids, name="sample_id").to_series().to_csv(
            os.path.join(prep_dir, "dedup_final_sample_ids.csv"), index=False
        )

    cleaned_data["meta"] = meta

    return cleaned_data, meta, y_all_enc, all_ids, dup_meta_audit_df


# ====================== Small helpers ======================
def _pam50_feature_from_meta(meta, pam50_col=None, require_variation=True):
    """
    If pam50_col is provided, create CLINICALblock__PAM50_LumB (1=LumB, 0=others).
    If pam50_col is None → return empty (feature disabled).
    If require_variation=True and only one class is present → return empty.
    """
    import pandas as pd
    if meta is None or not hasattr(meta, "columns") or pam50_col is None:
        return pd.DataFrame(index=(meta.index if hasattr(meta, "index") else pd.Index([], dtype=str)))

    if pam50_col not in meta.columns:
        return pd.DataFrame(index=meta.index)

    s = meta[pam50_col].astype(str).str.strip().str.lower()
    is_lumb = (
        (s == "lumb") | (s == "luminal b") | (s == "luminal_b") | (s == "luminalb") |
        s.str.contains(r"\blum(inal)?\s*b\b", regex=True)
    )

    if require_variation:
        if not (is_lumb.any() and (~is_lumb).any()):
            return pd.DataFrame(index=meta.index)

    df = pd.DataFrame({"CLINICALblock__PAM50_LumB": is_lumb.astype(float)}, index=meta.index)
    df.index = df.index.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return df


def _unwrap_for_explain(model):
    """Return a bare estimator for SHAP/coef extraction."""
    from sklearn.calibration import CalibratedClassifierCV
    m = model
    if hasattr(m, "named_steps"):
        m = m.named_steps.get("model", m)
    if isinstance(m, CalibratedClassifierCV):
        try:
            m = m.base_estimator
        except Exception:
            pass
    return m


def _save_coefficients_if_any(model, features, outpath):
    import pandas as pd
    est = _unwrap_for_explain(model)
    try:
        coef = getattr(est, "coef_", None)
        if coef is None:
            return
        coef = coef.ravel()
        pd.Series(coef, index=features).to_csv(outpath)
    except Exception:
        pass


@contextmanager
def pushd(new_dir):
    """Temporarily change working directory (ensures directory exists)."""
    prev = os.getcwd()
    os.makedirs(new_dir, exist_ok=True)
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev)


def _save_selected_matrix(df, path):
    import pandas as pd
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = df.copy()
    df.index = df.index.astype(str).str.strip()
    df.index = df.index.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    df.to_csv(path)


def _is_estimator_ok(est):
    try:
        est = clone(est)
        _ = est.get_params()
        if hasattr(est, "get_tags"):
            _ = est.get_tags()
        return True
    except Exception as e:
        print(f"[GRID DROP] {type(est).__name__}: {e}")
        return False


def _filter_param_grid(param_grid_dict):
    return {
        name: (est, params)
        for name, (est, params) in param_grid_dict.items()
        if _is_estimator_ok(est)
    }


def _dump_json_safely(obj, path):
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=str)
    except Exception as e:
        print(f"[WARN] Could not write JSON to {path}: {e}")


# ====================== Column selector by prefix (for block-aware stacking) ======================
class PrefixSelector(BaseEstimator, TransformerMixin):
    """
    Select columns whose names start with any of the given prefixes.
    Keeps pandas DataFrame shape with the original column names.
    """
    def __init__(self, prefixes):
        self.prefixes = tuple(prefixes)
        self.columns_ = None

    def fit(self, X, y=None):
        if not hasattr(X, "columns"):
            raise ValueError("PrefixSelector requires a pandas DataFrame with column names.")
        cols = [c for c in X.columns if any(str(c).startswith(p) for p in self.prefixes)]
        self.columns_ = cols
        return self

    def transform(self, X):
        if self.columns_ is None:
            return X
        Xout = X.copy()
        for c in self.columns_:
            if c not in Xout.columns:
                Xout[c] = 0.0
        return Xout.loc[:, self.columns_]


# ====================== Block helpers ======================
_BLOCK_PREFIX = {
    "RNA": "RNAblock__",
    "CNV": "CNVblock__",
    "MUT": "MUTblock__",
    "CLINICAL": "CLINICALblock__",
}


# Blocks that should still run per-block CV, but be excluded from the
# block-aware stacked model.
#_STACK_EXCLUDE_BLOCKS = {"CLINICAL"}
_STACK_EXCLUDE_BLOCKS = {}


def _block_columns(selected_cols, block_name):
    pref = _BLOCK_PREFIX[block_name]
    return [c for c in selected_cols if c.startswith(pref)]


# ====================== Oversampler (always oversample class=1) ======================
class RecurrentOverSampler(BaseEstimator):
    """Always oversample positive class (default=1) to ratio * (#non-positive)."""
    def __init__(self, ratio=1.0, pos_label=1, random_state=42):
        self.ratio = ratio
        self.pos_label = pos_label
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def fit_resample(self, X, y):
        import numpy as np
        y_arr = np.asarray(y)
        n_pos = (y_arr == self.pos_label).sum()
        n_nonpos = (y_arr != self.pos_label).sum()
        target_pos = max(int(np.ceil(self.ratio * n_nonpos)), n_pos)
        strategy = {self.pos_label: target_pos}
        ros = RandomOverSampler(sampling_strategy=strategy, random_state=self.random_state)
        return ros.fit_resample(X, y)


# ====================== Supervised FS (optional, inside CV) ======================
class SupervisedFS(BaseEstimator, TransformerMixin):
    def __init__(self,
                 fs_p_thresh=0.05,
                 fs_fdr_thresh=0.25,
                 fs_corr_threshold=0.70,
                 fs_topk_if_empty=200):
        self.fs_p_thresh = fs_p_thresh
        self.fs_fdr_thresh = fs_fdr_thresh
        self.fs_corr_threshold = fs_corr_threshold
        self.fs_topk_if_empty = fs_topk_if_empty
        self.keep_cols_ = None

    def _bh_fdr(self, pvals):
        import numpy as np
        p = np.asarray(pvals, dtype=float)
        n = p.size
        order = np.argsort(p)
        ranks = np.arange(1, n+1, dtype=float)
        q = p[order] * n / ranks
        q = np.minimum.accumulate(q[::-1])[::-1]
        q = np.clip(q, 0.0, 1.0)
        out = np.empty_like(q)
        out[order] = q
        return out

    def _mw_pvalues(self, X, y):
        import numpy as np
        import pandas as pd
        from scipy.stats import mannwhitneyu
        cols = list(X.columns)
        y = pd.Series(y).astype(int)
        g0 = y == 0
        g1 = y == 1
        pvals = []
        for c in cols:
            a = pd.to_numeric(X.loc[g0, c], errors="coerce").dropna().values
            b = pd.to_numeric(X.loc[g1, c], errors="coerce").dropna().values
            if a.size < 2 or b.size < 2:
                pvals.append(1.0)
                continue
            try:
                stat = mannwhitneyu(a, b, alternative="two-sided")
                pvals.append(float(stat.pvalue))
            except Exception:
                pvals.append(1.0)
        return cols, pvals

    def _corr_prune(self, X, ordered_cols):
        import numpy as np
        import pandas as pd
        keep = []
        for c in ordered_cols:
            ok = True
            for kc in keep:
                try:
                    r = pd.to_numeric(X[kc], errors="coerce").corr(pd.to_numeric(X[c], errors="coerce"))
                except Exception:
                    r = 0.0
                if r is None or np.isnan(r):
                    r = 0.0
                if abs(r) >= float(self.fs_corr_threshold):
                    ok = False
                    break
            if ok:
                keep.append(c)
        return keep

    def compute_selected_feature_list_(self, X, y):
        cols, p = self._mw_pvalues(X, y)
        import numpy as np
        import pandas as pd
        p = np.asarray(p, dtype=float)
        q = self._bh_fdr(p)
        df = pd.DataFrame({"feature": cols, "p": p, "q": q}).sort_values("p")
        df_filt = df[(df["p"] <= float(self.fs_p_thresh)) & (df["q"] <= float(self.fs_fdr_thresh))]
        if df_filt.empty:
            df_filt = df.head(int(self.fs_topk_if_empty)).copy()
        ordered = df_filt.sort_values("p")["feature"].tolist()
        keep = self._corr_prune(X, ordered)
        if len(keep) == 0:
            keep = df.head(min(int(self.fs_topk_if_empty), len(df)))["feature"].tolist()
        return keep

    def fit(self, X, y):
        import pandas as pd
        import numpy as np
        if not hasattr(X, "columns"):
            X = pd.DataFrame(X, columns=[f"f{i}" for i in range(np.asarray(X).shape[1])])
        keep = self.compute_selected_feature_list_(X, y)
        self.keep_cols_ = [c for c in keep if c in X.columns] or list(X.columns)
        return self

    def transform(self, X):
        import pandas as pd
        if not hasattr(X, "columns"):
            X = pd.DataFrame(X, columns=self.keep_cols_)
        for c in self.keep_cols_:
            if c not in X.columns:
                X[c] = 0.0
        return X.loc[:, self.keep_cols_]


# ====================== Param grid builder ======================
def _build_param_grid_for(
    subtype: Optional[str] = None,
    oversample_ratios=(1.0, 1.5, 2.0, 2.4, 3.0),
    cw_opts=("balanced", None, {0: 1.0, 1: 2.0}),
    include_xgboost: bool = True,
    fs_enable: bool = False,
    fs_grid=None,
    include_stacking: bool = False
):
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier,
        GradientBoostingClassifier, HistGradientBoostingClassifier,
        StackingClassifier,
    )

    if fs_grid is None:
        fs_grid = {
            "fs__fs_p_thresh": [0.01, 0.05, 0.1],
            "fs__fs_fdr_thresh": [0.05, 0.25, 1.0],
            "fs__fs_corr_threshold": [0.7, 0.9],
            "fs__fs_topk_if_empty": [200, 500],
        }

    def _pipe(est):
        steps = []
        if fs_enable:
            steps.append(("fs", SupervisedFS()))
        steps.append(("oversampler", RecurrentOverSampler()))
        steps.append(("model", est))
        return Pipeline(steps)

    def _maybe_merge_fs(params: dict):
        if fs_enable:
            p = params.copy()
            p.update(fs_grid)
            return p
        return params

    param_grid = {}

    # Logistic Regression families
    param_grid["LogReg_L2_liblinear"] = (
        _pipe(LogisticRegression(solver="liblinear")),
        _maybe_merge_fs({
            "oversampler__ratio": list(oversample_ratios),
            "model__penalty": ["l2"],
            "model__C": [1e-3, 1e-2, 1e-1, 1, 10],
            "model__class_weight": list(cw_opts),
            "model__max_iter": [1000, 5000],
        }),
    )
    param_grid["LogReg_L1_liblinear"] = (
        _pipe(LogisticRegression(solver="liblinear")),
        _maybe_merge_fs({
            "oversampler__ratio": list(oversample_ratios),
            "model__penalty": ["l1"],
            "model__C": [1e-3, 1e-2, 1e-1, 1, 10],
            "model__class_weight": list(cw_opts),
            "model__max_iter": [1000, 5000],
        }),
    )
    param_grid["LogReg_ElasticNet_saga"] = (
        _pipe(LogisticRegression(solver="saga")),
        _maybe_merge_fs({
            "oversampler__ratio": list(oversample_ratios),
            "model__penalty": ["elasticnet"],
            "model__C": [1e-3, 1e-2, 1e-1, 1, 10],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            "model__class_weight": list(cw_opts),
            "model__max_iter": [1000, 5000],
        }),
    )

    # SGD
    param_grid["SGD_LogLoss"] = (
        _pipe(SGDClassifier(loss="log_loss")),
        _maybe_merge_fs({
            "oversampler__ratio": list(oversample_ratios),
            "model__alpha": [1e-5, 1e-4, 1e-3],
            "model__penalty": ["l2", "l1", "elasticnet"],
            "model__l1_ratio": [0.15, 0.5, 0.9],
            "model__max_iter": [5000],
            "model__tol": [1e-3],
        }),
    )

    # SVM
    param_grid["SVM_RBF"] = (
        _pipe(SVC(probability=True)),
        _maybe_merge_fs({
            "oversampler__ratio": list(oversample_ratios),
            "model__C": [0.1, 1, 10],
            "model__gamma": ["scale", 0.01, 0.001],
            "model__kernel": ["rbf"],
            "model__class_weight": list(cw_opts),
        }),
    )

    # Calibrated LinearSVC
    param_grid["LinearSVC_Calibrated"] = (
        _pipe(CalibratedClassifierCV(estimator=LinearSVC())),
        _maybe_merge_fs({
            "oversampler__ratio": list(oversample_ratios),
            "model__estimator__C": [0.1, 1, 10],
            "model__estimator__class_weight": list(cw_opts),
            "model__method": ["sigmoid", "isotonic"],
        }),
    )

    # Tree ensembles
    param_grid["RandomForest"] = (
        _pipe(RandomForestClassifier()),
        _maybe_merge_fs({
            "oversampler__ratio": list(oversample_ratios),
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__max_features": ["sqrt", "log2"],
            "model__class_weight": list(cw_opts),
            "model__n_jobs": [1],
        }),
    )
    param_grid["ExtraTrees"] = (
        _pipe(ExtraTreesClassifier()),
        _maybe_merge_fs({
            "oversampler__ratio": list(oversample_ratios),
            "model__n_estimators": [300, 600],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__max_features": ["sqrt", "log2"],
            "model__class_weight": list(cw_opts),
            "model__n_jobs": [1],
        }),
    )
    param_grid["GradientBoosting"] = (
        _pipe(GradientBoostingClassifier()),
        _maybe_merge_fs({
            "oversampler__ratio": list(oversample_ratios),
            "model__n_estimators": [200, 400],
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [3, 5],
            "model__subsample": [0.8, 1.0],
        }),
    )
    param_grid["HistGradientBoosting"] = (
        _pipe(HistGradientBoostingClassifier(early_stopping=True)),
        _maybe_merge_fs({
            "oversampler__ratio": list(oversample_ratios),
            "model__max_depth": [None, 6, 10],
            "model__learning_rate": [0.03, 0.1],
            "model__l2_regularization": [0.0, 0.1, 1.0],
            "model__max_iter": [200, 500, 1000],
        }),
    )

    # XGBoost
    if include_xgboost:
        try:
            from xgboost import XGBClassifier
            param_grid["XGBoost"] = (
                _pipe(XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=42,
                    n_jobs=1,
                )),
                _maybe_merge_fs({
                    "oversampler__ratio": list(oversample_ratios),
                    "model__n_estimators": [200, 500, 1000],
                    "model__max_depth": [3, 5, 8],
                    "model__learning_rate": [0.03, 0.1],
                    "model__subsample": [0.7, 1.0],
                    "model__colsample_bytree": [0.7, 1.0],
                    "model__min_child_weight": [1, 5, 10],
                    "model__reg_lambda": [0.0, 1.0, 10.0],
                    "model__reg_alpha": [0.0, 0.1],
                }),
            )
        except Exception as e:
            print(f"[GRID DROP] XGBoost unavailable: {e}")

    # Plain stacking (same-features) if asked
    if include_stacking:
        base_lr = Pipeline([
            ("oversampler", RecurrentOverSampler(ratio=2.0)),
            ("model", LogisticRegression(solver="liblinear", penalty="l2", C=1.0, max_iter=5000))
        ])
        base_svm = Pipeline([
            ("oversampler", RecurrentOverSampler(ratio=2.0)),
            ("model", SVC(probability=True, kernel="rbf", C=1.0, gamma="scale"))
        ])
        base_rf = Pipeline([
            ("oversampler", RecurrentOverSampler(ratio=2.0)),
            ("model", RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=1, class_weight=None))
        ])

        meta = LogisticRegression(solver="liblinear", penalty="l2", max_iter=5000)

        stack = StackingClassifier(
            estimators=[("lr", base_lr), ("svm", base_svm), ("rf", base_rf)],
            final_estimator=meta,
            cv=5,
            passthrough=False,
            n_jobs=1,
        )

        param_grid["Stacking_SameFeatures"] = (
            Pipeline([("model", stack)]),
            {
                "model__final_estimator__C": [0.1, 1.0, 10.0],
                "model__estimators__svm__model__C": [0.3, 1, 3],
                "model__estimators__rf__model__n_estimators": [200, 400],
            },
        )

    return param_grid


# -------- helper to tune just one family (used for fixed-family stacking) --------
def _best_estimator_for_family(X, y, family_name, inner_folds,
                               include_xgboost=True,
                               calibrate=False, calibrate_method="sigmoid", calibrate_cv=5):
    """Grid-search a single model family and return the fitted best estimator."""
    grid_all = _filter_param_grid(
        _build_param_grid_for(include_xgboost=include_xgboost, fs_enable=False, include_stacking=False)
    )
    if family_name not in grid_all:
        raise ValueError(f"Unknown family {family_name}. Available: {list(grid_all.keys())}")
    est, params = grid_all[family_name]
    best_est, _, _, _ = _grid_select_and_fit(
        X, y, {family_name: (est, params)}, inner_folds,
        label=f"FIXED_FAMILY::{family_name}",
        calibrate=calibrate, calibrate_method=calibrate_method, calibrate_cv=calibrate_cv
    )
    return best_est


# ====================== Grid + robust ROC scorer ======================
def _grid_select_and_fit(X, y, param_grid_dict, cv_folds, label,
                         calibrate=False, calibrate_method="sigmoid", calibrate_cv=5):
    import numpy as np
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import roc_auc_score

    def _roc_auc_scorer_auto(estimator, X_test, y_true):
        proba_like = None
        if hasattr(estimator, "predict_proba"):
            try:
                P = estimator.predict_proba(X_test)
                proba_like = P[:, -1] if (P.ndim == 2 and P.shape[1] >= 2) else P.ravel()
            except Exception:
                proba_like = None
        if proba_like is None and hasattr(estimator, "decision_function"):
            try:
                proba_like = estimator.decision_function(X_test)
            except Exception:
                proba_like = None
        if proba_like is None:
            proba_like = estimator.predict(X_test)
        y_true = np.asarray(y_true).astype(int)
        return roc_auc_score(y_true, proba_like)

    best_auc, best_name, best_model, best_params = -float("inf"), None, None, None

    for name, (est, grid) in param_grid_dict.items():
        try:
            gs = GridSearchCV(
                est, grid, cv=cv_folds, scoring=_roc_auc_scorer_auto,
                n_jobs=8, refit=True, error_score="raise", pre_dispatch="2*n_jobs", verbose=0
            )
            gs.fit(X, y)
        except Exception as e:
            print(f"[SKIP MODEL] {label} :: {name} failed during fit: {e}")
            continue

        score = gs.best_score_

        if score is None or (isinstance(score, float) and np.isnan(score)):
            print(f"[SKIP MODEL] {label} :: {name} returned NaN score.")
            continue

        if score > best_auc:
            best_auc = float(score)
            best_name = name
            best_model = gs.best_estimator_
            best_params = gs.best_params_

    if best_model is None:
        raise RuntimeError(f"No model finished successfully in GridSearchCV for {label}.")

    best_model.fit(X, y)

    if calibrate and not isinstance(best_model, CalibratedClassifierCV):
        try:
            cal = CalibratedClassifierCV(estimator=best_model, method=calibrate_method, cv=calibrate_cv)
            cal.fit(X, y)
            best_model = cal
            best_name = f"{best_name}+Calibrated({calibrate_method},cv={calibrate_cv})"
        except Exception as e:
            print(f"[WARN] Calibration failed; using uncalibrated model: {e}")

    return best_model, best_name, best_auc, best_params


def _proba_pos(est, X, pos_label=1):
    import numpy as np
    if hasattr(est, "predict_proba"):
        P = est.predict_proba(X)
        classes_ = getattr(est, "classes_", np.arange(P.shape[1]))
        try:
            idx = np.where(classes_ == pos_label)[0]
            if idx.size:
                return P[:, int(idx[0])]
        except Exception:
            pass
        try:
            idx = [i for i, c in enumerate(classes_) if str(c).strip().lower() in ("recurrent", "1")]
            if idx:
                return P[:, idx[0]]
        except Exception:
            pass
        return P[:, -1]
    if hasattr(est, "decision_function"):
        s = est.decision_function(X).astype(float)
        return (s - s.min()) / (s.max() - s.min() + 1e-9)
    raise ValueError("Estimator lacks predict_proba/decision_function.")

from sklearn.calibration import CalibratedClassifierCV

def _unwrap_model_for_shap(model):
    """
    Return the most interpretable underlying estimator for SHAP.

    Priority:
      1) unwrap sklearn/imblearn pipelines to final step
      2) unwrap CalibratedClassifierCV to its fitted base estimator
      3) otherwise return model as-is

    Note:
      For CalibratedClassifierCV with multiple calibrated clones, this uses
      the first fitted calibrated estimator as a practical approximation.
    """
    est = model

    # unwrap pipeline -> final estimator
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline
    except Exception:
        ImbPipeline = tuple()

    try:
        from sklearn.pipeline import Pipeline as SkPipeline
    except Exception:
        SkPipeline = tuple()

    try:
        if isinstance(est, (ImbPipeline, SkPipeline)):
            if hasattr(est, "named_steps") and "model" in est.named_steps:
                est = est.named_steps["model"]
            elif hasattr(est, "steps") and len(est.steps) > 0:
                est = est.steps[-1][1]
    except Exception:
        pass

    # unwrap calibrated classifier
    try:
        if isinstance(est, CalibratedClassifierCV):
            # fitted internal calibrated estimators
            ccs = getattr(est, "calibrated_classifiers_", None)
            if ccs is not None and len(ccs) > 0:
                cc0 = ccs[0]

                # sklearn internal fitted estimator handle
                if hasattr(cc0, "estimator"):
                    base_est = cc0.estimator
                    if base_est is not None:
                        return base_est

                if hasattr(cc0, "base_estimator"):
                    base_est = cc0.base_estimator
                    if base_est is not None:
                        return base_est

            # fallback to public attr if present
            if hasattr(est, "estimator") and est.estimator is not None:
                return est.estimator
            if hasattr(est, "base_estimator") and est.base_estimator is not None:
                return est.base_estimator
    except Exception:
        pass

    return est


def _save_shap_if_any(model, X_df, shap_dir, tag):
    import os
    import pandas as pd

    if X_df is None or getattr(X_df, "empty", True):
        return

    os.makedirs(shap_dir, exist_ok=True)

    model_for_shap = _unwrap_model_for_shap(model)

    # optional note if we had to unwrap
    if type(model_for_shap) is not type(model):
        try:
            print(f"[SHAP] {tag}: using base estimator {type(model_for_shap).__name__} "
                  f"instead of wrapper {type(model).__name__}")
        except Exception:
            pass

    try:
        explainer = get_shap_explainer(model_for_shap, X_df)
    except Exception as e:
        print(f"[SHAP SKIP] Could not build explainer for {type(model_for_shap).__name__}: {e}")
        explainer = None

    if explainer is None:
        return

    try:
        sv = explainer(X_df)
        arr = sv.values

        # binary classification outputs sometimes come out as (n, p, 2)
        if getattr(arr, "ndim", 0) == 3 and arr.shape[2] == 2:
            arr = arr[:, :, 1]

        out = pd.DataFrame(arr, index=X_df.index, columns=X_df.columns)
        out.to_csv(os.path.join(shap_dir, f"shap_{tag}.csv"))
    except Exception as e:
        print(f"[SHAP SKIP] Could not compute/save SHAP for {tag}: {e}")



def _get_block(dct, key, index_like):
    import pandas as pd
    df = dct.get(key, None)
    if df is None or not isinstance(df, pd.DataFrame) or df.shape[1] == 0:
        return pd.DataFrame(index=pd.Index(index_like, dtype=str))
    out = df.copy()
    out.index = out.index.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return out

from pathlib import Path
import numpy as np
import pandas as pd

def export_meta_coefficients(meta_model, meta_feature_names, coeff_dir, prefix="meta"):
    """
    Exports a TSV of meta-model feature importance.
    - For linear models: coef_
    - For tree models: feature_importances_
    Falls back gracefully if neither exists.
    """
    coeff_dir = Path(coeff_dir)
    coeff_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if hasattr(meta_model, "coef_"):
        coef = np.asarray(meta_model.coef_)
        # Binary logistic regression: shape (1, n_features)
        if coef.ndim == 2 and coef.shape[0] == 1:
            coef = coef[0]
        # Multiclass: keep per-class columns
        if coef.ndim == 2:
            df = pd.DataFrame(coef.T, index=meta_feature_names,
                              columns=[f"coef_class{i}" for i in range(coef.shape[0])])
            df.insert(0, "feature", df.index)
            out_path = coeff_dir / f"{prefix}_coefficients.tsv"
            df.to_csv(out_path, sep="\t", index=False)
            return str(out_path)

        df = pd.DataFrame({"feature": meta_feature_names, "coef": coef})
        df["abs_coef"] = df["coef"].abs()
        df = df.sort_values("abs_coef", ascending=False)
        out_path = coeff_dir / f"{prefix}_coefficients.tsv"
        df.to_csv(out_path, sep="\t", index=False)
        return str(out_path)

    if hasattr(meta_model, "feature_importances_"):
        imp = np.asarray(meta_model.feature_importances_)
        df = pd.DataFrame({"feature": meta_feature_names, "importance": imp})
        df["abs_importance"] = df["importance"].abs()
        df = df.sort_values("abs_importance", ascending=False)
        out_path = coeff_dir / f"{prefix}_feature_importances.tsv"
        df.to_csv(out_path, sep="\t", index=False)
        return str(out_path)

    # fallback
    out_path = coeff_dir / f"{prefix}_coefficients_UNAVAILABLE.txt"
    out_path.write_text("Meta-model has no coef_ or feature_importances_. Cannot export coefficients.\n")
    return str(out_path)


def export_meta_shap(meta_model, X_meta_eval, meta_feature_names, shap_dir, prefix="meta", max_background=200):
    """
    Exports SHAP values for the meta model into shap_dir.
    Produces:
      - <prefix>_shap_values.npy
      - <prefix>_shap_summary.tsv  (mean |SHAP| per feature)
    """
    import shap  # keep local import so script runs even if shap isn't installed unless enabled

    shap_dir = Path(shap_dir)
    shap_dir.mkdir(parents=True, exist_ok=True)

    # Ensure ndarray + background sampling
    X = np.asarray(X_meta_eval)
    if X.shape[0] > max_background:
        idx = np.random.choice(X.shape[0], size=max_background, replace=False)
        background = X[idx]
    else:
        background = X

    # Use shap.Explainer for broad compatibility
    try:
        explainer = shap.Explainer(meta_model, background)
        shap_values = explainer(X)
        # shap_values.values shape: (n, p) or (n, p, k)
        vals = np.asarray(shap_values.values)
    except Exception:
        # conservative fallback
        explainer = shap.KernelExplainer(meta_model.predict_proba, background)
        vals = np.asarray(explainer.shap_values(X))
        # KernelExplainer can return list per class; convert to (n,p) for positive class if binary
        if isinstance(vals, list) and len(vals) == 2:
            vals = np.asarray(vals[1])

    # Save raw
    npy_path = shap_dir / f"{prefix}_shap_values.npy"
    np.save(npy_path, vals)

    # Summarize mean(|shap|) per feature
    if vals.ndim == 3:
        # (n,p,k) -> average over samples and classes
        mean_abs = np.mean(np.abs(vals), axis=(0, 2))
    else:
        mean_abs = np.mean(np.abs(vals), axis=0)

    summ = pd.DataFrame({"feature": meta_feature_names, "mean_abs_shap": mean_abs})
    summ = summ.sort_values("mean_abs_shap", ascending=False)
    tsv_path = shap_dir / f"{prefix}_shap_summary.tsv"
    summ.to_csv(tsv_path, sep="\t", index=False)

    return str(npy_path), str(tsv_path)


from pathlib import Path
import numpy as np

def unwrap_stacking_model(model):
    """
    If you calibrate the stack, you may get a CalibratedClassifierCV wrapper.
    This tries to unwrap to the underlying StackingClassifier (or Pipeline -> StackingClassifier).
    """
    m = model
    # Unwrap sklearn Pipeline if needed
    if hasattr(m, "named_steps") and hasattr(m, "steps"):
        m = m.steps[-1][1]

    # Unwrap calibration wrappers (different sklearn versions use estimator/base_estimator)
    for attr in ("estimator", "base_estimator"):
        if hasattr(m, attr):
            maybe = getattr(m, attr)
            if maybe is not None:
                m = maybe
                break

    # If still a pipeline after unwrapping
    if hasattr(m, "named_steps") and hasattr(m, "steps"):
        m = m.steps[-1][1]

    return m


def infer_stack_meta_feature_names(stack_core, n_meta_cols):
    """
    Best-effort names for meta-features coming out of StackingClassifier.transform().
    """
    base = []
    # Prefer init-time names
    if hasattr(stack_core, "estimators") and stack_core.estimators is not None:
        try:
            base = [name for name, _ in stack_core.estimators]
        except Exception:
            base = []
    # Fallback to fitted names
    if not base and hasattr(stack_core, "estimators_") and stack_core.estimators_ is not None:
        try:
            base = [f"est{i+1}" for i in range(len(stack_core.estimators_))]
        except Exception:
            base = []

    n_base = len(base)

    # Common binary case: one column per base estimator (proba of positive class)
    if n_base and n_meta_cols == n_base:
        return base

    # Multiclass-ish / expanded outputs: label generically but deterministically
    if n_base and n_meta_cols % n_base == 0:
        k = n_meta_cols // n_base
        names = []
        for b in base:
            for j in range(k):
                names.append(f"{b}__meta{j+1}")
        return names

    return [f"meta_{i+1}" for i in range(n_meta_cols)]



def _build_fold_matrices_with_fs(
    data_path,
    output_dir,
    subtype,
    status_col,
    pathway_options,
    train_ids,
    val_ids,
    test_ids=None,
    RNAgene_bypass=False,
    CNVgene_bypass=False,
    # ---- subset-FS control ----
    enable_subset_fs=True,
    subset_run_name=None,


    rna_gene_min_raw_count: int = 10,
    rna_gene_min_detected_frac: float = 0.25,
    cnv_gene_min_detected_frac: float = 0.20,
    cnv_arm_min_detected_frac: float = 0.20,
    cnv_band_min_detected_frac: float = 0.20,
    mut_min_detected_frac: float = 0.10,

    # ---- which supervised tests to run ----
    primary_test="mwu",          # "mwu" or "cox" (legacy convenience)
    run_mwu=None,                # if None, inferred from primary_test
    run_cox_uni=None,            # if None, inferred from primary_test
    run_cox_multi=False,         # optional extra test

    # ---- OPEN thresholds (per test) ----
    # legacy fallbacks (if per-test thresholds are None, we use these)
    fs_p_thresh=0.09,
    fs_fdr_thresh=0.10,

    # per-test overrides (set these from CLI to “open up” thresholds)
    mwu_p_thresh=None,
    mwu_fdr_thresh=None,
    cox_uni_p_thresh=None,
    cox_uni_fdr_thresh=None,
    cox_multi_p_thresh=None,
    cox_multi_fdr_thresh=None,

    # ---- Cox config ----
    time_col="OS_Time",
    event_col="OS_Status",
    cox_covariates=None,
    cox_hr_abs_min=0.0,
    cox_hr_direction="any",
    cox_min_n=10,
    cox_min_events=1,

    # ---- limma thresholds (already open; just pass-through) ----
    limma_pvalue_threshold=0.01,
    limma_padj_threshold=0.25,
    limma_logfc_threshold=0.75,

    # ---- variance filter knobs (label-free; full-cohort cache) ----
    var_min=0.0,
    var_min_nonmissing=10,
    variance_keep_file=None,
    var_keep_top_frac=None,

    # ---- label-based correlation prefilter knobs ----
    corr_prefilter=False,
    corr_abs_min=None,
    corr_min_n=10,
    corr_prune_sig_mode="hierarchical",

    # ---- redundancy pruning (feature-feature corr) ----
    fs_corr_threshold=0.90,
    fs_min_per_block=0,

    fs_corr_thresholds_by_block=None,

    fs_mwu_p_thresh_by_block=None,
    fs_mwu_fdr_thresh_by_block=None,

    fs_cox_uni_p_thresh_by_block=None,
    fs_cox_uni_fdr_thresh_by_block=None,

    fs_cox_multi_p_thresh_by_block=None,
    fs_cox_multi_fdr_thresh_by_block=None,

    fs_rna_pathway_dbs=None,
    fs_cnv_pathway_dbs=None,
    fs_mut_pathway_dbs=None,
    fs_ptmb_pathway_dbs=None,


    # ---- fallbacks ----
    fs_min_features=10,
    fs_topk_if_empty=200,

    # ---- MUT / PTMB ----
    include_mutated_pathways=True,

    # Binary mutated pathway feature source:
    #   snv    = original behavior, calculated from mutation/SNV data loaded by FS
    #   import = read a precomputed sample x pathway 0/1 matrix
    #   none   = skip binary mutated pathway features
    mut_pathway_source="snv",
    mut_binary_matrix_csv=None,
    mut_binary_sample_col=None,
    mut_binary_prefix="MUTblock__",

    # pTMB feature source:
    #   snv    = original behavior, calculate from MAF/GMT/GTF
    #   import = read a precomputed sample x pathway pTMB matrix
    #   none   = skip pTMB features
    enable_ptmb=False,
    ptmb_source="none",
    ptmb_matrix_csv=None,
    ptmb_sample_col=None,
    ptmb_prefix="MUTblock__pTMB__",
    ptmb_transform="none",

    # Existing SNV/MAF pTMB calculation inputs retained for backward compatibility
    ptmb_maf_path=None,
    ptmb_maf_sample_col=None,
    ptmb_maf_gene_col=None,
    ptmb_gtf_path=None,
    ptmb_cds_cache_csv=None,
    ptmb_denom_policy="all_genes_with_len",
    ptmb_min_genes_with_len=1,
    ptmb_gmt_path=None,

    # ---- optional protein-coding filter ----
    protein_coding_gtf=None,

    # ---- custom gene sets (kept; same constants you were using) ----
    custom_gene_sets_file="./data/RNAseq/genescores/custom_genesets.gmt",
    custom_gene_set_mode="median_log2",
    custom_gene_set_min_overlap=1,

    # ---- misc / compat ----
    prep_dir=None,
    cleaned_data=None,
    all_pathway_dicts=None,
    pam50_col=None,
    mwu_case_label="recurrent",
    mwu_ctrl_label="non-recurrent",
):
    """
    Leak-free supervised FS on TRAIN IDs only (subset FS), then build TRAIN/VAL(/TEST)
    matrices using the SAME up-to-date preprocessing artifacts written by run_feature_selection().

    Returns:
        Xtr_sc, Xva_sc, Xte_sc, selected_cols, ct_fold, fs_dir, Xtr, Xva, Xte
    """
    import os
    import pandas as pd
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from config import make_output_path

    # -----------------
    # helpers
    # -----------------
    def _norm(idx):
        if idx is None:
            return None
        return (pd.Index(idx).astype(str).str.strip()
                .str.replace(r"\.0$", "", regex=True)
                .drop_duplicates())

    def _read_csv_if_exists(path):
        if not path or (not os.path.exists(path)):
            return None
        df = pd.read_csv(path, index_col=0)
        df.index = (df.index.astype(str).str.strip().str.replace(r"\.0$", "", regex=True))
        # drop accidental index columns like "Unnamed: 0"
        df = df.loc[:, ~df.columns.astype(str).str.lower().str.startswith("unnamed")]
        return df

    def _ensure_cols(df, cols, fill_value=0.0):
        """Ensure df has all cols; add missing with fill_value."""
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"[ALIGN] {len(missing)} selected feature(s) missing from assembled blocks; adding zeros. e.g. {missing[:10]}")
            for c in missing:
                df[c] = fill_value
        return df

    # -----------------
    # normalize IDs
    # -----------------
    train_ids = _norm(train_ids)
    val_ids   = _norm(val_ids)
    test_ids  = _norm(test_ids)

    # -----------------
    # infer run flags from primary_test (unless caller overrides)
    # -----------------
    pt = str(primary_test).lower().strip()
    if run_mwu is None:
        run_mwu = (pt == "mwu")
    if run_cox_uni is None:
        run_cox_uni = (pt == "cox")

    # -----------------
    # open thresholds (per-test), defaulting to legacy fs_* values
    # -----------------
    mwu_p  = float(fs_p_thresh)   if mwu_p_thresh is None      else float(mwu_p_thresh)
    mwu_q  = float(fs_fdr_thresh) if mwu_fdr_thresh is None    else float(mwu_fdr_thresh)

    coxu_p = float(fs_p_thresh)   if cox_uni_p_thresh is None  else float(cox_uni_p_thresh)
    coxu_q = float(fs_fdr_thresh) if cox_uni_fdr_thresh is None else float(cox_uni_fdr_thresh)

    coxm_p = float(fs_p_thresh)   if cox_multi_p_thresh is None else float(cox_multi_p_thresh)
    coxm_q = float(fs_fdr_thresh) if cox_multi_fdr_thresh is None else float(cox_multi_fdr_thresh)

    # -----------------
    # Backward compatibility for direct/programmatic calls to this helper.
    # -----------------
    mut_pathway_source = str(mut_pathway_source or "snv").strip().lower()
    if not bool(include_mutated_pathways):
        mut_pathway_source = "none"

    ptmb_source = str(ptmb_source or "none").strip().lower()
    if bool(enable_ptmb) and ptmb_source == "none":
        ptmb_source = "snv"
    if ptmb_source != "none":
        enable_ptmb = True

    # -----------------
    # load/dedup once (for pam50 + IDs sanity); FS will also load internally
    # -----------------
    if cleaned_data is None or all_pathway_dicts is None:
        data_all, all_pathway_dicts = load_all_data(data_path, subtype, pathway_options)
        cleaned_data, _, _, _, _ = dedup_and_align_data(
            data_all, status_col=status_col, prep_dir=prep_dir, raise_on_conflict=True
        )

    for k, df in list(cleaned_data.items()):
        if isinstance(df, pd.DataFrame):
            df.index = (df.index.astype(str).str.strip().str.replace(r"\.0$", "", regex=True))
    meta = cleaned_data.get("meta", pd.DataFrame()).copy()
    cleaned_data["meta"] = meta

    pam50_all = _pam50_feature_from_meta(cleaned_data["meta"], pam50_col=pam50_col, require_variation=True)

    # -----------------
    # 1) RUN up-to-date FS on TRAIN ONLY
    # -----------------
    fs_dir = run_feature_selection(
        data_path=data_path,
        output_dir=output_dir,
        subtype=subtype,
        status_col=status_col,
        pathway_options=pathway_options,
        RNAgene_bypass=RNAgene_bypass,
        CNVgene_bypass=CNVgene_bypass,
        include_mutated_pathways=bool(include_mutated_pathways),
        mut_pathway_source=mut_pathway_source,
        mut_binary_matrix_csv=mut_binary_matrix_csv,
        mut_binary_sample_col=mut_binary_sample_col,
        mut_binary_prefix=mut_binary_prefix,

        enable_ptmb=bool(enable_ptmb),
        ptmb_source=ptmb_source,
        ptmb_matrix_csv=ptmb_matrix_csv,
        ptmb_sample_col=ptmb_sample_col,
        ptmb_prefix=ptmb_prefix,
        ptmb_transform=ptmb_transform,

        ptmb_maf_path=ptmb_maf_path,
        ptmb_maf_sample_col=ptmb_maf_sample_col,
        ptmb_maf_gene_col=ptmb_maf_gene_col,
        ptmb_gtf_path=ptmb_gtf_path,
        ptmb_cds_cache_csv=ptmb_cds_cache_csv,
        ptmb_denom_policy=str(ptmb_denom_policy),
        ptmb_min_genes_with_len=int(ptmb_min_genes_with_len),
        ptmb_gmt_path=ptmb_gmt_path,

        protein_coding_gtf=protein_coding_gtf,

        var_min=float(var_min),
        var_min_nonmissing=int(var_min_nonmissing),
        variance_keep_file=variance_keep_file,
        var_keep_top_frac=var_keep_top_frac,

        rna_gene_min_raw_count=int(rna_gene_min_raw_count),
        rna_gene_min_detected_frac=float(rna_gene_min_detected_frac),
        cnv_gene_min_detected_frac=float(cnv_gene_min_detected_frac),
        cnv_arm_min_detected_frac=float(cnv_arm_min_detected_frac),
        cnv_band_min_detected_frac=float(cnv_band_min_detected_frac),
        mut_min_detected_frac=float(mut_min_detected_frac),


        corr_prefilter=bool(corr_prefilter),
        corr_abs_min=corr_abs_min,
        corr_min_n=int(corr_min_n),
        corr_prune_sig_mode=corr_prune_sig_mode,

        run_mwu=bool(run_mwu),
        run_cox_uni=bool(run_cox_uni),
        run_cox_multi=bool(run_cox_multi),

        mwu_p_thresh=float(mwu_p),
        mwu_fdr_thresh=float(mwu_q),
        cox_uni_p_thresh=float(coxu_p),
        cox_uni_fdr_thresh=float(coxu_q),
        cox_multi_p_thresh=float(coxm_p),
        cox_multi_fdr_thresh=float(coxm_q),

        cox_time_col=time_col if run_cox_uni or run_cox_multi else None,
        cox_event_col=event_col if run_cox_uni or run_cox_multi else None,
        cox_covariates=cox_covariates,
        cox_hr_abs_min=float(cox_hr_abs_min),
        cox_hr_direction=str(cox_hr_direction),
        cox_min_n=int(cox_min_n),
        cox_min_events=int(cox_min_events),

        corr_threshold=float(fs_corr_threshold),

        sample_id_subset=train_ids,
        enable_subset_fs=bool(enable_subset_fs),
        subset_run_name=subset_run_name,

        limma_pvalue_threshold=float(limma_pvalue_threshold),
        limma_padj_threshold=float(limma_padj_threshold),
        limma_logfc_threshold=float(limma_logfc_threshold),

        fs_min_features=int(fs_min_features),
        fs_topk_if_empty=int(fs_topk_if_empty),
        fs_min_per_block=fs_min_per_block,

        corr_thresholds_by_block=fs_corr_thresholds_by_block,

        mwu_p_thresh_by_block=fs_mwu_p_thresh_by_block,
        mwu_fdr_thresh_by_block=fs_mwu_fdr_thresh_by_block,

        cox_uni_p_thresh_by_block=fs_cox_uni_p_thresh_by_block,
        cox_uni_fdr_thresh_by_block=fs_cox_uni_fdr_thresh_by_block,

        cox_multi_p_thresh_by_block=fs_cox_multi_p_thresh_by_block,
        cox_multi_fdr_thresh_by_block=fs_cox_multi_fdr_thresh_by_block,

        rna_pathway_dbs=fs_rna_pathway_dbs,
        cnv_pathway_dbs=fs_cnv_pathway_dbs,
        mut_pathway_dbs=fs_mut_pathway_dbs,
        ptmb_pathway_dbs=fs_ptmb_pathway_dbs,

        custom_gene_sets_file=custom_gene_sets_file,
        custom_gene_set_mode=custom_gene_set_mode,
        custom_gene_set_min_overlap=int(custom_gene_set_min_overlap),

        mwu_case_label=str(mwu_case_label),
        mwu_ctrl_label=str(mwu_ctrl_label),
    )

    # -----------------
    # 2) Load fold-selected blocks -> TRAIN matrix
    # -----------------
    def _load_fs_block(fname):
        df = _read_csv_if_exists(os.path.join(fs_dir, fname))
        return df

    pieces = []
    for fname in [
        "RNAblock_selected_post_redundancy.csv",
        "CNVblock_selected_post_redundancy.csv",
        "MUTblock_selected_post_redundancy.csv",
        "CLINICALblock_selected_post_redundancy.csv",
    ]:
        df = _load_fs_block(fname)
        if df is not None and isinstance(df, pd.DataFrame) and df.shape[1] > 0:
            pieces.append(df)

    if not pieces:
        raise RuntimeError("[FS] Produced no features in this fold (no *_selected_post_redundancy.csv).")

    Xtr = pd.concat(pieces, axis=1)
    Xtr.index = Xtr.index.astype(str).str.strip()

    # reindex to train_ids exactly
    Xtr = (Xtr.loc[~Xtr.index.duplicated(keep="first")]
              .reindex(train_ids)
              .apply(pd.to_numeric, errors="coerce")
              .fillna(0.0))

    # ALWAYS keep pam50 (if available)
    if isinstance(pam50_all, pd.DataFrame) and not pam50_all.empty:
        Xtr = pd.concat([Xtr, pam50_all.reindex(train_ids).fillna(0.0)], axis=1)

    selected_cols = list(Xtr.columns.astype(str))

    # -----------------
    # 3) Build ALL-blocks matrix from FS full preprocessing artifacts (most up-to-date)
    # -----------------
    out_root = make_output_path(output_dir, subtype)
    full_dir = os.path.join(out_root, "preprocessing_full")

    # core candidates
    rna_pw_all = _read_csv_if_exists(os.path.join(full_dir, "RNAblock_candidates_full.csv"))
    cnv_pw_all = _read_csv_if_exists(os.path.join(full_dir, "CNVblock_candidates_full.csv"))
    mut_all    = _read_csv_if_exists(os.path.join(full_dir, "MUTblock_candidates_full.csv"))
    clin_all   = _read_csv_if_exists(os.path.join(full_dir, "CLINICALblock_candidates_full.csv"))

    # optional add-ons (all already prefixed in FS outputs)
    cib_all     = _read_csv_if_exists(os.path.join(full_dir, "CIBERSORT_full.csv"))
    pga_all     = _read_csv_if_exists(os.path.join(full_dir, "CNV_pct_altered_full.csv"))
    tf_all      = _read_csv_if_exists(os.path.join(full_dir, "TF_full.csv"))
    hoadley_all = _read_csv_if_exists(os.path.join(full_dir, "Hoadley_full.csv"))
    tide_all    = _read_csv_if_exists(os.path.join(full_dir, "TIDE_full.csv"))
    ecot_all    = _read_csv_if_exists(os.path.join(full_dir, "ECOTYPER_full.csv"))
    rna_custom  = _read_csv_if_exists(os.path.join(full_dir, "RNA_custom_sets_full.csv"))
    cnv_gene    = _read_csv_if_exists(os.path.join(full_dir, "CNVblock_genelevel_all.csv"))  # may or may not exist

    # DE genes come from RNA_log2_full.csv (NO prefix in file)
    rna_log2 = _read_csv_if_exists(os.path.join(full_dir, "RNA_log2_full.csv"))
    de_expr_all = None
    if isinstance(rna_log2, pd.DataFrame) and not rna_log2.empty:
        # only keep those selected_cols that look like gene-level (suffix exists in RNA_log2 columns)
        de_genes = []
        rna_log2_cols = set(rna_log2.columns.astype(str))
        for c in selected_cols:
            if c.startswith("RNAblock__"):
                suf = c.split("__", 1)[1]
                if suf in rna_log2_cols:
                    de_genes.append(suf)
        if de_genes:
            tmp = rna_log2.loc[:, [g for g in de_genes if g in rna_log2.columns]].copy()
            tmp.columns = [f"RNAblock__{c}" for c in tmp.columns.astype(str)]
            de_expr_all = tmp

    # concat everything we have
    blocks = [
        rna_pw_all, cnv_pw_all, mut_all, clin_all,
        cib_all, pga_all, tf_all, hoadley_all,
        rna_custom, tide_all, ecot_all, cnv_gene,
        de_expr_all,
        pam50_all,  # keep available for slicing
    ]
    blocks = [b for b in blocks if isinstance(b, pd.DataFrame) and b.shape[1] > 0]

    if not blocks:
        raise RuntimeError(f"[CV] No preprocessing_full blocks found under: {full_dir}")

    X_all_blocks = pd.concat(blocks, axis=1)
    X_all_blocks.index = (X_all_blocks.index.astype(str).str.strip().str.replace(r"\.0$", "", regex=True))
    X_all_blocks = X_all_blocks.loc[~X_all_blocks.index.duplicated(keep="first")]
    X_all_blocks = X_all_blocks.loc[:, ~X_all_blocks.columns.astype(str).str.lower().str.startswith("unnamed")]

    # guard against duplicate column names
    if X_all_blocks.columns.duplicated().any():
        dups = X_all_blocks.columns[X_all_blocks.columns.duplicated()].unique().tolist()
        print(f"[CV] WARNING: {len(dups)} duplicate feature name(s) in assembled blocks; keeping first. e.g. {dups[:10]}")
        X_all_blocks = X_all_blocks.loc[:, ~X_all_blocks.columns.duplicated(keep="first")]

    # ensure all fold-selected cols exist (missing => zeros)
    X_all_blocks = _ensure_cols(X_all_blocks, selected_cols, fill_value=0.0)

    # -----------------
    # 4) Slice VAL/TEST using fold-selected columns
    # -----------------
    def _slice(ids, tag):
        if ids is None:
            return None
        ids = pd.Index(ids).astype(str).str.strip()
        miss = [i for i in ids if i not in X_all_blocks.index]
        if miss:
            print(f"[ALIGN] {tag}: {len(miss)} missing in assembled blocks; zero-filling rows. e.g. {miss[:5]}")
        out = (X_all_blocks.reindex(index=ids, columns=selected_cols)
                          .apply(pd.to_numeric, errors="coerce")
                          .fillna(0.0))
        return out

    Xva = _slice(val_ids, "val_ids")
    Xte = _slice(test_ids, "test_ids") if test_ids is not None else None

    # keep row uniqueness
    Xtr = Xtr.loc[~Xtr.index.duplicated(keep="first")]
    Xva = Xva.loc[~Xva.index.duplicated(keep="first")]
    if Xte is not None:
        Xte = Xte.loc[~Xte.index.duplicated(keep="first")]

    # -----------------
    # 5) Scale non-MUT / non-CLINICAL using fold-local scaler
    # -----------------
    Xtr = Xtr.apply(pd.to_numeric, errors="coerce").astype("float64").fillna(0.0)
    Xva = Xva.apply(pd.to_numeric, errors="coerce").astype("float64").fillna(0.0)
    if Xte is not None:
        Xte = Xte.apply(pd.to_numeric, errors="coerce").astype("float64").fillna(0.0)

    scale_cols = [c for c in selected_cols if not (c.startswith("MUTblock__") or c.startswith("CLINICALblock__"))]
    if len(scale_cols) == 0:
        ct_fold = ColumnTransformer(transformers=[], remainder="passthrough", verbose_feature_names_out=False)
    else:
        ct_fold = ColumnTransformer(
            [("scale", StandardScaler(), scale_cols)],
            remainder="passthrough",
            verbose_feature_names_out=False
        )

    Xtr_sc = pd.DataFrame(ct_fold.fit_transform(Xtr), index=Xtr.index, columns=ct_fold.get_feature_names_out())
    Xtr_sc = Xtr_sc[selected_cols].apply(pd.to_numeric, errors="coerce").astype("float64")

    Xva_sc = pd.DataFrame(ct_fold.transform(Xva), index=Xva.index, columns=ct_fold.get_feature_names_out())
    Xva_sc = Xva_sc[selected_cols].apply(pd.to_numeric, errors="coerce").astype("float64")

    if Xte is not None:
        Xte_sc = pd.DataFrame(ct_fold.transform(Xte), index=Xte.index, columns=ct_fold.get_feature_names_out())
        Xte_sc = Xte_sc[selected_cols].apply(pd.to_numeric, errors="coerce").astype("float64")
    else:
        Xte_sc = None

    return Xtr_sc, Xva_sc, Xte_sc, list(selected_cols), ct_fold, fs_dir, Xtr, Xva, Xte





# ====================== Per-block CV (overlay ROC + hists) ======================
def _cv_models_on_block(
    Xtr_block, ytr, Xva_block, yva,
    block_name, inner_folds, plots_dir, models_dir, fold_label,
    include_xgboost=True, calibrate=False, calibrate_method="sigmoid", calibrate_cv=5,
    Xte_block=None,   # <-- added default so references are safe
):
    import os
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve
    import joblib
    os.makedirs(plots_dir, exist_ok=True)

    grid = _filter_param_grid(
        _build_param_grid_for(
            include_xgboost=include_xgboost,
            fs_enable=False,
            include_stacking=False
        )
    )

    roc_curves = []
    per_model_rows = []
    preds_frames = []

    for model_name, (proto, param_grid) in grid.items():
        try:
            best_est, name_used, inner_auc, best_params = _grid_select_and_fit(
                Xtr_block, ytr, {model_name: (proto, param_grid)}, inner_folds,
                label=f"{fold_label}::{block_name}::{model_name}",
                calibrate=calibrate, calibrate_method=calibrate_method, calibrate_cv=calibrate_cv
            )
        except Exception as e:
            print(f"[{fold_label}][{block_name}] {model_name} failed: {e}")
            continue

        Xtr_block = _coerce_numeric_float32(Xtr_block)
        Xva_block = _coerce_numeric_float32(Xva_block)
        if Xte_block is not None:
            Xte_block = _coerce_numeric_float32(Xte_block)

        proba_va = _proba_pos(best_est, Xva_block)

        try:
            auc_va = roc_auc_score(yva.astype(int), proba_va)
        except Exception:
            auc_va = float("nan")

        mslug = _slug(model_name)
        tag = f"{fold_label}__{block_name}__{mslug}"
        pred_df = pd.DataFrame({
            "sample_id": Xva_block.index.astype(str),
            "y_true": yva.astype(int).values,
            "predicted_proba": proba_va,
            "model_name": model_name,
            "block": block_name,
            "fold": int(re.sub(r"[^0-9]", "", fold_label) or 0)
        })
        pred_df.to_csv(os.path.join(models_dir, f"PRED__{tag}.csv"), index=False)
        preds_frames.append(pred_df)

        try:
            mask = ~pd.isna(pred_df["y_true"])
            fpr, tpr, _ = roc_curve(pred_df.loc[mask, "y_true"], pred_df.loc[mask, "predicted_proba"], pos_label=1)
            roc_curves.append((model_name, fpr, tpr, float(auc_va)))
        except Exception as e:
            print(f"[{fold_label}][{block_name}] ROC failed for {model_name}: {e}")

        try:
            ap = average_precision_score(yva.astype(int), proba_va)
            prec, rec, _ = precision_recall_curve(yva.astype(int), proba_va)
            plt.figure()
            plt.plot(rec, prec, label=f"AP={ap:.3f}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend()
            plt.title(f"PR — {fold_label} — {block_name} — {model_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"pr__{tag}.png"), dpi=200)
            plt.close()
        except Exception as e:
            print(f"[{fold_label}][{block_name}] PR failed for {model_name}: {e}")

        try:
            plt.figure()
            yv = pred_df["y_true"].values.astype(int)
            pv = pred_df["predicted_proba"].values
            for lab, nice in [(0, "Non-recurrent"), (1, "Recurrent")]:
                m = (yv == lab)
                if m.sum() > 0:
                    plt.hist(pv[m], bins=30, alpha=0.6, label=f"{nice} (n={m.sum()})")
            plt.xlim(0, 1)
            plt.xlabel("Predicted probability (recurrent=1)")
            plt.ylabel("Count")
            plt.title(f"Hist — {fold_label} — {block_name} — {model_name}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"hist__{tag}.png"), dpi=200)
            plt.close()
        except Exception as e:
            print(f"[{fold_label}][{block_name}] Hist failed for {model_name}: {e}")

        per_model_rows.append({
            "fold": fold_label,
            "block": block_name,
            "model_name": model_name,
            "inner_auc_mean": float(inner_auc),
            "val_auc": float(auc_va),
            "params": best_params,
            "estimator": best_est
        })

        # save each best per-block model with meta
        try:
            dump_with_meta(best_est, os.path.join(models_dir, f"BEST__{tag}.pkl"), joblib)
        except Exception as e:
            print(f"[{fold_label}][{block_name}] Could not save best model: {e}")

    try:
        if roc_curves:
            import matplotlib.pyplot as plt
            plt.figure()
            for name, fpr, tpr, aucv in roc_curves:
                plt.plot(fpr, tpr, label=f"{name} (AUC={aucv:.3f})", linewidth=1.4)
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC (overlay) — {fold_label} — {block_name}")
            plt.legend(loc="lower right", fontsize=8)
            plt.tight_layout()
            outp = os.path.join(plots_dir, f"roc_overlay__{fold_label}__{block_name}.png")
            plt.savefig(outp, dpi=210)
            plt.close()
    except Exception as e:
        print(f"[{fold_label}][{block_name}] ROC overlay failed: {e}")

    df_rows = pd.DataFrame([{k: v for k, v in r.items() if k != "estimator"} for r in per_model_rows])
    preds = pd.concat(preds_frames, axis=0, ignore_index=True) if preds_frames else pd.DataFrame()
    best_fit = None
    if per_model_rows:
        best_fit = max(per_model_rows, key=lambda d: (d["val_auc"] if np.isfinite(d["val_auc"]) else -np.inf))["estimator"]
    return df_rows, preds, best_fit



# ====================== Fit block-aware StackingClassifier ======================


def fit_stack_from_block_bests(
    Xtr_full, ytr, Xva_full, yva,
    best_by_block: dict,
    plots_dir, models_dir, fold_label,
    meta_model="logreg", meta_kwargs=None, cv=5,
    calibrate_stack=False, calibrate_method="sigmoid", calibrate_cv=5
):
    import os
    import numpy as np
    import re
    import pandas as pd
    import matplotlib.pyplot as plt
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import StackingClassifier
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
    from sklearn.base import clone

    os.makedirs(plots_dir, exist_ok=True)

    if meta_kwargs is None:
        meta_kwargs = {}

    # Meta model (final estimator)
    if str(meta_model).lower() in ("logreg", "logistic", "lr"):
        meta = LogisticRegression(solver="liblinear", penalty="l2", max_iter=5000, **meta_kwargs)
    else:
        meta = LogisticRegression(solver="liblinear", penalty="l2", max_iter=5000, **meta_kwargs)

    estimators = []
    for block_name, best_est in best_by_block.items():
        if best_est is None:
            continue
        wrapped = Pipeline([
            ("sel", PrefixSelector([_BLOCK_PREFIX[block_name]])),
            ("model", clone(best_est))
        ])
        estimators.append((f"{block_name.lower()}", wrapped))

    if not estimators:
        print(f"[{fold_label}][STACK] No base estimators provided; skipping stacking.")
        return None, None

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        cv=cv,
        passthrough=False,
        n_jobs=1
    )

    model_to_use = stack
    if calibrate_stack:
        try:
            model_to_use = CalibratedClassifierCV(estimator=stack, method=calibrate_method, cv=calibrate_cv)
        except Exception as e:
            print(f"[{fold_label}][STACK] Calibration wrapper failed, using uncalibrated stack: {e}")
            model_to_use = stack

    # Fit model used for predictions/scoring (may be calibrated)
    model_to_use.fit(Xtr_full, ytr)

    proba_va = _proba_pos(model_to_use, Xva_full)
    auc_va = float(roc_auc_score(yva.astype(int), proba_va))

    tag = f"{fold_label}__STACK_BLOCKS"
    pred_df = pd.DataFrame({
        "sample_id": Xva_full.index.astype(str),
        "y_true": yva.astype(int).values,
        "predicted_proba": proba_va,
        "model_name": "StackingClassifier(block-aware)",
        "block": "STACK",
        "fold": int(re.sub(r"[^0-9]", "", fold_label) or 0)
    })
    pred_df.to_csv(os.path.join(models_dir, f"PRED__{tag}.csv"), index=False)

    # ----------------------------
    # NEW: Export META coefficients + SHAP (meta-level)
    # ----------------------------
    try:
        # For interpretability, we want a fitted *StackingClassifier* with transform().
        # If we're calibrated, model_to_use is CalibratedClassifierCV and doesn't expose transform(),
        # so we fit an uncalibrated stack clone just for coef/SHAP exports.
        if hasattr(model_to_use, "transform") and hasattr(model_to_use, "final_estimator_"):
            stack_interp = model_to_use
        else:
            stack_interp = clone(stack).fit(Xtr_full, ytr)

        # Meta-feature matrix (what the meta model actually sees)
        Ztr = stack_interp.transform(Xtr_full)
        Zva = stack_interp.transform(Xva_full)

        # Feature names for meta-features
        try:
            meta_feat_names = list(stack_interp.get_feature_names_out())
        except Exception:
            meta_feat_names = [name for name, _ in estimators]
            if Ztr.shape[1] != len(meta_feat_names):
                meta_feat_names = [f"meta_f{i}" for i in range(Ztr.shape[1])]

        Ztr_df = pd.DataFrame(Ztr, index=Xtr_full.index, columns=meta_feat_names)
        Zva_df = pd.DataFrame(Zva, index=Xva_full.index, columns=meta_feat_names)

        final_est = getattr(stack_interp, "final_estimator_", None)

        # --- Coefficients (LogReg) ---
        if final_est is not None and hasattr(final_est, "coef_"):
            coef = np.asarray(final_est.coef_).ravel()
            intercept = float(np.asarray(final_est.intercept_).ravel()[0]) if hasattr(final_est, "intercept_") else np.nan

            coef_df = pd.DataFrame({
                "meta_feature": meta_feat_names,
                "coef": coef,
                "abs_coef": np.abs(coef),
            }).sort_values("abs_coef", ascending=False)
            coef_df["intercept"] = intercept
            coef_df.to_csv(os.path.join(models_dir, f"META_COEF__{tag}.csv"), index=False)

            # quick bar plot (top 30)
            top = coef_df.head(30).iloc[::-1]
            plt.figure(figsize=(8, max(3, 0.25 * len(top))))
            plt.barh(top["meta_feature"], top["coef"])
            plt.xlabel("Meta LR coefficient (log-odds scale)")
            plt.title(f"Meta coefficients — {tag}")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"meta_coef__{tag}.png"), dpi=210)
            plt.close()
        else:
            print(f"[{fold_label}][STACK][META] Final estimator has no coef_ (not LogisticRegression?). Skipping coef export.")

        # --- SHAP (meta-level) ---
        # This explains final_est predictions using meta-features Z = base-model outputs.
        try:
            import shap  # optional dependency

            if final_est is not None:
                explainer = shap.LinearExplainer(final_est, Ztr_df, feature_perturbation="interventional")
                shap_tr = explainer.shap_values(Ztr_df)
                shap_va = explainer.shap_values(Zva_df)

                # shap may return list (per class). For binary, take class 1 if list-like.
                if isinstance(shap_tr, list):
                    shap_tr = shap_tr[1] if len(shap_tr) > 1 else shap_tr[0]
                if isinstance(shap_va, list):
                    shap_va = shap_va[1] if len(shap_va) > 1 else shap_va[0]

                shap_tr = np.asarray(shap_tr)
                shap_va = np.asarray(shap_va)

                shap_tr_df = pd.DataFrame(shap_tr, index=Ztr_df.index, columns=meta_feat_names)
                shap_va_df = pd.DataFrame(shap_va, index=Zva_df.index, columns=meta_feat_names)

                shap_tr_df.to_csv(os.path.join(models_dir, f"META_SHAP_VALUES_TRAIN__{tag}.csv"))
                shap_va_df.to_csv(os.path.join(models_dir, f"META_SHAP_VALUES_VALID__{tag}.csv"))

                meanabs = pd.DataFrame({
                    "meta_feature": meta_feat_names,
                    "mean_abs_shap_train": np.abs(shap_tr).mean(axis=0),
                    "mean_abs_shap_valid": np.abs(shap_va).mean(axis=0),
                }).sort_values("mean_abs_shap_valid", ascending=False)
                meanabs.to_csv(os.path.join(models_dir, f"META_SHAP_MEANABS__{tag}.csv"), index=False)

                top = meanabs.head(30).iloc[::-1]
                plt.figure(figsize=(8, max(3, 0.25 * len(top))))
                plt.barh(top["meta_feature"], top["mean_abs_shap_valid"])
                plt.xlabel("Mean |SHAP| (valid, meta-level)")
                plt.title(f"Meta SHAP importance — {tag}")
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"meta_shap_meanabs__{tag}.png"), dpi=210)
                plt.close()
        except ImportError:
            print(f"[{fold_label}][STACK][META] 'shap' not installed; skipping SHAP export.")
        except Exception as e:
            print(f"[{fold_label}][STACK][META] SHAP export failed: {e}")

    except Exception as e:
        print(f"[{fold_label}][STACK][META] Meta coef/SHAP export block failed: {e}")

    # ----------------------------
    # Existing plots
    # ----------------------------
    fpr, tpr, _ = roc_curve(yva.astype(int), proba_va, pos_label=1)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc_va:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC — {tag}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"roc__{tag}.png"), dpi=210)
    plt.close()

    prec, rec, _ = precision_recall_curve(yva.astype(int), proba_va)
    ap = average_precision_score(yva.astype(int), proba_va)
    plt.figure()
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.title(f"PR — {tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"pr__{tag}.png"), dpi=210)
    plt.close()

    plt.figure()
    yv = yva.astype(int).values
    for lab, nice in [(0, "Non-recurrent"), (1, "Recurrent")]:
        m = (yv == lab)
        if m.sum() > 0:
            plt.hist(proba_va[m], bins=30, alpha=0.6, label=f"{nice} (n={m.sum()})")
    plt.xlim(0, 1)
    plt.xlabel("Predicted probability (recurrent=1)")
    plt.ylabel("Count")
    plt.title(f"Hist — {tag}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"hist__{tag}.png"), dpi=210)
    plt.close()

    try:
        dump_with_meta(model_to_use, os.path.join(models_dir, f"STACK_BLOCKS__{fold_label}.pkl"), joblib)
    except Exception as e:
        print(f"[{fold_label}][STACK] Could not save stacked model: {e}")

    return model_to_use, auc_va


# ====================== TRAIN (with OUTER CV) ======================
def train_once_concat(args):
    import numpy as np
    import pandas as pd
    import joblib
    import matplotlib
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
    )

    # --- custom RNA set config (same as in _build_fold_matrices_with_fs) ---
    CUSTOM_GENE_SETS_FILE = "./data/RNAseq/genescores/custom_genesets.gmt"
    CUSTOM_GENE_SET_MODE = "median_log2"
    CUSTOM_GENE_SET_MIN_OVERLAP = 1

    # ---- sanity: outer CV only ----
    if int(getattr(args, "outer_folds", 0)) < 2:
        raise ValueError("This script supports ONLY outer CV. Use --outer_folds >= 2.")

    # ---- dirs ----
    fold_dirs = build_fold_dirs(args.output_dir, args.subtype)
    print("[DEBUG] output root:", fold_dirs["root"])
    print(f"[ENV] sys.executable = {sys.executable}")
    print(f"[ENV] sklearn = {sklearn.__version__}")

    models_dir = fold_dirs["models"]
    prep_dir = fold_dirs["preprocessing"]
    shap_dir = fold_dirs["shap"]
    coef_dir = fold_dirs["coefficients"]
    plots_dir = fold_dirs["plots"]
    mat_dir = fold_dirs["matrices"]

    # ---- load & dedup once ----
    data_all, all_pathway_dicts = load_all_data(args.data_path, args.subtype, args.pathway_options)
    cleaned_data, subtype_meta, y_all_enc, all_ids, _ = dedup_and_align_data(
        data_all, status_col=args.status_col, prep_dir=prep_dir, raise_on_conflict=True
    )
    subtype_meta = cleaned_data["meta"].copy()

    if "time_to_event" not in subtype_meta.columns:
        if ("time_recur" in subtype_meta.columns) or ("time_norecur" in subtype_meta.columns):
            subtype_meta["time_recur"] = subtype_meta.get("time_recur", pd.Series(index=subtype_meta.index, dtype=float))
            subtype_meta["time_norecur"] = subtype_meta.get("time_norecur", pd.Series(index=subtype_meta.index, dtype=float))
            subtype_meta["time_to_event"] = subtype_meta["time_recur"].fillna(subtype_meta["time_norecur"])
        else:
            subtype_meta["time_to_event"] = np.nan
    subtype_meta.to_csv(os.path.join(prep_dir, "subtype_meta_full.csv"))

    _dump_json_safely(
        {
            "subtype": args.subtype,
            "pathway_options": args.pathway_options,
            "outer_folds": int(args.outer_folds),
            "inner_folds": int(args.inner_folds),
        },
        os.path.join(prep_dir, "run_config.json"),
    )

    # === OUTER CV ===
    skf_outer = StratifiedKFold(n_splits=int(args.outer_folds), shuffle=True, random_state=42)
    outer_results, fold_preds = [], []
    all_va_y, all_va_p = [], []
    best_overall = {"auc_va": -1.0, "fold": None, "name": None, "params": None, "model": None}

    for fold_idx, (tr, va) in enumerate(skf_outer.split(np.zeros(len(all_ids)), y_all_enc), start=1):
        tr_ids = all_ids[tr]
        va_ids = all_ids[va]
        ytr = y_all_enc.loc[tr_ids]
        yva = y_all_enc.loc[va_ids]
        label = f"OUTER_FOLD_{fold_idx:02d}"
        print(f"[OUTER] {label}: train={len(tr_ids)}  val={len(va_ids)}")

        # === Build matrices with FS on TRAIN only ===
        Xtr_sc, Xva_sc, Xte_sc, selected_cols, ct_fold, fs_dir, Xtr_raw, Xva_raw, Xte_raw = _build_fold_matrices_with_fs(
        data_path=args.data_path,
        output_dir=args.output_dir,
        subtype=args.subtype,
        status_col=args.status_col,
        pathway_options=args.pathway_options,
        RNAgene_bypass=getattr(args, "RNAgene_bypass", False),
        CNVgene_bypass=getattr(args, "CNVgene_bypass", False),

        train_ids=tr_ids,
        val_ids=va_ids,
        test_ids=None,

        enable_subset_fs=True,
        subset_run_name=label,

        primary_test=args.primary_test,
        time_col=args.time_col,
        event_col=args.event_col,
        cox_covariates=getattr(args, "cox_covariates", None),



        fs_p_thresh=args.fs_p_thresh,
        fs_fdr_thresh=args.fs_fdr_thresh,
        fs_corr_threshold=args.fs_corr_threshold,

        limma_pvalue_threshold=args.limma_pvalue_threshold,
        limma_padj_threshold=args.limma_padj_threshold,
        limma_logfc_threshold=args.limma_logfc_threshold,

        fs_min_features=args.fs_min_features,
        fs_topk_if_empty=args.fs_topk_if_empty,

        cox_hr_abs_min=args.cox_hr_abs_min,
        cox_hr_direction=args.cox_hr_direction,

        prep_dir=prep_dir,
        cleaned_data=cleaned_data,
        all_pathway_dicts=all_pathway_dicts,
        pam50_col=args.pam50_col,

        rna_gene_min_raw_count=getattr(args, "rna_gene_min_raw_count", 10),
        rna_gene_min_detected_frac=getattr(args, "rna_gene_min_detected_frac", 0.25),
        cnv_gene_min_detected_frac=getattr(args, "cnv_gene_min_detected_frac", 0.20),
        cnv_arm_min_detected_frac=getattr(args, "cnv_arm_min_detected_frac", 0.20),
        cnv_band_min_detected_frac=getattr(args, "cnv_band_min_detected_frac", 0.20),
        mut_min_detected_frac=getattr(args, "mut_min_detected_frac", 0.10),

        # (optional, only if you added these to argparse)
        # ALWAYS run all supervised FS tests
        run_mwu=True,
        run_cox_uni=True,
        run_cox_multi=True,


        mwu_p_thresh=getattr(args, "mwu_p_thresh", None),
        mwu_fdr_thresh=getattr(args, "mwu_fdr_thresh", None),
        cox_uni_p_thresh=getattr(args, "cox_uni_p_thresh", None),
        cox_uni_fdr_thresh=getattr(args, "cox_uni_fdr_thresh", None),
        cox_multi_p_thresh=getattr(args, "cox_multi_p_thresh", None),
        cox_multi_fdr_thresh=getattr(args, "cox_multi_fdr_thresh", None),

        var_min=getattr(args, "var_min", 0.0),
        var_min_nonmissing=getattr(args, "var_min_nonmissing", 10),
        variance_keep_file=getattr(args, "variance_keep_file", None),
        var_keep_top_frac=getattr(args, "var_keep_top_frac", None),

        corr_prefilter=getattr(args, "corr_prefilter", False),
        corr_abs_min=getattr(args, "corr_abs_min", None),
        corr_min_n=getattr(args, "corr_min_n", 10),

        include_mutated_pathways=getattr(args, "include_mutated_pathways", True),
        mut_pathway_source=getattr(args, "mut_pathway_source", "snv"),
        mut_binary_matrix_csv=getattr(args, "mut_binary_matrix_csv", None),
        mut_binary_sample_col=getattr(args, "mut_binary_sample_col", None),
        mut_binary_prefix=getattr(args, "mut_binary_prefix", "MUTblock__"),

        enable_ptmb=getattr(args, "enable_ptmb", False),
        ptmb_source=getattr(args, "ptmb_source", "none"),
        ptmb_matrix_csv=getattr(args, "ptmb_matrix_csv", None),
        ptmb_sample_col=getattr(args, "ptmb_sample_col", None),
        ptmb_prefix=getattr(args, "ptmb_prefix", "MUTblock__pTMB__"),
        ptmb_transform=getattr(args, "ptmb_transform", "none"),

        ptmb_maf_path=getattr(args, "ptmb_maf_path", None),
        ptmb_maf_sample_col=getattr(args, "ptmb_maf_sample_col", None),
        ptmb_maf_gene_col=getattr(args, "ptmb_maf_gene_col", None),
        ptmb_gtf_path=getattr(args, "ptmb_gtf_path", None),
        ptmb_cds_cache_csv=getattr(args, "ptmb_cds_cache_csv", None),
        ptmb_denom_policy=getattr(args, "ptmb_denom_policy", "all_genes_with_len"),
        ptmb_min_genes_with_len=getattr(args, "ptmb_min_genes_with_len", 1),
        ptmb_gmt_path=getattr(args, "ptmb_gmt_path", None),

        protein_coding_gtf=getattr(args, "protein_coding_gtf", None),

        fs_corr_thresholds_by_block=getattr(args, "fs_corr_thresholds_by_block", None),

        fs_mwu_p_thresh_by_block=getattr(args, "fs_mwu_p_thresh_by_block", None),
        fs_mwu_fdr_thresh_by_block=getattr(args, "fs_mwu_fdr_thresh_by_block", None),

        fs_cox_uni_p_thresh_by_block=getattr(args, "fs_cox_uni_p_thresh_by_block", None),
        fs_cox_uni_fdr_thresh_by_block=getattr(args, "fs_cox_uni_fdr_thresh_by_block", None),

        fs_cox_multi_p_thresh_by_block=getattr(args, "fs_cox_multi_p_thresh_by_block", None),
        fs_cox_multi_fdr_thresh_by_block=getattr(args, "fs_cox_multi_fdr_thresh_by_block", None),

        fs_rna_pathway_dbs=getattr(args, "fs_rna_pathway_dbs", None),
        fs_cnv_pathway_dbs=getattr(args, "fs_cnv_pathway_dbs", None),
        fs_mut_pathway_dbs=getattr(args, "fs_mut_pathway_dbs", None),
        fs_ptmb_pathway_dbs=getattr(args, "fs_ptmb_pathway_dbs", None),




    )


        ytr = y_all_enc.reindex(Xtr_sc.index)
        yva = y_all_enc.reindex(Xva_sc.index)

        print(f"[OUTER] Fold {fold_idx:02d}: "
              f"Xtr_sc={len(Xtr_sc)} | ytr={len(ytr)} | "
              f"Xva_sc={len(Xva_sc)} | yva={len(yva)}")

        # ====== Save per-fold artifacts (with meta) ======
        dump_with_meta(ct_fold, os.path.join(prep_dir, f"column_transformer_{label}.joblib"), joblib)

        pd.Series(selected_cols, name="feature").to_csv(
            os.path.join(prep_dir, f"selected_features_{label}.csv"), index=False
        )
        scale_cols = [c for c in selected_cols if not (c.startswith("MUTblock__") or c.startswith("CLINICALblock__"))]
        pd.Series(scale_cols, name="scaled_feature").to_csv(
            os.path.join(prep_dir, f"scaled_columns_{label}.csv"), index=False
        )
        pd.Index(Xtr_sc.index, name="sample_id").to_series().to_csv(
            os.path.join(prep_dir, f"train_ids_{label}.csv"), index=False)
        pd.Index(Xva_sc.index, name="sample_id").to_series().to_csv(
            os.path.join(prep_dir, f"val_ids_{label}.csv"), index=False)

        _save_selected_matrix(Xtr_raw, os.path.join(mat_dir, f"X_train_unscaled_{label}.csv"))
        _save_selected_matrix(Xva_raw, os.path.join(mat_dir, f"X_val_unscaled_{label}.csv"))
        _save_selected_matrix(Xtr_sc, os.path.join(mat_dir, f"X_train_scaled_{label}.csv"))
        _save_selected_matrix(Xva_sc, os.path.join(mat_dir, f"X_val_scaled_{label}.csv"))

        fs_snap = os.path.join(prep_dir, f"fs_snapshot_{label}")
        os.makedirs(fs_snap, exist_ok=True)
        for src in glob.glob(os.path.join(fs_dir, "*.csv")) + glob.glob(os.path.join(fs_dir, "*.json")):
            shutil.copy2(src, fs_snap)

        # ====== NEW: Per-block CV ======
        block_summ_rows, block_preds_frames = [], []
        best_by_block = {}

        run_blocks = [b.strip().upper() for b in getattr(args, "cv_blocks", "RNA,CNV,MUT,CLINICAL").split(",") if b.strip()]
        for blk in run_blocks:
            blk_cols = _block_columns(selected_cols, blk)
            if len(blk_cols) == 0:
                print(f"[{label}] Skip block {blk}: no selected features.")
                best_by_block[blk] = None
                continue

            Xtr_b = Xtr_sc.loc[:, blk_cols]
            Xva_b = Xva_sc.loc[:, blk_cols]

            rows, preds, best_est = _cv_models_on_block(
                Xtr_b, ytr, Xva_b, yva,
                block_name=blk,
                inner_folds=int(args.inner_folds),
                plots_dir=plots_dir,
                models_dir=models_dir,
                fold_label=label,
                include_xgboost=bool(args.include_xgboost),
                calibrate=bool(args.calibrate),
                calibrate_method=args.calibrate_method,
                calibrate_cv=int(args.calibrate_cv),
            )
            if rows is not None and not rows.empty:
                block_summ_rows.append(rows)
            if preds is not None and not preds.empty:
                block_preds_frames.append(preds)
            best_by_block[blk] = best_est

        if block_summ_rows:
            pd.concat(block_summ_rows, axis=0, ignore_index=True)\
                .to_csv(os.path.join(models_dir, f"OUTER_SUMMARY__by_block__{label}.csv"), index=False)
        if block_preds_frames:
            pd.concat(block_preds_frames, axis=0, ignore_index=True)\
                .to_csv(os.path.join(models_dir, f"PRED_ALL__by_block__{label}.csv"), index=False)

        # ====== Choose base models for stacking ======
        stack_mode = str(getattr(args, "stack_base_mode", "best_per_block")).lower()
        if stack_mode == "fixed_family":
            fam = getattr(args, "stack_base_family", "LogReg_L2_liblinear")
            best_for_stack = {}
            for blk in run_blocks:
                # Skip blocks explicitly excluded from stacking (e.g. CLINICAL)
                if blk in _STACK_EXCLUDE_BLOCKS:
                    print(f"[{label}] Fixed-family stacking: skipping excluded block {blk}.")
                    continue

                blk_cols = _block_columns(selected_cols, blk)
                if len(blk_cols) == 0:
                    print(f"[{label}] Fixed-family stacking: block {blk} has no features.")
                    continue

                Xtr_b = Xtr_sc.loc[:, blk_cols]
                try:
                    best_for_stack[blk] = _best_estimator_for_family(
                        Xtr_b, ytr,
                        family_name=fam,
                        inner_folds=int(args.inner_folds),
                        include_xgboost=bool(args.include_xgboost),
                        calibrate=bool(args.calibrate),
                        calibrate_method=args.calibrate_method,
                        calibrate_cv=int(args.calibrate_cv),
                    )
                except Exception as e:
                    print(f"[{label}] Fixed-family tuning failed on {blk} ({fam}): {e}")
        else:
            # Use per-block bests but drop any blocks we don't want in the stack
            best_for_stack = {
                blk: est
                for blk, est in best_by_block.items()
                if (blk not in _STACK_EXCLUDE_BLOCKS) and (est is not None)
            }


        # ====== Stacked classifier (block-aware) ======
        try:
            _stack_model, _stack_auc = fit_stack_from_block_bests(
                Xtr_full=Xtr_sc, ytr=ytr, Xva_full=Xva_sc, yva=yva,
                best_by_block=best_for_stack,
                plots_dir=plots_dir,
                models_dir=models_dir,
                fold_label=label,
                meta_model=getattr(args, "stack_meta_model", "logreg"),
                meta_kwargs={},
                cv=max(3, int(args.stack_cv)),
                calibrate_stack=bool(getattr(args, "stack_calibrate", False)),
                calibrate_method=getattr(args, "stack_calibrate_method", "sigmoid"),
                calibrate_cv=int(getattr(args, "stack_calibrate_cv", 5)),
            )

            if _stack_model is not None:
                outer_results.append({
                    "fold": fold_idx,
                    "model_name": "StackingClassifier(block-aware)",
                    "inner_auc_mean": float("nan"),
                    "val_auc": float(_stack_auc),
                    "params": {
                        "meta": getattr(args, "stack_meta_model", "logreg"),
                        "bases": ("fixed_family:" + getattr(args, "stack_base_family", ""))
                                if stack_mode == "fixed_family" else "best-per-block"
                    }
                })

                stack_pred_csv = os.path.join(models_dir, f"PRED__{label}__STACK_BLOCKS.csv")
                if os.path.exists(stack_pred_csv):
                    fold_preds.append(pd.read_csv(stack_pred_csv))

                # --- META EXPORTS (to coeff/ and shap/ dirs) ---
                stack_core = unwrap_stacking_model(_stack_model)

                if hasattr(stack_core, "final_estimator_") and hasattr(stack_core, "transform"):
                    meta_model = stack_core.final_estimator_

                    # meta features for this fold (use val for fold-specific explanation; or train if you prefer)
                    X_meta_va = stack_core.transform(Xva_sc)
                    meta_feature_names = infer_stack_meta_feature_names(stack_core, X_meta_va.shape[1])

                    export_meta_coefficients(
                        meta_model,
                        meta_feature_names,
                        coeff_dir=coeff_dir,
                        prefix=f"{label}__META"
                    )
                    export_meta_shap(
                        meta_model,
                        X_meta_eval=X_meta_va,
                        meta_feature_names=meta_feature_names,
                        shap_dir=shap_dir,
                        prefix=f"{label}__META"
                    )
                else:
                    print(f"[{label}][STACK] No final_estimator_/transform available; cannot export meta coeff/SHAP.")

        except Exception as e:
            print(f"[{label}][STACK] Failed to fit/eval stacked classifier: {e}")

        # ====== Train ALL models (concat) ======
        concat_param_grid = _filter_param_grid(
            _build_param_grid_for(
                subtype=args.subtype,
                include_xgboost=args.include_xgboost,
                fs_enable=False,
                include_stacking=args.include_stacking,
            )
        )

        per_model_rows, per_model_preds = [], []

        for model_name, (proto, grid) in concat_param_grid.items():
            try:
                m_est, m_name, m_inner_auc, m_params = _grid_select_and_fit(
                    Xtr_sc, ytr, {model_name: (proto, grid)}, args.inner_folds,
                    label=f"{label}::{model_name}",
                    calibrate=args.calibrate, calibrate_method=args.calibrate_method, calibrate_cv=args.calibrate_cv
                )
            except Exception as e:
                print(f"[{label}] {model_name} failed during inner CV: {e}")
                continue

            m_proba_va = _proba_pos(m_est, Xva_sc)
            try:
                m_auc_va = roc_auc_score(yva.values.astype(int), m_proba_va)
            except Exception:
                m_auc_va = float("nan")

            mslug = _slug(model_name)
            fold_tag = f"{mslug}__{label}"

            try:
                _save_shap_if_any(m_est, Xtr_sc.loc[:, selected_cols], shap_dir, tag=fold_tag)
            except Exception as e:
                print(f"[{label}][WARN] SHAP failed for {model_name}: {e}")

            try:
                _save_coefficients_if_any(m_est, list(Xtr_sc.columns),
                                          os.path.join(coef_dir, f"coef_{fold_tag}.csv"))
            except Exception as e:
                print(f"[{label}][WARN] Coef export failed for {model_name}: {e}")

            try:
                dump_with_meta(m_est, os.path.join(models_dir, f"{fold_tag}.pkl"), joblib)
            except Exception as e:
                print(f"[{label}] Could not save model {model_name}: {e}")

            m_pred_df = pd.DataFrame({
                "sample_id": Xva_sc.index.astype(str),
                "y_true": yva.values.astype(int),
                "predicted_proba": m_proba_va,
                "model_name": model_name,
                "fold": fold_idx
            })
            m_pred_df["predicted_label"] = (
                m_pred_df["predicted_proba"] >= float(getattr(args, "prob_threshold", 0.5))
            ).astype(int)
            m_pred_df.to_csv(os.path.join(models_dir, f"{fold_tag}__preds.csv"), index=False)

            try:
                mask = ~pd.isna(m_pred_df["y_true"])
                fpr, tpr, _ = roc_curve(
                    m_pred_df.loc[mask, "y_true"].astype(int),
                    m_pred_df.loc[mask, "predicted_proba"], pos_label=1
                )
                auc_val = roc_auc_score(
                    m_pred_df.loc[mask, "y_true"].astype(int),
                    m_pred_df.loc[mask, "predicted_proba"]
                )
                plt.figure()
                plt.plot(fpr, tpr, label=f"AUC={auc_val:.3f}")
                plt.plot([0, 1], [0, 1], linestyle="--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC — {label} — {model_name}")
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"roc_{fold_tag}.png"), dpi=200)
                plt.close()
            except Exception as e:
                print(f"[{label}][WARN] ROC failed for {model_name}: {e}")

            try:
                ap = average_precision_score(yva.astype(int), m_proba_va)
                prec, rec, _ = precision_recall_curve(yva.astype(int), m_proba_va)
                plt.figure()
                plt.plot(rec, prec, label=f"AP={ap:.3f}")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(f"PR — {label} — {model_name}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"pr_{fold_tag}.png"), dpi=200)
                plt.close()
            except Exception as e:
                print(f"[{label}][WARN] PR failed for {model_name}: {e}")

            per_model_rows.append({
                "fold": fold_idx,
                "model_name": model_name,
                "inner_auc_mean": float(m_inner_auc),
                "val_auc": float(m_auc_va),
                "params": m_params
            })
            per_model_preds.append(m_pred_df)

        if per_model_rows:
            summary_txt = os.path.join(models_dir, f"CV_SUMMARY__{label}.txt")
            with open(summary_txt, "w") as f:
                for row in per_model_rows:
                    f.write(f"{row['model_name']}: inner_auc={row['inner_auc_mean']:.4f}, "
                            f"val_auc={row['val_auc']:.4f}\n")

        if per_model_preds:
            fold_preds.append(pd.concat(per_model_preds, axis=0, ignore_index=True))
        outer_results.extend(per_model_rows)

        if per_model_rows:
            fold_best = max(per_model_rows, key=lambda r: (r["val_auc"] if np.isfinite(r["val_auc"]) else -np.inf))
            best_model_name_fold = fold_best["model_name"]
            best_pred_df = [d for d in per_model_preds if d["model_name"].iloc[0] == best_model_name_fold][0]
            proba_va = best_pred_df["predicted_proba"].values
            auc_va = float(fold_best["val_auc"])
            if np.isfinite(auc_va) and auc_va > best_overall["auc_va"]:
                best_overall.update({
                    "auc_va": float(auc_va),
                    "fold": fold_idx,
                    "name": best_model_name_fold,
                    "params": fold_best["params"],
                    "model": None
                })
            all_va_y.append(yva.values.astype(int))
            all_va_p.append(proba_va)

    # === Per-model aggregation & plots ===
    outer_df = pd.DataFrame(outer_results)
    outer_df.to_csv(os.path.join(models_dir, "OUTER_CV_summary.csv"), index=False)

    all_val_preds = pd.concat(fold_preds, axis=0, ignore_index=True) if fold_preds else pd.DataFrame()
    if not all_val_preds.empty:
        all_val_preds.to_csv(os.path.join(models_dir, "OUTER_CV_all_val_preds_by_model.csv"), index=False)

    if not outer_df.empty:
        per_model_summary = (outer_df.groupby("model_name", as_index=False)
                             .agg(mean_auc=("val_auc", "mean"),
                                  sd_auc=("val_auc", "std"),
                                  median_auc=("val_auc", "median"),
                                  n=("val_auc", "size"))
                             .sort_values("mean_auc", ascending=False))
        per_model_summary.to_csv(os.path.join(models_dir, "OUTER_CV_summary_by_model.csv"), index=False)
    else:
        per_model_summary = pd.DataFrame(columns=["model_name", "mean_auc", "sd_auc", "median_auc", "n"])

    try:
        if not per_model_summary.empty:
            models = per_model_summary["model_name"].tolist()
            data = [outer_df.loc[outer_df["model_name"] == m, "val_auc"].values for m in models]
            plt.figure(figsize=(10, 5))
            plt.boxplot(data, labels=models, showfliers=False)
            for i, vals in enumerate(data, start=1):
                if len(vals) == 0:
                    continue
                x = np.random.normal(loc=i, scale=0.06, size=len(vals))
                plt.scatter(x, vals, s=20, alpha=0.6)
            plt.ylabel("Outer-fold AUC")
            plt.title("Outer CV — AUC by Model")
            plt.ylim(0.0, 1.0)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "auc_by_model_box_jitter.png"), dpi=200)
            plt.close()
    except Exception as e:
        print(f"[PER-MODEL][WARN] AUC box/jitter failed: {e}")

    # KM on OOF predictions
    try:
        if not all_val_preds.empty:
            from lifelines import KaplanMeierFitter
            from lifelines.statistics import logrank_test
            oof = all_val_preds.copy()
            oof["sample_id"] = oof["sample_id"].astype(str).str.strip()
            meta_aligned = subtype_meta.copy()
            meta_aligned.index = meta_aligned.index.astype(str).str.strip()
            oof = oof.merge(meta_aligned[["time_to_event"]], left_on="sample_id", right_index=True, how="left")

            for model_name, group in oof.groupby("model_name"):
                t = pd.to_numeric(group["time_to_event"], errors="coerce")
                e = pd.to_numeric(group["y_true"], errors="coerce")
                ok = (~t.isna()) & (~e.isna())
                if ok.sum() == 0:
                    print(f"[PER-MODEL] KM skipped for {model_name}: no time_to_event+event rows.")
                    continue
                thr = float(np.nanmedian(group.loc[ok, "predicted_proba"].values))
                grp = (group["predicted_proba"] >= thr).astype(int).values
                km = KaplanMeierFitter()
                plt.figure()
                mask0 = ok.values & (grp == 0)
                if mask0.sum() > 0:
                    km.fit(t[mask0], e[mask0], label=f"Low (< {thr:.2f}) (n={int(mask0.sum())}, events={int(e[mask0].sum())})")
                    ax = km.plot(ci_show=False)
                mask1 = ok.values & (grp == 1)
                if mask1.sum() > 0:
                    km.fit(t[mask1], e[mask1], label=f"High (≥ {thr:.2f}) (n={int(mask1.sum())}, events={int(e[mask1].sum())})")
                    km.plot(ci_show=False)
                title = f"KM by OOF risk — {model_name}"
                if mask0.sum() > 0 and mask1.sum() > 0:
                    lr = logrank_test(t[mask0], t[mask1], event_observed_A=e[mask0], event_observed_B=e[mask1])
                    title += f"  (log-rank p = {lr.p_value:.3e})"
                plt.title(title)
                plt.xlabel("Time to event (months)")
                plt.ylabel("Survival probability")
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"km_outer_cv_by_pred__{_slug(model_name)}.png"), dpi=200)
                plt.close()
    except Exception as e:
        print(f"[PER-MODEL][WARN] KM by model failed: {e}")

    # aggregate ROC across folds for fold-best models
    try:
        if all_va_y and all_va_p:
            y_cat = np.concatenate(all_va_y)
            p_cat = np.concatenate(all_va_p)
            fpr, tpr, _ = roc_curve(y_cat, p_cat, pos_label=1)
            auc_cat = roc_auc_score(y_cat, p_cat)
            plt.figure()
            plt.plot(fpr, tpr, label=f"Mean held-out AUC = {auc_cat:.3f}")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Outer-CV Aggregate ROC (best-per-fold)")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "roc_outer_cv_aggregate.png"), dpi=200)
            plt.close()
    except Exception as e:
        print(f"[OUTER][WARN] Aggregated ROC failed: {e}")

    # ---- final refit on ALL samples using best mean AUC model ----
    if per_model_summary.empty:
        raise RuntimeError("Outer CV finished but no per-model results were produced.")

    best_row = (
        per_model_summary
        .sort_values(["mean_auc", "sd_auc"], ascending=[False, True])
        .iloc[0]
    )
    win_name = best_row["model_name"]

    outer_df_model = outer_df[outer_df["model_name"] == win_name]
    best_fold_row = outer_df_model.sort_values("val_auc", ascending=False).iloc[0]
    win_params = best_fold_row["params"]

    # FS on FULL cohort
        # FS on FULL cohort (include custom RNA sets)
    fs_dir_full = run_feature_selection(
        data_path=args.data_path,
        output_dir=args.output_dir,
        subtype=args.subtype,
        status_col=args.status_col,
        pathway_options=args.pathway_options,
        RNAgene_bypass=getattr(args, "RNAgene_bypass", False),
        CNVgene_bypass=getattr(args, "CNVgene_bypass", False),
        include_mutated_pathways=getattr(args, "include_mutated_pathways", True),
        mut_pathway_source=getattr(args, "mut_pathway_source", "snv"),
        mut_binary_matrix_csv=getattr(args, "mut_binary_matrix_csv", None),
        mut_binary_sample_col=getattr(args, "mut_binary_sample_col", None),
        mut_binary_prefix=getattr(args, "mut_binary_prefix", "MUTblock__"),

        enable_ptmb=getattr(args, "enable_ptmb", False),
        ptmb_source=getattr(args, "ptmb_source", "none"),
        ptmb_matrix_csv=getattr(args, "ptmb_matrix_csv", None),
        ptmb_sample_col=getattr(args, "ptmb_sample_col", None),
        ptmb_prefix=getattr(args, "ptmb_prefix", "MUTblock__pTMB__"),
        ptmb_transform=getattr(args, "ptmb_transform", "none"),

        ptmb_maf_path=getattr(args, "ptmb_maf_path", None),
        ptmb_maf_sample_col=getattr(args, "ptmb_maf_sample_col", None),
        ptmb_maf_gene_col=getattr(args, "ptmb_maf_gene_col", None),
        ptmb_gtf_path=getattr(args, "ptmb_gtf_path", None),
        ptmb_cds_cache_csv=getattr(args, "ptmb_cds_cache_csv", None),
        ptmb_denom_policy=getattr(args, "ptmb_denom_policy", "all_genes_with_len"),
        ptmb_min_genes_with_len=getattr(args, "ptmb_min_genes_with_len", 1),
        ptmb_gmt_path=getattr(args, "ptmb_gmt_path", None),

        protein_coding_gtf=getattr(args, "protein_coding_gtf", None),

        var_min=getattr(args, "var_min", 0.0),
        var_min_nonmissing=getattr(args, "var_min_nonmissing", 10),
        variance_keep_file=getattr(args, "variance_keep_file", None),
        var_keep_top_frac=getattr(args, "var_keep_top_frac", None),

        rna_gene_min_raw_count=getattr(args, "rna_gene_min_raw_count", 10),
        rna_gene_min_detected_frac=getattr(args, "rna_gene_min_detected_frac", 0.25),
        cnv_gene_min_detected_frac=getattr(args, "cnv_gene_min_detected_frac", 0.20),
        cnv_arm_min_detected_frac=getattr(args, "cnv_arm_min_detected_frac", 0.20),
        cnv_band_min_detected_frac=getattr(args, "cnv_band_min_detected_frac", 0.20),
        mut_min_detected_frac=getattr(args, "mut_min_detected_frac", 0.10),

        corr_prefilter=getattr(args, "corr_prefilter", False),
        corr_abs_min=getattr(args, "corr_abs_min", None),
        corr_min_n=getattr(args, "corr_min_n", 10),
        corr_prune_sig_mode=getattr(args, "corr_prune_sig_mode", "hierarchical"),

        # ALWAYS run all supervised FS tests
        run_mwu=True,
        run_cox_uni=True,
        run_cox_multi=True,


        mwu_p_thresh=getattr(args, "mwu_p_thresh", args.fs_p_thresh),
        mwu_fdr_thresh=getattr(args, "mwu_fdr_thresh", args.fs_fdr_thresh),
        cox_uni_p_thresh=getattr(args, "cox_uni_p_thresh", args.fs_p_thresh),
        cox_uni_fdr_thresh=getattr(args, "cox_uni_fdr_thresh", args.fs_fdr_thresh),
        cox_multi_p_thresh=getattr(args, "cox_multi_p_thresh", args.fs_p_thresh),
        cox_multi_fdr_thresh=getattr(args, "cox_multi_fdr_thresh", args.fs_fdr_thresh),

        cox_time_col=args.time_col,
        cox_event_col=args.event_col,
        cox_covariates=getattr(args, "cox_covariates", None),
        cox_hr_abs_min=args.cox_hr_abs_min,
        cox_hr_direction=args.cox_hr_direction,
        cox_min_n=getattr(args, "cox_min_n", 10),
        cox_min_events=getattr(args, "cox_min_events", 1),

        corr_threshold=args.fs_corr_threshold,

        sample_id_subset=all_ids,
        enable_subset_fs=True,
        subset_run_name="FULL_COHORT",

        limma_pvalue_threshold=args.limma_pvalue_threshold,
        limma_padj_threshold=args.limma_padj_threshold,
        limma_logfc_threshold=args.limma_logfc_threshold,

        fs_min_features=args.fs_min_features,
        fs_topk_if_empty=args.fs_topk_if_empty,
        fs_min_per_block=getattr(args, "fs_min_per_block", 0),


        corr_thresholds_by_block=getattr(args, "fs_corr_thresholds_by_block", None),

        mwu_p_thresh_by_block=getattr(args, "fs_mwu_p_thresh_by_block", None),
        mwu_fdr_thresh_by_block=getattr(args, "fs_mwu_fdr_thresh_by_block", None),

        cox_uni_p_thresh_by_block=getattr(args, "fs_cox_uni_p_thresh_by_block", None),
        cox_uni_fdr_thresh_by_block=getattr(args, "fs_cox_uni_fdr_thresh_by_block", None),

        cox_multi_p_thresh_by_block=getattr(args, "fs_cox_multi_p_thresh_by_block", None),
        cox_multi_fdr_thresh_by_block=getattr(args, "fs_cox_multi_fdr_thresh_by_block", None),

        rna_pathway_dbs=getattr(args, "fs_rna_pathway_dbs", None),
        cnv_pathway_dbs=getattr(args, "fs_cnv_pathway_dbs", None),
        mut_pathway_dbs=getattr(args, "fs_mut_pathway_dbs", None),
        ptmb_pathway_dbs=getattr(args, "fs_ptmb_pathway_dbs", None),

        custom_gene_sets_file=CUSTOM_GENE_SETS_FILE,
        custom_gene_set_mode=CUSTOM_GENE_SET_MODE,
        custom_gene_set_min_overlap=int(CUSTOM_GENE_SET_MIN_OVERLAP),

        mwu_case_label=getattr(args, "mwu_case_label", "recurrent"),
        mwu_ctrl_label=getattr(args, "mwu_ctrl_label", "non-recurrent"),
    )



    def _load_fs_block(path):
        p = os.path.join(fs_dir_full, path)
        if not os.path.exists(p):
            return None
        import pandas as pd
        df = pd.read_csv(p, index_col=0)
        df.index = df.index.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        return df

    pieces = []
    for fname in [
        "RNAblock_selected_post_redundancy.csv",
        "CNVblock_selected_post_redundancy.csv",
        "MUTblock_selected_post_redundancy.csv",
        "CLINICALblock_selected_post_redundancy.csv",
    ]:
        df = _load_fs_block(fname)
        if df is not None and df.shape[1] > 0:
            pieces.append(df)
    if not pieces:
        raise RuntimeError("[FINAL REFIT] FS on full cohort kept no features.")
    X_all_full = pd.concat(pieces, axis=1).reindex(all_ids).apply(pd.to_numeric, errors="coerce").fillna(0.0)

    pam50_full = _pam50_feature_from_meta(subtype_meta, pam50_col=args.pam50_col)
    if not pam50_full.empty:
        X_all_full = pd.concat([X_all_full, pam50_full.reindex(all_ids).fillna(0.0)], axis=1)

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    scale_cols = [c for c in X_all_full.columns if not (c.startswith("MUTblock__") or c.startswith("CLINICALblock__"))]
    if len(scale_cols) == 0:
        ct_final = ColumnTransformer(transformers=[], remainder="passthrough", verbose_feature_names_out=False)
    else:
        ct_final = ColumnTransformer([("scale", StandardScaler(), scale_cols)],
                                     remainder="passthrough", verbose_feature_names_out=False)
    X_full_sc = pd.DataFrame(ct_final.fit_transform(X_all_full),
                             index=X_all_full.index, columns=ct_final.get_feature_names_out())
    X_full_sc = X_full_sc[X_all_full.columns]

    # ALWAYS save preprocessing artifacts before deciding on final model type
    dump_with_meta(ct_final, os.path.join(prep_dir, "global_cont_scaler.joblib"), joblib)
    pd.Series(X_all_full.columns, name="feature").to_csv(os.path.join(prep_dir, "ALL_BLOCKS_FINAL_features.csv"), index=False)
    pd.Series(X_all_full.columns, name="feature").to_csv(os.path.join(prep_dir, "train_feature_order.csv"), index=False)
    pd.DataFrame({"scaled_feature": scale_cols}).to_csv(os.path.join(prep_dir, "global_scaled_columns.csv"), index=False)

    # Pre-scaling audit & stats
    stats = pd.DataFrame({
        "mean": X_all_full.mean(),
        "std": X_all_full.std(ddof=0).replace(0, 1e-9),
        "p01": X_all_full.quantile(0.01),
        "p99": X_all_full.quantile(0.99),
        "min": X_all_full.min(),
        "max": X_all_full.max(),
    })
    stats.index.name = "feature"
    stats.to_csv(os.path.join(prep_dir, "train_feature_stats.csv"))

    # SPECIAL CASE: if the winner is the block-aware stack, rebuild it on FULL data
    if win_name == "StackingClassifier(block-aware)":
        # re-derive blocks
        run_blocks = [b.strip().upper() for b in getattr(args, "cv_blocks", "RNA,CNV,MUT,CLINICAL").split(",") if b.strip()]

        # Only keep blocks we actually want in the final stacked model
        run_blocks_for_stack = [b for b in run_blocks if b not in _STACK_EXCLUDE_BLOCKS]
        best_by_block_full = {}

        # collect all per-block summaries from outer folds
        block_summ_files = glob.glob(os.path.join(models_dir, "OUTER_SUMMARY__by_block__OUTER_FOLD_*.csv"))
        import pandas as pd
        if block_summ_files:
            block_summ = pd.concat([pd.read_csv(f) for f in block_summ_files], ignore_index=True)
        else:
            block_summ = pd.DataFrame(columns=["fold", "block", "model_name", "inner_auc_mean", "val_auc", "params"])

        for blk in run_blocks_for_stack:
            blk_cols = [c for c in X_full_sc.columns if c.startswith(_BLOCK_PREFIX[blk])]
            if not blk_cols:
                print(f"[FINAL_FULL] Stack: block {blk} has no features; skipping.")
                continue

            sub = block_summ.loc[block_summ["block"] == blk]
            if sub.empty:
                print(f"[FINAL_FULL] Stack: no summary rows for block {blk}; skipping.")
                continue
            best_row_blk = sub.sort_values("val_auc", ascending=False).iloc[0]
            best_model_name = best_row_blk["model_name"]

            grid_all = _filter_param_grid(
                _build_param_grid_for(
                    subtype=args.subtype,
                    include_xgboost=args.include_xgboost,
                    fs_enable=False,
                    include_stacking=False
                )
            )
            if best_model_name not in grid_all:
                print(f"[FINAL_FULL] Stack: best model {best_model_name} for block {blk} not in grid; skipping.")
                continue

            proto, grid = grid_all[best_model_name]
            X_blk_full = X_full_sc.loc[:, blk_cols]
            est_full, _, _, _ = _grid_select_and_fit(
                X_blk_full, y_all_enc,
                {best_model_name: (proto, grid)},
                cv_folds=args.inner_folds,
                label=f"FINAL_FULL::{blk}",
                calibrate=args.calibrate,
                calibrate_method=args.calibrate_method,
                calibrate_cv=args.calibrate_cv
            )
            best_by_block_full[blk] = est_full

        final_stack, final_stack_auc = fit_stack_from_block_bests(
            Xtr_full=X_full_sc,
            ytr=y_all_enc,
            Xva_full=X_full_sc,
            yva=y_all_enc,
            best_by_block=best_by_block_full,
            plots_dir=fold_dirs["plots"],
            models_dir=fold_dirs["models"],
            fold_label="FINAL_FULL",
            meta_model=getattr(args, "stack_meta_model", "logreg"),
            meta_kwargs={},
            cv=max(3, int(args.stack_cv)),
            calibrate_stack=bool(getattr(args, "stack_calibrate", False)),
            calibrate_method=getattr(args, "stack_calibrate_method", "sigmoid"),
            calibrate_cv=int(getattr(args, "stack_calibrate_cv", 5)),
        )

        dump_with_meta(final_stack, os.path.join(fold_dirs["models"], "ALL_BLOCKS_FINAL.pkl"), joblib)

        with open(os.path.join(models_dir, "ALL_BLOCKS_FINAL_info.txt"), "w") as f:
            f.write("Model Type (best mean AUC): StackingClassifier(block-aware) (refit on ALL after outer CV)\n")
            if not outer_df.empty:
                f.write(f"Outer-CV mean held-out AUC (all models pooled): {float(outer_df['val_auc'].dropna().mean()):.4f}\n")
            f.write("Final Params:\n")
            f.write("  stacked: block-aware (retrained from per-block bests)\n")

        versions = {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "sklearn": __import__("sklearn").__version__,
            "xgboost": getattr(sys.modules.get("xgboost"), "__version__", None),
            "lightgbm": getattr(sys.modules.get("lightgbm"), "__version__", None),
            "catboost": getattr(sys.modules.get("catboost"), "__version__", None),
        }
        with open(os.path.join(prep_dir, "versions.txt"), "w") as f:
            for k, v in versions.items():
                f.write(f"{k}: {v}\n")

        print(f"[TRAIN] ✅ Finished outer CV. Final (STACK) model refit on ALL.")
        print(f"[TRAIN] 📁 Outputs → {fold_dirs['root']}")
        return fold_dirs["root"]

    # ----- else: normal single-family final refit -----
    concat_param_grid_full = _filter_param_grid(
        _build_param_grid_for(args.subtype, fs_enable=False, include_xgboost=args.include_xgboost)
    )
    if win_name not in concat_param_grid_full:
        raise RuntimeError(f"Selected final model '{win_name}' not present in grid.")
    win_pipe, win_grid = concat_param_grid_full[win_name]
    winner_only = {win_name: (win_pipe, win_grid)}

    final_model, _, _, win_params = _grid_select_and_fit(
        X_full_sc, y_all_enc, winner_only, cv_folds=args.inner_folds,
        label="FINAL_FULL",
        calibrate=args.calibrate, calibrate_method=args.calibrate_method, calibrate_cv=args.calibrate_cv
    )

    dump_with_meta(final_model, os.path.join(fold_dirs["models"], "ALL_BLOCKS_FINAL.pkl"), joblib)

    with open(os.path.join(models_dir, "ALL_BLOCKS_FINAL_info.txt"), "w") as f:
        f.write(f"Model Type (best mean AUC): {win_name} (refit on ALL after outer CV)\n")
        if not outer_df.empty:
            f.write(f"Outer-CV mean held-out AUC (all models pooled): {float(outer_df['val_auc'].dropna().mean()):.4f}\n")
        f.write("Final Params:\n")
        for k, v in (win_params or {}).items():
            f.write(f"  {k}: {v}\n")

    versions = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "sklearn": __import__("sklearn").__version__,
        "xgboost": getattr(sys.modules.get("xgboost"), "__version__", None),
        "lightgbm": getattr(sys.modules.get("lightgbm"), "__version__", None),
        "catboost": getattr(sys.modules.get("catboost"), "__version__", None),
    }
    with open(os.path.join(prep_dir, "versions.txt"), "w") as f:
        for k, v in versions.items():
            f.write(f"{k}: {v}\n")

    print(f"[TRAIN] ✅ Finished outer CV. Final model refit on ALL.")
    print(f"[TRAIN] 📁 Outputs → {fold_dirs['root']}")
    return fold_dirs["root"]


# ====================== PREDICT ======================
def predict_concat_only(
    output_dir, subtype, pred_matrix_csv, sample_ids=None, pred_labels_csv=None,
    prob_threshold=0.5, threshold_mode="fixed", scale_mode="training", pam50_col=None
):
    import numpy as np
    import pandas as pd
    import joblib

    fold_dirs = build_fold_dirs(output_dir, subtype)
    models_dir = fold_dirs["models"]
    prep_dir = fold_dirs["preprocessing"]
    preds_dir = fold_dirs["predictions"]
    plots_dir = fold_dirs["plots"]

    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "ALL_BLOCKS_FINAL.pkl")
    ct_path = os.path.join(prep_dir, "global_cont_scaler.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"ColumnTransformer not found: {ct_path}")

    # STRICT version-checked loads
    model = load_with_strict_version(model_path, joblib)
    ct = load_with_strict_version(ct_path, joblib)

    train_order_csv = os.path.join(prep_dir, "train_feature_order.csv")
    scaled_cols_csv = os.path.join(prep_dir, "global_scaled_columns.csv")
    feat_list_csv = os.path.join(prep_dir, "ALL_BLOCKS_FINAL_features.csv")
    train_input_cols = pd.read_csv(train_order_csv)["feature"].astype(str).tolist()
    scale_cols = pd.read_csv(scaled_cols_csv)["scaled_feature"].astype(str).tolist()
    feat_list = pd.read_csv(feat_list_csv)["feature"].astype(str).tolist()

    scale_name = next((n for n in ct.named_transformers_.keys() if str(n).startswith("scale")), None)
    scaler = None
    if len(scale_cols) > 0:
        if scale_name is None:
            raise RuntimeError("Expected 'scale' transformer but none found; check training artifacts.")
        scaler = ct.named_transformers_[scale_name]

    X_pred = pd.read_csv(pred_matrix_csv, index_col=0)
    X_pred.index = X_pred.index.astype(str).str.strip()
    X_pred.columns = X_pred.columns.astype(str).str.strip()
    if sample_ids is not None:
        sample_ids = pd.Index(sample_ids).astype(str).str.strip()
        X_pred = X_pred.reindex(sample_ids)
    X_pred = X_pred.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    pam50_feat_name = "CLINICALblock__PAM50_LumB"
    if (pam50_col is not None) and (pam50_feat_name not in X_pred.columns):
        meta_csv = os.path.join(prep_dir, "subtype_meta_full.csv")
        try:
            meta_df = pd.read_csv(meta_csv, index_col=0)
            meta_df.index = meta_df.index.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
            pam50_extra = _pam50_feature_from_meta(meta_df, pam50_col=pam50_col)
            if not pam50_extra.empty:
                X_pred = X_pred.join(pam50_extra, how="left")
                X_pred[pam50_feat_name] = X_pred[pam50_feat_name].fillna(0.0)
                print(f"[PRED] Added {pam50_feat_name} from subtype_meta_full.csv")
        except Exception as e:
            print(f"[PRED] Could not add PAM50 feature from meta: {e}")

    stats_path = os.path.join(prep_dir, "train_feature_stats.csv")
    if str(scale_mode).lower() == "training":
        if os.path.exists(stats_path):
            stats = pd.read_csv(stats_path).set_index("feature")
            scale_set = set(scale_cols)
            common = [c for c in X_pred.columns if c in stats.index]
            for c in common:
                s = stats.loc[c]
                if c in scale_set:
                    lo = float(s.get("p01", s.get("min", 0.0)))
                    hi = float(s.get("p99", s.get("max", 0.0)))
                else:
                    lo = float(s.get("min", 0.0))
                    hi = float(s.get("max", 0.0))
                if lo != hi:
                    X_pred[c] = X_pred[c].clip(lower=lo, upper=hi)
        else:
            print("[SHIFT] No training stats found; skipping winsorization.")

    pred_cols = set(X_pred.columns)
    train_cols = set(train_input_cols)
    missing = [c for c in train_input_cols if c not in pred_cols]
    extra = [c for c in X_pred.columns if c not in train_cols]
    if extra:
        X_pred = X_pred.drop(columns=extra)
    if missing:
        if scaler is not None:
            scale_means = dict(zip(scale_cols, getattr(scaler, "mean_", [0.0] * len(scale_cols))))
        else:
            scale_means = {}
        for c in missing:
            X_pred[c] = float(scale_means.get(c, 0.0)) if c in scale_cols else 0.0
    X_pred = X_pred.reindex(columns=train_input_cols)

    scm = str(scale_mode).lower()
    if scm == "training":
        X_scaled_arr = ct.transform(X_pred)
        X_scaled_df = pd.DataFrame(X_scaled_arr, index=X_pred.index, columns=ct.get_feature_names_out())
        X_scaled_df = X_scaled_df[train_input_cols]
    elif scm == "prediction_z":
        X_scaled_df = X_pred.copy()
        if len(scale_cols) > 0:
            pred_means = X_scaled_df[scale_cols].mean(axis=0)
            pred_stds = X_scaled_df[scale_cols].std(axis=0, ddof=0).replace(0.0, np.nan)
            for c in scale_cols:
                if c in X_scaled_df.columns:
                    mu = float(pred_means.get(c, 0.0))
                    sd = float(pred_stds.get(c, np.nan))
                    X_scaled_df[c] = 0.0 if (np.isnan(sd) or sd == 0.0) else (X_scaled_df[c] - mu) / sd
    else:
        raise ValueError("scale_mode must be 'training' or 'prediction_z'.")

    excl_prefixes = ("MUTblock__", "CLINICALblock__")
    passthrough_cols = [c for c in train_input_cols if c.startswith(excl_prefixes)]
    if passthrough_cols:
        mx = (X_scaled_df[passthrough_cols] - X_pred[passthrough_cols]).abs().max().max()
        if mx > 1e-9:
            print(f"[WARN] Passthrough columns changed by up to {mx:g} after transform.")

    for m in [c for c in feat_list if c not in X_scaled_df.columns]:
        X_scaled_df[m] = 0.0
    X_scaled_df = X_scaled_df.reindex(columns=feat_list, fill_value=0.0)
  

    def _proba_pos_pred(m, Xdf):
        if hasattr(m, "predict_proba"):
            proba = m.predict_proba(Xdf)
            if hasattr(m, "classes_") and 1 in list(m.classes_):
                pos_idx = list(m.classes_).index(1)
            else:
                pos_idx = -1
            return proba[:, pos_idx]
        elif hasattr(m, "decision_function"):
            from scipy.special import expit
            return expit(m.decision_function(Xdf))
        else:
            return m.predict(Xdf).astype(float)

    X_scaled_df = _coerce_numeric_float32(X_scaled_df)
    probs = _proba_pos_pred(model, X_scaled_df)

    tmode = str(threshold_mode).lower()
    used_threshold = float(np.nanmedian(probs)) if tmode == "median" else float(prob_threshold)
    threshold_source = "cohort_median" if tmode == "median" else "fixed"

    out = pd.DataFrame({"sample_id": X_scaled_df.index.astype(str), "predicted_proba": probs})
    out["predicted_label"] = (out["predicted_proba"] >= used_threshold).astype(int)

    out_fname = f"predictions_ALL_BLOCKS_FINAL__{threshold_source}__{scm}.csv"
    out_path = os.path.join(preds_dir, out_fname)
    out.to_csv(out_path, index=False)
    print(f"💾 Saved predictions → {out_path}  ({len(out)} samples)")
    print("[PRED] threshold:", {"mode": threshold_mode, "value": used_threshold, "source": threshold_source})

    return out


# ====================== RESUME ======================
def resume_post_outer(args):
    import os
    import re
    import glob
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import joblib
    from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve

    fold_dirs = build_fold_dirs(args.output_dir, args.subtype)
    models_dir = fold_dirs["models"]
    prep_dir = fold_dirs["preprocessing"]
    plots_dir = fold_dirs["plots"]

    meta_path = os.path.join(prep_dir, "subtype_meta_full.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta: {meta_path}")
    subtype_meta = pd.read_csv(meta_path, index_col=0)
    subtype_meta.index = subtype_meta.index.astype(str).str.strip()
    if "time_to_event" not in subtype_meta.columns:
        subtype_meta["time_to_event"] = np.nan

    rows = []
    for jf in glob.glob(os.path.join(models_dir, "*__best_params.json")):
        try:
            with open(jf, "r") as f:
                d = json.load(f)
            fold_lab = d.get("fold", "")
            m = re.search(r"OUTER_FOLD_(\d+)", fold_lab)
            fold_idx = int(m.group(1)) if m else None
            rows.append({
                "fold": fold_idx,
                "model_name": d.get("model_name"),
                "val_auc": d.get("val_auc"),
                "params": d.get("best_params")
            })
        except Exception:
            pass
    outer_df = pd.DataFrame(rows).dropna(subset=["model_name"])
    if outer_df.empty:
        raise RuntimeError("No best_params JSONs found; cannot resume.")
    outer_df.to_csv(os.path.join(models_dir, "OUTER_CV_summary.csv"), index=False)

    pred_files = glob.glob(os.path.join(models_dir, "*__preds.csv"))
    if not pred_files:
        raise RuntimeError("No per-fold preds CSVs found; cannot resume.")
    all_val_preds = pd.concat([pd.read_csv(p) for p in pred_files], ignore_index=True)
    all_val_preds["sample_id"] = all_val_preds["sample_id"].astype(str).str.strip()
    all_val_preds.to_csv(os.path.join(models_dir, "OUTER_CV_all_val_preds_by_model.csv"), index=False)

    per_model_summary = (
        outer_df.groupby("model_name", as_index=False)
        .agg(mean_auc=("val_auc", "mean"),
             sd_auc=("val_auc", "std"),
             median_auc=("val_auc", "median"),
             n=("val_auc", "size"))
        .sort_values("mean_auc", ascending=False)
    )
    per_model_summary.to_csv(os.path.join(models_dir, "OUTER_CV_summary_by_model.csv"), index=False)

    try:
        models = per_model_summary["model_name"].tolist()
        data = [outer_df.loc[outer_df["model_name"] == m, "val_auc"].values for m in models]
        plt.figure(figsize=(10, 5))
        plt.boxplot(data, labels=models, showfliers=False)
        for i, vals in enumerate(data, start=1):
            if len(vals) == 0:
                continue
            x = np.random.normal(loc=i, scale=0.06, size=len(vals))
            plt.scatter(x, vals, s=20, alpha=0.6)
        plt.ylabel("Outer-fold AUC")
        plt.title("Outer CV — AUC by Model")
        plt.ylim(0.0, 1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "auc_by_model_box_jitter.png"), dpi=200)
        plt.close()
    except Exception as e:
        print(f"[WARN] resume: box/jitter failed: {e}")

    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
        oof = all_val_preds.copy()
        meta_aligned = subtype_meta[["time_to_event"]].copy()
        oof = oof.merge(meta_aligned, left_on="sample_id", right_index=True, how="left")

        for model_name, group in oof.groupby("model_name"):
            t = pd.to_numeric(group["time_to_event"], errors="coerce")
            e = pd.to_numeric(group["y_true"], errors="coerce")
            ok = (~t.isna()) & (~e.isna())
            if ok.sum() == 0:
                print(f"[KM] skip {model_name}: no time/event.")
                continue
            thr = float(np.nanmedian(group.loc[ok, "predicted_proba"].values))
            grp = (group["predicted_proba"] >= thr).astype(int).values
            km = KaplanMeierFitter()
            plt.figure()
            mask0 = ok.values & (grp == 0)
            if mask0.sum() > 0:
                km.fit(t[mask0], e[mask0], label=f"Low (< {thr:.2f}) (n={int(mask0.sum())}, events={int(e[mask0].sum())})")
                km.plot(ci_show=False)
            mask1 = ok.values & (grp == 1)
            if mask1.sum() > 0:
                km.fit(t[mask1], e[mask1], label=f"High (≥ {thr:.2f}) (n={int(mask1.sum())}, events={int(e[mask1].sum())})")
                km.plot(ci_show=False)
            title = f"KM by OOF risk — {model_name}"
            if mask0.sum() > 0 and mask1.sum() > 0:
                lr = logrank_test(t[mask0], t[mask1], event_observed_A=e[mask0], event_observed_B=e[mask1])
                title += f"  (log-rank p = {lr.p_value:.3e})"
            plt.title(title)
            plt.xlabel("Time to event (months)")
            plt.ylabel("Survival probability")
            plt.tight_layout()
            slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_name)).strip("_")
            plt.savefig(os.path.join(plots_dir, f"km_outer_cv_by_pred__{slug}.png"), dpi=200)
            plt.close()
    except Exception as e:
        print(f"[WARN] resume: KM failed: {e}")

    try:
        y_cat, p_cat = [], []
        for f_idx, df_fold in outer_df.groupby("fold"):
            top = df_fold.sort_values("val_auc", ascending=False).iloc[0]
            mname = top["model_name"]
            g = all_val_preds[(all_val_preds["fold"] == f_idx) & (all_val_preds["model_name"] == mname)]
            if g.empty:
                continue
            y_cat.append(g["y_true"].astype(int).values)
            p_cat.append(g["predicted_proba"].values)
        if y_cat and p_cat:
            y_cat = np.concatenate(y_cat)
            p_cat = np.concatenate(p_cat)
            fpr, tpr, _ = roc_curve(y_cat, p_cat, pos_label=1)
            auc_cat = roc_auc_score(y_cat, p_cat)
            plt.figure()
            plt.plot(fpr, tpr, label=f"Mean held-out AUC = {auc_cat:.3f}")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Outer-CV Aggregate ROC (best-per-fold)")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "roc_outer_cv_aggregate.png"), dpi=200)
            plt.close()
    except Exception as e:
        print(f"[WARN] resume: aggregate ROC failed: {e}")

    # --- Final refit on ALL samples (same logic as in train_once_concat) ---
    data_all, all_pathway_dicts = load_all_data(args.data_path, args.subtype, args.pathway_options)
    cleaned_data, subtype_meta2, y_all_enc, all_ids, _ = dedup_and_align_data(
        data_all, status_col=args.status_col, prep_dir=prep_dir, raise_on_conflict=True
    )

    winner = per_model_summary.sort_values(["mean_auc", "sd_auc"], ascending=[False, True]).iloc[0]["model_name"]
    win_params_all = outer_df[outer_df["model_name"] == winner].sort_values("val_auc", ascending=False).iloc[0]["params"]

    fs_dir_full = run_feature_selection(
        data_path=args.data_path,
        output_dir=args.output_dir,
        subtype=args.subtype,
        status_col=args.status_col,
        pathway_options=args.pathway_options,
        RNAgene_bypass=getattr(args, "RNAgene_bypass", False),
        CNVgene_bypass=getattr(args, "CNVgene_bypass", False),
        include_mutated_pathways=getattr(args, "include_mutated_pathways", True),
        mut_pathway_source=getattr(args, "mut_pathway_source", "snv"),
        mut_binary_matrix_csv=getattr(args, "mut_binary_matrix_csv", None),
        mut_binary_sample_col=getattr(args, "mut_binary_sample_col", None),
        mut_binary_prefix=getattr(args, "mut_binary_prefix", "MUTblock__"),

        enable_ptmb=getattr(args, "enable_ptmb", False),
        ptmb_source=getattr(args, "ptmb_source", "none"),
        ptmb_matrix_csv=getattr(args, "ptmb_matrix_csv", None),
        ptmb_sample_col=getattr(args, "ptmb_sample_col", None),
        ptmb_prefix=getattr(args, "ptmb_prefix", "MUTblock__pTMB__"),
        ptmb_transform=getattr(args, "ptmb_transform", "none"),

        ptmb_maf_path=getattr(args, "ptmb_maf_path", None),
        ptmb_maf_sample_col=getattr(args, "ptmb_maf_sample_col", None),
        ptmb_maf_gene_col=getattr(args, "ptmb_maf_gene_col", None),
        ptmb_gtf_path=getattr(args, "ptmb_gtf_path", None),
        ptmb_cds_cache_csv=getattr(args, "ptmb_cds_cache_csv", None),
        ptmb_denom_policy=getattr(args, "ptmb_denom_policy", "all_genes_with_len"),
        ptmb_min_genes_with_len=getattr(args, "ptmb_min_genes_with_len", 1),
        ptmb_gmt_path=getattr(args, "ptmb_gmt_path", None),

        protein_coding_gtf=getattr(args, "protein_coding_gtf", None),

        var_min=getattr(args, "var_min", 0.0),
        var_min_nonmissing=getattr(args, "var_min_nonmissing", 10),
        variance_keep_file=getattr(args, "variance_keep_file", None),
        var_keep_top_frac=getattr(args, "var_keep_top_frac", None),

        corr_prefilter=getattr(args, "corr_prefilter", False),
        corr_abs_min=getattr(args, "corr_abs_min", None),
        corr_min_n=getattr(args, "corr_min_n", 10),
        corr_prune_sig_mode=getattr(args, "corr_prune_sig_mode", "hierarchical"),

        run_mwu=(getattr(args, "run_mwu", None) if hasattr(args, "run_mwu") else (str(args.primary_test).lower() == "mwu")),
        run_cox_uni=(getattr(args, "run_cox_uni", None) if hasattr(args, "run_cox_uni") else (str(args.primary_test).lower() == "cox")),
        run_cox_multi=getattr(args, "run_cox_multi", False),

        mwu_p_thresh=getattr(args, "mwu_p_thresh", args.fs_p_thresh),
        mwu_fdr_thresh=getattr(args, "mwu_fdr_thresh", args.fs_fdr_thresh),
        cox_uni_p_thresh=getattr(args, "cox_uni_p_thresh", args.fs_p_thresh),
        cox_uni_fdr_thresh=getattr(args, "cox_uni_fdr_thresh", args.fs_fdr_thresh),
        cox_multi_p_thresh=getattr(args, "cox_multi_p_thresh", args.fs_p_thresh),
        cox_multi_fdr_thresh=getattr(args, "cox_multi_fdr_thresh", args.fs_fdr_thresh),

        cox_time_col=args.time_col,
        cox_event_col=args.event_col,
        cox_covariates=getattr(args, "cox_covariates", None),
        cox_hr_abs_min=args.cox_hr_abs_min,
        cox_hr_direction=args.cox_hr_direction,
        cox_min_n=getattr(args, "cox_min_n", 10),
        cox_min_events=getattr(args, "cox_min_events", 1),

        corr_threshold=args.fs_corr_threshold,

        sample_id_subset=all_ids,
        enable_subset_fs=True,
        subset_run_name="FULL_COHORT",

        limma_pvalue_threshold=args.limma_pvalue_threshold,
        limma_padj_threshold=args.limma_padj_threshold,
        limma_logfc_threshold=args.limma_logfc_threshold,

        fs_min_features=args.fs_min_features,
        fs_topk_if_empty=args.fs_topk_if_empty,
        fs_min_per_block=getattr(args, "fs_min_per_block", 0),

        custom_gene_sets_file=CUSTOM_GENE_SETS_FILE,
        custom_gene_set_mode=CUSTOM_GENE_SET_MODE,
        custom_gene_set_min_overlap=int(CUSTOM_GENE_SET_MIN_OVERLAP),

        mwu_case_label=getattr(args, "mwu_case_label", "recurrent"),
        mwu_ctrl_label=getattr(args, "mwu_ctrl_label", "non-recurrent"),
    )

    def _load_fs_block2(path):
        p = os.path.join(fs_dir_full, path)
        if not os.path.exists(p):
            return None
        df = pd.read_csv(p, index_col=0)
        df.index = df.index.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        return df

    pieces = []
    for fname in [
        "RNAblock_selected_post_redundancy.csv",
        "CNVblock_selected_post_redundancy.csv",
        "MUTblock_selected_post_redundancy.csv",
        "CLINICALblock_selected_post_redundancy.csv",
    ]:
        df = _load_fs_block2(fname)
        if df is not None and df.shape[1] > 0:
            pieces.append(df)
    if not pieces:
        raise RuntimeError("[FINAL REFIT] FS on full cohort kept no features.")
    X_all_full = pd.concat(pieces, axis=1).reindex(all_ids).apply(pd.to_numeric, errors="coerce").fillna(0.0)

    pam50_full = _pam50_feature_from_meta(cleaned_data["meta"], pam50_col=args.pam50_col)
    if not pam50_full.empty:
        X_all_full = pd.concat([X_all_full, pam50_full.reindex(all_ids).fillna(0.0)], axis=1)

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    scale_cols = [c for c in X_all_full.columns if not (c.startswith("MUTblock__") or c.startswith("CLINICALblock__"))]
    if len(scale_cols) == 0:
        ct_final = ColumnTransformer(transformers=[], remainder="passthrough", verbose_feature_names_out=False)
    else:
        ct_final = ColumnTransformer([("scale", StandardScaler(), scale_cols)],
                                     remainder="passthrough", verbose_feature_names_out=False)
    X_full_sc = pd.DataFrame(ct_final.fit_transform(X_all_full),
                             index=X_all_full.index, columns=ct_final.get_feature_names_out())
    X_full_sc = X_full_sc[X_all_full.columns]

    # save preprocessing artifacts first (with meta)
    dump_with_meta(ct_final, os.path.join(prep_dir, "global_cont_scaler.joblib"), joblib)
    pd.Series(X_all_full.columns, name="feature").to_csv(os.path.join(prep_dir, "ALL_BLOCKS_FINAL_features.csv"), index=False)
    pd.Series(X_all_full.columns, name="feature").to_csv(os.path.join(prep_dir, "train_feature_order.csv"), index=False)
    pd.DataFrame({"scaled_feature": scale_cols}).to_csv(os.path.join(prep_dir, "global_scaled_columns.csv"), index=False)

    # SPECIAL CASE: block-aware stack won in resume
    if winner == "StackingClassifier(block-aware)":
        run_blocks = [b.strip().upper() for b in getattr(args, "cv_blocks", "RNA,CNV,MUT,CLINICAL").split(",") if b.strip()]
        best_by_block_full = {}

        block_summ_files = glob.glob(os.path.join(models_dir, "OUTER_SUMMARY__by_block__OUTER_FOLD_*.csv"))
        if block_summ_files:
            block_summ = pd.concat([pd.read_csv(f) for f in block_summ_files], ignore_index=True)
        else:
            block_summ = pd.DataFrame(columns=["fold", "block", "model_name", "inner_auc_mean", "val_auc", "params"])

        for blk in run_blocks:
            blk_cols = [c for c in X_full_sc.columns if c.startswith(_BLOCK_PREFIX[blk])]
            if not blk_cols:
                best_by_block_full[blk] = None
                continue

            sub = block_summ.loc[block_summ["block"] == blk]
            if sub.empty:
                best_by_block_full[blk] = None
                continue
            best_row = sub.sort_values("val_auc", ascending=False).iloc[0]
            best_model_name = best_row["model_name"]

            grid_all = _filter_param_grid(
                _build_param_grid_for(
                    subtype=args.subtype,
                    include_xgboost=args.include_xgboost,
                    fs_enable=False,
                    include_stacking=False
                )
            )
            if best_model_name not in grid_all:
                best_by_block_full[blk] = None
                continue
            proto, grid = grid_all[best_model_name]
            X_blk_full = X_full_sc.loc[:, blk_cols]
            est_full, _, _, _ = _grid_select_and_fit(
                X_blk_full, y_all_enc,
                {best_model_name: (proto, grid)},
                cv_folds=args.inner_folds,
                label=f"FINAL_FULL::{blk}",
                calibrate=args.calibrate,
                calibrate_method=args.calibrate_method,
                calibrate_cv=args.calibrate_cv
            )
            best_by_block_full[blk] = est_full

        final_stack, final_stack_auc = fit_stack_from_block_bests(
            Xtr_full=X_full_sc,
            ytr=y_all_enc,
            Xva_full=X_full_sc,
            yva=y_all_enc,
            best_by_block=best_by_block_full,
            plots_dir=fold_dirs["plots"],
            models_dir=fold_dirs["models"],
            fold_label="FINAL_FULL",
            meta_model=getattr(args, "stack_meta_model", "logreg"),
            meta_kwargs={},
            cv=max(3, int(args.stack_cv)),
            calibrate_stack=bool(getattr(args, "stack_calibrate", False)),
            calibrate_method=getattr(args, "stack_calibrate_method", "sigmoid"),
            calibrate_cv=int(getattr(args, "stack_calibrate_cv", 5)),
        )

        dump_with_meta(final_stack, os.path.join(fold_dirs["models"], "ALL_BLOCKS_FINAL.pkl"), joblib)

        with open(os.path.join(models_dir, "ALL_BLOCKS_FINAL_info.txt"), "w") as f:
            f.write("Model Type (best mean AUC): StackingClassifier(block-aware) (refit on ALL after outer CV)\n")
            if not outer_df.empty:
                f.write(f"Outer-CV mean held-out AUC (all models pooled): {float(outer_df['val_auc'].dropna().mean()):.4f}\n")
            f.write("Final Params:\n")
            f.write("  stacked: block-aware (retrained from per-block bests in resume)\n")

        print("[RESUME] ✅ Rebuilt OOF summaries/plots and trained final STACK model.")
        return

    # else: single family
    concat_param_grid_full = _filter_param_grid(
        _build_param_grid_for(args.subtype, fs_enable=False, include_xgboost=args.include_xgboost, include_stacking=args.include_stacking)
    )
    if winner not in concat_param_grid_full:
        raise RuntimeError(f"Selected final model '{winner}' not present in grid.")
    win_pipe, win_grid = concat_param_grid_full[winner]
    winner_only = {winner: (win_pipe, win_grid)}

    final_model, _, _, final_params = _grid_select_and_fit(
        X_full_sc, y_all_enc, winner_only, cv_folds=args.inner_folds,
        label="FINAL_FULL",
        calibrate=args.calibrate, calibrate_method=args.calibrate_method, calibrate_cv=args.calibrate_cv
    )
    dump_with_meta(final_model, os.path.join(models_dir, "ALL_BLOCKS_FINAL.pkl"), joblib)

    with open(os.path.join(models_dir, "ALL_BLOCKS_FINAL_info.txt"), "w") as f:
        f.write(f"Model Type (best mean AUC): {winner} (refit on ALL after outer CV)\n")
        if not outer_df.empty:
            f.write(f"Outer-CV mean held-out AUC (all models pooled): {float(outer_df['val_auc'].dropna().mean()):.4f}\n")
        f.write("Final Params:\n")
        for k, v in (final_params or {}).items():
            f.write(f"  {k}: {v}\n")

    print("[RESUME] ✅ Rebuilt OOF summaries/plots and trained final model.")


# ====================== CLI ======================
import argparse
VALID_PATHWAY_CHOICES = {"gobp", "reactome", "kegg", "hallmark_kegg"}


def get_args():
    import argparse

    p = argparse.ArgumentParser()

    # --- Data / IO ---
    p.add_argument("--data_path", help="Root path to input matrices & meta.")
    p.add_argument("--output_dir", required=True, help="Folder where outputs will be written.")
    p.add_argument("--subtype", required=True, help="Subtype label, e.g. LumA, LumB, Basal.")
    p.add_argument("--status_col", default="Disease_Status",
                   help="Column in meta used as the target (default: Disease_Status).")
    p.add_argument("--pathway_options", nargs="+", default=["hallmark_kegg"],
                   help="One or more pathway sets: gobp reactome kegg hallmark_kegg")

    # --- CV settings ---
    p.add_argument("--inner_folds", type=int, default=3)
    p.add_argument("--outer_folds", type=int, default=3)

    # --- Training matrix shortcuts / feature lists (compat) ---
    p.add_argument("--selected_blocks", nargs="+", default=None,
                   help="(Compat only; unused in global-scaler path).")
    p.add_argument("--feature_list_csv", default=None,
                   help="Optional CSV with 'feature' column listing features to keep.")
    p.add_argument("--train_matrix_csv", default=None,
                   help="Provide a FULL matrix to bypass feature building.")

    # --- Prediction-only path ---
    p.add_argument("--predict_matrix_csv", default=None)
    p.add_argument("--predict_labels_csv", default=None)
    p.add_argument("--predict_ids_csv", default=None)

    # --- Thresholding / scaling / calibration ---
    p.add_argument("--prob_threshold", type=float, default=0.5)
    p.add_argument("--threshold_mode", default="median", choices=["median", "fixed"])
    p.add_argument("--scale_mode", default="training", choices=["training", "prediction_z"])
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--calibrate_method", choices=["sigmoid", "isotonic"], default="sigmoid")
    p.add_argument("--calibrate_cv", type=int, default=5)

    p.add_argument("--rna_gene_min_raw_count", type=int, default=10)
    p.add_argument("--rna_gene_min_detected_frac", type=float, default=0.25)

    p.add_argument("--cnv_gene_min_detected_frac", type=float, default=0.20)
    p.add_argument("--cnv_arm_min_detected_frac", type=float, default=0.20)
    p.add_argument("--cnv_band_min_detected_frac", type=float, default=0.20)

    p.add_argument("--mut_min_detected_frac", type=float, default=0.10)

    # --- FS master toggle ---
    p.add_argument("--fs_enable", action="store_true",
                   help="Enable in-pipeline FS (inside CV).")

    # --- LEGACY shared thresholds (still supported; act as defaults) ---
    p.add_argument("--fs_p_thresh", type=float, default=0.09,
                   help="Legacy p-threshold used as default for MWU/Cox if per-test thresholds not set.")
    p.add_argument("--fs_fdr_thresh", type=float, default=0.10,
                   help="Legacy FDR threshold used as default for MWU/Cox if per-test thresholds not set.")

    # --- Per-test global thresholds (optional overrides) ---
    p.add_argument("--mwu_p_thresh", type=float, default=None)
    p.add_argument("--mwu_fdr_thresh", type=float, default=None)

    p.add_argument("--cox_uni_p_thresh", type=float, default=None)
    p.add_argument("--cox_uni_fdr_thresh", type=float, default=None)

    p.add_argument("--cox_multi_p_thresh", type=float, default=None)
    p.add_argument("--cox_multi_fdr_thresh", type=float, default=None)

    # --- NEW: per-block p/FDR threshold overrides ---
    p.add_argument("--fs_mwu_p_thresh_by_block", default=None,
                   help="Optional per-block MWU/Fisher p thresholds, e.g. RNAblock__:0.01,CNVblock__:0.05,MUTblock__:0.10")
    p.add_argument("--fs_mwu_fdr_thresh_by_block", default=None,
                   help="Optional per-block MWU/Fisher FDR thresholds, e.g. RNAblock__:0.05,CNVblock__:0.20,MUTblock__:0.25")

    p.add_argument("--fs_cox_uni_p_thresh_by_block", default=None,
                   help="Optional per-block univariable Cox p thresholds.")
    p.add_argument("--fs_cox_uni_fdr_thresh_by_block", default=None,
                   help="Optional per-block univariable Cox FDR thresholds.")

    p.add_argument("--fs_cox_multi_p_thresh_by_block", default=None,
                   help="Optional per-block multivariable Cox p thresholds.")
    p.add_argument("--fs_cox_multi_fdr_thresh_by_block", default=None,
                   help="Optional per-block multivariable Cox FDR thresholds.")

    # redundancy pruning threshold (feature-feature corr)
    p.add_argument("--fs_corr_threshold", type=float, default=0.90)

    # --- NEW: per-block redundancy threshold overrides ---
    p.add_argument("--fs_corr_thresholds_by_block", default=None,
                   help="Optional per-block redundancy thresholds, e.g. RNAblock__:0.90,CNVblock__:0.85,MUTblock__:0.75")

    # fallbacks
    p.add_argument("--fs_min_features", type=int, default=10)
    p.add_argument("--fs_topk_if_empty", type=int, default=200)
    p.add_argument("--fs_min_per_block", type=int, default=0)

    # --- limma thresholds (already open; set to 1.0 if you want no filtering) ---
    p.add_argument("--limma_pvalue_threshold", type=float, default=0.01)
    p.add_argument("--limma_padj_threshold", type=float, default=0.25)
    p.add_argument("--limma_logfc_threshold", type=float, default=0.75)

    # --- primary test driver (legacy convenience) ---
    p.add_argument("--primary_test", choices=["mwu", "cox"], default="mwu")
    p.add_argument("--time_col", default="OS_Time")
    p.add_argument("--event_col", default="OS_Status")

    # --- Cox config ---
    p.add_argument("--cox_hr_abs_min", type=float, default=0.0)
    p.add_argument("--cox_hr_direction", choices=["any", "gt1", "lt1"], default="any")
    p.add_argument("--cox_min_n", type=int, default=10)
    p.add_argument("--cox_min_events", type=int, default=1)

    # multivariable Cox
    p.add_argument("--fs_run_cox_multi", action="store_true",
                   help="Also run multivariable Cox per feature (requires --cox_covariates).")
    p.add_argument("--cox_covariates", nargs="+", default=None,
                   help="Covariate column names in meta for multivariable Cox.")

    # --- Variance filter knobs (label-free) ---
    p.add_argument("--var_min", type=float, default=0.0)
    p.add_argument("--var_min_nonmissing", type=int, default=10)
    p.add_argument("--variance_keep_file", default=None)
    p.add_argument("--var_keep_top_frac", type=float, default=None)

    # --- label-based correlation prefilter ---
    p.add_argument("--corr_prefilter", action="store_true")
    p.add_argument("--corr_abs_min", type=float, default=None)
    p.add_argument("--corr_min_n", type=int, default=10)
    p.add_argument(
        "--corr_prune_sig_mode",
        default="hierarchical",
        choices=["hierarchical", "cox_multi", "cox_uni", "group", "mwu", "limma"],
        help=(
            "Which significance source to use when choosing the 'best' feature during "
            "correlation pruning. "
            "Default: hierarchical = cox_multi -> cox_uni -> group. "
            "'group' uses group_p (MWU for continuous features, Fisher for binary features)."
        ),
    )

    # --- MUT + PTMB ---
    p.add_argument("--skip_mutated_pathways", action="store_true",
                   help="Backward-compatible alias for --mut_pathway_source none.")
    p.add_argument(
        "--mut_pathway_source",
        default="snv",
        choices=["snv", "import", "none"],
        help=(
            "Source for binary mutated pathway features: "
            "snv=calculate from mutation/SNV data, import=read precomputed matrix, none=skip."
        ),
    )
    p.add_argument("--mut_binary_matrix_csv", default=None,
                   help="Precomputed binary mutation pathway matrix for --mut_pathway_source import.")
    p.add_argument("--mut_binary_sample_col", default=None,
                   help="Optional sample ID column in imported binary mutation pathway matrix.")
    p.add_argument("--mut_binary_prefix", default="MUTblock__",
                   help="Prefix added to imported binary mutation pathway columns if missing.")

    p.add_argument("--enable_ptmb", action="store_true",
                   help="Backward-compatible alias for --ptmb_source snv when --ptmb_source is none.")
    p.add_argument(
        "--ptmb_source",
        default="none",
        choices=["none", "snv", "import"],
        help="Source for pTMB features: none=skip, snv=calculate from MAF/GMT/GTF, import=read precomputed matrix.",
    )
    p.add_argument("--ptmb_matrix_csv", default=None,
                   help="Precomputed pTMB matrix for --ptmb_source import.")
    p.add_argument("--ptmb_sample_col", default=None,
                   help="Optional sample ID column in imported pTMB matrix.")
    p.add_argument("--ptmb_prefix", default="MUTblock__pTMB__",
                   help="Prefix added to imported pTMB columns if missing.")
    p.add_argument("--ptmb_transform", default="none", choices=["none", "log10", "log1p"],
                   help="Optional transform applied only to imported pTMB values.")

    # Existing SNV/MAF pTMB calculation inputs retained for backward compatibility
    p.add_argument("--ptmb_maf_path", default=None)
    p.add_argument("--ptmb_maf_sample_col", default=None)
    p.add_argument("--ptmb_maf_gene_col", default=None)
    p.add_argument("--ptmb_gtf_path", default=None)
    p.add_argument("--ptmb_cds_cache_csv", default=None)
    p.add_argument("--ptmb_denom_policy", default="all_genes_with_len")
    p.add_argument("--ptmb_min_genes_with_len", type=int, default=1)
    p.add_argument("--ptmb_gmt_path", nargs="+", default=None,
                   help="One or more GMT files for SNV-mode PTMB gene sets.")

    # --- NEW: per-block pathway DB filters ---
    p.add_argument("--fs_rna_pathway_dbs", default=None,
                   help="Comma-separated pathway DBs for RNA block, e.g. HALLMARK,REACTOME,GOBP")
    p.add_argument("--fs_cnv_pathway_dbs", default=None,
                   help="Comma-separated pathway DBs for CNV block, e.g. REACTOME")
    p.add_argument("--fs_mut_pathway_dbs", default=None,
                   help="Comma-separated pathway DBs for MUT block, e.g. REACTOME,GOBP")
    p.add_argument("--fs_ptmb_pathway_dbs", default=None,
                   help="Comma-separated pathway DBs for pTMB block, e.g. REACTOME")

    # optional protein-coding filter
    p.add_argument("--protein_coding_gtf", default=None)

    # --- Model grid toggles ---
    p.add_argument("--include_xgboost", action="store_true")

    # --- Predict-only switch ---
    p.add_argument("--predict_only", action="store_true")

    p.add_argument("--include_stacking", action="store_true")
    p.add_argument("--resume_post_outer", action="store_true")
    p.add_argument("--pam50_col", default=None)

    # --- stacking controls ---
    p.add_argument("--cv_blocks", default="RNA,CNV,MUT,CLINICAL")
    p.add_argument("--stack_meta_model", default="logreg")
    p.add_argument("--stack_base_mode", choices=["best_per_block", "fixed_family"], default="best_per_block")
    p.add_argument("--stack_base_family", default="LogReg_L2_liblinear")
    p.add_argument("--stack_cv", type=int, default=5)
    p.add_argument("--stack_calibrate", action="store_true")
    p.add_argument("--stack_calibrate_method", choices=["sigmoid", "isotonic"], default="sigmoid")
    p.add_argument("--stack_calibrate_cv", type=int, default=5)

    p.add_argument("--RNAgene-bypass", action="store_true", dest="RNAgene_bypass")
    p.add_argument("--CNVgene-bypass", action="store_true", dest="CNVgene_bypass")

    return p.parse_args()


def main():
    args = get_args()

    # normalize pathway options
    args.pathway_options = [s.lower() for s in args.pathway_options]
    unknown_pw = [x for x in args.pathway_options if x not in VALID_PATHWAY_CHOICES]
    if unknown_pw:
        print(f"⚠️ Unknown pathway_options ignored: {unknown_pw}")
        args.pathway_options = [x for x in args.pathway_options if x in VALID_PATHWAY_CHOICES]
        if not args.pathway_options:
            raise ValueError("No valid --pathway_options after filtering.")

    if args.selected_blocks:
        seen = set()
        args.selected_blocks = [b for b in args.selected_blocks if not (b in seen or seen.add(b))]

    # Backward compatibility + source-specific validation for updated FS mutation modes.
    # Old behavior:
    #   --skip_mutated_pathways disables binary mutated pathway features.
    #   --enable_ptmb implies SNV/MAF pTMB calculation unless --ptmb_source was set.
    if getattr(args, "skip_mutated_pathways", False):
        args.mut_pathway_source = "none"

    args.include_mutated_pathways = (getattr(args, "mut_pathway_source", "snv") != "none")

    if getattr(args, "enable_ptmb", False) and getattr(args, "ptmb_source", "none") == "none":
        args.ptmb_source = "snv"

    if getattr(args, "ptmb_source", "none") != "none":
        args.enable_ptmb = True

    if args.mut_pathway_source == "import" and not args.mut_binary_matrix_csv:
        raise ValueError("--mut_binary_matrix_csv is required when --mut_pathway_source import.")

    if args.ptmb_source == "import" and not args.ptmb_matrix_csv:
        raise ValueError("--ptmb_matrix_csv is required when --ptmb_source import.")

    if args.ptmb_source == "snv":
        if not args.ptmb_maf_path:
            raise ValueError("--ptmb_maf_path is required when --ptmb_source snv.")
        if not args.ptmb_gmt_path:
            raise ValueError("--ptmb_gmt_path is required when --ptmb_source snv.")

    print("🚀 Run configuration")
    print(f"   data_path              : {args.data_path}")
    print(f"   output_dir             : {args.output_dir}")
    print(f"   subtype                : {args.subtype}")
    print(f"   status_col             : {args.status_col}")
    print(f"   pathway_options        : {args.pathway_options}")
    print(f"   inner_folds            : {args.inner_folds}")
    print(f"   outer_folds            : {args.outer_folds}")

    print(f"   rna_gene_min_raw_count     : {getattr(args, 'rna_gene_min_raw_count', 10)}")
    print(f"   rna_gene_min_detected_frac : {getattr(args, 'rna_gene_min_detected_frac', 0.25)}")
    print(f"   cnv_gene_min_detected_frac : {getattr(args, 'cnv_gene_min_detected_frac', 0.20)}")
    print(f"   cnv_arm_min_detected_frac  : {getattr(args, 'cnv_arm_min_detected_frac', 0.20)}")
    print(f"   cnv_band_min_detected_frac : {getattr(args, 'cnv_band_min_detected_frac', 0.20)}")
    print(f"   mut_min_detected_frac      : {getattr(args, 'mut_min_detected_frac', 0.10)}")

    print(f"   mut_pathway_source     : {getattr(args, 'mut_pathway_source', 'snv')}")
    print(f"   include_mutated_pathways: {getattr(args, 'include_mutated_pathways', True)}")
    print(f"   mut_binary_matrix_csv  : {getattr(args, 'mut_binary_matrix_csv', None)}")
    print(f"   mut_binary_sample_col  : {getattr(args, 'mut_binary_sample_col', None)}")
    print(f"   mut_binary_prefix      : {getattr(args, 'mut_binary_prefix', 'MUTblock__')}")

    print(f"   enable_ptmb            : {getattr(args, 'enable_ptmb', False)}")
    print(f"   ptmb_source            : {getattr(args, 'ptmb_source', 'none')}")
    print(f"   ptmb_matrix_csv        : {getattr(args, 'ptmb_matrix_csv', None)}")
    print(f"   ptmb_sample_col        : {getattr(args, 'ptmb_sample_col', None)}")
    print(f"   ptmb_prefix            : {getattr(args, 'ptmb_prefix', 'MUTblock__pTMB__')}")
    print(f"   ptmb_transform         : {getattr(args, 'ptmb_transform', 'none')}")
    print(f"   ptmb_maf_path          : {getattr(args, 'ptmb_maf_path', None)}")
    print(f"   ptmb_gmt_path          : {getattr(args, 'ptmb_gmt_path', None)}")

    print(f"   primary_test           : {getattr(args, 'primary_test', 'mwu')}")
    print(f"   fs_enable              : {getattr(args, 'fs_enable', False)}")

    # legacy/global defaults
    print(f"   fs_p_thresh            : {getattr(args, 'fs_p_thresh', None)}")
    print(f"   fs_fdr_thresh          : {getattr(args, 'fs_fdr_thresh', None)}")
    print(f"   mwu_p_thresh           : {getattr(args, 'mwu_p_thresh', None)}")
    print(f"   mwu_fdr_thresh         : {getattr(args, 'mwu_fdr_thresh', None)}")
    print(f"   cox_uni_p_thresh       : {getattr(args, 'cox_uni_p_thresh', None)}")
    print(f"   cox_uni_fdr_thresh     : {getattr(args, 'cox_uni_fdr_thresh', None)}")
    print(f"   cox_multi_p_thresh     : {getattr(args, 'cox_multi_p_thresh', None)}")
    print(f"   cox_multi_fdr_thresh   : {getattr(args, 'cox_multi_fdr_thresh', None)}")

    # new per-block overrides
    print(f"   fs_mwu_p_thresh_by_block      : {getattr(args, 'fs_mwu_p_thresh_by_block', None)}")
    print(f"   fs_mwu_fdr_thresh_by_block    : {getattr(args, 'fs_mwu_fdr_thresh_by_block', None)}")
    print(f"   fs_cox_uni_p_thresh_by_block  : {getattr(args, 'fs_cox_uni_p_thresh_by_block', None)}")
    print(f"   fs_cox_uni_fdr_thresh_by_block: {getattr(args, 'fs_cox_uni_fdr_thresh_by_block', None)}")
    print(f"   fs_cox_multi_p_thresh_by_block  : {getattr(args, 'fs_cox_multi_p_thresh_by_block', None)}")
    print(f"   fs_cox_multi_fdr_thresh_by_block: {getattr(args, 'fs_cox_multi_fdr_thresh_by_block', None)}")

    print(f"   fs_corr_threshold      : {getattr(args, 'fs_corr_threshold', None)}")
    print(f"   fs_corr_thresholds_by_block : {getattr(args, 'fs_corr_thresholds_by_block', None)}")

    print(f"   fs_min_features        : {getattr(args, 'fs_min_features', None)}")
    print(f"   fs_topk_if_empty       : {getattr(args, 'fs_topk_if_empty', None)}")

    print(f"   limma_pvalue_threshold : {getattr(args, 'limma_pvalue_threshold', None)}")
    print(f"   limma_padj_threshold   : {getattr(args, 'limma_padj_threshold', None)}")
    print(f"   limma_logfc_threshold  : {getattr(args, 'limma_logfc_threshold', None)}")

    print(f"   fs_rna_pathway_dbs     : {getattr(args, 'fs_rna_pathway_dbs', None)}")
    print(f"   fs_cnv_pathway_dbs     : {getattr(args, 'fs_cnv_pathway_dbs', None)}")
    print(f"   fs_mut_pathway_dbs     : {getattr(args, 'fs_mut_pathway_dbs', None)}")
    print(f"   fs_ptmb_pathway_dbs    : {getattr(args, 'fs_ptmb_pathway_dbs', None)}")

    if str(args.primary_test).lower() == "cox":
        print(f"   time_col               : {args.time_col}")
        print(f"   event_col              : {args.event_col}")
        print(f"   cox_hr_abs_min         : {getattr(args, 'cox_hr_abs_min', None)}")
        print(f"   cox_hr_direction       : {getattr(args, 'cox_hr_direction', None)}")

    print(f"   predict_only           : {getattr(args, 'predict_only', False)}")
    print(f"   predict_matrix_csv     : {getattr(args, 'predict_matrix_csv', None)}")
    print(f"   include_xgboost        : {getattr(args, 'include_xgboost', False)}")
    print(f"   include_stacking       : {getattr(args, 'include_stacking', False)}")
    print(f"   pam50_col              : {getattr(args, 'pam50_col', None)}")
    print(f"   cv_blocks              : {getattr(args, 'cv_blocks', 'RNA,CNV,MUT,CLINICAL')}")
    print(f"   stack_meta_model       : {getattr(args, 'stack_meta_model', 'logreg')}")
    print(f"   stack_base_mode        : {getattr(args, 'stack_base_mode', 'best_per_block')}")
    print(f"   stack_base_family      : {getattr(args, 'stack_base_family', 'LogReg_L2_liblinear')}")
    print(f"   stack_cv               : {getattr(args, 'stack_cv', 5)}")
    print(f"   stack_calibrate        : {getattr(args, 'stack_calibrate', False)}")
    print(f"   stack_calibrate_method : {getattr(args, 'stack_calibrate_method', 'sigmoid')}")
    print(f"   stack_calibrate_cv     : {getattr(args, 'stack_calibrate_cv', 5)}")
    print(f"   RNAgene_bypass         : {getattr(args, 'RNAgene_bypass', False)}")
    print(f"   CNVgene_bypass         : {getattr(args, 'CNVgene_bypass', False)}")

    if getattr(args, "resume_post_outer", False):
        resume_post_outer(args)
        return

    if args.predict_only:
        sample_ids = None
        if args.predict_ids_csv:
            ids_df = pd.read_csv(args.predict_ids_csv)
            col = "sample_id" if "sample_id" in ids_df.columns else ids_df.columns[0]
            sample_ids = ids_df[col].astype(str).tolist()

        _ = predict_concat_only(
            output_dir=args.output_dir,
            subtype=args.subtype,
            pred_matrix_csv=args.predict_matrix_csv,
            sample_ids=sample_ids,
            pred_labels_csv=args.predict_labels_csv,
            prob_threshold=args.prob_threshold,
            scale_mode=args.scale_mode,
            threshold_mode=args.threshold_mode,
            pam50_col=args.pam50_col
        )
    else:
        out_dir = train_once_concat(args)
        sample_ids = None
        if args.predict_ids_csv:
            ids_df = pd.read_csv(args.predict_ids_csv)
            col = "sample_id" if "sample_id" in ids_df.columns else ids_df.columns[0]
            sample_ids = ids_df[col].astype(str).tolist()

        if args.predict_matrix_csv:
            _ = predict_concat_only(
                output_dir=out_dir,
                subtype=args.subtype,
                pred_matrix_csv=args.predict_matrix_csv,
                sample_ids=sample_ids,
                pred_labels_csv=args.predict_labels_csv,
                prob_threshold=args.prob_threshold,
                scale_mode=args.scale_mode,
                threshold_mode=args.threshold_mode,
                pam50_col=args.pam50_col
            )

    print("🎉 Done.")


if __name__ == "__main__":
    main()
