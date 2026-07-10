#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --- isolate process from user site very early ---
# ---- HARD ISOLATION: must be the first lines in the file ----
# import os, sys, site
# os.environ.pop("PYTHONPATH", None)           # neutralize any injected paths
# os.environ["PYTHONNOUSERSITE"] = "1"         # belt…
# if site.ENABLE_USER_SITE:                    # …and suspenders
#     raise SystemExit(f"User site enabled: {site.getusersitepackages()}")
# Pin critical versions so we fail fast if wrong ones sneak in
# REQUIRED = {"sklearn": "1.6.2", "imblearn": "0.14.0"}
# import importlib
# for name, ver in REQUIRED.items():
#     m = importlib.import_module(name)
#     if getattr(m, "__version__", None) != ver:
#         raise SystemExit(f"{name} {m.__version__} from {m.__file__} (expected {ver})")


"""
Final full-cohort training and prediction script.

This script trains the final locked recurrence-risk models after feature-selection,
model-family selection, and iterated cross-validation assessment have been completed.
It is intended to fit the selected models on the full available subtype-
specific cohort and save the model artifacts needed for downstream prediction,
external validation, and figure generation. This script uses the selected strategy from those
analyses to train the final model on all available training samples.

Overall training workflow:

1. Load subtype-specific metadata and input data.
2. Run feature selection on the full cohort or reuse an existing feature-selection
   directory supplied with --fs_dir_existing.
3. Load the selected post-redundancy feature matrices for each available block:
   RNA, CNV, MUT, and CLINICAL.
4. Assemble block-specific and combined multi-block design matrices.
5. Fit preprocessing on the full training cohort:

   * RNA/CNV-style continuous features are scaled.
   * MUT and CLINICAL features are passed through without scaling.
6. Train the final model family selected from prior CV analyses, either globally or
   by block.
7. Save the final fitted model, preprocessing object, feature order, feature list,
   training matrices, training predictions, and diagnostic plots.
8. Train a block-aware stacked model using previously trained block-level
   models.
9. Provide prediction utilities for applying the saved final model to an external
   validation or prediction matrix.

Feature-selection input:
The script can either run run_feature_selection() internally or reuse previously
generated feature-selection outputs. When feature selection is run internally, it is
performed on the full training cohort because this script is used for final model
training rather than cross-validation. When --fs_dir_existing is provided, the script
loads the selected block matrices from that directory.

Supported feature blocks:

* RNAblock__: RNA-derived features, including pathway scores, gene-level features,
  immune/TME features, custom gene sets, TIDE, and EcoTyper features when available.
* CNVblock__: copy-number-derived features, including arm-level, band-level,
  gene-level, pathway-level, tumor-fraction, and percent-altered features when
  available.
* MUTblock__: mutation-derived features, including binary mutated pathway features,
  TMB/pathogenic features, and optional pTMB features.
* CLINICALblock__: clinical covariates and derived clinical features when available.

Mutation and pTMB feature sources:
Mutation-derived pathway and pTMB features can be generated from raw mutation inputs
or imported from precomputed sample-by-feature matrices. Imported mutation and pTMB
features are incorporated into the MUTblock before final feature selection and model
training, allowing the final model workflow to be used when raw variant files cannot
be shared.

Model training:
The script can train final models for individual feature blocks and combined
multi-block matrices. Model families can be specified directly, supplied through a
per-block model CSV, or inferred from prior CV outputs. Supported model families
include logistic regression, SGD logistic models, support vector machines, random
forests, extra trees, gradient boosting, histogram gradient boosting, and optional
XGBoost. Recurrent samples are oversampled during model fitting using a recurrent-
label-aware oversampling strategy.

Combined and stacked models:
For combined multi-omic modeling, the script trains an ALL-block model using the
selected features across blocks. It can also train a block-aware stacked model using
previously trained block-level final models. In the stacked model, each base learner
uses its assigned feature block, and a meta-model combines the block-level predicted
probabilities.

Saved training artifacts:
The script saves the artifacts needed to reproduce training-time preprocessing and to
apply the model to new datasets, including:

* final fitted model files for each trained block or combined model
* ColumnTransformer preprocessing objects
* unscaled and scaled training matrices
* final feature lists and feature order files
* scaled column lists
* training feature statistics for prediction-time drift checks
* training predictions and predicted probabilities
* model coefficients or feature importances when available
* SHAP outputs when available
* ROC curves, prediction histograms, and KM plots when survival information is
  available
* run configuration and metadata files

Prediction workflow:
In prediction mode, the script loads a saved final model and its preprocessing
artifacts, reads an external prediction matrix, aligns the matrix to the exact
training feature schema, applies the saved preprocessing, and outputs predicted
recurrence-risk probabilities. If labels or survival information are available for
the prediction dataset, the script can also generate ROC, histogram, and KM-style
diagnostic plots.

"""


from sklearn.base import clone

from config import *
from dataloading_utils import *
from dataprocessing_utils import *
from pca_utils import *
from plot_utils import *
from output_utils import *
#from param_grids import *
from model_utils import *
from feature_selection_only import run_feature_selection  # NEW


# -------------------------
# Helpers
# -------------------------



# --- Per-block model CSV -----------------------------------------------------
def load_block_model_csv(path):
    """
    Read a CSV like:
      block,best_model
      RNA,SGD_LogLoss
      CNV,ExtraTrees
      MUT,LogReg_L2_liblinear
      CLINICAL,XGBoost
      STACK,logreg   # optional; controls stacked meta-learner if present
    Returns dict like {"RNA": "SGD_LogLoss", "CNV": "ExtraTrees", ..., "STACK": "logreg"}.
    Accepts either exact grid names (e.g., 'ExtraTrees') or alias keys (e.g., 'extratrees','xgb','logreg').
    """
    import os, pandas as pd
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    bcol = cols.get("block") or list(df.columns)[0]
    mcol = cols.get("best_model") or (list(df.columns)[1] if len(df.columns) > 1 else None)
    if mcol is None:
        return {}
    out = {}
    for _, row in df.iterrows():
        blk = str(row[bcol]).strip().upper()
        if not blk:
            continue
        mdl = str(row[mcol]).strip()
        if not mdl:
            continue
        out[blk] = mdl
    return out

# For the stacked meta-learner: keep it simple (we only support these three)
def _coerce_meta_model_name(s: str | None) -> str:
    if not s:
        return "logreg"
    x = str(s).strip().lower()
    if x in {"logreg", "lr", "sgd_logloss", "sgd"}:  # treat sgd hint as logreg-like meta
        return "logreg"
    if x in {"rf", "randomforest", "random_forest"}:
        return "rf"
    if x in {"gbdt", "gradientboosting", "gradient_boosting"}:
        return "gbdt"
    # Anything else → fall back to logreg
    return "logreg"



def _normalize_model_id(s: str, valid_grid_keys: set[str]) -> str | None:
    """
    Map a CSV 'best_model' value to either:
      - an exact grid name in `valid_grid_keys` (e.g., 'ExtraTrees', 'SGD_LogLoss'), OR
      - a family alias key in MODEL_ALIAS (e.g., 'extratrees', 'sgd', 'logreg').
    Returns None if it can't be recognized.
    """
    s0 = str(s or "").strip()
    if not s0:
        return None

    # exact grid name?
    for g in valid_grid_keys:
        if s0 == g or s0.lower() == g.lower():
            return g

    # compress for alias matching
    low = s0.lower().replace(" ", "").replace("-", "").replace("_", "")

    # common synonyms → alias keys where possible
    syn = {
        "lr": "logreg", "logisticregression": "logreg", "logreg": "logreg",
        "sgd": "sgd", "sgdlogloss": "SGD_LogLoss",
        "svm": "svm", "rbfsvm": "svm",
        "linearsvc": "linearsvc", "linearsvm": "linearsvc",
        "rf": "rf", "randomforest": "rf", "randomforestclassifier": "rf",
        "extratrees": "extratrees", "ext": "extratrees", "et": "extratrees",
        "gbdt": "gbdt", "gradientboosting": "gbdt",
        "hgb": "hgb", "histgradientboosting": "hgb",
        "xgb": "xgb", "xgboost": "xgb", "xgbclassifier": "xgb",
        "stackblocks": "STACK_BLOCKS", "stack": "STACK_BLOCKS"
    }
    if low in syn:
        alias_or_grid = syn[low]
        # Alias key for restrictor?
        if alias_or_grid in MODEL_ALIAS:
            return alias_or_grid
        # Or an actual grid key present?
        if alias_or_grid in valid_grid_keys:
            return alias_or_grid
        return alias_or_grid  # leave as-is; caller may still handle

    # final attempt: case-insensitive exact match to any grid key
    for g in valid_grid_keys:
        if g.lower() == s0.lower():
            return g

    return None


def _load_block_family_csv(path: str | None, valid_grid_keys: set[str]) -> dict[str, str]:
    """
    Read a CSV with columns {block, best_model}. Returns a dict like:
      {'RNA':'sgd', 'CNV':'ExtraTrees', 'MUT':'logreg', 'CLINICAL':'xgb', 'STACK':'rf'}
    Values can be grid names or family aliases.
    Unrecognized rows are skipped (with a print).
    """
    import os, pandas as pd
    out = {}
    if not path or not os.path.exists(path):
        return out

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    bcol = cols.get("block") or list(df.columns)[0]
    mcol = cols.get("best_model") or cols.get("model") or list(df.columns)[1]

    for _, r in df.iterrows():
        blk = str(r[bcol]).strip().upper()
        if blk not in {"RNA", "CNV", "MUT", "CLINICAL", "ALL", "STACK"}:
            continue
        mid = _normalize_model_id(r[mcol], valid_grid_keys)
        if mid is None:
            print(f"[BLOCK CSV] ⚠️ Unrecognized model '{r[mcol]}' for block {blk}; skipping.")
            continue
        out[blk] = mid
    if out:
        print(f"[BLOCK CSV] Loaded per-block families: {out}")
    else:
        print(f"[BLOCK CSV] No valid rows found in {path}")
    return out


# ---- Stacking helpers -------------------------------------------------------
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import StackingClassifier

from sklearn.base import BaseEstimator, TransformerMixin

# --- NEW: discover winners from prior CV "final" model(s) --------------------
def _model_alias_from_estimator(est):
    """Map a fitted estimator (or pipeline/calibrator) to our MODEL_ALIAS key."""
    try:
        from sklearn.calibration import CalibratedClassifierCV
        if isinstance(est, CalibratedClassifierCV):
            est = est.estimator
    except Exception:
        pass

    # unwrap imblearn/sklearn pipeline to reach the final classifier
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from sklearn.pipeline import Pipeline
        if isinstance(est, (ImbPipeline, Pipeline)):
            if "model" in est.named_steps:
                est = est.named_steps["model"]
            else:
                est = est.steps[-1][1]
    except Exception:
        pass

    # imports only for isinstance checks
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC, LinearSVC
    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier,
        GradientBoostingClassifier, HistGradientBoostingClassifier
    )
    try:
        from xgboost import XGBClassifier
    except Exception:
        XGBClassifier = None

    if isinstance(est, LogisticRegression):              return "logreg"
    if isinstance(est, LinearSVC):                       return "linearsvc"
    if isinstance(est, SVC):                             return "svm"
    if isinstance(est, RandomForestClassifier):          return "rf"
    if isinstance(est, ExtraTreesClassifier):            return "extratrees"
    if isinstance(est, GradientBoostingClassifier):      return "gbdt"
    if isinstance(est, HistGradientBoostingClassifier):  return "hgb"
    if XGBClassifier is not None and isinstance(est, XGBClassifier): return "xgb"
    return None


def discover_cv_winner_models(output_dir, subtype, source="auto"):
    """
    Return dict like {"RNA": "logreg", "CNV": "xgb", "MUT": "rf"} describing the
    model FAMILY that 'won' per block in your prior CV outputs.

    Detection order (source='auto'):
      1) cv_models/ALL_BLOCKS_FINAL.pkl (stack with block base estimators)
      2) Per-block finals: cv_models/RNA_FINAL.pkl, CNV_FINAL.pkl, MUT_FINAL.pkl
      3) best_models_summary.csv (reads 'model' column and maps to alias)
    """
    import os, re, joblib, pandas as pd

    fold_dir   = os.path.join(output_dir, subtype)
    models_dir = os.path.join(fold_dir, "cv_models")
    winners = {}

    def _lower_blocks(name):
        name = str(name).strip().lower()
        if name.startswith("rna"): return "RNA"
        if name.startswith("cnv"): return "CNV"
        if name.startswith("mut"): return "MUT"
        if name.startswith("clin"): return "CLINICAL"   # <-- add this
        return None

    def _alias_from_model_string(s):
        s = str(s).lower()
        for key, names in MODEL_ALIAS.items():
            if key in s:
                return key
            # allow direct grid key match (e.g., "RandomForest")
            for n in names:
                if str(n).lower() in s:
                    return key
        # direct keys like "xgboost", "randomforest"
        for k in MODEL_ALIAS:
            if k in s:
                return k
        if "xgboost" in s: return "xgb"
        if "randomforest" in s: return "rf"
        if "extratrees" in s: return "extratrees"
        if "gradientboosting" in s: return "gbdt"
        if "histgradient" in s: return "hgb"
        if "linearsvc" in s: return "linearsvc"
        if "svm" in s: return "svm"
        if "logreg" in s or "logistic" in s: return "logreg"
        return None

    # --- 1) Try ALL_BLOCKS_FINAL.pkl (stack that embeds base models)
    if source in ("auto", "stack"):
        stack_path = os.path.join(models_dir, "ALL_BLOCKS_FINAL.pkl")
        if os.path.exists(stack_path):
            try:
                stack = joblib.load(stack_path)
                # unwrap calibration if present
                from sklearn.calibration import CalibratedClassifierCV
                if isinstance(stack, CalibratedClassifierCV):
                    stack = stack.base_estimator
                # StackingClassifier: estimators_ has named subpipelines
                ests = getattr(stack, "estimators_", None) or []
                named = getattr(stack, "named_estimators_", {})
                if named:
                    for nm, est in named.items():
                        blk = _lower_blocks(nm)
                        if not blk: continue
                        winners[blk] = _model_alias_from_estimator(est) or winners.get(blk)
                else:
                    # fallback if only tuple list
                    for nm, est in ests:
                        blk = _lower_blocks(nm)
                        if not blk: continue
                        winners[blk] = _model_alias_from_estimator(est) or winners.get(blk)
            except Exception as e:
                print(f"[WINNERS] Stack parse failed: {e}")

    # --- 2) Per-block FINAL pkls
    if source in ("auto", "perblock"):
        for blk in ("RNA", "CNV", "MUT", "CLINICAL"):
            if blk in winners: continue
            pkl = os.path.join(models_dir, f"{blk}_FINAL.pkl")
            if os.path.exists(pkl):
                try:
                    est = joblib.load(pkl)
                    winners[blk] = _model_alias_from_estimator(est) or winners.get(blk)
                except Exception as e:
                    print(f"[WINNERS] Could not load {pkl}: {e}")

    # --- 3) best_models_summary.csv (string → alias)
    if source in ("auto", "summary") and len(winners) < 3:
        summ = os.path.join(models_dir, "best_models_summary.csv")
        if os.path.exists(summ):
            try:
                df = pd.read_csv(summ)
                if {"block","model"}.issubset(df.columns):
                    for _, row in df.iterrows():
                        blk = str(row["block"]).strip().upper()
                        if blk not in {"RNA","CNV","MUT"}: continue
                        if blk in winners: continue
                        winners[blk] = _alias_from_model_string(row["model"]) or winners.get(blk)
            except Exception as e:
                print(f"[WINNERS] Could not parse summary: {e}")

    # sanity: drop Nones
    winners = {k: v for k, v in winners.items() if v}
    if not winners:
        print("[WINNERS] No CV winners discovered; will default to --model_id or full grid.")
    else:
        print(f"[WINNERS] Using CV winners per block: {winners}")
    return winners


def _unwrap_final_estimator(est):
    from sklearn.calibration import CalibratedClassifierCV
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from sklearn.pipeline import Pipeline
        pipeline_types = (ImbPipeline, Pipeline)
    except Exception:
        pipeline_types = tuple()

    cur = est

    # unwrap calibration
    if isinstance(cur, CalibratedClassifierCV):
        cur = getattr(cur, "estimator", cur)

    # unwrap pipeline
    if pipeline_types and isinstance(cur, pipeline_types):
        if "model" in cur.named_steps:
            cur = cur.named_steps["model"]
        else:
            cur = cur.steps[-1][1]

    return cur

def _save_coefficients_if_any(model, features, outpath):
    import pandas as pd
    est = _unwrap_final_estimator(model)
    if hasattr(est, "coef_"):
        coef = est.coef_
        if getattr(coef, "ndim", 1) > 1:
            coef = coef[0]
        pd.Series(coef, index=features).to_csv(outpath)
        return True
    return False

def _save_shap_if_any(model, X_df, shap_dir, tag):
    import os, pandas as pd
    try:
        explainer = get_shap_explainer(model, X_df)
    except Exception as e:
        print(f"[SHAP][{tag}] explainer failed: {e}")
        return False

    if explainer is None:
        print(f"[SHAP][{tag}] no compatible explainer")
        return False

    try:
        sv = explainer(X_df)
        arr = sv.values
        if getattr(arr, "ndim", 0) == 3 and arr.shape[2] == 2:
            arr = arr[:, :, 1]
        pd.DataFrame(arr, index=X_df.index, columns=X_df.columns).to_csv(
            os.path.join(shap_dir, f"shap_{tag}.csv")
        )
        print(f"[SHAP][{tag}] saved")
        return True
    except Exception as e:
        print(f"[SHAP][{tag}] compute failed: {e}")
        return False


# ---- Stacking helpers -------------------------------------------------------
from sklearn.base import BaseEstimator, TransformerMixin, clone

class BlockPreprocessor(BaseEstimator, TransformerMixin):
    """
    Clone-safe, block-scoped column selector/preprocessor.

    Notes
    -----
    - __init__ stores params verbatim (no modification) so sklearn.clone works.
    - Derived attributes live in fit() with trailing underscores.
    - Accepts `scale_mode` for API compatibility (scaling behavior is controlled by
      `do_scale`; `scale_mode` is just recorded here).
    """
    def __init__(
        self,
        block="RNA",
        prep_dir=None,
        scale_mode="training",   # <— compatibility only
        do_scale=False,
        winsorize=False,
        p01=0.01,
        p99=0.99,
        prefix_map=None,
    ):
        self.block = block
        self.prep_dir = prep_dir
        self.scale_mode = scale_mode
        self.do_scale = do_scale
        self.winsorize = winsorize
        self.p01 = p01
        self.p99 = p99
        self.prefix_map = prefix_map

    def fit(self, X, y=None):
        import pandas as pd

        default_prefix = {
            "RNA": "RNAblock__",
            "CNV": "CNVblock__",
            "MUT": "MUTblock__",
            "CLINICAL": "CLINICALblock__",
        }
        self.prefix_map_ = self.prefix_map if self.prefix_map is not None else default_prefix
        self.block_norm_ = str(self.block)
        self.prefix_ = self.prefix_map_.get(self.block_norm_, "")

        cols = list(getattr(X, "columns", []))
        self.columns_ = [c for c in cols if str(c).startswith(self.prefix_)]

        self.mean_ = None
        self.std_ = None
        self.lo_ = None
        self.hi_ = None

        if self.columns_:
            Xsub = pd.DataFrame(X, columns=cols)[self.columns_]
            if self.winsorize:
                q = Xsub.quantile([float(self.p01), float(self.p99)])
                self.lo_ = q.iloc[0, :].copy()
                self.hi_ = q.iloc[1, :].copy()

            if self.do_scale:
                Xuse = Xsub.clip(lower=self.lo_, upper=self.hi_, axis=1) if self.winsorize else Xsub
                mu = Xuse.mean(axis=0)
                sd = Xuse.std(axis=0, ddof=0).replace(0.0, float("nan"))
                self.mean_ = mu
                self.std_ = sd.fillna(1.0)

        return self

    def transform(self, X):
        import pandas as pd
        if not hasattr(self, "columns_"):
            raise RuntimeError("BlockPreprocessor not fitted. Call fit(X, y) first.")

        Xout = pd.DataFrame(X).copy()

        # For any training column missing in the incoming matrix, insert the TRAINING MEAN
        # (so z-scoring centers to ~0 rather than (0 - mean)/std).
        for c in self.columns_:
            if c not in Xout.columns:
                val = 0.0
                if (self.mean_ is not None) and (c in getattr(self.mean_, "index", [])):
                    val = float(self.mean_[c])
                Xout[c] = val

        # keep only expected columns in training order
        Xout = Xout.loc[:, self.columns_]

        # sanitize infinities/NaNs produced upstream
        Xout = Xout.replace([float("inf"), float("-inf")], float("nan")).fillna(0.0)

        # optional winsorization + z-scoring with training stats
        if self.columns_ and (self.winsorize or self.do_scale):
            if self.winsorize and (self.lo_ is not None) and (self.hi_ is not None):
                Xout = Xout.clip(lower=self.lo_, upper=self.hi_, axis=1)
            if self.do_scale and (self.mean_ is not None) and (self.std_ is not None):
                Xout = (Xout - self.mean_) / self.std_
                Xout = Xout.replace([float("inf"), float("-inf")], float("nan")).fillna(0.0)

        return Xout

def fit_stack_from_trained_blocks(
    output_dir, subtype,
    blocks="RNA,CNV,MUT,CLINICAL",
    meta_model="logreg",
    scale_mode="training",
    cv=5,
    calibrate=False, calibrate_method="sigmoid", calibrate_cv=3,
    block_model_csv=None,                     # optional
    stack_meta_model=None,                    # optional override
):
    """
    Build a StackingClassifier over trained per-block models, save STACKED_FINAL.pkl,
    generate training-set diagnostics (plots + CSV), and append a 'STACKED' row to
    best_models_summary.csv using the training AUC.

    HARD-CODED in this version:
      - GridSearchCV for STACKED (meta-learner + passthrough), no CLI edits needed
    """
    import os, json, joblib, numpy as np, pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from imblearn.pipeline import Pipeline as ImbPipeline
    import matplotlib.pyplot as plt

    # ---------------------------
    # HARD-CODED GRIDSEARCH KNOBS
    # ---------------------------
    DO_STACKED_GRIDSEARCH = True
    GS_OUTER_CV = 5
    GS_SCORING = "roc_auc"
    GS_NJOBS = 8
    GS_VERBOSE = 1
    SEED = 0
    # (inner cv used by StackingClassifier to generate meta-features)
    STACK_INNER_CV = int(cv) if cv else 5

    # ---------- paths ----------
    fold_dir   = os.path.join(output_dir, subtype)
    models_dir = os.path.join(fold_dir, "cv_models")
    prep_dir   = os.path.join(fold_dir, "preprocessing")
    plots_dir  = os.path.join(fold_dir, "plots")
    preds_dir  = os.path.join(fold_dir, "predictions")
    coef_dir = os.path.join(fold_dir, "coefficients")
    shap_dir = os.path.join(fold_dir, "shap")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    # Resolve meta-learner: override → CSV 'STACK' row → default
    csv_models = load_block_model_csv(block_model_csv) if block_model_csv else {}
    if stack_meta_model:
        meta_model = _coerce_meta_model_name(stack_meta_model)
    elif "STACK" in csv_models:
        meta_model = _coerce_meta_model_name(csv_models["STACK"])
    else:
        meta_model = _coerce_meta_model_name(meta_model)

    print(f"[STACK] Meta-learner: {meta_model} (stack_inner_cv={STACK_INNER_CV}, gridsearch={DO_STACKED_GRIDSEARCH}, calibrate={calibrate})")

    # ---------- load design + labels ----------
    all_unscaled_csv = os.path.join(prep_dir, "ALL_BLOCK_unscaled.csv")
    if not os.path.exists(all_unscaled_csv):
        raise FileNotFoundError(f"Missing {all_unscaled_csv}. Train blocks first.")
    X_full = pd.read_csv(all_unscaled_csv, index_col=0)
    X_full.index = X_full.index.astype(str).str.strip()
    X_full.columns = X_full.columns.astype(str).str.strip()
    X_full = (
        X_full.apply(pd.to_numeric, errors="coerce")
              .replace([float("inf"), float("-inf")], float("nan"))
              .fillna(0.0)
    )

    meta_csv = os.path.join(prep_dir, "subtype_meta_full.csv")
    if not os.path.exists(meta_csv):
        raise FileNotFoundError("subtype_meta_full.csv not found; cannot locate labels.")
    meta = pd.read_csv(meta_csv, index_col=0)
    meta.index = meta.index.astype(str).str.strip()
    if "Disease_Status" not in meta.columns:
        raise KeyError("Disease_Status missing in subtype_meta_full.csv")
    y_series = meta.reindex(X_full.index)["Disease_Status"].astype(str).str.strip().str.lower().map(
        {"non-recurrent":0, "nonrecurrent":0, "0":0, "recurrent":1, "1":1}
    )
    yy = y_series.astype(int).values

    # ---------- build base estimators (prep + cloned estimator) ----------
    req_blocks = [b.strip().upper() for b in str(blocks).split(",") if b.strip()]
    from sklearn.base import clone
    base_pipes = []
    for b in req_blocks:
        model_path = os.path.join(models_dir, ("ALL_FINAL.pkl" if b == "ALL" else f"{b}_FINAL.pkl"))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[STACK] Base model not found for {b}: {model_path}")
        base_model = clone(joblib.load(model_path))  # keep hyperparams; reset fit state

        # mimic block-train scaling: scale RNA/CNV; do NOT scale MUT/CLINICAL
        do_scale = (b not in ("MUT", "CLINICAL"))
        pre = BlockPreprocessor(block=b, prep_dir=prep_dir, scale_mode=scale_mode, do_scale=do_scale)

        base_pipes.append((b.lower(), ImbPipeline([("prep", pre), ("est", base_model)])))


    # ---------- meta-learner + HARD-CODED param grids ----------
    mm = str(meta_model).strip().lower()    
    if mm in ("logreg","lr"):
        final_est = LogisticRegression(
            max_iter=5000,
            solver="saga",
            class_weight="balanced",
            random_state=SEED
        )
        param_grid = [
            {
                "final_estimator__penalty": ["l2"],
                "final_estimator__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
            {
                "final_estimator__penalty": ["l1"],
                "final_estimator__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
        ]
    elif mm in ("rf","randomforest"):
        final_est = RandomForestClassifier(
            n_estimators=500, class_weight="balanced_subsample", n_jobs=GS_NJOBS, random_state=SEED
        )
        param_grid = [
            {
                "final_estimator__n_estimators": [300, 500, 800],
                "final_estimator__max_depth": [None, 5, 10],
                "final_estimator__min_samples_leaf": [1, 2, 5],
            }
        ]
    elif mm in ("gbdt","gb"):
        final_est = GradientBoostingClassifier(random_state=SEED)
        param_grid = [
            {
                "final_estimator__n_estimators": [100, 300, 600],
                "final_estimator__learning_rate": [0.05, 0.1, 0.2],
                "final_estimator__max_depth": [2, 3, 4],
            }
        ]
    else:
        raise ValueError(f"Unknown meta_model: {meta_model}")

    stack_core = StackingClassifier(
        estimators=base_pipes,
        final_estimator=final_est,
        stack_method="predict_proba",
        passthrough=False,
        cv=STACK_INNER_CV,
        n_jobs=GS_NJOBS
    )

    # ---------- fit (HARD-CODED gridsearch) ----------
    gs_obj = None
    if DO_STACKED_GRIDSEARCH:
        outer = StratifiedKFold(n_splits=int(GS_OUTER_CV), shuffle=True, random_state=SEED)
        print(f"[STACK][GS] GridSearchCV: scoring={GS_SCORING}, outer_cv={GS_OUTER_CV}, stack_inner_cv={STACK_INNER_CV}")
        gs_obj = GridSearchCV(
            estimator=stack_core,
            param_grid=param_grid,
            scoring=GS_SCORING,
            cv=outer,
            n_jobs=GS_NJOBS,
            verbose=int(GS_VERBOSE),
            refit=True,
            return_train_score=True,
        )
        gs_obj.fit(X_full, yy)
        stack_core = gs_obj.best_estimator_

        # save gridsearch artifacts
        cvres = pd.DataFrame(gs_obj.cv_results_)
        cvres_path = os.path.join(models_dir, "STACKED_gridsearch_cv_results.tsv")
        cvres.to_csv(cvres_path, sep="\t", index=False)

        best_params_path = os.path.join(models_dir, "STACKED_best_params.json")
        with open(best_params_path, "w") as f:
            json.dump(gs_obj.best_params_, f, indent=2)

        best_score_path = os.path.join(models_dir, "STACKED_best_score.txt")
        with open(best_score_path, "w") as f:
            f.write(f"best_score ({GS_SCORING}): {gs_obj.best_score_}\n")

        print(f"[STACK][GS] ✅ best_score={gs_obj.best_score_:.4f}")
        print(f"[STACK][GS] ✅ saved: {cvres_path}")
        print(f"[STACK][GS] ✅ saved: {best_params_path}")
    else:
        stack_core.fit(X_full, yy)




    def _pos_col(P, classes_):
        if P is None:
            return None
        if classes_ is not None and 1 in set(classes_):
            return P[:, list(classes_).index(1)]
        return P[:, -1]

    # base predictions on training (before calibration)
    base_df = pd.DataFrame(index=X_full.index)
    for name, est in getattr(stack_core, "named_estimators_", {}).items():
        try:
            if hasattr(est, "predict_proba"):
                P = est.predict_proba(X_full)
            elif hasattr(est, "decision_function"):
                from scipy.special import expit
                P = np.column_stack([1 - expit(est.decision_function(X_full)), expit(est.decision_function(X_full))])
            else:
                pred = est.predict(X_full)
                P = np.column_stack([1 - pred.astype(float), pred.astype(float)])
            base_df[f"p_{name.upper()}"] = _pos_col(P, getattr(est, "classes_", None))
        except Exception:
            try:
                pred = est.predict(X_full)
                base_df[f"p_{name.upper()}"] = pred.astype(float)
            except Exception:
                pass

    # stacked proba (uncalibrated)
    if hasattr(stack_core, "predict_proba"):
        P_stack_uncal = stack_core.predict_proba(X_full)
        p_stack_uncal = _pos_col(P_stack_uncal, getattr(stack_core, "classes_", None))
    else:
        from scipy.special import expit
        p_stack_uncal = expit(stack_core.decision_function(X_full))

    # ---------- optional calibration ----------
    stack = stack_core
    was_calibrated = False
    if calibrate:
        cal = CalibratedClassifierCV(stack_core, method=calibrate_method, cv=calibrate_cv)
        cal.fit(X_full, yy)
        stack = cal
        was_calibrated = True

    # final stacked proba (calibrated if requested)
    if hasattr(stack, "predict_proba"):
        P_final = stack.predict_proba(X_full)
        p_stack = _pos_col(P_final, getattr(stack, "classes_", None))
    else:
        from scipy.special import expit
        p_stack = expit(stack.decision_function(X_full))

    # ---------- compute AUCs ----------
    aucs = {}
    for c in base_df.columns:
        try:
            aucs[c] = roc_auc_score(yy, base_df[c].values)
        except Exception:
            pass
    aucs["p_STACKED"] = roc_auc_score(yy, p_stack)

    # ---------- save model ----------
    out_pkl = os.path.join(models_dir, "STACKED_FINAL.pkl")
    joblib.dump(stack, out_pkl)
    with open(os.path.join(models_dir, "STACKED_FINAL_info.txt"), "w") as f:
        f.write(f"blocks: {','.join(req_blocks)}\n")
        f.write(f"meta_model: {meta_model}\n")
        f.write(f"stack_inner_cv: {STACK_INNER_CV}\n")
        f.write(f"gridsearch: {bool(DO_STACKED_GRIDSEARCH)}\n")
        if DO_STACKED_GRIDSEARCH and gs_obj is not None:
            f.write(f"gridsearch_scoring: {GS_SCORING}\n")
            f.write(f"gridsearch_outer_cv: {GS_OUTER_CV}\n")
            f.write(f"gridsearch_best_score: {gs_obj.best_score_}\n")
            f.write(f"gridsearch_best_params: {json.dumps(gs_obj.best_params_)}\n")
        f.write(f"scale_mode: {scale_mode}\n")
        f.write(f"calibrated: {bool(was_calibrated)} ({calibrate_method if was_calibrated else '-'})\n")
        f.write("\n[TRAIN] AUCs (training-set):\n")
        for k, v in sorted(aucs.items()):
            f.write(f"  {k}: {v:.4f}\n")

    print(f"[STACK] ✅ Saved stacked model → {out_pkl}")

    # ---------- save stacked interpretation artifacts ----------
    X_meta_arr = stack_core.transform(X_full)

    expected_n = len(base_pipes)
    if X_meta_arr.shape[1] != expected_n:
        raise RuntimeError(
            f"STACKED meta features should be {expected_n}, but got {X_meta_arr.shape[1]}. "
            f"This usually means passthrough=True."
        )

    meta_cols = [f"p_{name.upper()}" for name, _ in base_pipes]
    meta_X = pd.DataFrame(X_meta_arr, index=X_full.index, columns=meta_cols)
    meta_X.to_csv(os.path.join(prep_dir, "STACKED_meta_features.csv"))

    _save_coefficients_if_any(
        stack_core.final_estimator_,
        list(meta_X.columns),
        os.path.join(coef_dir, "coef_STACKED_FINAL.csv")
    )

    _save_shap_if_any(
        stack_core.final_estimator_,
        meta_X,
        shap_dir,
        tag="STACKED_FINAL"
    )
    # ---------- save per-sample training preds ----------
    out = base_df.copy()
    out["p_STACKED"] = p_stack
    out.insert(0, "y_true", yy.astype(int))
    out.insert(0, "sample_id", out.index.astype(str))
    preds_csv = os.path.join(preds_dir, "STACKED__train_predictions.csv")
    out.to_csv(preds_csv, index=False)
    print(f"[STACK] 💾 Saved training predictions → {preds_csv}")

    # ---------- plots: hist by label ----------
    try:
        plt.figure()
        y_str = pd.Series(yy, index=X_full.index).map({0:"Non-recurrent", 1:"Recurrent"})
        for lab in ["Non-recurrent", "Recurrent"]:
            mask = (y_str == lab).values
            if mask.sum() > 0:
                plt.hist(out.loc[mask, "p_STACKED"].values, bins=30, alpha=0.6, label=f"{lab} (n={mask.sum()})")
        plt.xlabel("Stacked probability (recurrent=1)"); plt.ylabel("Count")
        plt.title("STACKED — training predicted probabilities")
        plt.legend(); plt.tight_layout()
        fig1 = os.path.join(plots_dir, "STACKED_train_pred_proba_hist.png")
        plt.savefig(fig1, dpi=200); plt.close()
        print(f"[PLOT] {fig1}")
    except Exception as e:
        print(f"[WARN] STACKED hist failed: {e}")

    # ---------- plots: ROC (base vs stacked) ----------
    try:
        plt.figure()
        fpr, tpr, _ = roc_curve(yy, out["p_STACKED"].values, pos_label=1)
        plt.plot(fpr, tpr, linewidth=2.0, label=f"STACKED (AUC {aucs['p_STACKED']:.3f})")
        for c in sorted(base_df.columns):
            try:
                fpr_b, tpr_b, _ = roc_curve(yy, base_df[c].values, pos_label=1)
                plt.plot(fpr_b, tpr_b, linestyle="--", alpha=0.9, label=f"{c} (AUC {aucs[c]:.3f})")
            except Exception:
                pass
        plt.plot([0,1],[0,1], linestyle=":")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("Training ROC — STACKED vs base blocks")
        plt.legend(loc="lower right", fontsize=8); plt.tight_layout()
        fig2 = os.path.join(plots_dir, "STACKED_vs_base_train_ROC.png")
        plt.savefig(fig2, dpi=200); plt.close()
        print(f"[PLOT] {fig2}")
    except Exception as e:
        print(f"[WARN] STACKED ROC failed: {e}")

    # ---------- plots: KM @ median split (if time available) ----------
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test

        if "time_to_event" in meta.columns:
            t = pd.to_numeric(meta.reindex(X_full.index)["time_to_event"], errors="coerce")
            e = pd.Series(yy, index=X_full.index)
            ok = (~t.isna()) & (~e.isna())
            if ok.sum() > 0:
                thr = float(np.nanmedian(out["p_STACKED"].values))
                grp = (out["p_STACKED"] >= thr).astype(int)

                km = KaplanMeierFitter()
                plt.figure()
                m0 = ok.values & (grp.values == 0); m1 = ok.values & (grp.values == 1)
                if m0.sum() > 0:
                    km.fit(t[m0], e[m0], label=f"Low (< median)  n={int(m0.sum())}")
                    km.plot(ci_show=False)
                if m1.sum() > 0:
                    km.fit(t[m1], e[m1], label=f"High (≥ median) n={int(m1.sum())}")
                    km.plot(ci_show=False)
                title = "STACKED — KM by predicted (median) risk"
                if m0.sum() > 0 and m1.sum() > 0:
                    lr = logrank_test(t[m0], t[m1], event_observed_A=e[m0], event_observed_B=e[m1])
                    title += f"  (log-rank p = {lr.p_value:.2e})"
                plt.title(title); plt.xlabel("Time (months)"); plt.ylabel("Survival probability")
                plt.ylim(0,1); plt.tight_layout()
                fig3 = os.path.join(plots_dir, "STACKED_train_KM_by_median.png")
                plt.savefig(fig3, dpi=200); plt.close()
                print(f"[PLOT] {fig3}")
            else:
                print("[KM] No usable rows with time_to_event.")
        else:
            print("[KM] time_to_event not found; skipping KM.")
    except Exception as e:
        print(f"[WARN] STACKED KM failed: {e}")

    # ---------- append to best_models_summary.csv ----------
    summary_csv = os.path.join(models_dir, "best_models_summary.csv")
    row = {
        "block": "STACKED",
        "model": f"{meta_model}{'+GS' if DO_STACKED_GRIDSEARCH else ''}{'+Calibrated' if was_calibrated else ''}",
        "mean_cv_auc": float(aucs["p_STACKED"]),  # training AUC (schema compatibility)
        "n_features": int(len(req_blocks)),
        "n_samples": int(X_full.shape[0]),
    }
    try:
        if os.path.exists(summary_csv):
            df = pd.read_csv(summary_csv)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row], columns=["block","model","mean_cv_auc","n_features","n_samples"])
        df.to_csv(summary_csv, index=False)
        print(f"[STACK] 📄 Appended STACKED row to {summary_csv}")
    except Exception as e:
        print(f"[WARN] Could not update best_models_summary.csv: {e}")

    return out_pkl





def _add_block_prefix(df, prefix: str):
    """Add a block prefix to all columns, avoiding double-prefixing."""
    import pandas as pd
    if df is None:
        return df
    if isinstance(df, pd.DataFrame):
        if df.shape[1] == 0:
            return df
        out = df.copy()
        out.columns = out.columns.astype(str)
        if not all(c.startswith(prefix) for c in out.columns):
            out = out.add_prefix(prefix)
        return out
    # tolerate Series; convert to DF
    if hasattr(df, "name"):
        return _add_block_prefix(df.to_frame(), prefix)
    return df

def _save_selected_matrix(df, path):
    """Persist the selected matrix (pre-prefix, pre-preprocess)."""
    import os
    import pandas as pd
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = df.copy()
    df.index = df.index.astype(str).str.strip()
    df.to_csv(path)

def _is_estimator_ok(est):
    """Return False if cloning / getting tags triggers version/tag errors."""
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
    """Drop any grid entries whose estimator fails _is_estimator_ok."""
    return {
        name: (est, params)
        for name, (est, params) in param_grid_dict.items()
        if _is_estimator_ok(est)
    }

def _dump_json_safely(obj, path):
    import json, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=str)
    except Exception as e:
        print(f"[WARN] Could not write JSON to {path}: {e}")


# --- Oversampler that ALWAYS targets class=1 ("recurrent") ---
from sklearn.base import BaseEstimator
from imblearn.over_sampling import RandomOverSampler

class RecurrentOverSampler(BaseEstimator):
    """
    Always oversample the specified positive class (default=1) to
    ratio * (#non-positive). Example: ratio=2.4 => n_pos_target = ceil(2.4 * n_nonpos).
    """
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


def _build_param_grid_for(
    subtype: str | None = None,              # kept for signature compatibility; not used for branching
    oversample_ratios=(1.0, 1.5, 2.0, 2.4, 3.0),
    cw_opts = ("balanced", None, {0: 1.0, 1: 2.0}),
    include_xgboost: bool = True,
):
    """
    Return a dict: { name: (Pipeline(estimator), param_grid) }.
    - Applies RecurrentOverSampler in every pipeline and grids oversampler__ratio.
    - Sweeps class_weight where supported.
    - No special-casing by subtype (everything is tried for every subtype).
    """
    from imblearn.pipeline import Pipeline

    # --- Base imports ---
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier,
        GradientBoostingClassifier, HistGradientBoostingClassifier
    )

    param_grid = {}

    # 1) Logistic Regression (L2)
    param_grid["LogReg_L2_liblinear"] = (
        Pipeline([("oversampler", RecurrentOverSampler()),
                  ("model", LogisticRegression(solver="liblinear"))]),
        {
            "oversampler__ratio": list(oversample_ratios),
            "model__penalty": ["l2"],
            "model__C": [1e-3, 1e-2, 1e-1, 1, 10],
            "model__class_weight": list(cw_opts),
            "model__max_iter": [1000, 5000, 10000],
        },
    )

    # Logistic Regression (L1)
    param_grid["LogReg_L1_liblinear"] = (
        Pipeline([("oversampler", RecurrentOverSampler()),
                  ("model", LogisticRegression(solver="liblinear"))]),
        {
            "oversampler__ratio": list(oversample_ratios),
            "model__penalty": ["l1"],
            "model__C": [1e-3, 1e-2, 1e-1, 1, 10],
            "model__class_weight": list(cw_opts),
            "model__max_iter": [1000, 5000, 10000],
        },
    )

    # Logistic Regression (Elastic Net, saga)
    param_grid["LogReg_ElasticNet_saga"] = (
        Pipeline([("oversampler", RecurrentOverSampler()),
                  ("model", LogisticRegression(solver="saga"))]),
        {
            "oversampler__ratio": list(oversample_ratios),
            "model__penalty": ["elasticnet"],
            "model__C": [1e-3, 1e-2, 1e-1, 1, 10],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            "model__class_weight": list(cw_opts),
            "model__max_iter": [1000, 5000, 10000],
        },
    )

    # 2) SGD (log loss / logistic regression style)
    param_grid["SGD_LogLoss"] = (
        Pipeline([("oversampler", RecurrentOverSampler()),
                  ("model", SGDClassifier(loss="log_loss"))]),
        {
            "oversampler__ratio": list(oversample_ratios),
            "model__alpha": [1e-5, 1e-4, 1e-3],
            "model__penalty": ["l2", "l1", "elasticnet"],
            "model__l1_ratio": [0.15, 0.5, 0.9],
            "model__max_iter": [10000],
            "model__tol": [1e-3],
        },
    )

    # 3) SVM family
    param_grid["SVM_RBF"] = (
        Pipeline([("oversampler", RecurrentOverSampler()),
                  ("model", SVC(probability=True))]),
        {
            "oversampler__ratio": list(oversample_ratios),
            "model__C": [0.1, 1, 10],
            "model__gamma": ["scale", 0.01, 0.001],
            "model__kernel": ["rbf"],
            "model__class_weight": list(cw_opts),
        },
    )

    param_grid["LinearSVC_Calibrated"] = (
        Pipeline([("oversampler", RecurrentOverSampler()),
                  ("model", CalibratedClassifierCV(estimator=LinearSVC()))]),
        {
            "oversampler__ratio": list(oversample_ratios),
            "model__estimator__C": [0.1, 1, 10],
            "model__estimator__class_weight": list(cw_opts),
            "model__method": ["sigmoid", "isotonic"],
        },
    )

    # 4) Tree / ensemble models
    param_grid["RandomForest"] = (
        Pipeline([("oversampler", RecurrentOverSampler()),
                  ("model", RandomForestClassifier())]),
        {
            "oversampler__ratio": list(oversample_ratios),
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__max_features": ["sqrt", "log2"],
            "model__class_weight": list(cw_opts),
            "model__n_jobs": [1],
        },
    )

    param_grid["ExtraTrees"] = (
        Pipeline([("oversampler", RecurrentOverSampler()),
                  ("model", ExtraTreesClassifier())]),
        {
            "oversampler__ratio": list(oversample_ratios),
            "model__n_estimators": [300, 600],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__max_features": ["sqrt", "log2"],
            "model__class_weight": list(cw_opts),
            "model__n_jobs": [1],
        },
    )

    param_grid["GradientBoosting"] = (
        Pipeline([("oversampler", RecurrentOverSampler()),
                  ("model", GradientBoostingClassifier())]),
        {
            "oversampler__ratio": list(oversample_ratios),
            "model__n_estimators": [200, 400],
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [3, 5],
            "model__subsample": [0.8, 1.0],
        },
    )

    param_grid["HistGradientBoosting"] = (
        Pipeline([("oversampler", RecurrentOverSampler()),
                  ("model", HistGradientBoostingClassifier(early_stopping=True))]),
        {
            "oversampler__ratio": list(oversample_ratios),
            "model__max_depth": [None, 6, 10],
            "model__learning_rate": [0.03, 0.1],
            "model__l2_regularization": [0.0, 0.1, 1.0],
            "model__max_iter": [200, 500, 1000, 5000],
        },
    )

    # 5) XGBoost (optional)
    if include_xgboost:
        try:
            from xgboost import XGBClassifier
            param_grid["XGBoost"] = (
                Pipeline([("oversampler", RecurrentOverSampler()),
                          ("model", XGBClassifier(
                              objective="binary:logistic",
                              eval_metric="logloss",
                              tree_method="hist",
                              random_state=42,
                              n_jobs=1,
                          ))]),
                {
                    "oversampler__ratio": list(oversample_ratios),
                    "model__n_estimators": [200, 500, 1000],
                    "model__max_depth": [3, 5, 8],
                    "model__learning_rate": [0.03, 0.1],
                    "model__subsample": [0.7, 1.0],
                    "model__colsample_bytree": [0.7, 1.0],
                    "model__min_child_weight": [1, 5, 10],
                    "model__reg_lambda": [0.0, 1.0, 10.0],
                    "model__reg_alpha": [0.0, 0.1],
                },
            )
        except Exception as e:
            print(f"[GRID DROP] XGBoost unavailable: {e}")

    return param_grid


from sklearn.calibration import CalibratedClassifierCV

def _grid_select_and_fit(X, y, param_grid_dict, cv_folds, label,
                         calibrate=False, calibrate_method="sigmoid", calibrate_cv=5):
    import numpy as np
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import roc_auc_score
    from sklearn.calibration import CalibratedClassifierCV

    def _roc_auc_scorer_auto(estimator, X_test, y_true):
        import numpy as np
        from sklearn.metrics import roc_auc_score

        # get a score aligned to P(y=1)
        proba_like = None
        if hasattr(estimator, "predict_proba"):
            P = estimator.predict_proba(X_test)
            classes_ = getattr(estimator, "classes_", None)
            if classes_ is not None and 1 in set(classes_):
                pos_idx = list(classes_).index(1)
            else:
                pos_idx = None
                if classes_ is not None:
                    for i, c in enumerate(classes_):
                        if str(c).strip().lower() in ("recurrent", "1", "true", "case"):
                            pos_idx = i; break
                if pos_idx is None:
                    pos_idx = -1
            proba_like = P[:, pos_idx]
        elif hasattr(estimator, "decision_function"):
            proba_like = estimator.decision_function(X_test)
        else:
            proba_like = estimator.predict(X_test)

        y_true = np.asarray(y_true).astype(int)
        return roc_auc_score(y_true, proba_like)

    best_auc = -np.inf
    best_name = None
    best_model = None
    best_params = None

    for name, (est, grid) in param_grid_dict.items():
        try:
            gs = GridSearchCV(
                est, grid, cv=cv_folds,
                scoring=_roc_auc_scorer_auto,   # <-- use robust scorer
                n_jobs=1, refit=True, error_score=np.nan
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
            best_auc   = float(score)
            best_name  = name
            best_model = gs.best_estimator_
            best_params= gs.best_params_

    if best_model is None:
        raise RuntimeError(f"No model finished successfully in GridSearchCV for {label}.")

    # Fit on full training
    best_model.fit(X, y)

    # Optional: wrap with probability calibration
    if calibrate and not isinstance(best_model, CalibratedClassifierCV):
        try:
            cal = CalibratedClassifierCV(estimator=best_model,
                                         method=calibrate_method, cv=calibrate_cv)
            cal.fit(X, y)
            best_model = cal
            best_name = f"{best_name}+Calibrated({calibrate_method},cv={calibrate_cv})"
        except Exception as e:
            print(f"[WARN] Calibration failed; using uncalibrated model: {e}")

    return best_model, best_name, best_auc, best_params


def _proba_pos(est, X, pos_label=1):
    """
    Return P(y=1) (recurrent) if available; otherwise a min–max normalized decision score.
    Robustly finds the 'recurrent/1' column even if classes_ is string-typed or reversed.
    """
    import numpy as np
    if hasattr(est, "predict_proba"):
        P = est.predict_proba(X)
        classes_ = getattr(est, "classes_", np.arange(P.shape[1]))
        # Try numeric 1 first
        try:
            idx = np.where(classes_ == pos_label)[0]
            if idx.size:
                return P[:, int(idx[0])]
        except Exception:
            pass
        # Fallback: look for textual 'recurrent' / '1'
        try:
            idx = [i for i, c in enumerate(classes_) if str(c).strip().lower() in ("recurrent", "1")]
            if idx:
                return P[:, idx[0]]
        except Exception:
            pass
        # Last fallback
        return P[:, -1]

    if hasattr(est, "decision_function"):
        s = est.decision_function(X)
        s = s.astype(float)
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    raise ValueError("Estimator lacks predict_proba/decision_function.")

# -------------------------
# Model filtering helpers (NEW)
# -------------------------
MODEL_ALIAS = {
    "lr": {"LogReg_L2_liblinear", "LogReg_L1_liblinear", "LogReg_ElasticNet_saga"},
    "logreg": {"LogReg_L2_liblinear", "LogReg_L1_liblinear", "LogReg_ElasticNet_saga"},
    "sgd": {"SGD_LogLoss"},
    "svm": {"SVM_RBF", "LinearSVC_Calibrated"},
    "linearsvc": {"LinearSVC_Calibrated"},
    "rf": {"RandomForest"},
    "extratrees": {"ExtraTrees"},
    "gbdt": {"GradientBoosting"},
    "hgb": {"HistGradientBoosting"},
    "xgb": {"XGBoost"},
    # convenience group
    "trees": {"RandomForest", "ExtraTrees", "GradientBoosting", "HistGradientBoosting", "XGBoost"},
}

def restrict_param_grid_to_model(param_grid_dict, model_id: str):
    """
    Return a new dict containing only the grids matching model_id.
    model_id can be a key in MODEL_ALIAS or an exact grid name.
    """
    if not model_id:
        return param_grid_dict
    wanted = MODEL_ALIAS.get(str(model_id).lower())
    if wanted is None:
        wanted = {model_id}
    return {k: v for k, v in param_grid_dict.items() if k in wanted}

def _load_final_training_feature_whitelist(fs_dir: str) -> tuple[list[str], str | None]:
    """
    Prefer FINAL_feature_list.csv if present, else FINAL_features_only.csv.

    Supports two formats:
      1) one-column feature list
      2) full annotated feature table containing a 'feature' column and, optionally,
         a boolean keep column such as kept_after_corr / selected_any / keep / selected.
    """
    import os
    import pandas as pd

    candidates = [
        #os.path.join(fs_dir, "FINAL_feature_list.csv"),
        os.path.join(fs_dir, "FINAL_features_only.csv"),
    ]

    chosen = None
    for p in candidates:
        if os.path.exists(p):
            chosen = p
            break

    if chosen is None:
        return [], None

    try:
        df = pd.read_csv(chosen, sep=None, engine="python", comment="#")
    except Exception:
        df = pd.read_csv(chosen, comment="#")

    if df is None or df.empty:
        return [], chosen

    cols = [str(c).strip() for c in df.columns]
    low = {str(c).strip().lower(): c for c in df.columns}

    feat_col = None
    for k in ["feature", "features"]:
        if k in low:
            feat_col = low[k]
            break

    # one-column file: just use it
    if feat_col is None and df.shape[1] == 1:
        feats = df.iloc[:, 0].astype(str).str.strip()
        feats = [f for f in feats if f and f.lower() != "nan"]
        # preserve order, dedup
        seen = set()
        out = []
        for f in feats:
            if f not in seen:
                seen.add(f)
                out.append(f)
        return out, chosen

    if feat_col is None:
        return [], chosen

    keep_col = None
    for k in ["kept_after_corr", "selected_any", "selected", "keep", "final_keep"]:
        if k in low:
            keep_col = low[k]
            break

    feats_df = df.copy()
    if keep_col is not None:
        keep_mask = feats_df[keep_col].astype(str).str.strip().str.lower().isin(
            ["true", "1", "yes", "y", "t"]
        )
        feats_df = feats_df.loc[keep_mask].copy()

    feats = feats_df[feat_col].astype(str).str.strip()
    feats = [f for f in feats if f and f.lower() != "nan"]

    # preserve order, dedup
    seen = set()
    out = []
    for f in feats:
        if f not in seen:
            seen.add(f)
            out.append(f)

    return out, chosen

def train_once_concat(args):
    """
    One-shot pipeline:

    1) If no --fs_dir_existing is provided:
         - Run feature selection via `run_feature_selection(...)` using the
           supplied pathway & significance arguments.
         - Use the resulting FS directory as `fs_dir`.
       Else:
         - Reuse the existing FS outputs in --fs_dir_existing.

    2) Load FS-selected blocks:
         - RNAblock_selected_post_redundancy.csv
         - CNVblock_selected_post_redundancy.csv
         - MUTblock_selected_post_redundancy.csv
         - CLINICALblock_selected_post_redundancy.csv (if present)

    3) Build per-block design matrices (RNA, CNV, MUT, CLINICAL, ALL).

    4) For each block:
         - Apply a ColumnTransformer that z-scores all non-MUT / non-CLINICAL
           columns, passes MUTblock__/CLINICALblock__ through untouched.
         - Run GridSearchCV over the chosen model family(ies), where the
           family to search can be:
             a) global --model_id (applies to all blocks), OR
             b) per-block family from --block_model_csv, OR
             c) discovered CV winners from previous runs (if --use_cv_winners),
             d) otherwise: the full grid.
         - Fit the best model on all samples.
         - Save scaler, scaled/unscaled matrices, final feature list, SHAP,
           coefficients, and training plots.

    5) For the ALL block, also emit back-compat artifacts:
         - global_cont_scaler.joblib
         - train_feature_order.csv
         - global_scaled_columns.csv
         - ALL_BLOCKS_FINAL_features.csv
         - ALL_BLOCKS_CONCAT_unscaled.csv
         - ALL_BLOCKS_CONCAT_scaled.csv
         - ALL_BLOCKS_FINAL.pkl
         - train_feature_stats.csv

    This function is **training-only** and expects to be called when
    `args.predict_only` is False. Prediction is handled by predict_concat_only
    / predict_stacked.
    """
    import os, sys, platform, numpy as np, pandas as pd, joblib, matplotlib
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer

    # -------------------- Step 0: decide / run feature selection --------------------
    # If user passed an existing FS directory, reuse it. Otherwise, run FS inline.
    fs_dir = getattr(args, "fs_dir_existing", None)

    out_dir  = make_output_path(args.output_dir, args.subtype)
    fold_dir = os.path.join(out_dir, args.subtype)

    if not fs_dir:
        print("[FS] No --fs_dir_existing provided; running feature selection inside training.")

        fs_dir = run_feature_selection(
            data_path=args.data_path,
            output_dir=args.output_dir,
            subtype=args.subtype,
            status_col=getattr(args, "status_col", "Disease_Status"),
            pathway_options=getattr(args, "pathway_options", ["hallmark_kegg"]),

            # Required so selected block CSVs are created
            enable_subset_fs=True,
            subset_run_name=getattr(args, "subset_run_name", "training_fullcohort"),
            sample_id_subset=None,

            # Blocks / feature families
            include_mutated_pathways=(getattr(args, "mut_pathway_source", "snv") != "none"),
            mut_pathway_source=getattr(args, "mut_pathway_source", "snv"),
            mut_binary_matrix_csv=getattr(args, "mut_binary_matrix_csv", None),
            mut_binary_sample_col=getattr(args, "mut_binary_sample_col", None),
            mut_binary_prefix=getattr(args, "mut_binary_prefix", "MUTblock__"),
            include_binary_arms=True,
            include_binary_bands=True,
            include_estimate_values=True,
            RNAgene_bypass=getattr(args, "RNAgene_bypass", False),
            CNVgene_bypass=getattr(args, "CNVgene_bypass", False),

            # Variance filter
            var_min=getattr(args, "var_min", 0.0),
            var_min_nonmissing=getattr(args, "var_min_nonmissing", 10),
            variance_keep_file=getattr(args, "variance_keep_file", None),
            var_keep_top_frac=getattr(args, "var_keep_top_frac", None),

            # Optional label-correlation prefilter
            corr_prefilter=getattr(args, "corr_prefilter", False),
            corr_abs_min=getattr(args, "corr_abs_min", None),
            corr_min_n=getattr(args, "corr_min_n", 10),

            # Group comparison / Fisher for binary
            run_mwu=True,
            mwu_p_thresh=float(getattr(args, "fs_p_thresh", 0.05)),
            mwu_fdr_thresh=float(getattr(args, "fs_fdr_thresh", 0.25)),
            mwu_case_label="recurrent",
            mwu_ctrl_label="non-recurrent",

            # Cox FS options
            run_cox_uni=getattr(args, "run_cox_uni", False),
            run_cox_multi=getattr(args, "run_cox_multi", False),
            cox_time_col=getattr(args, "cox_time_col", None),
            cox_event_col=getattr(args, "cox_event_col", None),
            cox_covariates=getattr(args, "cox_covariates", None),
            cox_uni_p_thresh=getattr(args, "cox_uni_p_thresh", 0.05),
            cox_uni_fdr_thresh=getattr(args, "cox_uni_fdr_thresh", 0.25),
            cox_multi_p_thresh=getattr(args, "cox_multi_p_thresh", 0.05),
            cox_multi_fdr_thresh=getattr(args, "cox_multi_fdr_thresh", 0.25),
            cox_hr_abs_min=getattr(args, "cox_hr_abs_min", 0.0),
            cox_hr_direction=getattr(args, "cox_hr_direction", "any"),
            cox_min_n=getattr(args, "cox_min_n", 10),
            cox_min_events=getattr(args, "cox_min_events", 1),

            # Limma thresholds
            limma_pvalue_threshold=float(getattr(args, "limma_pvalue_threshold", 0.05)),
            limma_padj_threshold=float(getattr(args, "limma_padj_threshold", 0.25)),
            limma_logfc_threshold=float(getattr(args, "limma_logfc_threshold", 0.25)),

            # Redundancy pruning
            corr_threshold=float(getattr(args, "fs_corr_threshold", 0.70)),

            # Minimum fallback feature behavior
            fs_min_features=int(getattr(args, "fs_min_features", 10)),
            fs_topk_if_empty=int(getattr(args, "fs_topk_if_empty", 200)),

            # Optional custom gene sets / protein coding / pTMB
            custom_gene_sets_file=getattr(args, "custom_gene_sets_file", None),
            custom_gene_set_min_overlap=getattr(args, "custom_gene_set_min_overlap", 1),
            protein_coding_gtf=getattr(args, "protein_coding_gtf", None),

            enable_ptmb=getattr(args, "enable_ptmb", False),
            ptmb_source=getattr(args, "ptmb_source", "none"),

            # Imported pTMB mode
            ptmb_matrix_csv=getattr(args, "ptmb_matrix_csv", None),
            ptmb_sample_col=getattr(args, "ptmb_sample_col", None),
            ptmb_prefix=getattr(args, "ptmb_prefix", "MUTblock__pTMB__"),
            ptmb_transform=getattr(args, "ptmb_transform", "none"),

            # SNV/MAF pTMB calculation mode
            ptmb_maf_path=getattr(args, "ptmb_maf_path", None),
            ptmb_maf_sample_col=getattr(args, "ptmb_maf_sample_col", None),
            ptmb_maf_gene_col=getattr(args, "ptmb_maf_gene_col", None),
            ptmb_gtf_path=getattr(args, "ptmb_gtf_path", None),
            ptmb_cds_cache_csv=getattr(args, "ptmb_cds_cache_csv", None),
            ptmb_denom_policy=getattr(args, "ptmb_denom_policy", "all_genes_with_len"),
            ptmb_min_genes_with_len=getattr(args, "ptmb_min_genes_with_len", 1),
            ptmb_gmt_path=getattr(args, "ptmb_gmt_path", None),
        )

        print(f"[FS] Feature selection completed. Using FS dir: {fs_dir}")

    else:
        if not os.path.isdir(fs_dir):
            raise RuntimeError(f"--fs_dir_existing={fs_dir!r} does not exist or is not a directory.")
        print(f"[FS] Reusing existing FS outputs at: {fs_dir}")

    # Sanity: user-supplied training matrix is incompatible with this path
    if getattr(args, "train_matrix_csv", None):
        raise RuntimeError(
            "train_once_concat now always trains on FS-selected blocks. "
            "User-supplied --train_matrix_csv bypass is disabled in this mode."
        )

    # -------------------- Load lightweight containers (for meta/labels only) --------------------
    bypass_pathways = []  # avoid pathway lookups entirely in load_all_data
    data, _ = load_all_data(args.data_path, args.subtype, bypass_pathways)

    # Normalize indices for any DataFrames in `data`
    for k, df in list(data.items()):
        if isinstance(df, pd.DataFrame):
            df.index = df.index.astype(str).str.strip()

    # Meta & label encoding
    data['meta'] = data['meta'].rename(columns={"Disease Status": "Disease_Status"})
    if args.status_col not in data['meta'].columns:
        raise KeyError(
            f"Target column {args.status_col!r} not in meta "
            f"(columns: {list(data['meta'].columns)})"
        )

    for d in ["cv_models", "preprocessing", "shap", "coefficients", "plots", "predictions"]:
        os.makedirs(os.path.join(fold_dir, d), exist_ok=True)
    models_dir = os.path.join(fold_dir, "cv_models")
    prep_dir   = os.path.join(fold_dir, "preprocessing")
    shap_dir   = os.path.join(fold_dir, "shap")
    coef_dir   = os.path.join(fold_dir, "coefficients")
    plots_dir  = os.path.join(fold_dir, "plots")

    # Persist small config bits
    _dump_json_safely({"subtype": args.subtype, "block_as_subtype": True},
                      os.path.join(prep_dir, "block_as_subtype.json"))
    _dump_json_safely({"significance_p": float(args.cox_uni_p_thresh)},
                      os.path.join(prep_dir, "significance_config.json"))
    _dump_json_safely(
        {
            "fs_dir_existing": fs_dir,
            "mut_pathway_source": getattr(args, "mut_pathway_source", "snv"),
            "mut_binary_matrix_csv": getattr(args, "mut_binary_matrix_csv", None),
            "mut_binary_sample_col": getattr(args, "mut_binary_sample_col", None),
            "mut_binary_prefix": getattr(args, "mut_binary_prefix", "MUTblock__"),
            "enable_ptmb": getattr(args, "enable_ptmb", False),
            "ptmb_source": getattr(args, "ptmb_source", "none"),
            "ptmb_matrix_csv": getattr(args, "ptmb_matrix_csv", None),
            "ptmb_sample_col": getattr(args, "ptmb_sample_col", None),
            "ptmb_prefix": getattr(args, "ptmb_prefix", "MUTblock__pTMB__"),
            "ptmb_transform": getattr(args, "ptmb_transform", "none"),
            "ptmb_maf_path": getattr(args, "ptmb_maf_path", None),
            "ptmb_gmt_path": getattr(args, "ptmb_gmt_path", None),
            "ptmb_gtf_path": getattr(args, "ptmb_gtf_path", None),
            "ptmb_cds_cache_csv": getattr(args, "ptmb_cds_cache_csv", None),
        },
        os.path.join(prep_dir, "fs_inputs.json")
    )

    # Canonical sample set (meta index)
    all_ids = data['meta'].index.astype(str).str.strip()
    subtype_meta = data['meta'].reindex(all_ids).copy()

    # Binary labels (recurrent=1, non-recurrent=0)
    y_map = (
        subtype_meta[args.status_col]
        .astype(str).str.strip().str.lower()
        .map({"non-recurrent": 0, "nonrecurrent": 0, "0": 0, "recurrent": 1, "1": 1})
    )
    if y_map.isna().any():
        bad = sorted(set(
            subtype_meta[args.status_col].astype(str).str.strip().str.lower()[y_map.isna()]
        ))
        raise ValueError(f"Unrecognized labels in {args.status_col}: {bad}")
    y_all_enc = y_map.astype(int).reindex(all_ids)

    _dump_json_safely(
        {"positive_class": "recurrent",
         "coding": {"non-recurrent": 0, "recurrent": 1}},
        os.path.join(prep_dir, "status_mapping.json")
    )

    # Ensure time_to_event exists
    if "time_to_event" not in subtype_meta.columns:
        if ("time_recur" in subtype_meta.columns) or ("time_norecur" in subtype_meta.columns):
            subtype_meta["time_recur"]   = subtype_meta.get("time_recur",   pd.Series(index=subtype_meta.index, dtype=float))
            subtype_meta["time_norecur"] = subtype_meta.get("time_norecur", pd.Series(index=subtype_meta.index, dtype=float))
            subtype_meta["time_to_event"] = subtype_meta["time_recur"].fillna(subtype_meta["time_norecur"])
        else:
            subtype_meta["time_to_event"] = np.nan

    subtype_meta.to_csv(os.path.join(prep_dir, "subtype_meta_full.csv"))
    short_cols = ["tumor_filename", "RNA_filename", "pam50_subtype", "Disease_Status", "time_to_event"]
    short_cols = [c for c in short_cols if c in subtype_meta.columns]
    subtype_meta[short_cols].to_csv(os.path.join(prep_dir, "subtype_meta_short.csv"))

    # -------------------- Load FS-selected blocks --------------------
    def _load_block_csv(name):
        p = os.path.join(fs_dir, name)
        if not os.path.exists(p):
            return None
        df = pd.read_csv(p, index_col=0)
        df.index   = df.index.astype(str).str.strip()
        df.columns = df.columns.astype(str).str.strip()
        return df.reindex(all_ids)

    df_RNA  = _load_block_csv("RNAblock_selected_post_redundancy.csv")
    df_CNV  = _load_block_csv("CNVblock_selected_post_redundancy.csv")
    df_MUT  = _load_block_csv("MUTblock_selected_post_redundancy.csv")
    df_CLIN = _load_block_csv("CLINICALblock_selected_post_redundancy.csv")

    available = {
        k: v for k, v in {
            "RNA": df_RNA,
            "CNV": df_CNV,
            "MUT": df_MUT,
            "CLINICAL": df_CLIN,
        }.items() if v is not None and v.shape[1] > 0
    }

    if not available:
        raise RuntimeError(
            f"No usable FS block outputs found in {fs_dir} "
            "(expected RNA/CNV/MUT/CLINICAL *_selected_post_redundancy.csv)."
        )

    # Clean & numeric
    for k, df in list(available.items()):
        available[k] = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # -------------------- NEW: subset by FINAL feature whitelist if present --------------------
    final_keep, final_keep_path = _load_final_training_feature_whitelist(fs_dir)

    if final_keep_path:
        print(f"[FS] Applying final feature whitelist from: {final_keep_path}")

    if final_keep:
        final_keep_set = set(final_keep)
        filtered = {}

        for tag, df in available.items():
            keep_cols = [f for f in final_keep if f in df.columns]
            missing_cols = [f for f in final_keep if f.startswith(f"{tag}block__") and f not in df.columns]

            if keep_cols:
                filtered[tag] = df.loc[:, keep_cols].copy()
                print(f"[FS] {tag}: kept {len(keep_cols)}/{df.shape[1]} columns after FINAL_feature_list filter.")
            else:
                print(f"[FS] {tag}: no columns survived FINAL_feature_list filter; block will be skipped.")

            if missing_cols:
                print(f"[FS] {tag}: {len(missing_cols)} final-list features not found in block CSV. "
                      f"Examples: {missing_cols[:10]}")

        available = {k: v for k, v in filtered.items() if v is not None and v.shape[1] > 0}

        if not available:
            raise RuntimeError(
                f"After applying FINAL_feature_list whitelist from {final_keep_path}, "
                f"no usable block features remained."
            )

    # Build ALL = concat of all available blocks
    pieces = [available[k] for k in ["RNA", "CNV", "MUT", "CLINICAL"] if k in available]
    X_ALL = pd.concat(pieces, axis=1).reindex(all_ids).fillna(0.0)

    # keep ALL block in FINAL_feature_list order if whitelist exists
    if final_keep:
        all_keep_cols = [f for f in final_keep if f in X_ALL.columns]
        X_ALL = X_ALL.loc[:, all_keep_cols].copy()

    block_mats = dict(available)
    block_mats["ALL"] = X_ALL

    # Persist selected features for audit
    for tag, X in block_mats.items():
        X.to_csv(os.path.join(prep_dir, f"{tag}_BLOCK_unscaled_SELECTED.csv"))
        pd.Series(list(X.columns), name="feature").to_csv(
            os.path.join(prep_dir, f"{tag}_BLOCK_SELECTED_feature_list.csv"), index=False
        )

    # -------------------- Grids & per-block selection --------------------
    all_grids = _filter_param_grid(_build_param_grid_for(args.subtype))
    valid_grid_keys = set(all_grids.keys())

    csv_models = _load_block_family_csv(
        getattr(args, "block_model_csv", None),
        valid_grid_keys
    ) or {}

    if csv_models:
        print(f"[MODELS CSV] Per-block selections: {csv_models}")

    block_winners = {}
    if bool(getattr(args, "use_cv_winners", False)):
        block_winners = discover_cv_winner_models(
            args.output_dir, args.subtype,
            source=str(getattr(args, "winners_source", "auto"))
        )

    def _grid_for_block(tag: str):
        # 1) global override
        if getattr(args, "model_id", None):
            return restrict_param_grid_to_model(all_grids, args.model_id)
        # 2) CSV specification
        if tag in csv_models:
            return restrict_param_grid_to_model(all_grids, csv_models[tag])
        # 3) auto winners
        if block_winners.get(tag):
            return restrict_param_grid_to_model(all_grids, block_winners[tag])
        # 4) full pool
        return all_grids

    # -------------------- Helper for fitting a block --------------------
    def _fit_block(tag: str, X_all: pd.DataFrame, grid_pool):
        """
        Scale (excluding MUTblock__* / CLINICALblock__*), tune model on inner CV,
        save artifacts for this block.
        Returns (model_name, auc, best_params, n_features, n_samples).
        """
        # ---- Train stats
        stats = pd.DataFrame({
            "mean": X_all.mean(),
            "std": X_all.std(ddof=0).replace(0, 1e-9),
            "p01": X_all.quantile(0.01),
            "p99": X_all.quantile(0.99),
            "min": X_all.min(),
            "max": X_all.max(),
        })
        stats.index.name = "feature"
        stats.to_csv(os.path.join(prep_dir, f"train_feature_stats__{tag}.csv"))


        _DISCRETE_CNV_STATES = {-2.0, -1.0, 0.0, 1.0, 2.0}
        _BINARY_STATES = {0.0, 1.0}

        def _rounded_unique_numeric_values(s: pd.Series, ndigits: int = 8):
            x = pd.to_numeric(s, errors="coerce").dropna()
            if x.empty:
                return set()
            return set(np.round(x.astype(float).unique(), ndigits))

        def _is_binary_feature(s: pd.Series) -> bool:
            vals = _rounded_unique_numeric_values(s)
            return len(vals) > 0 and vals.issubset(_BINARY_STATES)

        def _is_discrete_cnv_state_feature(s: pd.Series) -> bool:
            vals = _rounded_unique_numeric_values(s)
            return len(vals) > 0 and vals.issubset(_DISCRETE_CNV_STATES)

        def _should_scale_feature(col_name: str, s: pd.Series) -> bool:
            # Never scale MUT / CLINICAL
            if col_name.startswith(("MUTblock__", "CLINICALblock__")):
                return False

            # Never scale binary features anywhere
            if _is_binary_feature(s):
                return False

            # For CNV: leave discrete state calls unscaled, scale continuous scores
            if col_name.startswith("CNVblock__"):
                if _is_discrete_cnv_state_feature(s):
                    return False
                return True

            # Default: scale other continuous features (e.g. RNA, continuous pathway scores)
            return True


        # ---- ColumnTransformer (scale only continuous features)
        scale_cols = [c for c in X_all.columns if _should_scale_feature(c, X_all[c])]
        passthrough_cols = [c for c in X_all.columns if c not in scale_cols]

        if len(scale_cols) == 0:
            ct = ColumnTransformer(transformers=[], remainder="passthrough", verbose_feature_names_out=False)
        else:
            ct = ColumnTransformer(
                transformers=[("scale", StandardScaler(), scale_cols)],
                remainder="passthrough",
                verbose_feature_names_out=False
            )

        X_scaled_arr = ct.fit_transform(X_all)
        X_scaled_df = pd.DataFrame(X_scaled_arr, index=X_all.index, columns=ct.get_feature_names_out())
        X_scaled_df = X_scaled_df[X_all.columns]  # restore original order

        # Contracts & saves
        if passthrough_cols:
            mx = (X_scaled_df[passthrough_cols] - X_all[passthrough_cols]).abs().max().max()
            if mx > 1e-9:
                raise AssertionError(f"[TRAIN][{tag}] Passthrough columns changed by {mx:g} during scaling.")

        joblib.dump(ct, os.path.join(prep_dir, f"global_cont_scaler__{tag}.joblib"))
        pd.Series(scale_cols, name="scaled_feature").to_csv(
            os.path.join(prep_dir, f"global_scaled_columns__{tag}.csv"), index=False
        )
        pd.Series(passthrough_cols, name="passthrough_feature").to_csv(
            os.path.join(prep_dir, f"global_passthrough_columns__{tag}.csv"), index=False
        )
        pd.Series(X_all.columns, name="feature").to_csv(
            os.path.join(prep_dir, f"train_feature_order__{tag}.csv"), index=False
        )

        X_all.to_csv(os.path.join(prep_dir, f"{tag}_BLOCK_unscaled.csv"))
        X_scaled_df.to_csv(os.path.join(prep_dir, f"{tag}_BLOCK_scaled.csv"))

        # ---- ColumnTransformer (exclude MUT/CLINICAL from scaling)
        # scale_cols = [c for c in X_all.columns if not (c.startswith("MUTblock__") or c.startswith("CLINICALblock__"))]
        # if len(scale_cols) == 0:
        #     ct = ColumnTransformer(transformers=[], remainder="passthrough", verbose_feature_names_out=False)
        # else:
        #     ct = ColumnTransformer(
        #         transformers=[("scale", StandardScaler(), scale_cols)],
        #         remainder="passthrough",
        #         verbose_feature_names_out=False
        #     )

        # X_scaled_arr = ct.fit_transform(X_all)
        # X_scaled_df = pd.DataFrame(X_scaled_arr, index=X_all.index, columns=ct.get_feature_names_out())
        # X_scaled_df = X_scaled_df[X_all.columns]  # restore original order

        # # Contracts & saves
        # passthrough_cols = [c for c in X_all.columns if c.startswith(("MUTblock__", "CLINICALblock__"))]
        # if passthrough_cols:
        #     mx = (X_scaled_df[passthrough_cols] - X_all[passthrough_cols]).abs().max().max()
        #     if mx > 1e-9:
        #         raise AssertionError(f"[TRAIN][{tag}] Passthrough columns changed by {mx:g} during scaling.")

        # joblib.dump(ct, os.path.join(prep_dir, f"global_cont_scaler__{tag}.joblib"))
        # pd.Series(scale_cols, name="scaled_feature").to_csv(
        #     os.path.join(prep_dir, f"global_scaled_columns__{tag}.csv"), index=False
        # )
        # pd.Series(X_all.columns, name="feature").to_csv(
        #     os.path.join(prep_dir, f"train_feature_order__{tag}.csv"), index=False
        # )

        # X_all.to_csv(os.path.join(prep_dir, f"{tag}_BLOCK_unscaled.csv"))
        # X_scaled_df.to_csv(os.path.join(prep_dir, f"{tag}_BLOCK_scaled.csv"))

        # ---- Optional whitelist
        if getattr(args, "feature_list_csv", None):
            fl = pd.read_csv(args.feature_list_csv)
            col = "feature" if "feature" in fl.columns else fl.columns[0]
            user_feats = [str(x) for x in fl[col].astype(str)]
            keep = [f for f in user_feats if f in X_scaled_df.columns]
            if not keep:
                raise ValueError(f"[TRAIN][{tag}] None of the supplied whitelist features are present.")
            missing = [f for f in user_feats if f not in X_scaled_df.columns]
            for m in missing:
                X_scaled_df[m] = 0.0
            X_scaled_df = X_scaled_df.reindex(columns=user_feats)

        # ---- Tune & fit
        model, model_name, mean_auc, best_params = _grid_select_and_fit(
            X_scaled_df, y_all_enc, grid_pool, args.inner_folds,
            label=f"{tag}_BLOCK",
            calibrate=getattr(args, "calibrate", False),
            calibrate_method=getattr(args, "calibrate_method", "isotonic"),
            calibrate_cv=getattr(args, "calibrate_cv", 3)
        )

        # ---- Save artifacts
        joblib.dump(model, os.path.join(models_dir, f"{tag}_FINAL.pkl"))
        pd.Index(list(X_scaled_df.columns), name="feature").to_series().to_csv(
            os.path.join(prep_dir, f"{tag}_FINAL_features.csv"), index=False
        )

        with open(os.path.join(models_dir, f"{tag}_FINAL_info.txt"), "w") as f:
            f.write(f"Model Type: {model_name}\n")
            f.write(f"GridSearchCV mean ROC AUC (inner CV): {mean_auc:.4f}\n")
            f.write("Best Params:\n")
            for k, v in (best_params or {}).items():
                f.write(f"  {k}: {v}\n")

        _save_coefficients_if_any(
            model, list(X_scaled_df.columns),
            os.path.join(coef_dir, f"coef_{tag}_FINAL.csv")
        )
        _save_shap_if_any(model, X_scaled_df, shap_dir, tag=f"{tag}_FINAL")

        Xw = X_scaled_df.copy()
        Xw.insert(0, "sample_id", Xw.index.astype(str))
        Xw.to_csv(os.path.join(prep_dir, f"{tag}_FINAL_training_features_wide.csv"), index=False)

        Xl = X_scaled_df.stack().rename("value").reset_index()
        Xl.columns = ["sample_id", "feature", "value"]
        Xl.to_csv(os.path.join(prep_dir, f"{tag}_FINAL_training_features_long.csv"), index=False)

        return model_name, float(mean_auc), (best_params or {}), int(X_scaled_df.shape[1]), int(X_scaled_df.shape[0])

    # -------------------- Train all blocks --------------------
    summary_rows = []

    stack_blocks = [
        b.strip().upper()
        for b in getattr(args, "stack_blocks", "RNA,CNV,MUT,CLINICAL").split(",")
        if b.strip()
    ]

    seen = set()
    stack_blocks = [b for b in stack_blocks if not (b in seen or seen.add(b))]

    train_blocks = list(stack_blocks)
    if "ALL" not in train_blocks:
        train_blocks.append("ALL")

    print(f"[TRAIN] Requested stack blocks from --stack_blocks: {stack_blocks}")
    print(f"[TRAIN] Actual training blocks: {train_blocks}")

    for tag in train_blocks:
        if tag not in {"RNA", "CNV", "MUT", "CLINICAL", "ALL"}:
            print(f"[INFO] Skipping unknown block '{tag}'")
            continue

        if tag not in block_mats:
            print(f"[INFO] Skipping {tag}: no features found.")
            continue

        Xmat = block_mats[tag]
        grid_pool = _grid_for_block(tag)
        chosen_families = sorted(list(grid_pool.keys()))
        print(
            f"[TRAIN] === {tag} block: {Xmat.shape[0]}×{Xmat.shape[1]} "
            f"(searching {len(chosen_families)} grid(s): {chosen_families}) ==="
        )
        name, auc, params, nfeat, nsamp = _fit_block(tag, Xmat, grid_pool)
        summary_rows.append({
            "block": tag, "model": name, "mean_cv_auc": auc,
            "n_features": nfeat, "n_samples": nsamp
        })

    # -------------------- Backward-compat copies for ALL --------------------
    trained_blocks = {r["block"] for r in summary_rows}
    if "ALL" in trained_blocks:
        def _copy(src, dst):
            import shutil
            if os.path.exists(src):
                shutil.copy2(src, dst)

        _copy(os.path.join(prep_dir, "global_cont_scaler__ALL.joblib"),
              os.path.join(prep_dir, "global_cont_scaler.joblib"))
        _copy(os.path.join(prep_dir, "train_feature_order__ALL.csv"),
              os.path.join(prep_dir, "train_feature_order.csv"))
        _copy(os.path.join(prep_dir, "global_scaled_columns__ALL.csv"),
              os.path.join(prep_dir, "global_scaled_columns.csv"))
        _copy(os.path.join(prep_dir, "ALL_FINAL_features.csv"),
              os.path.join(prep_dir, "ALL_BLOCKS_FINAL_features.csv"))
        _copy(os.path.join(prep_dir, "ALL_BLOCK_unscaled.csv"),
              os.path.join(prep_dir, "ALL_BLOCKS_CONCAT_unscaled.csv"))
        _copy(os.path.join(prep_dir, "ALL_BLOCK_scaled.csv"),
              os.path.join(prep_dir, "ALL_BLOCKS_CONCAT_scaled.csv"))
        _copy(os.path.join(models_dir, "ALL_FINAL.pkl"),
              os.path.join(models_dir, "ALL_BLOCKS_FINAL.pkl"))
        _copy(os.path.join(prep_dir, "train_feature_stats__ALL.csv"),
              os.path.join(prep_dir, "train_feature_stats.csv"))

    # -------------------- Summary table & versions --------------------
    if summary_rows:
        summ = pd.DataFrame(summary_rows).sort_values(["block"])
        summ.to_csv(os.path.join(models_dir, "best_models_summary.csv"), index=False)
        try:
            import json
            with open(os.path.join(models_dir, "best_models_summary.json"), "w") as f:
                json.dump(summary_rows, f, indent=2)
        except Exception:
            pass

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

    print("[TRAIN] ✅ Ran FS (if needed) and trained RNA/CNV/MUT/CLINICAL/ALL models.")
    print(f"[TRAIN] 📁 Outputs → {out_dir}")
    return out_dir



# =========================
#   Predict on full matrix
# =========================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _to_binary_event_for_survival(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1, "0": 0,
        "true": 1, "false": 0,
        "yes": 1, "no": 0,
        "event": 1, "censored": 0,
        "death": 1, "alive": 0,
        "recurrent": 1, "non-recurrent": 0, "nonrecurrent": 0,
        "case": 1, "control": 0,
    }
    return s.map(mapping).astype(float)


def _plot_simple_cox_forest(summary_df: pd.DataFrame, title: str, out_png: str):
    if summary_df is None or summary_df.empty:
        return

    plot_df = summary_df.copy().replace([np.inf, -np.inf], np.nan)

    if "exp(coef)" in plot_df.columns:
        plot_df["HR"] = pd.to_numeric(plot_df["exp(coef)"], errors="coerce")
    else:
        plot_df["HR"] = np.exp(pd.to_numeric(plot_df["coef"], errors="coerce"))

    if "exp(coef) lower 95%" in plot_df.columns:
        plot_df["HR_lo"] = pd.to_numeric(plot_df["exp(coef) lower 95%"], errors="coerce")
        plot_df["HR_hi"] = pd.to_numeric(plot_df["exp(coef) upper 95%"], errors="coerce")
    else:
        plot_df["HR_lo"] = np.exp(pd.to_numeric(plot_df["coef lower 95%"], errors="coerce"))
        plot_df["HR_hi"] = np.exp(pd.to_numeric(plot_df["coef upper 95%"], errors="coerce"))

    plot_df["p"] = pd.to_numeric(plot_df["p"], errors="coerce")
    plot_df = plot_df.dropna(subset=["HR", "HR_lo", "HR_hi", "p"])

    if plot_df.empty:
        return

    plot_df = plot_df.sort_values("HR", ascending=True)

    plt.figure(figsize=(8.0, max(3.0, 0.55 * len(plot_df) + 1.0)))
    ax = plt.gca()

    y = np.arange(len(plot_df))
    hr = plot_df["HR"].to_numpy()
    lo = plot_df["HR_lo"].to_numpy()
    hi = plot_df["HR_hi"].to_numpy()
    xerr = np.vstack([hr - lo, hi - hr])

    ax.errorbar(hr, y, xerr=xerr, fmt="o", capsize=3)
    ax.axvline(1.0, linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Hazard ratio (log scale)")
    ax.set_yticks(y)
    ax.set_yticklabels([
        f"{idx} (p={p:.2e})" if p >= 1e-16 else f"{idx} (p<1e-16)"
        for idx, p in zip(plot_df.index.astype(str), plot_df["p"].astype(float))
    ])
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _run_prediction_cox_simple(
    out_df: pd.DataFrame,
    labels_map: dict,
    plots_dir: str,
    preds_dir: str,
    prefix: str,
    token: str,
    labels_order=None,
    penalizer: float = 0.01,
):
    try:
        from lifelines import CoxPHFitter
    except Exception:
        print(f"[{prefix}] lifelines not available; skipping Cox.")
        return

    if ("time_to_event" not in labels_map) or (
        ("event_observed" not in labels_map) and ("disease_status" not in labels_map)
    ):
        print(f"[{prefix}] Cox skipped (time or event not available).")
        return

    if "event_observed" in labels_map and labels_map["event_observed"] is not None:
        ev = _to_binary_event_for_survival(labels_map["event_observed"])
    else:
        ev = labels_map["disease_status"].map({
            "recurrent": 1,
            "non-recurrent": 0,
            "nonrecurrent": 0
        })

    tm = pd.to_numeric(labels_map["time_to_event"], errors="coerce")

    base_df = out_df.set_index("sample_id")[["predicted_proba", "pct_group"]].join(
        pd.DataFrame({"event": ev, "time": tm})
    )
    base_df = base_df.dropna(subset=["predicted_proba", "event", "time"]).copy()

    if base_df.empty or base_df["event"].sum() < 3:
        print(f"[{prefix}] Cox skipped: insufficient usable rows/events.")
        return

    # 1) Continuous Cox on predicted probability
    try:
        cont_df = base_df[["time", "event", "predicted_proba"]].copy()
        cont_df["predicted_proba_per_0p1"] = cont_df["predicted_proba"] / 0.1
        cont_df = cont_df.drop(columns=["predicted_proba"])

        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(cont_df, duration_col="time", event_col="event")

        cont_csv = os.path.join(preds_dir, f"{prefix}__cox_continuous__{token}.csv")
        cont_png = os.path.join(plots_dir, f"{prefix}__cox_forest_continuous__{token}.png")
        cph.summary.to_csv(cont_csv)
        _plot_simple_cox_forest(
            cph.summary,
            f"[{prefix}] Cox: predicted score (per 0.1)",
            cont_png
        )
        print(f"[PLOT] {cont_png}")
    except Exception as e:
        print(f"[WARN][{prefix}] Continuous Cox failed: {e}")

    # 2) Group Cox matching KM percentile groups
    try:
        order = labels_order or list(pd.Series(base_df["pct_group"]).dropna().astype(str).unique())
        order = [g for g in order if (base_df["pct_group"].astype(str) == g).any()]

        grp = pd.Categorical(base_df["pct_group"].astype(str), categories=order, ordered=True)
        grp_dummies = pd.get_dummies(grp, prefix="pct_group", drop_first=True)

        grp_df = pd.concat([base_df[["time", "event"]], grp_dummies], axis=1).dropna()
        grp_terms = [c for c in grp_df.columns if c.startswith("pct_group_")]

        if grp_terms and grp_df["event"].sum() >= 3:
            cph_grp = CoxPHFitter(penalizer=penalizer)
            cph_grp.fit(grp_df, duration_col="time", event_col="event")

            grp_csv = os.path.join(preds_dir, f"{prefix}__cox_groups__{token}.csv")
            grp_png = os.path.join(plots_dir, f"{prefix}__cox_forest_groups__{token}.png")
            cph_grp.summary.to_csv(grp_csv)
            _plot_simple_cox_forest(
                cph_grp.summary,
                f"[{prefix}] Cox: percentile groups",
                grp_png
            )
            print(f"[PLOT] {grp_png}")
    except Exception as e:
        print(f"[WARN][{prefix}] Group Cox failed: {e}")


def predict_concat_only(
    output_dir,
    subtype,
    pred_matrix_csv,          # path to full prediction matrix CSV (all features; we’ll subset per block)
    sample_ids=None,          # optional list of sample IDs to restrict to (intersected with labels if provided)
    pred_labels_csv=None,     # optional CSV with true labels to enable ROC, bars, KM
    percentiles="50",         # CSV (e.g., "20,80") OR keywords: "median","quartiles","tertiles","deciles","none"
    scale_mode="training",    # "training" (apply saved scaler) or "prediction_z" (z-score within cohort)
    blocks="RNA,CNV,MUT,CLINICAL,ALL",             # NEW: comma-list from {RNA,CNV,MUT,ALL}; e.g. "RNA,CNV" or "ALL"
    #blocks="RNA,CNV,CLINICAL,ALL",             # NEW: comma-list from {RNA,CNV,MUT,ALL}; e.g. "RNA,CNV" or "ALL"

):
    """
    Predict per requested block(s) using that block's trained artifacts.
    Subsets features to the block, applies the block's scaler, and writes
    block-tagged outputs (CSVs + plots with block in filename).

    `blocks` examples:
      - "ALL"           → uses generic ALL artifacts (back-compat)
      - "RNA"           → RNA-only model and features
      - "RNA,CNV,MUT"   → run 3 models and return dict {"RNA": df, "CNV": df, "MUT": df}
    """
    import os, numpy as np, pandas as pd, joblib
    import matplotlib.pyplot as plt

    # ---------- Small helpers ----------
    def _parse_percentiles(s: str):
        if s is None:
            return [], "none"
        raw = str(s).strip()
        key = raw.lower().replace(" ", "")
        if key in ("none", ""):
            return [], "none"
        if key in ("median", "med", "p50", "q2"):
            return [50.0], "median"
        if key in ("quartiles", "quartile", "iqr"):
            return [25.0, 50.0, 75.0], "quartiles"
        if key in ("tertiles", "tertile"):
            return [100.0/3.0, 200.0/3.0], "tertiles"
        if key in ("deciles", "decile"):
            return [float(x) for x in range(10, 100, 10)], "deciles"
        vals = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            v = float(part)
            if not (0.0 < v < 100.0):
                raise ValueError("--percentiles must be in (0,100)")
            vals.append(v)
        vals = sorted(set(vals))
        return vals, "pct" + raw.replace(",", "-")

    def _labels_from_pcts(pcts):
        if not pcts:
            return ["All"]
        labs = [f"P≤{pcts[0]:g}"]
        for i in range(len(pcts) - 1):
            labs.append(f"P({pcts[i]:g},{pcts[i+1]:g}]")
        labs.append(f"P>{pcts[-1]:g}")
        return labs

    def _to_binary_event(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return series.astype(float).fillna(0).astype(int)
        s = series.astype(str).str.strip().str.lower()
        mapping = {
            "1":1, "0":0, "true":1, "false":0, "yes":1, "no":0,
            "event":1, "none":0, "censored":0,
            "recurrent":1, "non-recurrent":0, "nonrecurrent":0,
            "case":1, "control":0, "death":1, "alive":0
        }
        return s.map(mapping).fillna(0).astype(int)

    def _rescale_unit(x: pd.Series) -> pd.Series:
        xmin, xmax = float(x.min()), float(x.max())
        if xmax <= xmin:
            return pd.Series(np.zeros_like(x, dtype=float), index=x.index)
        return ((x - xmin) / (xmax - xmin)).astype(float)

    def _add_corner_text(ax, text: str):
        if not text:
            return
        ax.text(0.98, 0.98, text,
                transform=ax.transAxes, ha="right", va="top",
                bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.3"),
                fontsize=9)

    def _plot_hist(df, score_col_display, group_col, labels_order, title, out_png, bins, vlines=None, info=""):
        plt.figure(figsize=(7.2, 4.6)); ax = plt.gca()
        col = df[group_col].astype(str)
        present_labels = [lab for lab in labels_order if (col == lab).any()]
        if not present_labels:
            present_labels = [str(x) for x in pd.Series(col).dropna().unique().tolist()]
        drew_any = False
        for lab in present_labels:
            vals = df.loc[col == lab, score_col_display].astype(float).values
            if len(vals):
                ax.hist(vals, bins=bins, alpha=0.6, label=lab); drew_any = True
        if not drew_any:
            ax.hist(df[score_col_display].astype(float).values, bins=bins, alpha=0.8)
        else:
            ax.legend()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Score (0–1, display scaling)"); ax.set_ylabel("Count"); ax.set_title(title)
        if vlines:
            for xv in vlines:
                ax.axvline(np.clip(float(xv), 0, 1), linestyle="--", linewidth=1.3)
        _add_corner_text(ax, info)
        plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

    def _plot_event_bars(n, e, rate, labels, title, out_png, info=""):
        plt.figure(figsize=(6.4, 4.2)); ax = plt.gca()
        vals = [rate.get(l,0) for l in labels]
        ax.bar(labels, vals); ax.set_ylim(0,1)
        ax.set_ylabel("Event rate"); ax.set_title(title)
        for i, lab in enumerate(labels):
            ax.text(i, min(vals[i]+0.02,0.98), f"{int(e.get(lab,0))}/{int(n.get(lab,0))}",
                    ha="center", va="bottom", fontsize=9)
        _add_corner_text(ax, info)
        plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

    def _plot_km(df, time_col, event_col, group_col, labels, title, out_png, extra_text=""):
        try:
            from lifelines import KaplanMeierFitter
            from lifelines.statistics import logrank_test, multivariate_logrank_test
        except Exception:
            print("[KM] lifelines not available; skipping KM."); return
        plt.figure(figsize=(6.8,4.6)); ax = plt.gca()
        present = []
        km = KaplanMeierFitter()
        for lab in labels:
            dd = df[df[group_col]==lab]
            if dd.empty: continue
            present.append(lab)
            km.fit(dd[time_col].astype(float).values, dd[event_col].astype(int).values, label=lab)
            km.plot(ax=ax, ci_show=False)
        ptxt = ""
        try:
            if logrank_test is not None and len(present) >= 2:
                if len(present)==2:
                    A,B = present[:2]
                    A_df = df[df[group_col]==A]; B_df = df[df[group_col]==B]
                    res = logrank_test(A_df[time_col].astype(float), B_df[time_col].astype(float),
                                       event_observed_A=A_df[event_col].astype(int),
                                       event_observed_B=B_df[event_col].astype(int))
                    ptxt = f"Log-rank p ({A} vs {B}): {('<1e-16' if res.p_value<1e-16 else f'{res.p_value:.2e}')}"
                else:
                    if multivariate_logrank_test is not None:
                        glob = multivariate_logrank_test(df[time_col].astype(float), df[group_col], df[event_col].astype(int))
                        ptxt = f"Global p: {('<1e-16' if glob.p_value<1e-16 else f'{glob.p_value:.2e}')}"
        except Exception as e:
            print(f"[KM] p-value calc failed: {e}")
        ax.set_title(title); ax.set_xlabel("Time"); ax.set_ylabel("Survival probability"); ax.set_ylim(0,1)
        _add_corner_text(ax, (extra_text + ("\n" if extra_text and ptxt else "") + ptxt).strip())
        plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

    def _pick_col(cols_lower, *cands):
        for c in cands:
            if c in cols_lower:
                return cols_lower[c]
        return None

    # ---- Static paths ----
    fold_dir   = os.path.join(output_dir, subtype)
    models_dir = os.path.join(fold_dir, "cv_models")
    prep_dir   = os.path.join(fold_dir, "preprocessing")
    preds_dir  = os.path.join(fold_dir, "predictions")
    plots_dir  = os.path.join(fold_dir, "plots")
    os.makedirs(preds_dir, exist_ok=True); os.makedirs(plots_dir, exist_ok=True)

    # ---- Load full prediction matrix once ----
    X_pred_full = pd.read_csv(pred_matrix_csv, index_col=0)
    X_pred_full.index = X_pred_full.index.astype(str).str.strip()
    X_pred_full.columns = X_pred_full.columns.astype(str).str.strip()

    def _norm_ids(idx_like):
        s = pd.Index(idx_like).astype(str).str.strip()
        return s.str.replace(r"\.0$", "", regex=True)
    X_pred_full.index = _norm_ids(X_pred_full.index)

    # ---- Sample selection (shared across blocks) ----
    def _filter_labels_to_subtype(df, subtype_str):
        cols_lower = {c.lower(): c for c in df.columns}
        # Try several common names for subtype in the clinical sheet
        for cand in ["pam50_subtype", "subtype", "pam50", "molecular_subtype", "pam50 subtype"]:
            if cand in cols_lower:
                col = cols_lower[cand]
                mask = df[col].astype(str).str.strip().str.lower() == str(subtype_str).strip().lower()
                return df[mask]
        return df  # if no subtype column present, leave as-is

    labs_df_eager, sid_col_eager, label_ids = None, None, None
    if pred_labels_csv is not None and os.path.exists(pred_labels_csv):
        labs_df_eager = pd.read_csv(pred_labels_csv)
        # filter to the requested subtype HERE (critical: before we pick IDs)
        labs_df_eager = _filter_labels_to_subtype(labs_df_eager, subtype)

        cols_lower = {c.lower(): c for c in labs_df_eager.columns}
        sid_col_eager = _pick_col(cols_lower, "sample_id") or labs_df_eager.columns[0]
        label_ids = _norm_ids(labs_df_eager[sid_col_eager])



    sample_ids_norm = None
    if sample_ids is not None:
        sample_ids_norm = _norm_ids(sample_ids)

    if label_ids is not None:
        base = [s for s in label_ids if s in X_pred_full.index]
        if sample_ids_norm is not None:
            keep_set = set(sample_ids_norm)
            base = [s for s in base if s in keep_set]
    elif sample_ids_norm is not None:
        base = [s for s in sample_ids_norm if s in X_pred_full.index]
    else:
        base = list(X_pred_full.index)

    X_pred_full = X_pred_full.reindex(base)
    if len(X_pred_full) == 0:
        raise ValueError("After subsetting to label/sample IDs, no samples remain.")

    X_pred_full = X_pred_full.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # ---- Percentiles setup (shared) ----
    pcts, pct_key = _parse_percentiles(percentiles)

    # ---- Which blocks to run ----
    req_blocks = [b.strip().upper() for b in str(blocks).split(",") if b.strip()]
    allowed = {"RNA","CNV","MUT","CLINICAL","ALL"}
    bad = [b for b in req_blocks if b not in allowed]
    if bad:
        raise ValueError(f"Unknown block(s): {bad}. Allowed: {sorted(allowed)}")

    # ---- Small alias to load block-specific artifacts (with ALL back-compat) ----
    def _paths_for_block(block):
        if block == "ALL":
            # Back-compat generic names
            model_path = os.path.join(models_dir, "ALL_BLOCKS_FINAL.pkl")
            ct_path    = os.path.join(prep_dir,   "global_cont_scaler.joblib")
            train_order_csv = os.path.join(prep_dir, "train_feature_order.csv")
            scaled_cols_csv = os.path.join(prep_dir, "global_scaled_columns.csv")
            feat_list_csv   = os.path.join(prep_dir, "ALL_BLOCKS_FINAL_features.csv")
            stats_csv       = os.path.join(prep_dir, "train_feature_stats.csv")
        else:
            model_path = os.path.join(models_dir, f"{block}_FINAL.pkl")
            ct_path    = os.path.join(prep_dir,   f"global_cont_scaler__{block}.joblib")
            train_order_csv = os.path.join(prep_dir, f"train_feature_order__{block}.csv")
            scaled_cols_csv = os.path.join(prep_dir, f"global_scaled_columns__{block}.csv")
            feat_list_csv   = os.path.join(prep_dir, f"{block}_FINAL_features.csv")
            stats_csv       = os.path.join(prep_dir, f"train_feature_stats__{block}.csv")
        return model_path, ct_path, train_order_csv, scaled_cols_csv, feat_list_csv, stats_csv

    # ---- Shared labels map builder (used later in plots) ----
    def _build_labels_map(sample_ids_series):
        labels_map = {}
        try:
            if labs_df_eager is None:
                if pred_labels_csv is not None and os.path.exists(pred_labels_csv):
                    df = pd.read_csv(pred_labels_csv)
                    cols_lower = {c.lower(): c for c in df.columns}
                    sid = _pick_col(cols_lower, "sample_id") or df.columns[0]
                    idxed = df.set_index(sid)
                else:
                    return {}
            else:
                df = labs_df_eager
                cols_lower = {c.lower(): c for c in df.columns}
                sid = sid_col_eager
                idxed = df.set_index(sid)

            cols_lower = {c.lower(): c for c in idxed.columns}
            ds_col = _pick_col(cols_lower, "disease_status", "label", "y_label")
            if ds_col:
                ds = idxed[ds_col].astype(str).str.strip().str.lower()
                labels_map["disease_status"] = ds.reindex(sample_ids_series.values)

            ev_col = _pick_col(cols_lower, "event_observed", "dfs_event", "os_event", "event")
            if ev_col:
                ev = pd.to_numeric(idxed[ev_col], errors="coerce")
                labels_map["event_observed"] = ev.reindex(sample_ids_series.values)

            te_col = _pick_col(cols_lower, "time_to_event", "dfs_months", "os_months", "time")
            if te_col:
                te = pd.to_numeric(idxed[te_col], errors="coerce")
                labels_map["time_to_event"] = te.reindex(sample_ids_series.values)
        except Exception as e:
            print(f"[WARN] Could not load/align labels meta: {e}")
            labels_map = {}
        return labels_map

    # ---- Core per-block runner ----
    def _run_block(block):
        model_path, ct_path, train_order_csv, scaled_cols_csv, feat_list_csv, stats_path = _paths_for_block(block)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[{block}] Model not found: {model_path}")
        if not os.path.exists(ct_path):
            raise FileNotFoundError(f"[{block}] ColumnTransformer not found: {ct_path}")
        for pth in [train_order_csv, scaled_cols_csv, feat_list_csv]:
            if not os.path.exists(pth):
                raise FileNotFoundError(f"[{block}] Missing artifact: {pth}")

        model = joblib.load(model_path)
        ct    = joblib.load(ct_path)

        train_input_cols = pd.read_csv(train_order_csv)["feature"].astype(str).tolist()
        scale_cols       = pd.read_csv(scaled_cols_csv)["scaled_feature"].astype(str).tolist()
        feat_list        = pd.read_csv(feat_list_csv)["feature"].astype(str).tolist()

        # Subset feature columns to this block's training schema
        X_pred = X_pred_full.copy()
        pred_cols  = set(X_pred.columns)
        train_cols = set(train_input_cols)
        missing = [c for c in train_input_cols if c not in pred_cols]
        extra   = [c for c in X_pred.columns if c not in train_cols]
        if extra:
            X_pred = X_pred.drop(columns=extra)
        if missing:
            # For missing scaled cols, use training mean from the scaler.
            # For passthrough cols (including MUT/CLINICAL), fill with 0.0.
            scale_name = next((n for n in ct.named_transformers_.keys() if str(n).startswith("scale")), None)

            if scale_name is not None:
                scaler = ct.named_transformers_[scale_name]
                scale_means = dict(zip(scale_cols, getattr(scaler, "mean_", np.zeros(len(scale_cols)))))
            else:
                scale_means = {}

            for c in missing:
                X_pred[c] = float(scale_means.get(c, 0.0)) if c in scale_cols else 0.0

        X_pred = X_pred.reindex(columns=train_input_cols)

        # Optional: training-range winsorization (if present)
        if str(scale_mode).lower() == "training" and os.path.exists(stats_path):
            try:
                stats = pd.read_csv(stats_path)
                stats.columns = [str(c).strip().lower() for c in stats.columns]
                idx_col = "feature" if "feature" in stats.columns else stats.columns[0]
                stats = stats.set_index(idx_col)
                have_p01 = "p01" in stats.columns; have_p99 = "p99" in stats.columns
                have_min = "min" in stats.columns;  have_max = "max" in stats.columns
                if (have_p01 and have_p99) or (have_min and have_max):
                    scale_set = set(scale_cols)
                    common = [c for c in X_pred.columns if c in stats.index]
                    
                    # for c in common:
                    #     s = stats.loc[c]
                    #     if have_p01 and have_p99 and c in scale_set:
                    #         lo, hi = float(s.get("p01")), float(s.get("p99"))
                    #     else:
                    #         lo = float(s.get("min", 0.0)) if have_min else None
                    #         hi = float(s.get("max", 0.0)) if have_max else None
                    #     if (lo is not None) and (hi is not None) and lo != hi:
                    #         X_pred[c] = X_pred[c].clip(lower=lo, upper=hi)


                    passthrough_prefixes = ("MUTblock__", "CLINICALblock__")

                    for c in common:
                        if c.startswith(passthrough_prefixes):
                            continue

                        s = stats.loc[c]
                        if have_p01 and have_p99 and c in scale_set:
                            lo, hi = float(s.get("p01")), float(s.get("p99"))
                        else:
                            lo = float(s.get("min", 0.0)) if have_min else None
                            hi = float(s.get("max", 0.0)) if have_max else None

                        if (lo is not None) and (hi is not None) and lo != hi:
                            X_pred[c] = X_pred[c].clip(lower=lo, upper=hi)
                    
            except Exception as e:
                print(f"[{block}] Winsorization skipped: {e}")

        # Transform according to requested scale_mode
        scm = str(scale_mode).lower()
        if scm == "training":
            X_scaled_arr = ct.transform(X_pred)
            X_scaled_df  = pd.DataFrame(X_scaled_arr, index=X_pred.index,
                                        columns=ct.get_feature_names_out())
            X_scaled_df  = X_scaled_df[train_input_cols]
        elif scm == "prediction_z":
            # Manually z-score ONLY columns that were scaled in training.
            X_scaled_df = X_pred.copy()
            pred_means = X_scaled_df[scale_cols].mean(axis=0)
            pred_stds  = X_scaled_df[scale_cols].std(axis=0, ddof=0).replace(0.0, np.nan)
            for c in scale_cols:
                if c in X_scaled_df.columns:
                    mu = float(pred_means.get(c, 0.0))
                    sd = float(pred_stds.get(c, np.nan))
                    X_scaled_df[c] = 0.0 if (np.isnan(sd) or sd == 0.0) else (X_scaled_df[c] - mu) / sd
        else:
            raise ValueError("scale_mode must be 'training' or 'prediction_z'.")

        # Align to model's FINAL feature set
        for m in [c for c in feat_list if c not in X_scaled_df.columns]:
            X_scaled_df[m] = 0.0
        X_scaled_df = X_scaled_df.reindex(columns=feat_list, fill_value=0.0)

        # Predict probabilities
        def _proba_pos(m, Xdf):
            if hasattr(m, "predict_proba"):
                proba = m.predict_proba(Xdf)
                if hasattr(m, "classes_"):
                    classes = list(m.classes_)
                    pos_idx = classes.index(1) if 1 in classes else -1
                else:
                    pos_idx = -1
                return proba[:, pos_idx]
            elif hasattr(m, "decision_function"):
                from scipy.special import expit
                return expit(m.decision_function(Xdf))
            else:
                return m.predict(Xdf).astype(float)

        probs = _proba_pos(model, X_scaled_df)

        # Percentile/median split
        cut_raw = [float(np.nanquantile(probs, p/100.0)) for p in pcts] if pcts else []
        uniq_cuts = []
        for v in cut_raw:
            if not uniq_cuts or abs(v - uniq_cuts[-1]) > 1e-12:
                uniq_cuts.append(v)
        edges  = [-np.inf] + uniq_cuts + [np.inf]
        labels = _labels_from_pcts(pcts if len(uniq_cuts)==len(pcts) else
                                   [round(100*(i+1)/(len(uniq_cuts)+1),2) for i in range(len(uniq_cuts))])

        # out = pd.DataFrame({"sample_id": X_scaled_df.index.astype(str),
        #                     "predicted_proba": probs})
        # out["pct_group"] = pd.cut(out["predicted_proba"], bins=edges, labels=labels,
        #                           include_lowest=True, duplicates="drop").astype(str)
        # out["percentiles"]   = str(percentiles)
        # out["cutpoints_raw"] = ";".join(f"{c:.10g}" for c in uniq_cuts)
        # out["block"] = block

        out = pd.DataFrame({"sample_id": X_scaled_df.index.astype(str),
                    "predicted_proba": probs})
        out["pct_group"] = pd.cut(out["predicted_proba"], bins=edges, labels=labels,
                                include_lowest=True, duplicates="drop").astype(str)
        out["percentiles"]   = str(percentiles)
        out["cutpoints_raw"] = ";".join(f"{c:.10g}" for c in uniq_cuts)
        out["block"] = block

        token = (pct_key if pct_key else str(percentiles)).replace(",", "-")

        # Single-cut label
        if len(uniq_cuts) == 1:
            ptxt = (pct_key if pct_key in ("median",) else str(int(round(pcts[0]))))
            out[f"pred_label_at_P{ptxt}"] = (out["predicted_proba"] >= uniq_cuts[0]).astype(int)
            pred_fname = f"predictions__{block}__pct{pct_key if pct_key!='none' else 'none'}.csv"
        else:
            for p, c in zip(pcts, cut_raw):
                out[f"pred_label_at_P{int(round(p))}"] = (out["predicted_proba"] >= c).astype(int)
            pred_fname = f"predictions__{block}__{pct_key if pct_key!='none' else 'pctnone'}.csv"

        out_path = os.path.join(preds_dir, pred_fname)
        out.to_csv(out_path, index=False)
        print(f"💾 [{block}] Saved predictions → {out_path}  ({len(out)} samples)")

        # Probability summary
        print(f"[{block}] prob summary:",
              {"min": float(out.predicted_proba.min()),
               "p25": float(out.predicted_proba.quantile(0.25)),
               "median": float(out.predicted_proba.median()),
               "p75": float(out.predicted_proba.quantile(0.75)),
               "max": float(out.predicted_proba.max())})

        # Labels map for plots
        labels_map = _build_labels_map(out["sample_id"])

        # Plots with BLOCK in filenames
        try:
            disp = _rescale_unit(out["predicted_proba"])
            smin, smax = float(out["predicted_proba"].min()), float(out["predicted_proba"].max())
            vlines = [0.0 if smax<=smin else (c - smin)/(smax - smin) for c in uniq_cuts]
            out["_display"] = disp.values
            plt_title = f"[{block}] Predicted score distribution (percentile/median groups)"
            token = (pct_key if pct_key else str(percentiles)).replace(",", "-")
            plt_path  = os.path.join(plots_dir, f"{block}__hist__{token}.png")
            _plot_hist(out, "_display", "pct_group", _labels_from_pcts(pcts), plt_title, plt_path, bins=30,
                       vlines=vlines, info=f"Cutpoints(raw): {', '.join(f'{c:.4g}' for c in uniq_cuts)}\nTotal n={len(out)}")
            print(f"[PLOT] {plt_path}")
        except Exception as e:
            print(f"[WARN][{block}] Failed histogram: {e}")

        try:
            s_vec = out["predicted_proba"].astype(float).to_numpy()
            plt.figure(); painted = False
            if ("event_observed" in labels_map) and (labels_map["event_observed"] is not None):
                e_series = pd.to_numeric(labels_map["event_observed"], errors="coerce").reindex(out["sample_id"])
                mask_event = (e_series == 1).to_numpy(); mask_cens = (e_series == 0).to_numpy()
                if np.nan_to_num(mask_cens).sum() > 0:
                    plt.hist(s_vec[mask_cens], bins=30, alpha=0.6, label=f"Censored (n={int(mask_cens.sum())})"); painted = True
                if np.nan_to_num(mask_event).sum() > 0:
                    plt.hist(s_vec[mask_event], bins=30, alpha=0.6, label=f"Event (n={int(mask_event.sum())})"); painted = True
            elif ("disease_status" in labels_map) and (labels_map["disease_status"] is not None):
                ds = labels_map["disease_status"].reindex(out["sample_id"]).astype(str).str.strip().str.lower()
                for lab, nice in [("non-recurrent","Non-recurrent"),("nonrecurrent","Non-recurrent"),("recurrent","Recurrent")]:
                    m = (ds == lab).to_numpy()
                    if m.sum() > 0:
                        plt.hist(s_vec[m], bins=30, alpha=0.6, label=f"{nice} (n={int(m.sum())})"); painted = True
            if not painted:
                plt.hist(s_vec, bins=30)
            else:
                plt.legend()
            for c in uniq_cuts:
                plt.axvline(float(c), linestyle="--", linewidth=1.3)
            plt.xlabel("Predicted probability (used for split)"); plt.ylabel("Count")
            plt.title(f"[{block}] Distribution by label (on split score)")
            token = (pct_key if pct_key else str(percentiles)).replace(",", "-")
            plt.tight_layout()
            path = os.path.join(plots_dir, f"{block}__hist_labels__{token}.png")
            plt.savefig(path, dpi=200); plt.close(); print(f"[PLOT] {path}")
        except Exception as e:
            print(f"[WARN][{block}] Label-overlay histogram failed: {e}")

        # Event-rate bars
        try:
            if ("event_observed" in labels_map) or ("disease_status" in labels_map):
                if "event_observed" in labels_map:
                    ev_series = _to_binary_event(labels_map["event_observed"])
                else:
                    ev_series = labels_map["disease_status"].map({"recurrent":1, "non-recurrent":0, "nonrecurrent":0})
                ev_series = ev_series.reindex(out["sample_id"]).astype(float)

                tmp = out[["sample_id", "pct_group"]].set_index("sample_id").join(ev_series.rename("event"))
                ok = ~tmp["event"].isna()
                tmp = tmp[ok]
                n = tmp.groupby("pct_group").size().to_dict()
                e = tmp.groupby("pct_group")["event"].sum(min_count=1).fillna(0).to_dict()
                rate = {k: (e.get(k,0) / max(n.get(k,1), 1)) for k in set(n)|set(e)}

                token = (pct_key if pct_key else str(percentiles)).replace(",", "-")
                bars_path = os.path.join(plots_dir, f"{block}__eventrate__{token}.png")
                _plot_event_bars(n, e, rate, _labels_from_pcts(pcts), f"[{block}] Event rate by group", bars_path,
                                 info=f"n={int(ok.sum())}")
                print(f"[PLOT] {bars_path}")
            else:
                print(f"[{block}] Event column not available; skipping event-rate bars.")
        except Exception as e:
            print(f"[WARN][{block}] Failed event-rate bars: {e}")

        # KM
        try:
            if ("time_to_event" in labels_map) and (("event_observed" in labels_map) or ("disease_status" in labels_map)):
                if "event_observed" in labels_map:
                    ev = _to_binary_event(labels_map["event_observed"])
                else:
                    ev = labels_map["disease_status"].map({"recurrent":1, "non-recurrent":0, "nonrecurrent":0})
                tm = labels_map["time_to_event"]
                km_df = out.set_index("sample_id")[["pct_group"]].join(pd.DataFrame({"event": ev, "time": tm})).dropna(subset=["event","time"])
                if km_df.empty:
                    print(f"[{block}] KM: no usable rows with time+event.")
                else:
                    km_path = os.path.join(plots_dir, f"{block}__KM__{token}.png")
                    _plot_km(km_df.reset_index(), "time", "event", "pct_group",
                             _labels_from_pcts(pcts), f"[{block}] KM by groups", km_path,
                             extra_text=f"Cutpoints(raw): {', '.join(f'{c:.4g}' for c in uniq_cuts)}")
                    print(f"[PLOT] {km_path}")
            else:
                print(f"[{block}] KM skipped (time or event not available).")
        except Exception as e:
            print(f"[WARN][{block}] Failed KM plotting: {e}")

        # Cox
        try:
            _run_prediction_cox_simple(
                out_df=out,
                labels_map=labels_map,
                plots_dir=plots_dir,
                preds_dir=preds_dir,
                prefix=block,
                token=token,
                labels_order=_labels_from_pcts(pcts),
            )
        except Exception as e:
            print(f"[WARN][{block}] Failed Cox plotting: {e}")


        # ROC
        try:
            from sklearn.metrics import roc_auc_score, roc_curve
            y_vec = None
            if "disease_status" in labels_map:
                mapping = {"non-recurrent": 0, "nonrecurrent": 0, "0": 0, "0.0": 0, "false": 0, "no": 0,
                           "recurrent": 1, "1": 1, "1.0": 1, "true": 1, "yes": 1}
                y_vec = labels_map["disease_status"].map(mapping)
                if y_vec is not None and y_vec.isna().any():
                    y_num = pd.to_numeric(labels_map["disease_status"], errors="coerce")
                    y_vec = y_vec.where(~y_vec.isna(), y_num)
            if y_vec is None and "event_observed" in labels_map:
                y_vec = labels_map["event_observed"]

            if y_vec is not None:
                mask = ~y_vec.isna(); n_use = int(mask.sum())
                if n_use == 0:
                    print(f"[{block}] ROC skipped: no usable labels.")
                else:
                    n0 = int((y_vec[mask] == 0).sum()); n1 = int((y_vec[mask] == 1).sum())
                    if n0 == 0 or n1 == 0:
                        print(f"[{block}] ROC skipped: need both classes (neg={n0}, pos={n1}).")
                    else:
                        fpr, tpr, _ = roc_curve(y_vec[mask].astype(int).values, out["predicted_proba"].values[mask], pos_label=1)
                        auc_val = roc_auc_score(y_vec[mask].astype(int).values, out["predicted_proba"].values[mask])
                        plt.figure()
                        plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
                        plt.plot([0,1],[0,1], linestyle="--")
                        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
                        plt.title(f"[{block}] Prediction ROC curve"); plt.legend(loc="lower right")
                        plt.tight_layout()
                        roc_path = os.path.join(plots_dir, f"{block}__predict_ROC__{token}.png")
                        plt.savefig(roc_path, dpi=200); plt.close()
                        print(f"[PLOT] {roc_path}")
            else:
                print(f"[{block}] ROC skipped: no disease_status or event_observed available.")
        except Exception as e:
            print(f"[WARN][{block}] Failed to generate prediction ROC: {e}")

        # Audit file
        info_path = os.path.join(preds_dir, f"predict_info__{block}__{token}.txt")
        with open(info_path, "w") as f:
            f.write(f"block: {block}\n")
            f.write(f"pred_matrix_csv: {pred_matrix_csv}\n")
            f.write(f"samples: {X_scaled_df.shape[0]}\nfeatures: {X_scaled_df.shape[1]}\n")
            f.write(f"scale_mode: {scale_mode}\npercentiles: {percentiles}\n")
            if pred_labels_csv: f.write(f"labels_csv: {pred_labels_csv}\n")
            if 'missing' in locals() and missing: f.write(f"filled_missing_features: {len(missing)}\n")
            if 'extra' in locals() and extra:     f.write(f"dropped_extra_features: {len(extra)}\n")

        # Debug coverage
        coverage = len(set(train_input_cols) & set(X_pred.columns)) / max(1, len(train_input_cols))
        print(f"[{block}] Column coverage vs training: {coverage:.2%}")
        return out

    # ---- Run all requested blocks ----
    results = {blk: _run_block(blk) for blk in req_blocks}
    return results[req_blocks[0]] if len(results) == 1 else results

def predict_stacked(
    output_dir,
    subtype,
    pred_matrix_csv,
    sample_ids=None,          # optional list of sample IDs to restrict to
    pred_labels_csv=None,     # optional CSV with labels/time to enable ROC/KM
    percentiles="50",         # e.g., "50" (median), "25,50,75", "quartiles", etc.
    scale_mode="training",    # kept for API symmetry; stacked uses training stats
):
    """
    Predict with STACKED_FINAL.pkl and generate:
      - histogram(s) (overall + by label)
      - event-rate bars by percentile groups
      - ROC (if labels available)
      - KM curves by percentile groups (if time+event available)

    IMPORTANT:
    - Subsets to the intersection of (labels if provided) ∩ (sample_ids if provided) ∩ (matrix)
    - No cohort z-score; the stacked model internally uses training-time stats via its
      BlockPreprocessor components.
    """
    import os, numpy as np, pandas as pd, joblib, matplotlib.pyplot as plt

    # ---------- Paths & guards ----------
    fold_dir   = os.path.join(output_dir, subtype)
    models_dir = os.path.join(fold_dir, "cv_models")
    prep_dir   = os.path.join(fold_dir, "preprocessing")
    preds_dir  = os.path.join(fold_dir, "predictions")
    plots_dir  = os.path.join(fold_dir, "plots")
    os.makedirs(preds_dir, exist_ok=True); os.makedirs(plots_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "STACKED_FINAL.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"STACKED model not found: {model_path}")

    stack = joblib.load(model_path)

    # ---------- Small helpers ----------
    def _parse_percentiles(s: str):
        if s is None: return [], "none"
        key = str(s).strip().lower().replace(" ", "")
        if key in ("none",""): return [], "none"
        if key in ("median","med","p50","q2"): return [50.0], "median"
        if key in ("quartiles","quartile","iqr"): return [25.0, 50.0, 75.0], "quartiles"
        if key in ("tertiles","tertile"): return [100.0/3.0, 200.0/3.0], "tertiles"
        if key in ("deciles","decile"): return [float(x) for x in range(10,100,10)], "deciles"
        vals = sorted({float(x) for x in str(s).split(",") if x.strip()})
        for v in vals:
            if not (0.0 < v < 100.0):
                raise ValueError("--percentiles must be in (0,100)")
        return vals, "pct" + str(s).replace(",", "-")

    # def _parse_percentiles(s: str):
    #     if s is None:
    #         return [], "none"

    #     key = str(s).strip().lower().replace(" ", "")

    #     if key in ("none", ""):
    #         return [], "none"

    #     # true sample median mode
    #     if key in ("median", "med"):
    #         return [], "median"

    #     # explicit 50th percentile mode
    #     if key in ("p50", "q2"):
    #         return [50.0], "pct50"

    #     if key in ("quartiles", "quartile", "iqr"):
    #         return [25.0, 50.0, 75.0], "quartiles"

    #     if key in ("tertiles", "tertile"):
    #         return [100.0 / 3.0, 200.0 / 3.0], "tertiles"

    #     if key in ("deciles", "decile"):
    #         return [float(x) for x in range(10, 100, 10)], "deciles"

    #     vals = sorted({float(x) for x in str(s).split(",") if x.strip()})
    #     for v in vals:
    #         if not (0.0 < v < 100.0):
    #             raise ValueError("--percentiles must be in (0,100)")
    #     return vals, "pct" + str(s).replace(",", "-")


    def _labels_from_pcts(pcts):
        if not pcts: return ["All"]
        labs = [f"P≤{pcts[0]:g}"]
        for i in range(len(pcts)-1): labs.append(f"P({pcts[i]:g},{pcts[i+1]:g}]")
        labs.append(f"P>{pcts[-1]:g}")
        return labs

    def _norm_ids(idx_like):
        s = pd.Index(idx_like).astype(str).str.strip()
        return s.str.replace(r"\.0$", "", regex=True)

    def _pick_col(cols_lower, *cands):
        for c in cands:
            if c in cols_lower: return cols_lower[c]
        return None

    def _build_labels_map(labels_csv, ids_order):
        if labels_csv is None or (not os.path.exists(labels_csv)): return {}
        df = pd.read_csv(labels_csv)
        cols_lower = {c.lower(): c for c in df.columns}
        sid = _pick_col(cols_lower, "sample_id") or df.columns[0]
        idxed = df.set_index(sid)
        cols_lower = {c.lower(): c for c in idxed.columns}
        out = {}
        ds_col = _pick_col(cols_lower, "disease_status", "label", "y_label")
        if ds_col is not None:
            out["disease_status"] = idxed[ds_col].astype(str).str.strip().str.lower().reindex(ids_order)
        ev_col = _pick_col(cols_lower, "event_observed", "dfs_event", "os_event", "event")
        if ev_col is not None:
            out["event_observed"] = pd.to_numeric(idxed[ev_col], errors="coerce").reindex(ids_order)
        te_col = _pick_col(cols_lower, "time_to_event", "dfs_months", "os_months", "time")
        if te_col is not None:
            out["time_to_event"] = pd.to_numeric(idxed[te_col], errors="coerce").reindex(ids_order)
        return out

    def _to_binary_event(series: pd.Series) -> pd.Series:
        if series is None: return pd.Series(dtype=int)
        if pd.api.types.is_numeric_dtype(series): return series.fillna(0).astype(int)
        s = series.astype(str).str.strip().str.lower()
        mapping = {"1":1,"0":0,"true":1,"false":0,"yes":1,"no":0,
                   "recurrent":1,"non-recurrent":0,"nonrecurrent":0,"event":1,"censored":0}
        return s.map(mapping).fillna(0).astype(int)

    def _pos_proba(est, Xdf):
        if hasattr(est, "predict_proba"):
            P = est.predict_proba(Xdf)
            classes_ = getattr(est, "classes_", None)
            if classes_ is not None and 1 in set(classes_):
                return P[:, list(classes_).index(1)]
            return P[:, -1]
        elif hasattr(est, "decision_function"):
            from scipy.special import expit
            return expit(est.decision_function(Xdf))
        else:
            return est.predict(Xdf).astype(float)

    def _rescale_unit(x: pd.Series) -> pd.Series:
        xmin, xmax = float(x.min()), float(x.max())
        if xmax <= xmin: return pd.Series(np.zeros_like(x), index=x.index, dtype=float)
        return ((x - xmin) / (xmax - xmin)).astype(float)

    def _add_corner_text(ax, text: str):
        if text:
            ax.text(0.98, 0.98, text, transform=ax.transAxes, ha="right", va="top",
                    bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.3"), fontsize=9)

    # ---------- Load prediction matrix & subset rows BEFORE anything else ----------
    X = pd.read_csv(pred_matrix_csv, index_col=0)
    X.index = _norm_ids(X.index); X.columns = X.columns.astype(str).str.strip()

    # load label IDs (if any) to drive strict subsetting
    label_ids = None
    if pred_labels_csv and os.path.exists(pred_labels_csv):
        tmp = pd.read_csv(pred_labels_csv)
        cols_lower = {c.lower(): c for c in tmp.columns}
        sid_col = _pick_col(cols_lower, "sample_id") or tmp.columns[0]
        label_ids = _norm_ids(tmp[sid_col])

    if sample_ids is not None:
        sample_ids = _norm_ids(sample_ids)

    if label_ids is not None:
        keep = [s for s in label_ids if s in X.index]
        if sample_ids is not None:
            keep = [s for s in keep if s in set(sample_ids)]
    elif sample_ids is not None:
        keep = [s for s in sample_ids if s in X.index]
    else:
        keep = list(X.index)

    X = X.reindex(keep)
    if X.shape[0] == 0:
        raise ValueError("After subsetting to label/sample IDs, no samples remain for STACKED prediction.")

    # sanitize numerics
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

 #   ---------- Predict ----------
    probs = _pos_proba(stack, X)
    pcts, pct_key = _parse_percentiles(percentiles)
    cuts = [float(np.nanquantile(probs, p/100.0)) for p in pcts] if pcts else []
    # make unique, stable cutpoints
    uniq_cuts = []
    for v in cuts:
        if not uniq_cuts or abs(v - uniq_cuts[-1]) > 1e-12:
            uniq_cuts.append(v)
    edges  = [-np.inf] + uniq_cuts + [np.inf]
    labels = _labels_from_pcts(pcts if len(uniq_cuts)==len(pcts) else
                               [round(100*(i+1)/(len(uniq_cuts)+1),2) for i in range(len(uniq_cuts))])

    out = pd.DataFrame({"sample_id": X.index.astype(str), "predicted_proba": probs})
    out["pct_group"] = pd.cut(out["predicted_proba"], bins=edges, labels=labels,
                              include_lowest=True, duplicates="drop").astype(str)
    out["percentiles"]   = str(percentiles)
    out["cutpoints_raw"] = ";".join(f"{c:.10g}" for c in uniq_cuts)
    out["block"] = "STACKED"

    token = (pct_key if pct_key else str(percentiles)).replace(",", "-")
    pred_csv = os.path.join(preds_dir, f"STACKED__predictions__{token}.csv")
    out.to_csv(pred_csv, index=False)
    print(f"💾 [STACKED] Saved predictions → {pred_csv}  (n={len(out)})")

 
    # ---------- Labels map & plots ----------
    labels_map = _build_labels_map(pred_labels_csv, out["sample_id"])

    # Hist (overall, display-scaled to [0,1]) + vlines at cutpoints
    try:
        disp = _rescale_unit(out["predicted_proba"])
        vlines = []
        smin, smax = float(out["predicted_proba"].min()), float(out["predicted_proba"].max())
        if smax > smin: vlines = [(c - smin)/(smax - smin) for c in uniq_cuts]
        out["_display"] = disp.values

        plt.figure(figsize=(7.2,4.6)); ax = plt.gca()
        # by groups
        present = [lab for lab in labels if (out["pct_group"]==lab).any()]
        drawn = False
        for lab in present:
            vals = out.loc[out["pct_group"]==lab, "_display"].astype(float).values
            if len(vals): ax.hist(vals, bins=30, alpha=0.6, label=lab); drawn=True
        if drawn: ax.legend()
        else: ax.hist(out["_display"].values, bins=30, alpha=0.8)
        ax.set_xlim(0,1)
        ax.set_title("[STACKED] Predicted score distribution")
        ax.set_xlabel("Score (0–1, display scaling)"); ax.set_ylabel("Count")
        for xv in vlines: ax.axvline(np.clip(float(xv),0,1), linestyle="--", linewidth=1.3)
        ax.text(0.98, 0.98, f"Cutpoints(raw): {', '.join(f'{c:.4g}' for c in uniq_cuts)}\nTotal n={len(out)}",
                transform=ax.transAxes, ha="right", va="top",
                bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.3"), fontsize=9)
        path = os.path.join(plots_dir, f"STACKED__hist__{token}.png")
        plt.tight_layout(); plt.savefig(path, dpi=200); plt.close(); print(f"[PLOT] {path}")
    except Exception as e:
        print(f"[WARN][STACKED] Histogram failed: {e}")

    # Hist by labels (disease_status or event)
    try:
        plt.figure(); drawn=False
        if "disease_status" in labels_map and labels_map["disease_status"] is not None:
            ds = labels_map["disease_status"].reindex(out["sample_id"]).astype(str).str.strip().str.lower()
            for lab,nice in [("non-recurrent","Non-recurrent"),("nonrecurrent","Non-recurrent"),("recurrent","Recurrent")]:
                m = (ds==lab).to_numpy()
                if m.sum()>0:
                    plt.hist(out["predicted_proba"].values[m], bins=30, alpha=0.6, label=f"{nice} (n={int(m.sum())})"); drawn=True
        elif "event_observed" in labels_map and labels_map["event_observed"] is not None:
            evec = pd.to_numeric(labels_map["event_observed"], errors="coerce").reindex(out["sample_id"]).to_numpy()
            m0 = (evec==0); m1 = (evec==1)
            if np.nan_to_num(m0).sum()>0:
                plt.hist(out["predicted_proba"].values[m0], bins=30, alpha=0.6, label=f"Censored (n={int(m0.sum())})"); drawn=True
            if np.nan_to_num(m1).sum()>0:
                plt.hist(out["predicted_proba"].values[m1], bins=30, alpha=0.6, label=f"Event (n={int(m1.sum())})"); drawn=True
        if not drawn: plt.hist(out["predicted_proba"].values, bins=30)
        else: plt.legend()
        for c in uniq_cuts: plt.axvline(float(c), linestyle="--", linewidth=1.3)
        plt.xlabel("Predicted probability (used for split)"); plt.ylabel("Count")
        plt.title("[STACKED] Distribution by label")
        path = os.path.join(plots_dir, f"STACKED__hist_labels__{token}.png")
        plt.tight_layout(); plt.savefig(path, dpi=200); plt.close(); print(f"[PLOT] {path}")
    except Exception as e:
        print(f"[WARN][STACKED] Label-overlay histogram failed: {e}")

    # Event-rate bars
    try:
        if ("event_observed" in labels_map) or ("disease_status" in labels_map):
            if "event_observed" in labels_map and labels_map["event_observed"] is not None:
                ev_series = _to_binary_event(labels_map["event_observed"])
            else:
                ev_series = labels_map["disease_status"].map({"recurrent":1, "non-recurrent":0, "nonrecurrent":0})
            tmp = out.set_index("sample_id")[["pct_group"]].join(ev_series.rename("event"))
            tmp = tmp.dropna(subset=["event"])
            n = tmp.groupby("pct_group").size().to_dict()
            e = tmp.groupby("pct_group")["event"].sum(min_count=1).fillna(0).to_dict()
            rate = {k: (e.get(k,0) / max(n.get(k,1), 1)) for k in set(n)|set(e)}
            plt.figure(figsize=(6.4,4.2)); ax = plt.gca()
            order = _labels_from_pcts(pcts)
            vals = [rate.get(k,0) for k in order]
            ax.bar(order, vals); ax.set_ylim(0,1)
            for i,lab in enumerate(order):
                ax.text(i, min(vals[i]+0.02,0.98), f"{int(e.get(lab,0))}/{int(n.get(lab,0))}",
                        ha="center", va="bottom", fontsize=9)
            ax.set_title("[STACKED] Event rate by group"); ax.set_ylabel("Event rate")
            path = os.path.join(plots_dir, f"STACKED__eventrate__{token}.png")
            plt.tight_layout(); plt.savefig(path, dpi=200); plt.close(); print(f"[PLOT] {path}")
        else:
            print("[STACKED] Event column not available; skipping event-rate bars.")
    except Exception as e:
        print(f"[WARN][STACKED] Event-rate bars failed: {e}")

    # KM curves (needs time + event)
    try:
        if ("time_to_event" in labels_map) and (("event_observed" in labels_map) or ("disease_status" in labels_map)):
            try:
                from lifelines import KaplanMeierFitter
                from lifelines.statistics import logrank_test, multivariate_logrank_test
            except Exception:
                print("[KM] lifelines not available; skipping KM.")
                return

            if "event_observed" in labels_map and labels_map["event_observed"] is not None:
                ev = _to_binary_event(labels_map["event_observed"])
            else:
                ev = labels_map["disease_status"].map({"recurrent":1, "non-recurrent":0, "nonrecurrent":0})
            tm = labels_map["time_to_event"]

            km_df = out.set_index("sample_id")[["pct_group"]].join(
                pd.DataFrame({"event": ev, "time": tm})
            ).dropna(subset=["event", "time"])

            if km_df.empty:
                print("[STACKED] KM: no usable rows with time+event.")
            else:
                plt.figure(figsize=(6.8,4.6)); ax = plt.gca()
                km = KaplanMeierFitter()
                present = []
                order = _labels_from_pcts(pcts)
                for lab in order:
                    dd = km_df[km_df["pct_group"] == lab]
                    if dd.empty: continue
                    present.append(lab)
                    km.fit(dd["time"].astype(float).values, dd["event"].astype(int).values, label=lab)
                    km.plot(ax=ax, ci_show=False)
                ptxt = ""
                try:
                    if len(present) >= 2:
                        if len(present) == 2:
                            A, B = present[:2]
                            A_df, B_df = km_df[km_df["pct_group"]==A], km_df[km_df["pct_group"]==B]
                            res = logrank_test(A_df["time"], B_df["time"],
                                               event_observed_A=A_df["event"].astype(int),
                                               event_observed_B=B_df["event"].astype(int))
                            ptxt = f"Log-rank p ({A} vs {B}): {('<1e-16' if res.p_value<1e-16 else f'{res.p_value:.2e}')}"
                        else:
                            glob = multivariate_logrank_test(km_df["time"], km_df["pct_group"], km_df["event"].astype(int))
                            ptxt = f"Global p: {('<1e-16' if glob.p_value<1e-16 else f'{glob.p_value:.2e}')}"
                except Exception as e:
                    print(f"[KM] p-value calc failed: {e}")
                ax.set_title("[STACKED] KM by percentile groups")
                ax.set_xlabel("Time"); ax.set_ylabel("Survival probability"); ax.set_ylim(0,1)
                _add_corner_text(ax, f"Cutpoints: {', '.join(f'{c:.4g}' for c in uniq_cuts)}\n{ptxt}".strip())
                km_path = os.path.join(plots_dir, f"STACKED__KM__{token}.png")
                plt.tight_layout(); plt.savefig(km_path, dpi=200); plt.close(); print(f"[PLOT] {km_path}")
        else:
            print("[STACKED] KM skipped (time or event not available).")
    except Exception as e:
        print(f"[WARN][STACKED] KM plotting failed: {e}")


    # Cox
    try:
        _run_prediction_cox_simple(
            out_df=out,
            labels_map=labels_map,
            plots_dir=plots_dir,
            preds_dir=preds_dir,
            prefix="STACKED",
            token=token,
            labels_order=_labels_from_pcts(pcts),
        )
    except Exception as e:
        print(f"[WARN][STACKED] Failed Cox plotting: {e}")

    # ROC
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        y_vec = None
        if "disease_status" in labels_map and labels_map["disease_status"] is not None:
            mapping = {"non-recurrent":0, "nonrecurrent":0, "0":0, "0.0":0, "false":0, "no":0,
                       "recurrent":1, "1":1, "1.0":1, "true":1, "yes":1}
            y_vec = labels_map["disease_status"].map(mapping)
            if y_vec is not None and y_vec.isna().any():
                y_num = pd.to_numeric(labels_map["disease_status"], errors="coerce")
                y_vec = y_vec.where(~y_vec.isna(), y_num)
        if y_vec is None and "event_observed" in labels_map and labels_map["event_observed"] is not None:
            y_vec = labels_map["event_observed"]
        if y_vec is not None:
            mask = ~y_vec.isna(); n_use = int(mask.sum())
            if n_use > 0 and y_vec[mask].nunique() == 2:
                fpr, tpr, _ = roc_curve(y_vec[mask].astype(int).values, out["predicted_proba"].values[mask], pos_label=1)
                auc_val = roc_auc_score(y_vec[mask].astype(int).values, out["predicted_proba"].values[mask])
                plt.figure()
                plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
                plt.plot([0,1],[0,1], linestyle="--")
                plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
                plt.title("[STACKED] Prediction ROC curve"); plt.legend(loc="lower right")
                roc_path = os.path.join(plots_dir, f"STACKED__predict_ROC__{token}.png")
                plt.tight_layout(); plt.savefig(roc_path, dpi=200); plt.close(); print(f"[PLOT] {roc_path}")
            else:
                print("[STACKED] ROC skipped: need two classes with usable labels.")
        else:
            print("[STACKED] ROC skipped: no disease_status or event_observed available.")
    except Exception as e:
        print(f"[WARN][STACKED] ROC failed: {e}")

    # Audit file
    info_path = os.path.join(preds_dir, f"predict_info__STACKED__{token}.txt")
    with open(info_path, "w") as f:
        f.write(f"pred_matrix_csv: {pred_matrix_csv}\n")
        f.write(f"samples: {X.shape[0]}\n")
        f.write(f"scale_mode: {scale_mode}\npercentiles: {percentiles}\n")
        if pred_labels_csv: f.write(f"labels_csv: {pred_labels_csv}\n")


# =========================
#           CLI
# =========================
import os
import argparse
import pandas as pd

VALID_PATHWAY_CHOICES = {"gobp", "reactome", "kegg", "hallmark_kegg"}

def get_args():
    p = argparse.ArgumentParser(
        description="Train/predict recurrence models with inline feature selection.",
        allow_abbrev=False,
    )

    # -------------------------
    # Core paths / cohort
    # -------------------------
    p.add_argument("--data_path", default=None,
                   help="Root path to input matrices and metadata. Required for training.")
    p.add_argument("--output_dir", required=True,
                   help="Folder where outputs will be written.")
    p.add_argument("--subtype", required=True,
                   help="Subtype label, e.g. LumA, LumB, Basal, Luminal, ALL.")
    p.add_argument("--status_col", default="Disease_Status",
                   help="Column in meta used as target label.")
    p.add_argument("--pathway_options", nargs="+", default=["hallmark_kegg"],
                   help="Pathway sets to use, e.g. gobp reactome kegg hallmark_kegg.")

    # -------------------------
    # Training / model selection
    # -------------------------
    p.add_argument("--inner_folds", type=int, default=3,
                   help="Folds for GridSearchCV model selection.")
    p.add_argument("--model_id", default=None,
                   help=(
                       "Model family to tune: lr/logreg/sgd/svm/linearsvc/rf/"
                       "extratrees/gbdt/hgb/xgb, or exact grid key."
                   ))

    p.add_argument("--calibrate", action="store_true",
                   help="Wrap chosen model with CalibratedClassifierCV.")
    p.add_argument("--calibrate_method", choices=["sigmoid", "isotonic"], default="sigmoid")
    p.add_argument("--calibrate_cv", type=int, default=5)

    p.add_argument("--scale_mode", default="training",
                   help="Scaling mode for train/predict compatibility.")

    p.add_argument("--use_cv_winners", action="store_true",
                   help="Restrict per-block model grids using previous CV winners.")
    p.add_argument("--winners_source", choices=["auto", "stack", "perblock", "summary"],
                   default="auto")
    p.add_argument("--block_model_csv", default=None,
                   help="CSV mapping block -> best_model to restrict model families.")

    # -------------------------
    # Inline feature selection
    # -------------------------
    p.add_argument("--fs_dir_existing", default=None,
                   help=(
                       "Existing FS folder containing *_selected_post_redundancy.csv files. "
                       "If provided, skip inline FS and reuse this folder."
                   ))
    p.add_argument("--subset_run_name", default="training_fullcohort",
                   help="Name for the inline FS output folder.")

    # Group-test thresholds
    p.add_argument("--fs_p_thresh", type=float, default=0.05,
                   help="MWU/Fisher p-value cutoff for FS.")
    p.add_argument("--fs_fdr_thresh", type=float, default=0.25,
                   help="MWU/Fisher BH-FDR cutoff for FS.")
    p.add_argument("--fs_corr_threshold", type=float, default=0.70,
                   help="Feature-feature correlation threshold for redundancy pruning.")

    # Variance filter
    p.add_argument("--var_min", type=float, default=0.0)
    p.add_argument("--var_min_nonmissing", type=int, default=10)
    p.add_argument("--variance_keep_file", default=None)
    p.add_argument("--var_keep_top_frac", type=float, default=None)

    # Optional label-correlation prefilter
    p.add_argument("--corr_prefilter", action="store_true")
    p.add_argument("--corr_abs_min", type=float, default=None)
    p.add_argument("--corr_min_n", type=int, default=10)

    # Limma thresholds
    p.add_argument("--limma_pvalue_threshold", type=float, default=0.05)
    p.add_argument("--limma_padj_threshold", type=float, default=0.25)
    p.add_argument("--limma_logfc_threshold", type=float, default=0.25)

    # Cox feature-selection options
    p.add_argument("--run_cox_uni", action="store_true")
    p.add_argument("--run_cox_multi", action="store_true")
    p.add_argument("--cox_time_col", default=None)
    p.add_argument("--cox_event_col", default=None)
    p.add_argument("--cox_covariates", nargs="*", default=None)
    p.add_argument("--cox_uni_p_thresh", type=float, default=0.05)
    p.add_argument("--cox_uni_fdr_thresh", type=float, default=0.25)
    p.add_argument("--cox_multi_p_thresh", type=float, default=0.05)
    p.add_argument("--cox_multi_fdr_thresh", type=float, default=0.25)
    p.add_argument("--cox_hr_abs_min", type=float, default=0.0)
    p.add_argument("--cox_hr_direction", default="any", choices=["any", "gt1", "lt1"])
    p.add_argument("--cox_min_n", type=int, default=10)
    p.add_argument("--cox_min_events", type=int, default=1)

    # Minimum feature fallback
    p.add_argument("--fs_min_features", type=int, default=10)
    p.add_argument("--fs_topk_if_empty", type=int, default=200)

    # Optional extra feature inputs
    p.add_argument("--custom_gene_sets_file", default=None)
    p.add_argument("--custom_gene_set_min_overlap", type=int, default=1)

    p.add_argument("--protein_coding_gtf", default=None)
    p.add_argument("--RNAgene-bypass", action="store_true", dest="RNAgene_bypass")
    p.add_argument("--CNVgene-bypass", action="store_true", dest="CNVgene_bypass")

    # Mutation-derived features
    # Backward-compatible behavior:
    #   --no_mutated_pathways forces mut_pathway_source='none'
    #   --enable_ptmb with no --ptmb_source forces ptmb_source='snv'
    p.add_argument("--no_mutated_pathways", action="store_true",
                   help="Backward-compatible shortcut: disable binary mutated pathway features.")
    p.add_argument("--mut_pathway_source", default="snv",
                   choices=["snv", "import", "none"],
                   help=(
                       "Source for binary mutated pathway features: "
                       "'snv' = calculate from mutation/SNV data in data_path; "
                       "'import' = load precomputed sample x pathway binary matrix; "
                       "'none' = skip."
                   ))
    p.add_argument("--mut_binary_matrix_csv", default=None,
                   help="Precomputed binary mutation pathway matrix for --mut_pathway_source import.")
    p.add_argument("--mut_binary_sample_col", default=None,
                   help="Optional sample ID column in --mut_binary_matrix_csv; if omitted, first column/index is used.")
    p.add_argument("--mut_binary_prefix", default="MUTblock__",
                   help="Prefix added to imported binary mutation pathway columns if not already present.")

    p.add_argument("--enable_ptmb", action="store_true",
                   help="Backward-compatible shortcut: enable pTMB calculation from SNV/MAF inputs unless --ptmb_source is set.")
    p.add_argument("--ptmb_source", default="none",
                   choices=["none", "snv", "import"],
                   help=(
                       "Source for pathway TMB features: "
                       "'snv' = calculate from MAF/GMT/GTF inputs; "
                       "'import' = load precomputed sample x pTMB matrix; "
                       "'none' = skip."
                   ))

    # Imported pTMB mode
    p.add_argument("--ptmb_matrix_csv", default=None,
                   help="Precomputed pathway TMB matrix for --ptmb_source import.")
    p.add_argument("--ptmb_sample_col", default=None,
                   help="Optional sample ID column in --ptmb_matrix_csv; if omitted, first column/index is used.")
    p.add_argument("--ptmb_prefix", default="MUTblock__pTMB__",
                   help="Prefix added to imported pTMB columns if not already present.")
    p.add_argument("--ptmb_transform", default="none", choices=["none", "log10", "log1p"],
                   help="Optional transform applied only to imported pTMB values.")

    # SNV/MAF pTMB calculation mode
    p.add_argument("--ptmb_maf_path", default=None)
    p.add_argument("--ptmb_maf_sample_col", default=None)
    p.add_argument("--ptmb_maf_gene_col", default=None)
    p.add_argument("--ptmb_gtf_path", default=None)
    p.add_argument("--ptmb_cds_cache_csv", default=None)
    p.add_argument("--ptmb_denom_policy", default="all_genes_with_len",
                   choices=["all_genes_with_len", "all_genes", "mutated_genes_with_len"])
    p.add_argument("--ptmb_min_genes_with_len", type=int, default=1)
    p.add_argument("--ptmb_gmt_path", nargs="+", default=None)

    # -------------------------
    # Prediction
    # -------------------------
    p.add_argument("--predict_only", action="store_true",
                   help="Skip training and run prediction only.")
    p.add_argument("--predict_ids_csv", default=None,
                   help="Optional CSV with sample_id column to restrict prediction IDs.")
    p.add_argument("--predict_matrix_csv", default=None,
                   help="Full prediction matrix CSV. Required for predict_only.")
    p.add_argument("--predict_labels_csv", default=None,
                   help="Optional labels/survival CSV for prediction ROC/KM.")
    p.add_argument("--predict_blocks", default="RNA,CNV,MUT,CLINICAL,ALL",
                   help="Comma-list of blocks to run at prediction time.")

    p.add_argument("--prob_threshold", type=float, default=0.5,
                   help="Fixed probability threshold if threshold_mode=fixed.")
    p.add_argument("--threshold_mode", default="median",
                   choices=["median", "fixed"],
                   help="Risk-group thresholding mode.")
    p.add_argument("--percentiles", default="50",
                   help="Percentile split(s) for KM, e.g. 50 or 25,50,75.")

    # -------------------------
    # Stacking
    # -------------------------
    p.add_argument("--enable_stacking", action="store_true",
                   help="After training blocks, fit stacked model.")
    p.add_argument("--stack_blocks", default="RNA,CNV,MUT,CLINICAL",
                   help="Comma-list of blocks to include in stack.")
    p.add_argument("--stack_meta_model", default="logreg",
                   choices=["logreg", "rf", "gbdt"],
                   help="Stacked meta-learner.")
    p.add_argument("--stack_cv", type=int, default=5)
    p.add_argument("--predict_stacked", action="store_true",
                   help="Run prediction using STACKED_FINAL.pkl.")

    return p.parse_args()

def main():
    args = get_args()


    # -------------------------
    # Validate required paths
    # -------------------------
    if not args.predict_only and not args.data_path:
        raise ValueError("--data_path is required for training.")

    if args.predict_only and not args.predict_matrix_csv:
        raise ValueError("--predict_matrix_csv is required when --predict_only is used.")

    # -------------------------
    # Normalize mutation-derived feature modes
    # -------------------------
    if args.no_mutated_pathways:
        args.mut_pathway_source = "none"

    # Backward compatibility with older commands:
    #   --enable_ptmb previously meant "calculate pTMB from MAF/GMT/GTF".
    if args.enable_ptmb and args.ptmb_source == "none":
        args.ptmb_source = "snv"

    if args.ptmb_source != "none":
        args.enable_ptmb = True

    # Source-specific validation is only needed when inline FS will run.
    # If --fs_dir_existing is provided, training reuses selected FS outputs and does not
    # need raw/imported mutation inputs.
    needs_inline_fs = (not args.predict_only) and (not args.fs_dir_existing)

    if needs_inline_fs:
        if args.mut_pathway_source == "import" and not args.mut_binary_matrix_csv:
            raise ValueError("--mut_binary_matrix_csv is required when --mut_pathway_source import.")

        if args.ptmb_source == "import" and not args.ptmb_matrix_csv:
            raise ValueError("--ptmb_matrix_csv is required when --ptmb_source import.")

        if args.ptmb_source == "snv":
            if not args.ptmb_maf_path:
                raise ValueError("--ptmb_maf_path is required when --ptmb_source snv.")
            if not args.ptmb_gmt_path:
                raise ValueError("--ptmb_gmt_path is required when --ptmb_source snv.")

    # -------------------------
    # Normalize pathway options
    # -------------------------
    args.pathway_options = [str(s).lower() for s in args.pathway_options]

    unknown_pw = [x for x in args.pathway_options if x not in VALID_PATHWAY_CHOICES]
    if unknown_pw:
        print(f"⚠️ Unknown pathway_options ignored: {unknown_pw}")
        args.pathway_options = [x for x in args.pathway_options if x in VALID_PATHWAY_CHOICES]

    if not args.pathway_options:
        raise ValueError("No valid --pathway_options after filtering.")

    # -------------------------
    # Run configuration printout
    # -------------------------
    fs_mode = "reuse existing FS dir" if args.fs_dir_existing else "run inline FS during training"

    print("\n🚀 Run configuration")
    print("────────────────────────────────────────")
    print("[Core]")
    print(f"   data_path              : {args.data_path}")
    print(f"   output_dir             : {args.output_dir}")
    print(f"   subtype                : {args.subtype}")
    print(f"   status_col             : {args.status_col}")
    print(f"   pathway_options        : {args.pathway_options}")

    print("\n[Mode]")
    print(f"   predict_only           : {args.predict_only}")
    print(f"   fs_mode                : {fs_mode}")
    print(f"   fs_dir_existing        : {args.fs_dir_existing}")

    print("\n[Training / model selection]")
    print(f"   model_id               : {args.model_id}")
    print(f"   inner_folds            : {args.inner_folds}")
    print(f"   scale_mode             : {args.scale_mode}")
    print(f"   calibrate              : {args.calibrate}")
    print(f"   calibrate_method       : {args.calibrate_method}")
    print(f"   calibrate_cv           : {args.calibrate_cv}")
    print(f"   use_cv_winners         : {args.use_cv_winners}")
    print(f"   winners_source         : {args.winners_source}")
    print(f"   block_model_csv        : {args.block_model_csv}")

    print("\n[Inline feature selection]")
    print(f"   subset_run_name        : {args.subset_run_name}")
    print(f"   fs_p_thresh            : {args.fs_p_thresh}")
    print(f"   fs_fdr_thresh          : {args.fs_fdr_thresh}")
    print(f"   fs_corr_threshold      : {args.fs_corr_threshold}")
    print(f"   var_min                : {args.var_min}")
    print(f"   var_min_nonmissing     : {args.var_min_nonmissing}")
    print(f"   var_keep_top_frac      : {args.var_keep_top_frac}")
    print(f"   variance_keep_file     : {args.variance_keep_file}")
    print(f"   corr_prefilter         : {args.corr_prefilter}")
    print(f"   corr_abs_min           : {args.corr_abs_min}")
    print(f"   corr_min_n             : {args.corr_min_n}")

    print("\n[Limma feature selection]")
    print(f"   limma_pvalue_threshold : {args.limma_pvalue_threshold}")
    print(f"   limma_padj_threshold   : {args.limma_padj_threshold}")
    print(f"   limma_logfc_threshold  : {args.limma_logfc_threshold}")

    print("\n[Cox feature selection]")
    print(f"   run_cox_uni            : {args.run_cox_uni}")
    print(f"   run_cox_multi          : {args.run_cox_multi}")
    print(f"   cox_time_col           : {args.cox_time_col}")
    print(f"   cox_event_col          : {args.cox_event_col}")
    print(f"   cox_covariates         : {args.cox_covariates}")
    print(f"   cox_uni_p_thresh       : {args.cox_uni_p_thresh}")
    print(f"   cox_uni_fdr_thresh     : {args.cox_uni_fdr_thresh}")
    print(f"   cox_multi_p_thresh     : {args.cox_multi_p_thresh}")
    print(f"   cox_multi_fdr_thresh   : {args.cox_multi_fdr_thresh}")
    print(f"   cox_hr_abs_min         : {args.cox_hr_abs_min}")
    print(f"   cox_hr_direction       : {args.cox_hr_direction}")
    print(f"   cox_min_n              : {args.cox_min_n}")
    print(f"   cox_min_events         : {args.cox_min_events}")

    print("\n[Optional feature inputs]")
    print(f"   custom_gene_sets_file  : {args.custom_gene_sets_file}")
    print(f"   protein_coding_gtf     : {args.protein_coding_gtf}")
    print(f"   RNAgene_bypass         : {args.RNAgene_bypass}")
    print(f"   CNVgene_bypass         : {args.CNVgene_bypass}")
    print(f"   no_mutated_pathways    : {args.no_mutated_pathways}")
    print(f"   mut_pathway_source     : {args.mut_pathway_source}")
    print(f"   mut_binary_matrix_csv  : {args.mut_binary_matrix_csv}")
    print(f"   mut_binary_sample_col  : {args.mut_binary_sample_col}")
    print(f"   mut_binary_prefix      : {args.mut_binary_prefix}")
    print(f"   enable_ptmb            : {args.enable_ptmb}")
    print(f"   ptmb_source            : {args.ptmb_source}")
    print(f"   ptmb_matrix_csv        : {args.ptmb_matrix_csv}")
    print(f"   ptmb_sample_col        : {args.ptmb_sample_col}")
    print(f"   ptmb_prefix            : {args.ptmb_prefix}")
    print(f"   ptmb_transform         : {args.ptmb_transform}")
    print(f"   ptmb_maf_path          : {args.ptmb_maf_path}")
    print(f"   ptmb_gmt_path          : {args.ptmb_gmt_path}")
    print(f"   ptmb_gtf_path          : {args.ptmb_gtf_path}")
    print(f"   ptmb_cds_cache_csv     : {args.ptmb_cds_cache_csv}")

    print("\n[Prediction]")
    print(f"   predict_matrix_csv     : {args.predict_matrix_csv}")
    print(f"   predict_labels_csv     : {args.predict_labels_csv}")
    print(f"   predict_ids_csv        : {args.predict_ids_csv}")
    print(f"   predict_blocks         : {args.predict_blocks}")
    print(f"   threshold_mode         : {args.threshold_mode}")
    print(f"   prob_threshold         : {args.prob_threshold}")
    print(f"   percentiles            : {args.percentiles}")

    print("\n[Stacking]")
    print(f"   enable_stacking        : {args.enable_stacking}")
    print(f"   stack_blocks           : {args.stack_blocks}")
    print(f"   stack_meta_model       : {args.stack_meta_model}")
    print(f"   stack_cv               : {args.stack_cv}")
    print(f"   predict_stacked        : {args.predict_stacked}")
    print("────────────────────────────────────────\n")

    # ---- tiny helper to read sample IDs if provided ----
    def _load_sample_ids(path):
        import os, pandas as pd
        if not path or not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        cols_lower = {c.lower(): c for c in df.columns}
        sid_col = cols_lower.get("sample_id", df.columns[0])
        s = df[sid_col].astype(str).str.strip().tolist()
        return s

    # =========================
    # Prediction-only branch
    # =========================
    if args.predict_only:
        if not args.predict_matrix_csv:
            raise ValueError("--predict_only requires --predict_matrix_csv")

        sample_ids = _load_sample_ids(getattr(args, "predict_ids_csv", None))

        # 1) Always run per-block / ALL predictions
        print("🔮 Predicting with per-block/ALL models ...")
        predict_concat_only(
            output_dir=args.output_dir,
            subtype=args.subtype,
            pred_matrix_csv=args.predict_matrix_csv,
            sample_ids=sample_ids,
            pred_labels_csv=args.predict_labels_csv,
            percentiles=getattr(args, "percentiles", "50"),
            scale_mode=getattr(args, "scale_mode", "training"),
            blocks=getattr(args, "predict_blocks", "RNA,CNV,MUT,ALL"),
            #blocks=getattr(args, "predict_blocks", "RNA,CNV,MUT,ALL"),

        )

        # 2) If requested, ALSO run stacked predictions (no early return!)
        if getattr(args, "predict_stacked", False):
            print("🔮 Predicting with STACKED_FINAL.pkl ...")
            predict_stacked(
                output_dir=args.output_dir,
                subtype=args.subtype,
                pred_matrix_csv=args.predict_matrix_csv,
                sample_ids=sample_ids,
                pred_labels_csv=args.predict_labels_csv,
                percentiles=getattr(args, "percentiles", "50"),
                scale_mode=getattr(args, "scale_mode", "training"),
            )

        return


    # =========================
    # Training branch
    # =========================
    # This path uses precomputed FS outputs (no on-the-fly FS here).
    out_dir = train_once_concat(args)

    # Optionally fit the stacked model on top of trained block models
    if getattr(args, "enable_stacking", False):
        print("🧱 Building stacked meta-model over trained block models ...")
        fit_stack_from_trained_blocks(
            output_dir=args.output_dir,
            subtype=args.subtype,
            blocks=getattr(args, "stack_blocks", "RNA,CNV,MUT"),
            meta_model=getattr(args, "stack_meta_model", "logreg"),
            scale_mode=getattr(args, "scale_mode", "training"),
            cv=int(getattr(args, "stack_cv", 5)),
            calibrate=bool(getattr(args, "calibrate", False)),
            calibrate_method=getattr(args, "calibrate_method", "sigmoid"),
            calibrate_cv=int(getattr(args, "calibrate_cv", 3)),
            block_model_csv=getattr(args, "block_model_csv", None),
            stack_meta_model=getattr(args, "stack_meta_model", None),
        )

    print("✅ Done.")

if __name__ == "__main__":
    main()






# source /fh/fast/ha_g/user/acruiksh/BRAVO/paper/scripts/bravo_env3/bin/activate && /fh/fast/ha_g/user/acruiksh/BRAVO/paper/scripts/bravo_env3/bin/python /fh/fast/ha_g/user/acruiksh/BRAVO/github/scripts/training_script_with_import_modes.py --data_path /fh/fast/ha_g/user/acruiksh/BRAVO/github/data   --output_dir /fh/fast/ha_g/user/acruiksh/BRAVO/github/test   --subtype LumA   --status_col Disease_Status   --pathway_options hallmark_kegg reactome   --subset_run_name fullcohort   --protein_coding_gtf /fh/fast/ha_g/user/acruiksh/BRAVO/github/data/gencode.v44.annotation.gtf   --var_min_nonmissing 5   --var_keep_top_frac 0.50   --corr_prefilter --corr_abs_min 0.010  --corr_min_n 5   --run_cox_uni --run_cox_multi   --fs_p_thresh 0.05 --fs_fdr_thresh 1   --cox_uni_p_thresh 0.05 --cox_uni_fdr_thresh 1   --cox_multi_p_thresh 0.05 --cox_multi_fdr_thresh 1   --limma_pvalue_threshold 0.05 --limma_padj_threshold 1 --limma_logfc_threshold 0.25   --cox_time_col RFS_Time --cox_event_col RFS_Status   --cox_covariates age_dx chemo_med hormtxt_evr herceptin_med   --fs_corr_threshold 0.80   --custom_gene_sets_file /fh/fast/ha_g/user/acruiksh/BRAVO/github/data/RNAseq/genescores/custom_genesets.gmt  --mut_pathway_source import --mut_binary_matrix_csv /fh/fast/ha_g/user/acruiksh/BRAVO/github/data/mutations/LumA/MUT_binary_pathways_full_snv_calculated.csv --mut_binary_sample_col patient_id --ptmb_source import --ptmb_matrix_csv /fh/fast/ha_g/user/acruiksh/BRAVO/github/data/mutations/LumA/PTMB_full_raw.csv --ptmb_sample_col patient_id


# source /fh/fast/ha_g/user/acruiksh/BRAVO/paper/scripts/bravo_env3/bin/activate && /fh/fast/ha_g/user/acruiksh/BRAVO/paper/scripts/bravo_env3/bin/python /fh/fast/ha_g/user/acruiksh/BRAVO/github/scripts/training_script_with_import_modes.py --data_path /fh/fast/ha_g/user/acruiksh/BRAVO/github/data   --output_dir /fh/fast/ha_g/user/acruiksh/BRAVO/github/test   --subtype LumB   --status_col Disease_Status   --pathway_options hallmark_kegg reactome   --subset_run_name fullcohort   --protein_coding_gtf /fh/fast/ha_g/user/acruiksh/BRAVO/github/data/gencode.v44.annotation.gtf   --var_min_nonmissing 5   --var_keep_top_frac 0.50   --corr_prefilter --corr_abs_min 0.010  --corr_min_n 5   --run_cox_uni --run_cox_multi   --fs_p_thresh 0.05 --fs_fdr_thresh 1   --cox_uni_p_thresh 0.05 --cox_uni_fdr_thresh 1   --cox_multi_p_thresh 0.05 --cox_multi_fdr_thresh 1   --limma_pvalue_threshold 0.05 --limma_padj_threshold 1 --limma_logfc_threshold 0.25   --cox_time_col RFS_Time --cox_event_col RFS_Status   --cox_covariates age_dx chemo_med hormtxt_evr herceptin_med   --fs_corr_threshold 0.80   --custom_gene_sets_file /fh/fast/ha_g/user/acruiksh/BRAVO/github/data/RNAseq/genescores/custom_genesets.gmt  --enable_ptmb   --ptmb_maf_path /fh/fast/ha_g/user/acruiksh/BRAVO/github/data/mutations/20241022_BRAVO_mutect2_strelka2_passed_filterpass_runc_to_t_vaf_SOBdetector_nonsynSNVs_ONLY.csv --ptmb_gtf_path /fh/fast/ha_g/user/acruiksh/BRAVO/github/data/gencode.v44.annotation.gtf   --ptmb_gmt_path  /fh/fast/ha_g/user/acruiksh/BRAVO/github/data/pathways/h.all.v2023.2.Hs.symbols.gmt /fh/fast/ha_g/user/acruiksh/BRAVO/github/data/pathways/c2.cp.kegg_legacy.v2023.2.Hs.symbols.gmt /fh/fast/ha_g/user/acruiksh/BRAVO/github/data/pathways/c2.cp.reactome.v2023.2.Hs.symbols.gmt  --ptmb_maf_sample_col patient_id   --ptmb_maf_gene_col "Gene.refGene"

# source /fh/fast/ha_g/user/acruiksh/BRAVO/paper/scripts/bravo_env3/bin/activate && /fh/fast/ha_g/user/acruiksh/BRAVO/paper/scripts/bravo_env3/bin/python /fh/fast/ha_g/user/acruiksh/BRAVO/github/scripts/training_script_with_import_modes.py --data_path /fh/fast/ha_g/user/acruiksh/BRAVO/github/data   --output_dir /fh/fast/ha_g/user/acruiksh/BRAVO/github/test   --subtype Basal   --status_col Disease_Status   --pathway_options hallmark_kegg reactome   --subset_run_name fullcohort   --protein_coding_gtf /fh/fast/ha_g/user/acruiksh/BRAVO/github/data/gencode.v44.annotation.gtf   --var_min_nonmissing 5   --var_keep_top_frac 0.50   --corr_prefilter --corr_abs_min 0.010  --corr_min_n 5   --run_cox_uni --run_cox_multi   --fs_p_thresh 0.05 --fs_fdr_thresh 1   --cox_uni_p_thresh 0.05 --cox_uni_fdr_thresh 1   --cox_multi_p_thresh 0.05 --cox_multi_fdr_thresh 1   --limma_pvalue_threshold 0.05 --limma_padj_threshold 1 --limma_logfc_threshold 0.25   --cox_time_col RFS_Time --cox_event_col RFS_Status   --cox_covariates age_dx chemo_med hormtxt_evr herceptin_med   --fs_corr_threshold 0.80   --custom_gene_sets_file /fh/fast/ha_g/user/acruiksh/BRAVO/github/data/RNAseq/genescores/custom_genesets.gmt  --enable_ptmb   --ptmb_maf_path /fh/fast/ha_g/user/acruiksh/BRAVO/github/data/mutations/20241022_BRAVO_mutect2_strelka2_passed_filterpass_runc_to_t_vaf_SOBdetector_nonsynSNVs_ONLY.csv --ptmb_gtf_path /fh/fast/ha_g/user/acruiksh/BRAVO/github/data/gencode.v44.annotation.gtf   --ptmb_gmt_path  /fh/fast/ha_g/user/acruiksh/BRAVO/github/data/pathways/h.all.v2023.2.Hs.symbols.gmt /fh/fast/ha_g/user/acruiksh/BRAVO/github/data/pathways/c2.cp.kegg_legacy.v2023.2.Hs.symbols.gmt /fh/fast/ha_g/user/acruiksh/BRAVO/github/data/pathways/c2.cp.reactome.v2023.2.Hs.symbols.gmt  --ptmb_maf_sample_col patient_id   --ptmb_maf_gene_col "Gene.refGene"


