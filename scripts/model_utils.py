import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    average_precision_score
)
#from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier
#from catboost import CatBoostClassifier
import shap
import statsmodels.api as sm
from output_utils import *
from pca_utils import * 
from dataloading_utils import *
from dataprocessing_utils import *
from pca_utils import *
from plot_utils import *
from output_utils import *
from param_grids import *


def preprocess_block(
    X_train,
    process_type,
    block_name,
    X_test=None,
    y_train=None,
    output_path=None,
    fold=None,
    scale=True,
    top_k_sig=20,
    min_per_group=2,
    pca_n_components=10,
    pca_min_components=1,
    *,
    p_threshold=0.01,      # NEW: significance cutoff
    fallback_top_k=10      # NEW: if none pass threshold, take this many smallest p-values
):
    """
    Preprocess a feature block using the given process_type.
    process_type in {"pca", "sig_lax", "as_is", "clinical"}.

    sig_lax behavior:
      - run significance test; keep only features with p <= p_threshold
      - if none pass, keep top `fallback_top_k` by smallest p
      - (then optional scaling)

    Saves a joblib bundle with everything needed to reproduce the exact transform on validation/test:
      - columns (training feature list / ordering)
      - scaler (if used)
      - pca (if used)
      - selected_features (sig_lax)
      - metadata (block_name, fold, process_type)
    """
    import os
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # -------- persistence helpers --------
    def _fold_suffix(f):
        if f is None or (isinstance(f, str) and f.strip() == ""):
            return ""
        if isinstance(f, (int, np.integer)) or (isinstance(f, str) and str(f).isdigit()):
            return f"_fold{int(f) + 1}"
        return f"_{str(f)}"

    def _bundle_path(kind="bundle"):
        if output_path is None:
            return None
        os.makedirs(output_path, exist_ok=True)
        return os.path.join(output_path, f"{kind}_{block_name}{_fold_suffix(fold)}.joblib")

    def _save_bundle(bundle, kind="bundle"):
        path = _bundle_path(kind)
        if path:
            try:
                import joblib
            except Exception:
                from sklearn.externals import joblib  # fallback (older sklearn)
            joblib.dump(bundle, path)
            print(f"💾 Saved {kind} for '{block_name}': {path}")

    # ---------- small utils ----------
    def _ensure_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # booleans → ints
        bool_cols = df.select_dtypes(include=["bool"]).columns
        if len(bool_cols):
            df[bool_cols] = df[bool_cols].astype(int)
        # one-hot for object/category
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols):
            df = pd.get_dummies(df, columns=list(cat_cols), dummy_na=False)
        # numeric coercion + clean
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df.astype(float)

    def _safe_extract_sig_features(
        sig_results,
        p_col='p_value',
        feat_col='feature',
        p_thresh=p_threshold,
        fallback_k=fallback_top_k
    ):
        """
        Return features with p <= p_thresh; if none, take top fallback_k by smallest p.
        Caps by top_k_sig if there are many that pass the threshold.
        """
        if sig_results is None or len(sig_results) == 0:
            return []
        df = sig_results.copy()
        if feat_col not in df.columns:
            df[feat_col] = df.index.astype(str)
        if p_col not in df.columns:
            return []
        df[p_col] = pd.to_numeric(df[p_col], errors='coerce')
        df = df.dropna(subset=[p_col])
        if df.empty:
            return []
        # keep significant
        sig = df[df[p_col] <= p_thresh].sort_values(p_col)
        if not sig.empty:
            k = min(top_k_sig, len(sig))
            return sig.iloc[:k][feat_col].astype(str).tolist()
        # none significant → take top fallback_k overall
        return df.nsmallest(fallback_k, p_col)[feat_col].astype(str).tolist()

    def _fallback_unsupervised(X_tr, top_k=50):
        var = X_tr.var(axis=0, ddof=1)
        return var.sort_values(ascending=False).head(min(top_k, len(var))).index.tolist()

    # ---------- index alignment ----------
    X_train = X_train.copy()
    if X_test is not None:
        X_test = X_test.copy()

    X_train.index = X_train.index.astype(str).str.strip()
    if X_test is not None:
        X_test.index = X_test.index.astype(str).str.strip()

    y_series = None
    if y_train is not None:
        y_series = pd.Series(y_train).copy()
        y_series.index = y_series.index.astype(str).str.strip()
        y_series = y_series.reindex(X_train.index)

    # ---------- processing ----------
    if process_type == "pca":
        print("🧪 PCA preprocessing (fit on TRAIN only)...")

        # 0) Ensure numeric
        Xtr = _ensure_numeric_frame(X_train)
        Xte = _ensure_numeric_frame(X_test) if X_test is not None else None

        # 1) Align test to train BEFORE any fitting
        if Xte is not None:
            Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)

        # 2) Fit scaler on TRAIN only; transform both
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte) if Xte is not None else None

        # 3) Fit PCA on TRAIN only; transform both
        max_components = min(pca_n_components, Xtr_s.shape[0], Xtr_s.shape[1])
        if max_components < pca_min_components:
            print(f"⚠️ Skipping PCA: only {max_components} comps possible (< min {pca_min_components}). Using scaled features as-is.")
            # Fall back to scaled features
            X_train_sig = pd.DataFrame(Xtr_s, index=Xtr.index, columns=Xtr.columns)
            X_test_sig  = None if Xte_s is None else pd.DataFrame(Xte_s, index=Xte.index, columns=Xtr.columns)
            # Save bundle (scaler only)
            _save_bundle({
                "process_type": "pca_fallback_scaled",
                "block_name": block_name,
                "fold": fold,
                "columns": list(Xtr.columns),
                "scaler": scaler,
                "pca": None,
                "pca_n_components": pca_n_components,
                "pca_min_components": pca_min_components,
            })
            return X_train_sig, X_test_sig

        pca = PCA(n_components=max_components)
        Ztr = pca.fit_transform(Xtr_s)
        Zte = pca.transform(Xte_s) if Xte_s is not None else None

        pc_cols = [f"PC{i+1}" for i in range(Ztr.shape[1])]
        X_train_sig = pd.DataFrame(Ztr, index=Xtr.index, columns=pc_cols)
        X_test_sig  = None if Zte is None else pd.DataFrame(Zte, index=Xte.index, columns=pc_cols)

        # 4) Prefix PC names to avoid duplicates across blocks
        X_train_sig.columns = [f"{block_name}__{c}" for c in X_train_sig.columns]
        if X_test_sig is not None:
            X_test_sig.columns = X_train_sig.columns

        # 5) Save artifacts needed for validation
        _save_bundle({
            "process_type": "pca",
            "block_name": block_name,
            "fold": fold,
            "columns": list(Xtr.columns),   # training feature order
            "scaler": scaler,               # fitted scaler
            "pca": pca,                     # fitted PCA
            "pca_n_components": pca_n_components,
            "pca_min_components": pca_min_components,
        })

        # 6) (Optional) write CSV artifacts similar to your run_pca helper
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            suf = _fold_suffix(fold)
            # loadings (features × PCs)
            loadings = pd.DataFrame(pca.components_.T, index=Xtr.columns,
                                    columns=[f"{c}_loading" for c in pc_cols])
            loadings.to_csv(os.path.join(output_path, f"PCA_loadings_{block_name}{suf}.csv"))
            # scores
            X_train_sig.to_csv(os.path.join(output_path, f"PCA_scores_{block_name}_TRAIN{suf}.csv"))
            if X_test_sig is not None:
                X_test_sig.to_csv(os.path.join(output_path, f"PCA_scores_{block_name}_TEST{suf}.csv"))
            # explained variance
            pd.DataFrame({
                "explained_variance": pca.explained_variance_,
                "explained_variance_ratio": pca.explained_variance_ratio_,
            }, index=pc_cols).to_csv(os.path.join(output_path, f"PCA_explained_variance_{block_name}{suf}.csv"))

        return X_train_sig, X_test_sig

    elif process_type == "sig_lax":
        print("🧪 Significance-based feature selection (fit on TRAIN only)...")

        # Decide grouping mode based on y_series
        if y_series is None or y_series.isna().all():
            print("ℹ️ No labels provided; using variance fallback.")
            sig_features = _fallback_unsupervised(X_train, top_k=10)
        else:
            uniques = pd.unique(y_series.dropna())
            if np.all(np.isin(uniques, [0, 1, 2])):  # risk tertiles → compare 0 vs 2
                mask_ext = y_series.isin([0, 2])
                y_ext = y_series[mask_ext]
                from collections import Counter
                print(f"[{block_name}] Risk extremes counts:", dict(Counter(y_ext)))
                if set(pd.unique(y_ext)) == {0, 2} and y_ext.value_counts().min() >= min_per_group:
                    meta_df = pd.DataFrame({"group": y_ext.map({0: "low", 2: "high"})}, index=y_ext.index)
                    X_sel = X_train.loc[y_ext.index]
                    from dataprocessing_utils import run_significance_tests
                    sig_results = run_significance_tests(
                        data_df=X_sel, meta_df=meta_df,
                        group_col='group', group1='high', group2='low',
                        use_mannwhitney=True
                    )
                    sig_features = _safe_extract_sig_features(sig_results)
                    if not sig_features:
                        print(f"⚠️ No features after threshold; taking top {fallback_top_k} by p.")
                else:
                    print("❌ No features tested — risk extremes too small; using variance fallback.")
                    sig_features = _fallback_unsupervised(X_train, top_k=10)
            else:
                # recurrence mode
                y_norm = y_series.copy()
                if set(pd.unique(y_norm.dropna())) <= {0, 1}:
                    y_norm = y_norm.map({1: "recurrent", 0: "non-recurrent"})
                vc = y_norm.value_counts()
                print(f"[{block_name}] Recurrence counts:", vc.to_dict())
                if {"recurrent", "non-recurrent"}.issubset(set(vc.index)) and vc.min() >= min_per_group:
                    meta_df = pd.DataFrame({"group": y_norm}, index=y_norm.index)
                    from dataprocessing_utils import run_significance_tests
                    sig_results = run_significance_tests(
                        data_df=X_train, meta_df=meta_df,
                        group_col='group', group1='recurrent', group2='non-recurrent',
                        use_mannwhitney=True
                    )
                    sig_features = _safe_extract_sig_features(sig_results)
                    if not sig_features:
                        print(f"⚠️ No features after threshold; taking top {fallback_top_k} by p.")
                else:
                    print("❌ No features tested — recurrence groups invalid; using variance fallback.")
                    sig_features = _fallback_unsupervised(X_train, top_k=50)

        # keep only features present
        sig_features = [f for f in sig_features if f in X_train.columns]
        if not sig_features:
            print("⚠️ No usable features after selection; using variance fallback.")
            sig_features = _fallback_unsupervised(X_train, top_k=50)

        X_train_sig = X_train.loc[:, sig_features].copy()
        X_test_sig  = (None if X_test is None else X_test.reindex(columns=sig_features, fill_value=0.0).copy())

        fitted_scaler = None
        if scale:
            fitted_scaler = StandardScaler()
            X_train_sig.loc[:, :] = fitted_scaler.fit_transform(X_train_sig)
            if X_test_sig is not None:
                X_test_sig.loc[:, :] = fitted_scaler.transform(X_test_sig)

        # Save bundle
        _save_bundle({
            "process_type": "sig_lax",
            "block_name": block_name,
            "fold": fold,
            "columns": sig_features,       # selected features (order)
            "scaler": fitted_scaler,       # may be None if scale=False
            "pca": None,
            "selected_features": sig_features,
            "top_k_sig": top_k_sig,
            "min_per_group": min_per_group,
            "p_threshold": p_threshold,        # NEW
            "fallback_top_k": fallback_top_k,  # NEW
        })

        return X_train_sig, X_test_sig
   
    elif process_type == "as_is":
        print("🧪 Using features as-is (fit scaler on TRAIN only if scale=True)...")

        def _coerce_numeric(df):
            return (df.apply(pd.to_numeric, errors='coerce')
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .astype(float))

        X_train_sig = _coerce_numeric(X_train)
        X_test_sig  = None if X_test is None else _coerce_numeric(X_test)

        # Align test columns to train
        if X_test_sig is not None:
            X_test_sig = X_test_sig.reindex(columns=X_train_sig.columns, fill_value=0.0)

        fitted_scaler = None

        # --- Conditional scaling logic ---
        if scale:
            if block_name == "MUTblock":
                # Only scale these if present; otherwise skip scaling entirely
                target_cols = ["MUTblock__TMB", "MUTblock__pathogenic_mutation"]
                scale_cols = [c for c in target_cols if c in X_train_sig.columns]

                if len(scale_cols) > 0:
                    fitted_scaler = StandardScaler()
                    # Fit on TRAIN subset
                    X_train_sig.loc[:, scale_cols] = fitted_scaler.fit_transform(X_train_sig[scale_cols])

                    if X_test_sig is not None:
                        # Transform TEST subset (same column order)
                        X_test_sig.loc[:, scale_cols] = fitted_scaler.transform(X_test_sig[scale_cols])

                    print(f"🔧 Scaled MUTblock columns: {scale_cols}")
                else:
                    print("ℹ️ MUTblock target columns not found; skipping scaling for this block.")
                    fitted_scaler = None
            else:
                # Original behavior for non-MUT blocks: scale all columns
                fitted_scaler = StandardScaler()
                X_train_sig.loc[:, :] = fitted_scaler.fit_transform(X_train_sig)
                if X_test_sig is not None:
                    X_test_sig.loc[:, :] = fitted_scaler.transform(X_test_sig)
                print("🔧 Scaled all columns (non-MUT block).")

        # Save bundle (record which columns were scaled if MUTblock)
        scaled_columns = None
        if fitted_scaler is not None:
            if block_name == "MUTblock":
                scaled_columns = [c for c in ["MUTblock__TMB", "MUTblock__pathogenic_mutation"] if c in X_train_sig.columns]
            else:
                scaled_columns = list(X_train_sig.columns)

        _save_bundle({
            "process_type": "as_is",
            "block_name": block_name,
            "fold": fold,
            "columns": list(X_train_sig.columns),
            "scaler": fitted_scaler,          # may be None
            "scaled_columns": scaled_columns, # may be None
            "pca": None,
        })

        return X_train_sig, X_test_sig



    elif process_type == "clinical":
        print("🧪 Converting Clinical Features (fit on TRAIN only)...")
        # from dataprocessing_utils import encode_clinical_features

        # train
        X_train_clinical_df = encode_clinical_features(X_train)
        X_train_sig = X_train_clinical_df.astype({
            col: int for col in X_train_clinical_df.select_dtypes(bool).columns
        })
        X_train_sig = X_train_sig.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)

        # test
        X_test_sig = None
        if X_test is not None:
            X_test_clinical_df = encode_clinical_features(X_test)
            X_test_sig = X_test_clinical_df.astype({
                col: int for col in X_test_clinical_df.select_dtypes(bool).columns
            })
            X_test_sig = X_test_sig.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)
            X_test_sig = X_test_sig.reindex(columns=X_train_sig.columns, fill_value=0.0)

        # Save bundle (no scaler/pca by default)
        _save_bundle({
            "process_type": "clinical",
            "block_name": block_name,
            "fold": fold,
            "columns": list(X_train_sig.columns),
            "scaler": None,
            "pca": None,
        })

        return X_train_sig, X_test_sig

    else:
        raise ValueError(
            f"Unknown process_type '{process_type}'. Expected one of: ['pca','sig_lax','as_is','clinical']."
        )



\
import os

def make_output_path(output_dir, subtype=None, include_subtype=False):
    """Create and return the base output directory.
    If include_subtype=True, also create/append the subtype subfolder.
    """
    base = os.path.abspath(output_dir)
    os.makedirs(base, exist_ok=True)
    if include_subtype and subtype:
        base = os.path.join(base, subtype)
        os.makedirs(base, exist_ok=True)
    return base



def get_model(name):
    if name == 'Naive Bayes':
        return GaussianNB()
    elif 'Logistic' in name:
        return LogisticRegression(max_iter=1000, solver='liblinear')
    elif 'Random Forest' in name:
        return RandomForestClassifier(random_state=42)
    elif 'Gradient Boosting' in name:
        return GradientBoostingClassifier()
    elif 'XGBoost' in name:
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    elif 'LightGBM' in name:
        return LGBMClassifier()
    elif 'CatBoost' in name:
        return CatBoostClassifier(verbose=0)
    elif 'SVM' in name:
        return SVC(probability=True,max_iter=1000)
    else:
        raise ValueError(f"Unknown model: {name}")

def run_stacked_cv_blocks_only(data, meta, y, label_encoder, all_pathway_dicts, pathway_options,
                                gsva_scores_rna_all, gsva_scores_cnv_all, model_grid,
                                subtype, target_col='Disease_Status',
                                outer_folds=5, inner_folds=3, fold_output_dir=None):
    import numpy as np
    import pandas as pd
    import shap
    import matplotlib.pyplot as plt
    import os
    import joblib
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.base import clone

        
    # Output directories
    if fold_output_dir:
        os.makedirs(fold_output_dir, exist_ok=True)
        fold_output_dir = os.path.join(fold_output_dir, subtype)

        training_data_dir = os.path.join(fold_output_dir, "training_data")
        roc_dir = os.path.join(fold_output_dir, "roc_plots")
        cv_models_dir = os.path.join(fold_output_dir, "cv_models")
        preprocessing_dir = os.path.join(fold_output_dir, "preprocessing")
        shap_dir = os.path.join(fold_output_dir, "shap")
        coef_dir = os.path.join(fold_output_dir, "coefficients")
        final_model_dir = os.path.join(fold_output_dir, "final_models")
        for d in [training_data_dir, roc_dir, cv_models_dir, preprocessing_dir, shap_dir, coef_dir, final_model_dir]:
            os.makedirs(d, exist_ok=True)

    y_encoded = label_encoder.transform(y)
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)

    oof_meta_features = []
    oof_true_labels = []
    block_auc_tracking = {}
    best_models_by_block = {}
    block_roc_data = {}

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(meta, meta[target_col])):
        print(f"\n🔁 Outer Fold {fold + 1}/{outer_folds}")


        train_ids = meta.iloc[train_idx].index.astype(str).str.strip()
        test_ids = meta.iloc[test_idx].index.astype(str).str.strip()

        subtype_meta_train = meta.loc[train_ids].copy()
        subtype_meta_test = meta.loc[test_ids].copy()

        y_train = y.loc[train_ids]
        y_test = y.loc[test_ids]
        y_train_enc = pd.Series(label_encoder.transform(y_train), index=y_train.index)
        y_test_enc = label_encoder.transform(y_test)  # For roc_curve later

        mutation_pathways_train = process_mutation_pathways(
            data['mut'].loc[train_ids], subtype_meta_train, all_pathway_dicts,
            pathway_options, subtype=subtype, output_dir=None
        )
        #mutation_pathways_test = mutation_pathways_train.reindex(test_ids).fillna(0)
        significant_mutation_features = mutation_pathways_train.columns
        mutation_pathways_test = data['mut'].loc[test_ids].reindex(columns=significant_mutation_features).fillna(0)

        rna_results = run_significance_tests(gsva_scores_rna_all.loc[train_ids], subtype_meta_train,
                                             group_col='Disease_Status', group1='recurrent', group2='non-recurrent', use_mannwhitney=True)
        # sig_rna = rna_results[rna_results['p_value'] <= 0.015]['feature']
        # rna_pathways = gsva_scores_rna_all.loc[train_ids, sig_rna]
        #         # Get significant features or top 10 by p-value
        sig_rna = rna_results[rna_results['p_value'] <= 0.015]['feature']
        if sig_rna.empty:
            sig_rna = rna_results.nsmallest(10, 'p_value')['feature']

        # Subset the GSVA matrix
        rna_pathways = gsva_scores_rna_all.loc[train_ids, sig_rna]



        cnv_results = run_significance_tests(gsva_scores_cnv_all.loc[train_ids], subtype_meta_train,
                                             group_col='Disease_Status', group1='recurrent', group2='non-recurrent', use_mannwhitney=True)
        # sig_cnv = cnv_results[cnv_results['p_value'] <= 0.015]['feature']
        # cnv_pathways = gsva_scores_cnv_all.loc[train_ids, sig_cnv]
        sig_cnv = cnv_results[cnv_results['p_value'] <= 0.015]['feature']
        if sig_cnv.empty:
            sig_cnv = cnv_results.nsmallest(10, 'p_value')['feature']

        # Subset the GSVA matrix
        cnv_pathways = gsva_scores_cnv_all.loc[train_ids, sig_cnv]



        subtype_meta_train['Disease_Status'] = pd.Categorical(subtype_meta_train['Disease_Status'],
                                                              categories=["non-recurrent", "recurrent"])

        de_results = process_deseq_results(
            rna_expression=data['rna_raw'].loc[train_ids],
            metadata=subtype_meta_train[['Disease_Status']],
            design_factors=['Disease_Status'],
            contrast=["Disease_Status", "recurrent", "non-recurrent"],
            pvalue_threshold=0.01,
            padj_threshold=0.25,
            logfold_threshold=1,
            base_mean_threshold=50,
            gene_sum_threshold=50
        )
        rna_log2 = np.log2(data['rna'] + 1)
        de_genes = [g for g in de_results.index if g in rna_log2.columns]
        de_expr_train = rna_log2.loc[train_ids, de_genes]
        de_expr_test = rna_log2.loc[test_ids, de_genes]

        X_train_full, X_train_blocks = merge_feature_blocks(
            subtype_meta=subtype_meta_train,
            mutation_df=mutation_pathways_train,
            rna_df=rna_pathways,
            cnv_df=cnv_pathways,
            tf_df=data['tf'].loc[train_ids],
            tmb_df=data['tmb'].loc[train_ids],
            hoadley_df=data['hoadley'].loc[train_ids],
            cibersort_df=data['cibersort'].loc[train_ids],
            pathogenic_df=data['pathogenic_count'].loc[train_ids],
            cnv_pct_df=data['cnv_pct_altered'].loc[train_ids],
            clinical_df=data['clinical'].loc[train_ids],
            de_gene_df=de_expr_train,
            return_separate_blocks=True
        )
        X_test_full, X_test_blocks = merge_feature_blocks(
            subtype_meta=subtype_meta_test,
            mutation_df=mutation_pathways_test,
            rna_df=gsva_scores_rna_all.loc[test_ids, sig_rna],
            cnv_df=gsva_scores_cnv_all.loc[test_ids, sig_cnv],
            tf_df=data['tf'].loc[test_ids],
            tmb_df=data['tmb'].loc[test_ids],
            hoadley_df=data['hoadley'].loc[test_ids],
            cibersort_df=data['cibersort'].loc[test_ids],
            pathogenic_df=data['pathogenic_count'].loc[test_ids],
            cnv_pct_df=data['cnv_pct_altered'].loc[test_ids],
            clinical_df=data['clinical'].loc[test_ids],
            de_gene_df=de_expr_test,
            return_separate_blocks=True
        )

        meta_features_fold = []

        for block_name, block_info in X_train_blocks.items():
            X_block_train = block_info["data"]
            X_block_test = X_test_blocks[block_name]["data"]
            process_type = block_info["process"]

            print(f"\n📦 Block: {block_name} | Train shape: {X_block_train.shape} | Test shape: {X_block_test.shape}")
            print(f"Non-zero entries in training set: {(X_block_train != 0).sum().sum()}")

            X_train_proc, X_test_proc = preprocess_block(
                X_block_train, X_block_test, process_type, block_name, y_train, output_path=fold_output_dir, fold=fold
            )

            best_model, model_type = tune_model(X_train_proc, y_train_enc.loc[X_train_proc.index], model_grid, cv=inner_folds)
            y_test_pred = best_model.predict_proba(X_test_proc)[:, 1]

            fpr, tpr, _ = roc_curve(y_test_enc, y_test_pred)
            auc = roc_auc_score(y_test_enc, y_test_pred)
            block_roc_data.setdefault(block_name, []).append((fpr, tpr, auc))

            joblib.dump(best_model, os.path.join(cv_models_dir, f"{block_name}_fold{fold + 1}.pkl"))
            print(f"  📎 Saved CV model for block '{block_name}' fold {fold + 1}")
            print(f"  🔍 Selected model type: {model_type} | Coef available: {hasattr(best_model, 'coef_')}")

            explainer = shap.Explainer(best_model.predict_proba, X_train_proc)
            shap_values = explainer(X_train_proc)

            shap_array = shap_values.values
            if shap_array.ndim == 3 and shap_array.shape[2] == 2:
                shap_array = shap_array[:, :, 1]
            shap_df = pd.DataFrame(shap_array, columns=X_train_proc.columns)
            shap_df.to_csv(os.path.join(shap_dir, f"shap_{block_name}_fold{fold + 1}_{model_type}.csv"))
            joblib.dump(shap_values, os.path.join(shap_dir, f"shap_{block_name}_fold{fold + 1}_{model_type}.pkl"))

            if hasattr(best_model, 'coef_'):
                coefs = pd.Series(best_model.coef_[0], index=X_train_proc.columns)
                coefs.to_csv(os.path.join(coef_dir, f"coef_{block_name}_fold{fold + 1}_{model_type}.csv"))

            best_models_by_block[block_name] = (model_type, best_model)
            meta_features_fold.append(y_test_pred)

        if meta_features_fold:
            oof_meta_features.append(np.vstack(meta_features_fold).T)
            oof_true_labels.append(y_test_enc)

    # === Save individual ROC curves per block ===
    for block_name, curves in block_roc_data.items():
        plt.figure()
        for i, (fpr, tpr, auc) in enumerate(curves):
            plt.plot(fpr, tpr, label=f"Fold {i+1} AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title(f"ROC Curve for Block: {block_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(os.path.join(roc_dir, f"roc_block_{block_name}.png"))
        plt.close()

    # === Plot mean ROC curves for all blocks on one shared plot ===
    plt.figure(figsize=(8, 6))
    mean_fpr = np.linspace(0, 1, 100)

    for block_name, curves in block_roc_data.items():
        tprs = []
        aucs = []
        for fpr, tpr, auc in curves:
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        plt.plot(mean_fpr, mean_tpr, label=f"{block_name} (AUC = {mean_auc:.2f})", lw=2)

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Mean ROC Curves Across Folds by Block")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    combined_roc_path = os.path.join(roc_dir, "combined_block_roc_curves.png")
    plt.savefig(combined_roc_path)
    plt.close()
    print(f"📈 Saved combined ROC curve for all blocks: {combined_roc_path}")

    return (
        np.vstack(oof_meta_features) if oof_meta_features else np.array([]),
        np.concatenate(oof_true_labels) if oof_true_labels else np.array([]),
        best_models_by_block
    )

from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from collections import Counter

def get_safe_cv(y, desired_folds=3, min_class_size=5):
    """
    Return a safe cross-validation splitter depending on class distribution.
    Uses LeaveOneOut if any class has fewer than `min_class_size` samples.
    """
    counts = Counter(y)
    if min(counts.values()) < min_class_size:
        print(f"⚠️ Too few samples in at least one class (<{min_class_size}); using Leave-One-Out CV")
        return LeaveOneOut()
    return StratifiedKFold(n_splits=desired_folds, shuffle=True, random_state=42)





import shap
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



def get_shap_explainer(model, X, max_background=50):
    import numpy as np
    import shap
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier

    # Ensure numpy array for background selection
    X_array = X.values if hasattr(X, "values") else np.asarray(X)

    # Prefer a small kmeans background for KernelExplainer (speed)
    n_bg = min(max_background, len(X_array))
    try:
        background = shap.kmeans(X_array, n_bg) if len(X_array) > n_bg else X_array
    except Exception:
        background = X_array[:n_bg]

    # Tree-based → TreeExplainer
    if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)) or hasattr(model, "feature_importances_"):
        return shap.TreeExplainer(model)

    # Linear models → LinearExplainer
    if isinstance(model, LogisticRegression) and hasattr(model, "coef_"):
        try:
            return shap.LinearExplainer(model, X_array, feature_perturbation="interventional")
        except Exception:
            # older SHAP fallback
            return shap.LinearExplainer(model, X_array)

    # Black-box (SVC/MLP/NB/etc.) → KernelExplainer on class-1 proba if available
    if hasattr(model, "predict_proba"):
        f = lambda data: model.predict_proba(data)[:, 1]
    elif hasattr(model, "decision_function"):
        f = lambda data: model.decision_function(data)
    else:
        f = lambda data: model.predict(data)

    return shap.KernelExplainer(f, background)


def compute_shap_values(explainer, X):
    """Works for both new-style (__call__) and KernelExplainer (shap_values)."""
    try:
        res = explainer(X)
        return getattr(res, "values", res)
    except TypeError:
        # KernelExplainer in some versions
        return explainer.shap_values(X)


def run_stacked_cv_blocks_risk_assessment(data, meta, y, label_encoder, all_pathway_dicts, pathway_options,
                                gsva_scores_rna_all, gsva_scores_cnv_all, model_grid,
                                subtype, target_col='Disease_Status',
                                outer_folds=5, inner_folds=3, fold_output_dir=None):
    import numpy as np
    import pandas as pd
    import shap
    import matplotlib.pyplot as plt
    import os
    import joblib
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if fold_output_dir:
        os.makedirs(fold_output_dir, exist_ok=True)
        fold_output_dir = os.path.join(fold_output_dir, subtype)

        training_data_dir = os.path.join(fold_output_dir, "training_data")
        roc_dir = os.path.join(fold_output_dir, "roc_plots")
        cv_models_dir = os.path.join(fold_output_dir, "cv_models")
        preprocessing_dir = os.path.join(fold_output_dir, "preprocessing")
        shap_dir = os.path.join(fold_output_dir, "shap")
        coef_dir = os.path.join(fold_output_dir, "coefficients")
        final_model_dir = os.path.join(fold_output_dir, "final_models")
        for d in [training_data_dir, roc_dir, cv_models_dir, preprocessing_dir, shap_dir, coef_dir, final_model_dir]:
            os.makedirs(d, exist_ok=True)

    y_encoded = label_encoder.transform(y)
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)

    oof_meta_features = []
    oof_true_labels = []
    block_auc_tracking = {}
    best_models_by_block = {}
    block_roc_data = {}

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(meta, meta[target_col])):
        print(f"\n🔁 Outer Fold {fold + 1}/{outer_folds}")

        train_ids = meta.iloc[train_idx].index.astype(str).str.strip()
        test_ids = meta.iloc[test_idx].index.astype(str).str.strip()

        subtype_meta_train = meta.loc[train_ids].copy()
        subtype_meta_test = meta.loc[test_ids].copy()

        y_train = y.loc[train_ids]
        y_test = y.loc[test_ids]
        y_train_enc = pd.Series(label_encoder.transform(y_train), index=y_train.index)
        y_test_enc = label_encoder.transform(y_test)

        mutation_pathways_train = process_mutation_pathways(
            data['mut'].loc[train_ids], subtype_meta_train, all_pathway_dicts,
            pathway_options, subtype=subtype, output_dir=None
        )
        significant_mutation_features = mutation_pathways_train.columns
        mutation_pathways_test = data['mut'].loc[test_ids].reindex(columns=significant_mutation_features).fillna(0)

        rna_results = run_significance_tests(gsva_scores_rna_all.loc[train_ids], subtype_meta_train,
                                             group_col='Disease_Status', group1='recurrent', group2='non-recurrent', use_mannwhitney=True)
        sig_rna = rna_results[rna_results['p_value'] <= 0.05]['feature']
        if sig_rna.empty:
            sig_rna = rna_results.nsmallest(10, 'p_value')['feature']
        rna_pathways = gsva_scores_rna_all.loc[train_ids, sig_rna]

        cnv_results = run_significance_tests(gsva_scores_cnv_all.loc[train_ids], subtype_meta_train,
                                             group_col='Disease_Status', group1='recurrent', group2='non-recurrent', use_mannwhitney=True)
        sig_cnv = cnv_results[cnv_results['p_value'] <= 0.05]['feature']
        if sig_cnv.empty:
            sig_cnv = cnv_results.nsmallest(10, 'p_value')['feature']
        cnv_pathways = gsva_scores_cnv_all.loc[train_ids, sig_cnv]

        subtype_meta_train['Disease_Status'] = pd.Categorical(subtype_meta_train['Disease_Status'],
                                                              categories=["non-recurrent", "recurrent"])

        de_results = process_deseq_results(
            rna_expression=data['rna_raw'].loc[train_ids],
            metadata=subtype_meta_train[['Disease_Status']],
            design_factors=['Disease_Status'],
            contrast=["Disease_Status", "recurrent", "non-recurrent"],
            pvalue_threshold=0.05,
            padj_threshold=0.25,
            logfold_threshold=1,
            base_mean_threshold=50,
            gene_sum_threshold=50
        )
        rna_log2 = np.log2(data['rna'] + 1)
        de_genes = [g for g in de_results.index if g in rna_log2.columns]
        de_expr_train = rna_log2.loc[train_ids, de_genes]
        de_expr_test = rna_log2.loc[test_ids, de_genes]

        X_train_full, X_train_blocks = merge_feature_blocks(
            subtype_meta=subtype_meta_train,
            mutation_df=mutation_pathways_train,
            rna_df=rna_pathways,
            cnv_df=cnv_pathways,
            tf_df=data['tf'].loc[train_ids],
            tmb_df=data['tmb'].loc[train_ids],
            hoadley_df=data['hoadley'].loc[train_ids],
            cibersort_df=data['cibersort'].loc[train_ids],
            pathogenic_df=data['pathogenic_count'].loc[train_ids],
            cnv_pct_df=data['cnv_pct_altered'].loc[train_ids],
            clinical_df=data['clinical'].loc[train_ids],
            de_gene_df=de_expr_train,
            return_separate_blocks=True
        )

        X_test_full, X_test_blocks = merge_feature_blocks(
            subtype_meta=subtype_meta_test,
            mutation_df=mutation_pathways_test,
            rna_df=gsva_scores_rna_all.loc[test_ids, sig_rna],
            cnv_df=gsva_scores_cnv_all.loc[test_ids, sig_cnv],
            tf_df=data['tf'].loc[test_ids],
            tmb_df=data['tmb'].loc[test_ids],
            hoadley_df=data['hoadley'].loc[test_ids],
            cibersort_df=data['cibersort'].loc[test_ids],
            pathogenic_df=data['pathogenic_count'].loc[test_ids],
            cnv_pct_df=data['cnv_pct_altered'].loc[test_ids],
            clinical_df=data['clinical'].loc[test_ids],
            de_gene_df=de_expr_test,
            return_separate_blocks=True
        )

        meta_features_fold = []

        for block_name, block_info in X_train_blocks.items():
            X_block_train = block_info["data"]
            X_block_test = X_test_blocks[block_name]["data"]
            process_type = block_info["process"]

            print(f"\n📦 Block: {block_name} | Train shape: {X_block_train.shape} | Test shape: {X_block_test.shape}")
            print(f"Non-zero entries in training set: {(X_block_train != 0).sum().sum()}")

            # Preprocess block
            X_train_proc, X_test_proc = preprocess_block(
                X_block_train, X_block_test, process_type, block_name, y_train, output_path=fold_output_dir, fold=fold
            )

            # # === Train block-specific risk classifier ===
            # scaler = StandardScaler()
            # X_scaled = pd.DataFrame(scaler.fit_transform(X_train_proc), index=X_train_proc.index, columns=X_train_proc.columns)

            init_clf = LogisticRegression(max_iter=1000, random_state=42)
            init_clf.fit(X_train_proc, y_train_enc.loc[X_train_proc.index])
            risk_probs = init_clf.predict_proba(X_train_proc)[:, 1]

            risk_df = pd.DataFrame({
                "Predicted_Risk": risk_probs,
                "True_Label": y_train_enc.loc[X_train_proc.index]
            }, index=X_train_proc.index)

            high_risk_df = risk_df[risk_df["Predicted_Risk"] >= 0.8]
            low_risk_df = risk_df[risk_df["Predicted_Risk"] <= 0.2]
            confident_ids = high_risk_df.index.union(low_risk_df.index)

            print(f"🧪 Block-specific confident samples: {len(confident_ids)} ({len(high_risk_df)} high-risk, {len(low_risk_df)} low-risk)")

            # === Step 1: Select confident samples ===
            if confident_ids.empty:
                print(f"⚠️ No confident samples for block '{block_name}'. Training on all training samples.")
                X_train_conf = X_train_proc
                y_train_conf = y_train_enc
            else:
                X_train_conf = X_train_proc.loc[confident_ids]
                y_train_conf = y_train_enc.loc[confident_ids]

            # === Step 2: Check class balance in confident samples ===
            MIN_CONFIDENT_CLASS_SIZE = 5

            class_counts = Counter(y_train_conf)
            if len(class_counts) < 2 or any(v < MIN_CONFIDENT_CLASS_SIZE for v in class_counts.values()):
                print(f"⚠️ Confident sample set for block '{block_name}' has fewer than {MIN_CONFIDENT_CLASS_SIZE} samples in a class. Using full training data.")
                X_train_conf = X_train_proc
                y_train_conf = y_train_enc


            best_model, model_type, best_auc, best_params = tune_model(X_train_proc, y_train_enc, model_grid, cv=safe_cv)
            print(best_auc)
            print(best_params)

            safe_cv = get_safe_cv(y_train_conf, desired_folds=inner_folds)
            best_model, model_type = tune_model(X_train_conf, y_train_conf, model_grid, cv=safe_cv)

            
            y_test_pred = best_model.predict_proba(X_test_proc)[:, 1]
            y_pred_labels = (y_test_pred >= 0.5).astype(int)


            from sklearn.metrics import roc_curve, roc_auc_score

            # Compute ROC metrics
            fpr, tpr, thresholds = roc_curve(y_test_enc, y_test_pred)
            auc = roc_auc_score(y_test_enc, y_test_pred)

            # Track in-memory (optional aggregate use)
            block_roc_data.setdefault(block_name, []).append((fpr, tpr, auc))

            # Save ROC curve data to CSV
            roc_df = pd.DataFrame({
                'fpr': fpr,
                'tpr': tpr,
                'threshold': thresholds,
                'auc': auc  # repeated for convenience
            })
            roc_path = os.path.join(cv_models_dir, f"{block_name}_fold{fold + 1}_roc.csv")
            roc_df.to_csv(roc_path, index=False)



            joblib.dump(best_model, os.path.join(cv_models_dir, f"{block_name}_fold{fold + 1}.pkl"))
            # Save model metadata (type + hyperparameters)
            model_txt_path = os.path.join(cv_models_dir, f"{block_name}_fold{fold + 1}_info.txt")
            with open(model_txt_path, 'w') as f:
                f.write(f"Model Type: {model_type}\n")
                f.write("Hyperparameters:\n")
                if hasattr(best_model, "get_params"):
                    for k, v in best_model.get_params().items():
                        f.write(f"  {k}: {v}\n")

            print(f"  📎 Saved CV model for block '{block_name}' fold {fold + 1}")
            print(f"  🔍 Selected model type: {model_type} | Coef available: {hasattr(best_model, 'coef_')}")

            # ⬇️ Record metrics for this fold of this block
            metrics = {
                'block_name': block_name,
                'fold': fold + 1,
                'auc': roc_auc_score(y_test_enc, y_test_pred),
                'accuracy': accuracy_score(y_test_enc, y_pred_labels),
                'precision': precision_score(y_test_enc, y_pred_labels, zero_division=0),
                'recall': recall_score(y_test_enc, y_pred_labels, zero_division=0),
                'f1': f1_score(y_test_enc, y_pred_labels, zero_division=0),
                'n_test_samples': len(y_test_enc),
                'model_name': type(best_model).__name__,
                'params': str(best_model.get_params())
            }
            performance_metrics.append(metrics)

            explainer = get_shap_explainer(best_model, X_train_final)
            shap_vals = compute_shap_values(explainer, X_train_final)

            # If binary classifier returns a list (older SHAP), take class 1
            import numpy as np
            if isinstance(shap_vals, list) and len(shap_vals) == 2:
                shap_vals = shap_vals[1]

            shap_df = pd.DataFrame(shap_vals, columns=X_train_final.columns, index=X_train_final.index)


            # shap_df = pd.DataFrame(shap_array, columns=X_train_conf.columns)
            shap_df.to_csv(os.path.join(shap_dir, f"shap_{block_name}_fold{fold + 1}_{model_type}.csv"))
            joblib.dump(shap_values, os.path.join(shap_dir, f"shap_{block_name}_fold{fold + 1}_{model_type}.pkl"))

            if hasattr(best_model, 'coef_'):
                coefs = pd.Series(best_model.coef_[0], index=X_train_conf.columns)
                coefs.to_csv(os.path.join(coef_dir, f"coef_{block_name}_fold{fold + 1}_{model_type}.csv"))

            best_models_by_block[block_name] = (model_type, best_model)
            meta_features_fold.append(y_test_pred)

        if meta_features_fold:
            oof_meta_features.append(np.vstack(meta_features_fold).T)
            oof_true_labels.append(y_test_enc)

    # Save ROC plots
    for block_name, curves in block_roc_data.items():
        plt.figure()
        for i, (fpr, tpr, auc) in enumerate(curves):
            plt.plot(fpr, tpr, label=f"Fold {i+1} AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title(f"ROC Curve for Block: {block_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(os.path.join(roc_dir, f"roc_block_{block_name}.png"))
        plt.close()

    # Combined plot
    plt.figure(figsize=(8, 6))
    mean_fpr = np.linspace(0, 1, 100)
    for block_name, curves in block_roc_data.items():
        tprs, aucs = [], []
        for fpr, tpr, auc in curves:
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        plt.plot(mean_fpr, mean_tpr, label=f"{block_name} (AUC = {mean_auc:.2f})", lw=2)

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Mean ROC Curves Across Folds by Block")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    combined_roc_path = os.path.join(roc_dir, "combined_block_roc_curves.png")
    plt.savefig(combined_roc_path)
    plt.close()
    print(f"📈 Saved combined ROC curve for all blocks: {combined_roc_path}")

    return (
        np.vstack(oof_meta_features) if oof_meta_features else np.array([]),
        np.concatenate(oof_true_labels) if oof_true_labels else np.array([]),
        best_models_by_block
    )



#from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

def run_rsf_risk_classifier(
    X_train_concat,
    subtype_meta_train,
    top_pct=0.30,
    low_pct=0.30,
    random_state=42
):
    """
    Fit a Random Survival Forest (RSF) and return high/low risk IDs using RSF's overall risk (predict()).

    Parameters
    ----------
    X_train_concat : pd.DataFrame
        Feature matrix (rows = samples, cols = features).
    subtype_meta_train : pd.DataFrame
        Must contain 'time' and 'event' columns (event: 1=event, 0=censored).
    top_pct : float
        Proportion to label as high-risk, e.g., 0.30 = top 30%.
    low_pct : float
        Proportion to label as low-risk, e.g., 0.30 = bottom 30%.
    random_state : int
        RSF random seed.

    Returns
    -------
    high_risk_ids : pd.Index
    low_risk_ids  : pd.Index
    risk_scores   : pd.Series
        RSF overall risk (higher = worse).
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GridSearchCV
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    from sksurv.metrics import concordance_index_censored

    # --- 1) Prepare survival targets ---
    assert {'time','event'}.issubset(subtype_meta_train.columns), "meta needs 'time' and 'event'."
    y_struct = Surv.from_dataframe(event='event', time='time', data=subtype_meta_train.loc[X_train_concat.index])

    # --- 2) Define RSF + c-index scorer (ranking-focused) ---
    rsf = RandomSurvivalForest(random_state=random_state)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt", "log2"],
    }

    def cindex_scorer(estimator, X, y_structured):
        # RSF.predict(X) returns a risk score (higher = worse)
        risk = estimator.predict(X)
        e = y_structured["event"]
        t = y_structured["time"]
        return concordance_index_censored(e, t, risk)[0]

    print("🔍 Grid-searching RSF by c-index...")
    grid = GridSearchCV(
        estimator=rsf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring=cindex_scorer
    )
    grid.fit(X_train_concat, y_struct)
    best_rsf = grid.best_estimator_
    print(f"✅ Best RSF params: {grid.best_params_} | best c-index: {grid.best_score_:.3f}")

    # --- 3) RSF overall risk scores (cumulative hazard–based) ---
    risk_scores = pd.Series(best_rsf.predict(X_train_concat), index=X_train_concat.index)

    # --- 4) Label top/bottom percentiles ---
    top_cut  = risk_scores.quantile(1 - top_pct)
    low_cut  = risk_scores.quantile(low_pct)
    high_risk_ids = risk_scores.index[risk_scores >= top_cut]
    low_risk_ids  = risk_scores.index[risk_scores <= low_cut]

    print(f"📊 Top {int(top_pct*100)}% cutoff: {top_cut:.4f} → {len(high_risk_ids)} high-risk")
    print(f"📊 Bottom {int(low_pct*100)}% cutoff: {low_cut:.4f} → {len(low_risk_ids)} low-risk")

    return high_risk_ids, low_risk_ids, risk_scores


import os
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
import numpy as np



def run_stacked_cv_blocks_risk_assessment_refinement(
    data, meta, y, label_encoder, all_pathway_dicts, pathway_options,
    gsva_scores_rna_all, gsva_scores_cnv_all, model_grid,
    subtype, target_col='Disease_Status',
    outer_folds=5, inner_folds=3, fold_output_dir=None, selected_blocks=None,

):
    import numpy as np
    import pandas as pd
    import shap
    import matplotlib.pyplot as plt
    import os
    import joblib
    from sklearn.model_selection import StratifiedKFold, LeaveOneOut
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from collections import Counter
    from sklearn.neural_network import MLPClassifier


    # === Create output directories for saving models, plots, and SHAP results ===
    if fold_output_dir:
        os.makedirs(fold_output_dir, exist_ok=True)
        fold_output_dir = os.path.join(fold_output_dir, subtype)
        training_data_dir = os.path.join(fold_output_dir, "training_data")
        roc_dir = os.path.join(fold_output_dir, "roc_plots")
        cv_models_dir = os.path.join(fold_output_dir, "cv_models")
        preprocessing_dir = os.path.join(fold_output_dir, "preprocessing")
        shap_dir = os.path.join(fold_output_dir, "shap")
        coef_dir = os.path.join(fold_output_dir, "coefficients")
        for d in [training_data_dir, roc_dir, cv_models_dir, preprocessing_dir, shap_dir, coef_dir]:
            os.makedirs(d, exist_ok=True)

    # === Encode labels ===
    y_encoded = label_encoder.transform(y)

    # === Set up outer cross-validation ===
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)

    # === Initialize outputs for meta-model training and tracking ===
    oof_meta_features = []        # Out-of-fold predictions for meta-model
    oof_true_labels = []          # True labels for meta-model
    block_auc_tracking = {}       # AUC tracking per block
    best_models_by_block = {}     # Store best model for each block
    block_roc_data = {}           # Store ROC curve per block

    oof_predictions = pd.DataFrame(columns=['sample_id', 'true_label', 'predicted_proba', 'block_name'])
    training_predictions = pd.DataFrame(columns=['sample_id', 'true_label', 'predicted_proba', 'block_name'])
    performance_metrics =[]
    # === Outer CV loop ===
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(meta, meta[target_col])):
        print(f"\n🔁 Outer Fold {fold + 1}/{outer_folds}")

        # Split metadata and labels into train/test
        train_ids = meta.iloc[train_idx].index.astype(str).str.strip()
        test_ids = meta.iloc[test_idx].index.astype(str).str.strip()
        subtype_meta_train = meta.loc[train_ids].copy()
        subtype_meta_test = meta.loc[test_ids].copy()
        y_train = y.loc[train_ids]
        y_test = y.loc[test_ids]
        y_train_enc = pd.Series(label_encoder.transform(y_train), index=y_train.index)
        y_test_enc = label_encoder.transform(y_test)
        assert all(train_ids == y_train.index.astype(str).str.strip()), "❌ Misaligned training features and labels!"

        # === Step 1: Process mutation burden + significance for train set
        mutation_pathways_train, _ = process_mutation_pathways(
            M_matrix=data['mut'].loc[train_ids],
            subtype_meta=subtype_meta_train,
            pathway_dicts=all_pathway_dicts,
            pathway_options=pathway_options,
            output_dir=None,
            subtype=subtype
        )

        # === Step 2: Compute full burden matrix on test set (no significance testing needed)
        _, mutation_full_test = process_mutation_pathways(
            M_matrix=data['mut'].loc[test_ids],
            subtype_meta=meta.loc[test_ids].copy(),  # Just used to align indices; status column not needed
            pathway_dicts=all_pathway_dicts,
            pathway_options=pathway_options,
            output_dir=None,
            subtype=subtype
        )

        # === Step 3: Subset test burden matrix to significant features
        significant_mutation_features = mutation_pathways_train.columns
        mutation_pathways_test = mutation_full_test[significant_mutation_features].fillna(0)
      
        print("Mutation features:", significant_mutation_features)
        print("Non-zero features in X_train_final:", (mutation_pathways_train != 0).sum().sum())
        print("Non-zero features in X_test_proc:", (mutation_pathways_test != 0).sum().sum())


        # === Step 2: Run 1st round significance filtering for RNA GSVA pathways ===
        rna_results = run_significance_tests(gsva_scores_rna_all.loc[train_ids], subtype_meta_train,
                                             group_col='Disease_Status', group1='recurrent', group2='non-recurrent', use_mannwhitney=True)
        sig_rna = rna_results[rna_results['p_value'] <= 0.01]['feature']
        if sig_rna.empty:
            sig_rna = rna_results.nsmallest(5, 'p_value')['feature']
        rna_pathways_initial = gsva_scores_rna_all.loc[train_ids, sig_rna]

        # === Step 3: Run 1st round significance filtering for CNV GSVA pathways ===
        cnv_results = run_significance_tests(gsva_scores_cnv_all.loc[train_ids], subtype_meta_train,
                                             group_col='Disease_Status', group1='recurrent', group2='non-recurrent', use_mannwhitney=True)
        sig_cnv = cnv_results[cnv_results['p_value'] <= 0.01]['feature']
        if sig_cnv.empty:
            sig_cnv = cnv_results.nsmallest(5, 'p_value')['feature']
        cnv_pathways_initial = gsva_scores_cnv_all.loc[train_ids, sig_cnv]

        # === Step 4: Run DESeq2-style filtering for raw RNA ===
        subtype_meta_train['Disease_Status'] = pd.Categorical(subtype_meta_train['Disease_Status'], categories=["non-recurrent", "recurrent"])
        de_results = process_deseq_results(
            rna_expression=data['rna_raw'].loc[train_ids],
            metadata=subtype_meta_train[['Disease_Status']],
            design_factors=['Disease_Status'],
            contrast=["Disease_Status", "recurrent", "non-recurrent"],
            pvalue_threshold=0.01, padj_threshold=0.25, logfold_threshold=1.5,
            base_mean_threshold=100, gene_sum_threshold=100
        )
        rna_log2 = np.log2(data['rna'] + 1)
        de_genes = [g for g in de_results.index if g in rna_log2.columns]
        de_expr_train_initial = rna_log2.loc[train_ids, de_genes]
        de_expr_test = rna_log2.loc[test_ids, de_genes]

        # === Step 5: Merge features into blocks ===
        X_train_full, X_train_blocks = merge_feature_blocks(
            subtype_meta=subtype_meta_train,
            mutation_df=mutation_pathways_train,
            rna_df=rna_pathways_initial,
            cnv_df=cnv_pathways_initial,
            tf_df=data['tf'].loc[train_ids],
            tmb_df=data['tmb'].loc[train_ids],
            hoadley_df=data['hoadley'].loc[train_ids],
            cibersort_df=data['cibersort'].loc[train_ids],
            pathogenic_df=data['pathogenic_count'].loc[train_ids],
            cnv_pct_df=data['cnv_pct_altered'].loc[train_ids],
            clinical_df=data['clinical'].loc[train_ids],
            de_gene_df=de_expr_train_initial,
            return_separate_blocks=True
        )

        X_test_full, X_test_blocks = merge_feature_blocks(
            subtype_meta=subtype_meta_test,
            mutation_df=mutation_pathways_test,
            rna_df=gsva_scores_rna_all.loc[test_ids, sig_rna],
            cnv_df=gsva_scores_cnv_all.loc[test_ids, sig_cnv],
            tf_df=data['tf'].loc[test_ids],
            tmb_df=data['tmb'].loc[test_ids],
            hoadley_df=data['hoadley'].loc[test_ids],
            cibersort_df=data['cibersort'].loc[test_ids],
            pathogenic_df=data['pathogenic_count'].loc[test_ids],
            cnv_pct_df=data['cnv_pct_altered'].loc[test_ids],
            clinical_df=data['clinical'].loc[test_ids],
            de_gene_df=de_expr_test,
            return_separate_blocks=True
        )


        # ✅ Filter to selected blocks (both times after merge)
        if selected_blocks is not None:
            X_train_blocks = {k: v for k, v in X_train_blocks.items() if k in selected_blocks}
            X_test_blocks = {k: v for k, v in X_test_blocks.items() if k in selected_blocks}
            print(f"🔎 Using selected blocks only: {list(X_train_blocks.keys())}")

        meta_features_fold = []  # Collect per-block test predictions for this fold

        # === Step 6: Loop through each block and train model ===
        X_train_proc_blocks = {}
        X_test_proc_blocks = {}

        for block_name, block_info in X_train_blocks.items():
            print(f"\n🧪 Preprocessing Block: {block_name}")
            X_block_train = block_info['data']
            X_block_test = X_test_blocks[block_name]['data']
            process_type = block_info['process']


            X_train_proc, X_test_proc = preprocess_block(
                X_block_train, process_type, block_name,X_test=X_block_test,
                y_train=y_train, output_path=fold_output_dir, fold=fold
            )

            X_train_proc_blocks[block_name] = X_train_proc
            X_test_proc_blocks[block_name] = X_test_proc


        # === STEP 2: Concatenate all processed training blocks ===
        print("🔗 Concatenating all preprocessed training blocks...")
        X_train_concat = pd.concat(X_train_proc_blocks.values(), axis=1).fillna(0)

        print("⚙️ Running risk classifier for confident sample selection...")

        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import GridSearchCV
        import numpy as np
        import pandas as pd

        # ---- helper to always return P(positive_class) on the same scale ----
        def _proba_pos(est, X, pos_label=1):
            # If predict_proba exists, select the column that corresponds to pos_label
            if hasattr(est, "predict_proba"):
                P = est.predict_proba(X)
                # find the column index for pos_label (fallback to last column if not found)
                classes_ = getattr(est, "classes_", np.arange(P.shape[1]))
                idx = np.where(classes_ == pos_label)[0]
                j = int(idx[0]) if idx.size else (P.shape[1] - 1)
                return P[:, j]
            # Otherwise, use decision_function and min-max map to [0,1] (AUC is rank-based)
            if hasattr(est, "decision_function"):
                s = est.decision_function(X)
                s = (s - s.min()) / (s.max() - s.min() + 1e-9)
                return s
            raise ValueError("Estimator exposes neither predict_proba nor decision_function.")

        # sanity: ensure labels are 0/1 with 1 == recurrent (positive)
        # y_train_enc must be a Series indexed like X_train_concat with values in {0,1}
        y_train_here = y_train_enc.loc[X_train_concat.index]
        if set(pd.unique(y_train_here)) - {0, 1}:
            raise ValueError("y_train_enc must contain only {0,1} where 1 = recurrent.")
#         # === STEP 3: Fit risk classifier on full concatenated processed features ===

        # ----- model grids (your originals) -----
        if subtype == "LumA":
            param_grid = {
                "Logistic Regression L2": (
                    LogisticRegression(),
                    {
                        "penalty": ["l2"],
                        "C": [0.001, 0.01, 0.1, 1, 10],
                        "class_weight": ["balanced", None],
                        "solver": ["liblinear"],
                        "max_iter": [500, 1000]
                    }
                ),
                "Logistic Regression L1": (
                    LogisticRegression(),
                    {
                        "penalty": ["l1"],
                        "C": [0.001, 0.01, 0.1, 1, 10],
                        "class_weight": ["balanced", None],
                        "solver": ["liblinear"],
                        "max_iter": [500, 1000]
                    }
                ),
                "Random Forest": (
                    RandomForestClassifier(),
                    {
                        "n_estimators": [100, 200],
                        "max_depth": [None, 10, 20],
                        "min_samples_split": [2, 5],
                        "max_features": ["sqrt", "log2"]
                    }
                ),
                "SVM": (
                    SVC(probability=True),
                    {
                        "C": [0.01, 0.1, 1],
                        "kernel": ["linear", "rbf"],
                        "gamma": ["scale"],
                        "degree": [3]
                    }
                ),
                "Naive Bayes": (
                    GaussianNB(),
                    {}
                ),
                "Gradient Boosting": (
                    GradientBoostingClassifier(),
                    {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.01, 0.1],
                        "max_depth": [3, 5],
                        "subsample": [0.8, 1.0]
                    }
                ),
                "MLP": (
                    MLPClassifier(max_iter=1000),
                    {
                        "hidden_layer_sizes": [(50,), (100,), (100, 50)],
                        "activation": ["relu", "tanh"],
                        "solver": ["adam"],
                        "alpha": [0.0001, 0.001],
                        "learning_rate": ["constant", "adaptive"]
                    }
                )
            }
        else:
            param_grid = {
                "Logistic Regression L2": (
                    LogisticRegression(),
                    {
                        "penalty": ["l2"],
                        "C": [0.001, 0.01, 0.1, 1, 10],
                        "class_weight": [None],
                        "solver": ["liblinear"],
                        "max_iter": [500, 1000]
                    }
                ),
                "Logistic Regression L1": (
                    LogisticRegression(),
                    {
                        "penalty": ["l1"],
                        "C": [0.001, 0.01, 0.1, 1, 10],
                        "class_weight": [None],
                        "solver": ["liblinear"],
                        "max_iter": [500, 1000]
                    }
                ),
                "Random Forest": (
                    RandomForestClassifier(),
                    {
                        "n_estimators": [100, 200],
                        "max_depth": [None, 10, 20],
                        "min_samples_split": [2, 5],
                        "max_features": ["sqrt", "log2"]
                    }
                ),
                "SVM": (
                    SVC(probability=True),
                    {
                        "C": [0.01, 0.1, 1],
                        "kernel": ["linear", "rbf"],
                        "gamma": ["scale"],
                        "degree": [3]
                    }
                ),
                "Naive Bayes": (
                    GaussianNB(),
                    {}
                ),
                "Gradient Boosting": (
                    GradientBoostingClassifier(),
                    {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.01, 0.1],
                        "max_depth": [3, 5],
                        "subsample": [0.8, 1.0]
                    }
                ),
                "MLP": (
                    MLPClassifier(max_iter=1000),
                    {
                        "hidden_layer_sizes": [(50,), (100,), (100, 50)],
                        "activation": ["relu", "tanh"],
                        "solver": ["adam"],
                        "alpha": [0.0001, 0.001],
                        "learning_rate": ["constant", "adaptive"]
                    }
                )
            }

        best_auc = -np.inf
        best_model_name = None
        best_model = None
        best_params = None

        for model_name, (model, grid_params) in param_grid.items():
            print(f"🔍 Running risk classifier: {model_name}")
            grid = GridSearchCV(model, grid_params, cv=3, scoring="roc_auc", n_jobs=-1, refit=True)
            grid.fit(X_train_concat, y_train_here)

            mean_auc = grid.best_score_
            print(f"✅ {model_name} Mean CV AUC: {mean_auc:.4f}")
            print(f"📌 Best params: {grid.best_params_}")

            if mean_auc > best_auc:
                best_auc = mean_auc
                best_model_name = model_name
                best_model = grid.best_estimator_
                best_params = grid.best_params_

        if best_model is None:
            raise RuntimeError("No model finished successfully in GridSearchCV.")

        print("🎯 Selected Best Model")
        print(f"📌 Model: {best_model_name}")
        print(f"📌 Best AUC: {best_auc:.4f}")
        print(f"📌 Parameters: {best_params}")

        # Fit best model on ALL train data for confident-sample selection
        risk_model = best_model
        risk_model.fit(X_train_concat, y_train_here)

        # Always get P(recurrent=1)
        risk_probs = _proba_pos(risk_model, X_train_concat, pos_label=1)

        risk_df = pd.DataFrame(
            {'Predicted_Risk': risk_probs, 'True_Label': y_train_here},
            index=X_train_concat.index
        )

        # Fixed probability thresholds (you can swap to quantiles if you prefer coverage control)
        high_risk_ids = risk_df.index[risk_df['Predicted_Risk'] >= 0.80]
        low_risk_ids  = risk_df.index[risk_df['Predicted_Risk'] <= 0.20]
        confident_ids = high_risk_ids.union(low_risk_ids)

        print(f"✅ Confident sample IDs identified: {len(confident_ids)} "
            f"(high={len(high_risk_ids)}, low={len(low_risk_ids)})")


        confident_ids = confident_ids.astype(str).str.strip()
        ### NEW TRAIN IDS 
        train_ids = confident_ids.copy()
        test_ids = meta.iloc[test_idx].index.astype(str).str.strip()
        subtype_meta_train = meta.loc[train_ids].copy()
        subtype_meta_test = meta.loc[test_ids].copy()
        y_train = y.loc[train_ids]
        y_test = y.loc[test_ids]
        y_train_enc = pd.Series(label_encoder.transform(y_train), index=y_train.index)
        y_test_enc = label_encoder.transform(y_test)
        assert all(train_ids == y_train.index.astype(str).str.strip()), "❌ Misaligned training features and labels!"

        # === Step 1: Process mutation burden + significance for train set
        mutation_pathways_train, _ = process_mutation_pathways(
            M_matrix=data['mut'].loc[train_ids],
            subtype_meta=subtype_meta_train,
            pathway_dicts=all_pathway_dicts,
            pathway_options=pathway_options,
            output_dir=None,
            subtype=subtype,
            threshold=0.01
        )

        # === Step 2: Compute full burden matrix on test set (no significance testing needed)
        _, mutation_full_test = process_mutation_pathways(
            M_matrix=data['mut'].loc[test_ids],
            subtype_meta=meta.loc[test_ids].copy(),  # Just used to align indices; status column not needed
            pathway_dicts=all_pathway_dicts,
            pathway_options=pathway_options,
            output_dir=None,
            subtype=subtype
        )

        # === Step 3: Subset test burden matrix to significant features
        significant_mutation_features = mutation_pathways_train.columns
        mutation_pathways_test = mutation_full_test[significant_mutation_features].fillna(0)
      
        print("Mutation features:", significant_mutation_features)
        print("Non-zero features in X_train_final:", (mutation_pathways_train != 0).sum().sum())
        print("Non-zero features in X_test_proc:", (mutation_pathways_test != 0).sum().sum())


        # === Step 2: Run 1st round significance filtering for RNA GSVA pathways ===
        rna_results = run_significance_tests(gsva_scores_rna_all.loc[train_ids], subtype_meta_train,
                                             group_col='Disease_Status', group1='recurrent', group2='non-recurrent', use_mannwhitney=True)
        sig_rna = rna_results[rna_results['p_value'] <= 0.01]['feature']
        if sig_rna.empty:
            sig_rna = rna_results.nsmallest(5, 'p_value')['feature']
        rna_pathways_initial = gsva_scores_rna_all.loc[train_ids, sig_rna]

        # === Step 3: Run 1st round significance filtering for CNV GSVA pathways ===
        cnv_results = run_significance_tests(gsva_scores_cnv_all.loc[train_ids], subtype_meta_train,
                                             group_col='Disease_Status', group1='recurrent', group2='non-recurrent', use_mannwhitney=True)
        sig_cnv = cnv_results[cnv_results['p_value'] <= 0.01]['feature']
        if sig_cnv.empty:
            sig_cnv = cnv_results.nsmallest(5, 'p_value')['feature']
        cnv_pathways_initial = gsva_scores_cnv_all.loc[train_ids, sig_cnv]

        # === Step 4: Run DESeq2-style filtering for raw RNA ===
        subtype_meta_train['Disease_Status'] = pd.Categorical(subtype_meta_train['Disease_Status'], categories=["non-recurrent", "recurrent"])
        de_results = process_deseq_results(
            rna_expression=data['rna_raw'].loc[train_ids],
            metadata=subtype_meta_train[['Disease_Status']],
            design_factors=['Disease_Status'],
            contrast=["Disease_Status", "recurrent", "non-recurrent"],
            pvalue_threshold=0.01, padj_threshold=0.25, logfold_threshold=1.5,
            base_mean_threshold=100, gene_sum_threshold=100
        )
        rna_log2 = np.log2(data['rna'] + 1)
        de_genes = [g for g in de_results.index if g in rna_log2.columns]
        de_expr_train_initial = rna_log2.loc[train_ids, de_genes]
        de_expr_test = rna_log2.loc[test_ids, de_genes]

        # === Step 5: Merge features into blocks ===
        X_train_full, X_train_blocks = merge_feature_blocks(
            subtype_meta=subtype_meta_train,
            mutation_df=mutation_pathways_train,
            rna_df=rna_pathways_initial,
            cnv_df=cnv_pathways_initial,
            tf_df=data['tf'].loc[train_ids],
            tmb_df=data['tmb'].loc[train_ids],
            hoadley_df=data['hoadley'].loc[train_ids],
            cibersort_df=data['cibersort'].loc[train_ids],
            pathogenic_df=data['pathogenic_count'].loc[train_ids],
            cnv_pct_df=data['cnv_pct_altered'].loc[train_ids],
            clinical_df=data['clinical'].loc[train_ids],
            de_gene_df=de_expr_train_initial,
            return_separate_blocks=True
        )

        X_test_full, X_test_blocks = merge_feature_blocks(
            subtype_meta=subtype_meta_test,
            mutation_df=mutation_pathways_test,
            rna_df=gsva_scores_rna_all.loc[test_ids, sig_rna],
            cnv_df=gsva_scores_cnv_all.loc[test_ids, sig_cnv],
            tf_df=data['tf'].loc[test_ids],
            tmb_df=data['tmb'].loc[test_ids],
            hoadley_df=data['hoadley'].loc[test_ids],
            cibersort_df=data['cibersort'].loc[test_ids],
            pathogenic_df=data['pathogenic_count'].loc[test_ids],
            cnv_pct_df=data['cnv_pct_altered'].loc[test_ids],
            clinical_df=data['clinical'].loc[test_ids],
            de_gene_df=de_expr_test,
            return_separate_blocks=True
        )

        # ✅ Filter to selected blocks (both times after merge)
        if selected_blocks is not None:
            X_train_blocks = {k: v for k, v in X_train_blocks.items() if k in selected_blocks}
            X_test_blocks = {k: v for k, v in X_test_blocks.items() if k in selected_blocks}
            print(f"🔎 Using selected blocks only: {list(X_train_blocks.keys())}")

        meta_features_fold = []  # Collect per-block test predictions for this fold


        # === STEP 4: Subset each block's preprocessed data ===
        for block_name, block_info in X_train_blocks.items():
            X_block_train = block_info["data"]
            X_block_test = X_test_blocks[block_name]["data"]
            process_type = block_info["process"]
            print(f"📦 Block: {block_name} | Train shape: {X_block_train.shape} | Test shape: {X_block_test.shape}")
            print(f"Non-zero entries in training set: {(X_block_train != 0).sum().sum()}")

            X_train_proc, X_test_proc = preprocess_block(
                X_block_train, process_type, block_name, y_train=y_train,X_test=X_block_test, output_path=fold_output_dir, fold=fold
            )

            X_train_final = X_train_proc
            y_train_final = y_train_enc

            # Final model training
            safe_cv = get_safe_cv(y_train_final, desired_folds=inner_folds)
            #best_model, model_type = tune_model(X_train_proc, y_train_enc, model_grid, cv=safe_cv)
            best_model, model_type, best_auc, best_params = tune_model(X_train_proc, y_train_enc, model_grid, cv=safe_cv)
            print(best_model)
            print(model_type)
            print(best_auc)
            print(best_params)

            # Save training feature names
            train_features = X_train_final.columns

            # Align test features
            X_test_proc = X_test_proc.reindex(columns=train_features, fill_value=0)

            # Predict on test
            y_test_pred = best_model.predict_proba(X_test_proc)[:, 1]
            
            y_pred_labels = (y_test_pred >= 0.5).astype(int)


            fold_preds = pd.DataFrame({
                'sample_id': X_test_proc.index,
                'true_label': y_test.loc[X_test_proc.index].values,
                'predicted_proba': y_test_pred,
                'block_name': block_name,  # this assumes you're looping over blocks
                'fold': fold

            })

            oof_predictions = pd.concat([oof_predictions, fold_preds], axis=0)

            # Predict on training and test sets
            y_train_pred = best_model.predict_proba(X_train_final)[:, 1]

            train_preds = pd.DataFrame({
                'sample_id': X_train_final.index,
                'true_label': y_train.loc[X_train_final.index].values,
                'predicted_proba': y_train_pred,
                'block_name': block_name,
                'set': 'train',
                'fold': fold
            })

            training_predictions = pd.concat([training_predictions, train_preds], axis=0)

            from sklearn.metrics import roc_curve, roc_auc_score

            # Compute ROC metrics
            y_test_fold = y_test.loc[X_test_proc.index]
            y_test_enc_fold = label_encoder.transform(y_test_fold)
            fpr, tpr, thresholds = roc_curve(y_test_enc_fold, y_test_pred)
            auc = roc_auc_score(y_test_enc, y_test_pred)

            # Track in-memory (optional aggregate use)
            block_roc_data.setdefault(block_name, []).append((fpr, tpr, auc))

            # Save ROC curve data to CSV
            roc_df = pd.DataFrame({
                'fpr': fpr,
                'tpr': tpr,
                'threshold': thresholds,
                'auc': auc  # repeated for convenience
            })
            roc_path = os.path.join(cv_models_dir, f"{block_name}_fold{fold + 1}_roc.csv")
            roc_df.to_csv(roc_path, index=False)


            # === Save trained model to disk
            joblib.dump(best_model, os.path.join(cv_models_dir, f"{block_name}_fold{fold + 1}.pkl"))
            # Save model metadata (type + hyperparameters)
            model_txt_path = os.path.join(cv_models_dir, f"{block_name}_fold{fold + 1}_info.txt")
            with open(model_txt_path, 'w') as f:
                f.write(f"Model Type: {model_type}\n")
                f.write("Hyperparameters:\n")
                if hasattr(best_model, "get_params"):
                    for k, v in best_model.get_params().items():
                        f.write(f"  {k}: {v}\n")

            print(f"  📎 Saved CV model for block '{block_name}' fold {fold + 1}")
            print(f"  🔍 Selected model type: {model_type} | Coef available: {hasattr(best_model, 'coef_')}")

            # ⬇️ Record metrics for this fold of this block
            metrics = {
                'block_name': block_name,
                'fold': fold + 1,
                'auc': roc_auc_score(y_test_enc, y_test_pred),
                'accuracy': accuracy_score(y_test_enc, y_pred_labels),
                'precision': precision_score(y_test_enc, y_pred_labels, zero_division=0),
                'recall': recall_score(y_test_enc, y_pred_labels, zero_division=0),
                'f1': f1_score(y_test_enc, y_pred_labels, zero_division=0),
                'n_test_samples': len(y_test_enc),
                'model_name': type(best_model).__name__,
                'params': str(best_model.get_params())
            }

            performance_metrics.append(metrics)


            # === Compute SHAP values for final training data
            explainer = get_shap_explainer(best_model, X_train_final)
            if explainer is None:
                print(f"❌ Skipping SHAP for model {model_type} due to error.")
            else:
                shap_values = explainer(X_train_final)
                shap_array = shap_values.values

                # SHAP output can be (n_samples, n_features) or (n_samples, n_features, 2)
                if shap_array.ndim == 3 and shap_array.shape[2] == 2:
                    shap_array = shap_array[:, :, 1]  # Use class 1 SHAP values

                shap_df = pd.DataFrame(shap_array, columns=X_train_final.columns)
                shap_df.to_csv(os.path.join(shap_dir, f"shap_{block_name}_fold{fold + 1}_{model_type}.csv"))
                joblib.dump(shap_values, os.path.join(shap_dir, f"shap_{block_name}_fold{fold + 1}_{model_type}.pkl"))

            # === Save model coefficients (for linear models)
            if hasattr(best_model, 'coef_'):
                coefs = pd.Series(best_model.coef_[0], index=X_train_final.columns)
                coefs.to_csv(os.path.join(coef_dir, f"coef_{block_name}_fold{fold + 1}_{model_type}.csv"))

            # === Store block-level model and predictions
            best_models_by_block[block_name] = (model_type, best_model)
            meta_features_fold.append(y_test_pred)

                # === Collect OOF predictions across all blocks for this fold
        if meta_features_fold:
            # Stack predicted probabilities from each block (one row per test sample)
            oof_meta_features.append(np.vstack(meta_features_fold).T)
            oof_true_labels.append(y_test_enc)
   

    # === Save per-block ROC curve plots (one per fold)
    for block_name, curves in block_roc_data.items():
        plt.figure()
        for i, (fpr, tpr, auc) in enumerate(curves):
            plt.plot(fpr, tpr, label=f"Fold {i+1} AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title(f"ROC Curve for Block: {block_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(os.path.join(roc_dir, f"roc_block_{block_name}.png"))
        plt.close()
    #save out of fold predictions for each block
    oof_predictions.to_csv(os.path.join(fold_output_dir, f"oofold_predictions.csv"), index=False)
    training_predictions.to_csv(os.path.join(fold_output_dir, 'training_predictions.csv'), index=False)
    perf_df = pd.DataFrame(performance_metrics)
    perf_path = os.path.join(fold_output_dir, "performance_summary.csv")
    perf_df.to_csv(perf_path, index=False)

    # === Save combined ROC curve plot across folds for each block
    plt.figure(figsize=(8, 6))
    mean_fpr = np.linspace(0, 1, 100)

    for block_name, curves in block_roc_data.items():
        tprs, aucs = [], []
        for fpr, tpr, auc in curves:
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        plt.plot(mean_fpr, mean_tpr, label=f"{block_name} (AUC = {mean_auc:.2f})", lw=2)

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Mean ROC Curves Across Folds by Block")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    combined_roc_path = os.path.join(roc_dir, "combined_block_roc_curves.png")
    plt.savefig(combined_roc_path)
    plt.close()
    print(f"📈 Saved combined ROC curve for all blocks: {combined_roc_path}")



    return (
        np.vstack(oof_meta_features) if oof_meta_features else np.array([]),
        np.concatenate(oof_true_labels) if oof_true_labels else np.array([]),
        best_models_by_block
    )


from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

def tune_model_multiclass(X, y, model_grid, cv):
    """
    Returns: best_estimator, model_name, best_score, best_params
    Uses AUC-OVO macro if proba is available; otherwise calibrates or falls back to F1-macro.
    """
    best = (None, None, float("-inf"), None)  # (estimator, name, score, params)

    for name, (estimator, param_grid) in model_grid.items():
        est = estimator

        # Ensure probability for common culprits
        # SVC: set probability=True in grid or here
        try:
            from sklearn.svm import SVC
            if isinstance(est, SVC):
                est.set_params(probability=True)
        except Exception:
            pass

        # Choose scorer
        supports_proba = hasattr(est, "predict_proba")
        scorer = AUC_OVO_MACRO if supports_proba else F1_MACRO

        # If no proba, try calibrated wrapper to enable AUC
        use_calibrated = not supports_proba and hasattr(est, "decision_function")
        if use_calibrated:
            est = CalibratedClassifierCV(base_estimator=est, method="sigmoid", cv=3)
            scorer = AUC_OVO_MACRO  # now we have predict_proba

        try:
            gcv = GridSearchCV(est, param_grid, cv=cv, scoring=scorer, n_jobs=-1, refit=True, error_score="raise")
            gcv.fit(X, y)
            score = gcv.best_score_
            if score is None or score != score:  # NaN
                continue
            if score > best[2]:
                best = (gcv.best_estimator_, name, float(score), gcv.best_params_)
        except Exception as e:
            print(f"⚠️ Skipping {name} due to scoring/fit error: {e}")
            continue

    if best[0] is None:
        raise RuntimeError("All model candidates failed; check scorer and grids.")
    return best  # (estimator, name, score, params)


def run_cv_blocks_risk_prediction(
    data, meta, y, label_encoder, all_pathway_dicts, pathway_options,
    gsva_scores_rna_all, gsva_scores_cnv_all, model_grid,
    subtype, target_col='Disease_Status',
    outer_folds=5, inner_folds=3, fold_output_dir=None, selected_blocks=None,
):
    """
    Multiclass risk-tertile training with high-vs-low-only feature selection (CV-safe).

    Per outer fold:
      1) Preprocess all blocks -> concatenate as X_concat (train/test)
      2) Fit RSF on TRAIN ONLY -> risk scores (train/test)
      3) Bin TRAIN scores into tertiles -> derive TRAIN cutpoints -> bin TEST with same cutpoints
      4) Select features using TRAIN extremes (high vs low) only
      5) Train per-block multiclass classifier (0/1/2) with inner CV tuning
      6) Save OOF multiclass probabilities, metrics, ROC plots, and models

    Returns:
      X_meta (np.ndarray), y_meta (np.ndarray), best_models_by_block (dict)
    """
    import numpy as np
    import pandas as pd
    import os, joblib, warnings
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, precision_recall_fscore_support
    )
    from sklearn.exceptions import UndefinedMetricWarning
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import f1_score   # add; used in alignment summaries below


    # ───────────────────────────────
    # Output dirs
    # ───────────────────────────────
    if fold_output_dir:
        os.makedirs(fold_output_dir, exist_ok=True)
        fold_output_dir = os.path.join(fold_output_dir, subtype)
        training_data_dir = os.path.join(fold_output_dir, "training_data")
        roc_dir          = os.path.join(fold_output_dir, "roc_plots")
        cv_models_dir    = os.path.join(fold_output_dir, "cv_models")
        preprocessing_dir= os.path.join(fold_output_dir, "preprocessing")
        shap_dir         = os.path.join(fold_output_dir, "shap")
        coef_dir         = os.path.join(fold_output_dir, "coefficients")
        for d in [training_data_dir, roc_dir, cv_models_dir, preprocessing_dir, shap_dir, coef_dir]:
            os.makedirs(d, exist_ok=True)

    # ───────────────────────────────
    # Helpers
    # ───────────────────────────────
    def multiclass_auc(y_true, proba):
        # proba: (n, 3)
        return roc_auc_score(y_true, proba, multi_class='ovo', average='macro')

    from sklearn.utils.extmath import softmax

    def safe_multiclass_auc(y_true, proba, model_classes=None):
        """
        - 3 classes present  -> macro OVO AUC
        - exactly 2 classes  -> binary AUC using prob of the positive class
        - <2 classes         -> np.nan
        """
        y_true = np.asarray(y_true)
        if model_classes is None:
            model_classes = np.arange(proba.shape[1])
        model_classes = np.asarray(model_classes)

        present = np.unique(y_true)
        if present.size < 2:
            return np.nan

        # map class -> proba column
        col_map = {}
        for c in present:
            hit = np.where(model_classes == c)[0]
            if hit.size:
                col_map[int(c)] = int(hit[0])
        if len(col_map) < 2:
            return np.nan

        if present.size == 2:
            pos = int(np.max(present))               # pick larger as positive
            j = col_map[pos]
            y_bin = (y_true == pos).astype(int)
            return roc_auc_score(y_bin, proba[:, j])

        # 3+ classes present → multiclass OVO (only for present classes)
        cols = [col_map[int(c)] for c in present]
        return roc_auc_score(y_true, proba[:, cols], labels=present, multi_class="ovo", average="macro")

    def predict_proba_safe(model, X):
        if hasattr(model, "predict_proba"):
            P = model.predict_proba(X)
            if isinstance(P, list):
                P = np.column_stack(P)
            return P
        if hasattr(model, "decision_function"):
            D = model.decision_function(X)
            if D.ndim == 1:  # binary margin -> 2-class probs
                D = np.column_stack([-D, D])
            return softmax(D)
        raise AttributeError("Model exposes neither predict_proba nor decision_function.")

    def auc_ovo_macro_scorer(estimator, X, y):
        # Works across sklearn versions, SVC, etc.
        if hasattr(estimator, "predict_proba"):
            P = estimator.predict_proba(X)
            if isinstance(P, list):
                P = np.column_stack(P)
            return roc_auc_score(y, P, multi_class="ovo", average="macro")
        if hasattr(estimator, "decision_function"):
            D = estimator.decision_function(X)
            if D.ndim == 1:
                D = np.column_stack([-D, D])
            P = softmax(D)
            return roc_auc_score(y, P, multi_class="ovo", average="macro")
        raise ValueError("Estimator exposes neither predict_proba nor decision_function.")

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import SVC

    def tune_model_multiclass(X, y, model_grid, cv):
        best = (None, None, float("-inf"), None)
        for name, (est, param_grid) in model_grid.items():
            # Ensure SVC can output probs
            if isinstance(est, SVC):
                est.set_params(probability=True)

            needs_calibration = (not hasattr(est, "predict_proba")) and hasattr(est, "decision_function")
            final_est = CalibratedClassifierCV(est, method="sigmoid", cv=3) if needs_calibration else est

            try:
                gcv = GridSearchCV(
                    final_est, param_grid, cv=cv,
                    scoring=auc_ovo_macro_scorer,  # callable scorer
                    n_jobs=-1, refit=True, error_score="raise"
                )
                gcv.fit(X, y)
                score = gcv.best_score_
                if score > best[2]:
                    best = (gcv.best_estimator_, name, float(score), gcv.best_params_)
            except Exception as e:
                print(f"⚠️ Skipping {name} due to scoring/fit error: {e}")
                continue

        if best[0] is None:
            raise RuntimeError("All model candidates failed; check grids/scorers.")
        return best

    def assign_risk_tertiles(train_scores, test_scores):
        low_cut  = np.percentile(train_scores, 33.333)
        high_cut = np.percentile(train_scores, 66.667)

        def bin_scores(scores):
            y_mc = np.full(len(scores), 1, dtype=int)  # intermediate=1
            y_mc[scores <= low_cut] = 0                # low
            y_mc[scores >  high_cut] = 2               # high
            return y_mc

        return bin_scores(train_scores), bin_scores(test_scores), (low_cut, high_cut)

    def fit_rsf_and_predict(X_train_concat, subtype_meta_train, X_test_concat, random_state=42):
        """
        Fit RSF on TRAIN only (grid by c-index), return train & test risk scores.
        Requires subtype_meta_train to contain 'time' and 'event'.
        """
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
        from sksurv.metrics import concordance_index_censored

        assert {'time', 'event'}.issubset(subtype_meta_train.columns), "meta needs 'time' and 'event'."

        y_struct = Surv.from_dataframe(event='event', time='time',
                                       data=subtype_meta_train.loc[X_train_concat.index])

        rsf = RandomSurvivalForest(random_state=random_state)
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
            "max_features": ["sqrt", "log2"],
        }

        def cindex_scorer(estimator, X, y_structured):
            risk = estimator.predict(X)             # higher = worse
            e = y_structured["event"]
            t = y_structured["time"]
            return concordance_index_censored(e, t, risk)[0]

        grid = GridSearchCV(
            estimator=rsf,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            scoring=cindex_scorer
        )
        print("🔍 Grid-searching RSF by c-index...")
        grid.fit(X_train_concat, y_struct)
        best_rsf = grid.best_estimator_
        print(f"✅ Best RSF params: {grid.best_params_} | best c-index: {grid.best_score_:.3f}")

        train_scores = best_rsf.predict(X_train_concat)
        test_scores  = best_rsf.predict(X_test_concat)
        return np.asarray(train_scores), np.asarray(test_scores)

    # ROC helpers
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    from sklearn.preprocessing import label_binarize

    def collapse_proba_to_binary(proba, rule="ignore_mid", alpha=0.5, eps=1e-9):
        """
        proba: (n,3) for classes [0,1,2] = [low, mid, high]
        Returns a (n,) recurrence score in [0,1].
        """
        if rule == "ignore_mid":
            p_rec = proba[:, 2]; p_non = proba[:, 0]
            return p_rec / (p_rec + p_non + eps)
        elif rule == "weighted":
            return proba[:, 2] + alpha * proba[:, 1]
        elif rule == "hard_side":
            return (proba[:, 2] > proba[:, 0]).astype(float)
        else:
            raise ValueError(f"Unknown collapse rule: {rule}")

    def plot_multiclass_roc(y_true, proba, model_classes=None, title="", outpath=None):
        """
        Plots OvR ROC curves for the classes present in y_true.
        - If only 2 classes are present, computes a single binary ROC using the higher label as positive.
        Returns dict with per-class AUCs and 'micro_auc'.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        y_true = np.asarray(y_true)
        if model_classes is None:
            model_classes = np.arange(proba.shape[1])
        model_classes = np.asarray(model_classes)

        present = np.unique(y_true)
        if present.size < 2:
            print("⚠️ Multiclass ROC skipped: <2 classes present.")
            return {}

        # Map class -> proba column
        col_idx = {}
        for c in present:
            hits = np.where(model_classes == c)[0]
            if hits.size:
                col_idx[int(c)] = int(hits[0])
        if len(col_idx) < 2:
            print("⚠️ Multiclass ROC skipped: missing matching class columns.")
            return {}

        plt.figure(figsize=(6, 5))
        per_auc = {}

        # Per-class OvR curves
        for c, j in sorted(col_idx.items()):
            y_bin = (y_true == c).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == y_bin.size:
                continue
            fpr, tpr, _ = roc_curve(y_bin, proba[:, j])
            auc_c = auc(fpr, tpr)
            per_auc[c] = auc_c
            plt.plot(fpr, tpr, lw=1.6, label=f"class {c} vs rest (AUC={auc_c:.2f})")

        # Micro-avg:
        if len(col_idx) >= 3:
            # true multiclass micro-average
            pres_sorted = sorted(col_idx.keys())
            Y = label_binarize(y_true, classes=pres_sorted)              # (n, k)
            P = proba[:, [col_idx[c] for c in pres_sorted]]              # (n, k)
            fpr_micro, tpr_micro, _ = roc_curve(Y.ravel(), P.ravel())    # (nk,) vs (nk,)
            auc_micro = auc(fpr_micro, tpr_micro)
            plt.plot(fpr_micro, tpr_micro, lw=2, linestyle="--", label=f"micro (AUC={auc_micro:.2f})")
        else:
            # binary case → use the prob of the higher label as positive
            pos = int(max(col_idx.keys()))
            j = col_idx[pos]
            y_bin = (y_true == pos).astype(int)                          # (n,)
            scores = proba[:, j]                                         # (n,)
            fpr_micro, tpr_micro, _ = roc_curve(y_bin, scores)
            auc_micro = auc(fpr_micro, tpr_micro)
            plt.plot(fpr_micro, tpr_micro, lw=2, linestyle="--", label=f"binary (AUC={auc_micro:.2f})")

        plt.plot([0, 1], [0, 1], linestyle=":", color="gray")
        plt.xlim([0, 1]); plt.ylim([0, 1.02])
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(title or "Multiclass ROC (OvR)")
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        if outpath:
            plt.savefig(outpath, dpi=160)
            plt.close()

        return {"per_class_auc": per_auc, "micro_auc": auc_micro}

    def plot_binary_roc(y_true_bin, score, title="", outpath=None):
        fpr, tpr, _ = roc_curve(y_true_bin, score)
        auc_b = roc_auc_score(y_true_bin, score)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"AUC={auc_b:.2f}")
        plt.plot([0,1],[0,1], linestyle=":", color="gray")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(title or "Binary ROC")
        plt.legend(loc="lower right")
        plt.tight_layout()
        if outpath:
            plt.savefig(outpath, dpi=160); plt.close()
        return float(auc_b)

    def summarize_metrics(perf_df, outdir, metrics=(
            "auc_macro_ovo","accuracy","precision_macro","recall_macro","f1_macro",
            "auc_binary_ignore_mid","auc_binary_weighted_0.5"
        )):
        """
        Save mean±std and 95% CI by block, and overall.
        """
        import pandas as pd, numpy as np, os
        df = perf_df.copy()
        # By block
        gb = df.groupby("block_name")[list(set(metrics) & set(df.columns))]
        mean = gb.mean(numeric_only=True)
        std  = gb.std(numeric_only=True, ddof=1)
        cnt  = gb.count()
        ci95 = 1.96 * std / np.sqrt(cnt.replace(0, np.nan))

        block_summary = mean.add_suffix("_mean")
        for col in std.columns:  block_summary[f"{col}_std"] = std[col]
        for col in cnt.columns:  block_summary[f"{col}_n"]   = cnt[col]
        for col in ci95.columns: block_summary[f"{col}_ci95"] = ci95[col]
        block_summary.to_csv(os.path.join(outdir, "performance_summary_mc_block_mean.csv"))

        # Overall
        cols = list(set(metrics) & set(df.columns))
        overall = pd.DataFrame({
            "metric": cols,
            "mean":   [df[c].mean(skipna=True) for c in cols],
            "std":    [df[c].std(skipna=True, ddof=1) for c in cols],
            "n":      [df[c].count() for c in cols],
        })
        overall["ci95"] = 1.96 * overall["std"] / np.sqrt(overall["n"].replace(0, np.nan))
        overall.to_csv(os.path.join(outdir, "performance_summary_mc_overall_mean.csv"), index=False)


    # ── Helpers to make averaged-over-folds ROC plots from OOF predictions ──
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    from sklearn.preprocessing import label_binarize

    def plot_summary_ovr_micro_rocs(oof_df: pd.DataFrame, outpath: str):
        """
        One curve per block/model, using OOF predictions pooled across folds.
        Robust to messy true_label_mc dtype (arrays/strings/NaN).
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        def _coerce_labels_012(series: pd.Series):
            """Return clean int labels {0,1,2} and a boolean mask of valid rows."""
            arr = series.to_numpy()

            def to_scalar(x):
                # pull scalar out of [x] or array([x]); leave plain scalars
                if isinstance(x, (list, tuple, np.ndarray)):
                    x = np.array(x)
                    if x.size == 1:
                        return x.reshape(-1)[0]
                return x

            arr = np.array([to_scalar(x) for x in arr], dtype=object)
            # coerce to numeric; invalid -> NaN
            y_num = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy()
            mask = np.isfinite(y_num)
            y_clean = y_num[mask].astype(int)
            return y_clean, mask

        plt.figure(figsize=(7, 6))
        blocks = sorted(oof_df["block_name"].unique())
        any_plotted = False

        for blk in blocks:
            sub = oof_df[oof_df["block_name"] == blk].copy()

            # clean labels & align probabilities with valid rows
            y_true, mask = _coerce_labels_012(sub["true_label_mc"])
            P = sub[["pred_proba_0", "pred_proba_1", "pred_proba_2"]].to_numpy()
            P = P[mask, :]

            # need at least 2 classes to draw a ROC
            present = np.unique(y_true)
            if present.size < 2:
                print(f"⚠️ Skipping {blk}: <2 classes present in OOF after cleaning.")
                continue

            if present.size >= 3:
                # true multiclass micro-average: stack one-vs-rest
                Y = label_binarize(y_true, classes=[0, 1, 2])   # (n, 3)
                fpr, tpr, _ = roc_curve(Y.ravel(), P.ravel())
                auc_micro = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f"{blk} (micro AUC={auc_micro:.2f})")
                any_plotted = True
            else:
                # binary fallback: use the higher class as positive
                pos = int(np.max(present))                      # 1 or 2
                y_bin = (y_true == pos).astype(int)
                scores = P[:, pos]
                fpr, tpr, _ = roc_curve(y_bin, scores)
                auc_bin = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f"{blk} (binary AUC={auc_bin:.2f})")
                any_plotted = True

        plt.plot([0, 1], [0, 1], linestyle=":", color="gray")
        plt.xlim([0, 1]); plt.ylim([0, 1.02])
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("OOF Multiclass ROC (micro-avg) per Block")
        if any_plotted:
            plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(outpath, dpi=160)
        plt.close()


    def collapse_proba_to_binary(proba: np.ndarray, rule="ignore_mid", alpha=0.5, eps=1e-9):
        """
        proba: (n,3) for classes [0,1,2] = [low, mid, high]
        Returns a (n,) recurrence-like score in [0,1].
        """
        if rule == "ignore_mid":
            p_rec = proba[:, 2]; p_non = proba[:, 0]
            return p_rec / (p_rec + p_non + eps)
        elif rule == "weighted":
            return proba[:, 2] + alpha * proba[:, 1]
        elif rule == "hard_side":
            return (proba[:, 2] > proba[:, 0]).astype(float)
        else:
            raise ValueError(f"Unknown collapse rule: {rule}")


    def plot_summary_binary_rocs(
        oof_df, y_series, label_encoder, outpath_ignore, outpath_weighted, alpha=0.5
    ):
        """
        One curve per block/model, comparing collapsed risk→binary score vs. true recurrence label.
        Uses pooled OOF predictions. Robust to messy dtypes and missing labels.
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        # clean up y_series index so we can reindex by sample_id strings
        y_series = y_series.copy()
        y_series.index = y_series.index.astype(str).str.strip()

        # helper: coerce multiclass labels {0,1,2} and return mask of valid rows
        def _oof_label_mask(series: pd.Series):
            arr = series.to_numpy()

            def to_scalar(x):
                if isinstance(x, (list, tuple, np.ndarray)):
                    x = np.array(x)
                    if x.size == 1:
                        return x.reshape(-1)[0]
                return x

            arr = np.array([to_scalar(x) for x in arr], dtype=object)
            y_num = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy()
            return np.isfinite(y_num)  # we only need the mask here


        blocks = sorted(oof_df["block_name"].unique())

        # === ignore_mid ===
        plt.figure(figsize=(7, 6))
        any_plotted = False
        for blk in blocks:
            sub = oof_df[oof_df["block_name"] == blk].copy()

            # 1) keep rows where OOF multiclass label is sane (for alignment)
            mask1 = _oof_label_mask(sub["true_label_mc"])
            sub = sub.loc[mask1]

            # 2) align true recurrence labels to the same sample order; drop missing
            ids = sub["sample_id"].astype(str).values
            y_true_series = y_series.reindex(ids)
            mask2 = y_true_series.notna().to_numpy()

            if mask2.sum() < 2:
                print(f"⚠️ Skipping {blk}: <2 valid binary labels after cleaning.")
                continue

            ids = ids[mask2]
            y_true_bin = label_encoder.transform(y_true_series[mask2].values)

            P = sub[["pred_proba_0", "pred_proba_1", "pred_proba_2"]].to_numpy()[mask2, :]
            scores = collapse_proba_to_binary(P, rule="ignore_mid")
            fpr, tpr, _ = roc_curve(y_true_bin, scores)
            auc_b = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=2, label=f"{blk} (AUC={auc_b:.2f})")
            any_plotted = True

        plt.plot([0, 1], [0, 1], linestyle=":", color="gray")
        plt.xlim([0, 1]); plt.ylim([0, 1.02])
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("OOF Binary ROC per Block (collapse = ignore_mid)")
        if any_plotted:
            plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(outpath_ignore, dpi=160)
        plt.close()

        # === weighted (alpha) ===
        plt.figure(figsize=(7, 6))
        any_plotted = False
        for blk in blocks:
            sub = oof_df[oof_df["block_name"] == blk].copy()

            mask1 = _oof_label_mask(sub["true_label_mc"])
            sub = sub.loc[mask1]

            ids = sub["sample_id"].astype(str).values
            y_true_series = y_series.reindex(ids)
            mask2 = y_true_series.notna().to_numpy()

            if mask2.sum() < 2:
                print(f"⚠️ Skipping {blk}: <2 valid binary labels after cleaning.")
                continue

            ids = ids[mask2]
            y_true_bin = label_encoder.transform(y_true_series[mask2].values)

            P = sub[["pred_proba_0", "pred_proba_1", "pred_proba_2"]].to_numpy()[mask2, :]
            scores = collapse_proba_to_binary(P, rule="weighted", alpha=alpha)
            fpr, tpr, _ = roc_curve(y_true_bin, scores)
            auc_b = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=2, label=f"{blk} (AUC={auc_b:.2f})")
            any_plotted = True

        plt.plot([0, 1], [0, 1], linestyle=":", color="gray")
        plt.xlim([0, 1]); plt.ylim([0, 1.02])
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"OOF Binary ROC per Block (collapse = weighted, α={alpha})")
        if any_plotted:
            plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(outpath_weighted, dpi=160)
        plt.close()

        import numpy as np
        import pandas as pd
        import os
        import matplotlib.pyplot as plt


    def _clean_true_labels_012(series: pd.Series):
        """Coerce to clean {0,1,2} ints and return labels + valid mask."""
        arr = series.to_numpy()

        def to_scalar(x):
            if isinstance(x, (list, tuple, np.ndarray)):
                x = np.array(x)
                if x.size == 1:
                    return x.reshape(-1)[0]
            return x

        arr = np.array([to_scalar(x) for x in arr], dtype=object)
        y_num = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy()
        mask = np.isfinite(y_num)
        return y_num[mask].astype(int), mask

    def _extreme_binary_score(P, eps=1e-9):
        """
        P: (n,3) for classes [0,1,2] = [low, mid, high]
        Return score in [0,1] for 'high vs low' ignoring mid.
        """
        p_low  = P[:, 0]
        p_high = P[:, 2]
        return p_high / (p_high + p_low + eps)

    def evaluate_high_low_from_oof(oof_df: pd.DataFrame, outdir: str, ks=(0.05, 0.10, 0.20), make_plots=True):
        """
        Evaluate how accurately we pick HIGH(2) vs LOW(0) using pooled OOF predictions.
        Saves a per-block CSV of metrics and optional ROC plots (extremes only).
        """
        os.makedirs(outdir, exist_ok=True)
        blocks = sorted(oof_df["block_name"].unique())
        rows = []
        plt_handles = []

        # figure for one ROC per block (extremes only)
        if make_plots:
            plt.figure(figsize=(7, 6))

        for blk in blocks:
            sub = oof_df[oof_df["block_name"] == blk].copy()

            # 1) clean labels and align probas
            y_all, m1 = _clean_true_labels_012(sub["true_label_mc"])
            P_all = sub[["pred_proba_0","pred_proba_1","pred_proba_2"]].to_numpy()
            P_all = P_all[m1, :]

            # 2) keep only extremes (0 or 2)
            m_ext = np.isin(y_all, [0, 2])
            y = y_all[m_ext]
            P = P_all[m_ext, :]

            if y.size < 2 or (np.unique(y).size < 2):
                print(f"⚠️ {blk}: not enough extremes to evaluate (n={y.size}, classes={np.unique(y)}). Skipping.")
                continue

            # Binary ground truth: 1 = HIGH(2), 0 = LOW(0)
            y_bin = (y == 2).astype(int)

            # 3) continuous score (ignore mid)
            s = _extreme_binary_score(P)

            # 4) thresholded predictions: compare p_high vs p_low
            y_pred = (P[:, 2] > P[:, 0]).astype(int)

            # 5) metrics
            auc_ = roc_auc_score(y_bin, s)
            acc  = accuracy_score(y_bin, y_pred)
            f1   = f1_score(y_bin, y_pred, zero_division=0)
            prec, rec, f1_macro, _ = precision_recall_fscore_support(y_bin, y_pred, average="macro", zero_division=0)
            cm = confusion_matrix(y_bin, y_pred, labels=[0,1])  # [[TN,FP],[FN,TP]]

            # average precision (PR-AUC) for HIGH as positive
            ap_high = average_precision_score(y_bin, s)

            # 6) precision@k (how pure are the top-k% as HIGH?)
            # sort by score descending; count true positives among top k%
            order = np.argsort(-s)
            y_sorted = y_bin[order]
            n = y_sorted.size
            p_at_k = {}
            for k in ks:
                k_abs = max(1, int(round(k * n)))
                p_at_k[f"precision_at_{int(k*100)}%"] = y_sorted[:k_abs].mean()  # fraction of highs in top k%

            # 7) coverage vs accuracy when we require confidence on extremes
            # define conf = max(p_high, p_low); mask where conf >= tau
            taus = np.linspace(0.5, 0.95, 10)
            cov_list, acc_list = [], []
            conf = np.maximum(P[:, 2], P[:, 0])
            for tau in taus:
                m_cov = conf >= tau
                if m_cov.sum() == 0:
                    cov_list.append(0.0); acc_list.append(np.nan); continue
                acc_tau = accuracy_score(y_bin[m_cov], y_pred[m_cov])
                cov_list.append(m_cov.mean())
                acc_list.append(acc_tau)

            # add to table
            row = {
                "block_name": blk,
                "n_extreme": int(n),
                "pct_high": float(y_bin.mean()),
                "auc_high_vs_low": float(auc_),
                "ap_high": float(ap_high),
                "accuracy_label_argmax": float(acc),
                "f1_binary": float(f1),
                "precision_macro": float(prec),
                "recall_macro": float(rec),
                "TN": int(cm[0,0]), "FP": int(cm[0,1]), "FN": int(cm[1,0]), "TP": int(cm[1,1]),
                **p_at_k
            }
            rows.append(row)

            # extremes-only ROC line
            if make_plots:
                fpr, tpr, _ = roc_curve(y_bin, s)
                plt.plot(fpr, tpr, lw=2, label=f"{blk} (AUC={auc_:.2f})")

            # save coverage curve per block
            cov_df = pd.DataFrame({"tau": taus, "coverage": cov_list, "accuracy": acc_list})
            cov_df.to_csv(os.path.join(outdir, f"coverage_accuracy_{blk}.csv"), index=False)

        # save summary table
        if rows:
            summ = pd.DataFrame(rows).sort_values("auc_high_vs_low", ascending=False)
            summ.to_csv(os.path.join(outdir, "extremes_eval_summary.csv"), index=False)

        # finish ROC plot
        if make_plots and rows:
            plt.plot([0,1],[0,1], linestyle=":", color="gray")
            plt.xlim([0,1]); plt.ylim([0,1.02])
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.title("OOF ROC — HIGH vs LOW (extremes only)")
            plt.legend(loc="lower right", fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "roc_extremes_only.png"), dpi=160)
            plt.close()

    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    # Encoded original labels (not used for training here, but kept for compatibility)
    _ = label_encoder.transform(y)

    # Outer CV
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)

    # Tracking
    oof_predictions = pd.DataFrame(
        columns=['sample_id', 'true_label_mc', 'pred_proba_0', 'pred_proba_1', 'pred_proba_2', 'block_name', 'fold']
    )
    training_predictions = pd.DataFrame(
        columns=['sample_id', 'true_label_mc', 'pred_proba_0', 'pred_proba_1', 'pred_proba_2', 'block_name', 'set', 'fold']
    )
    performance_metrics = []
    best_models_by_block = {}

    # ───────────────────────────────
    # Outer folds
    # ───────────────────────────────
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(meta, meta[target_col])):
        print(f"\n🔁 Outer Fold {fold + 1}/{outer_folds}")

        # Split metadata and labels into train/test
        train_ids = meta.iloc[train_idx].index.astype(str).str.strip()
        test_ids = meta.iloc[test_idx].index.astype(str).str.strip()
        subtype_meta_train = meta.loc[train_ids].copy()
        subtype_meta_test = meta.loc[test_ids].copy()
        y_train = y.loc[train_ids]
        y_test = y.loc[test_ids]
        y_train_enc = pd.Series(label_encoder.transform(y_train), index=y_train.index)
        y_test_enc = label_encoder.transform(y_test)
        assert all(train_ids == y_train.index.astype(str).str.strip()), "❌ Misaligned training features and labels!"

        # === Step 1: Process mutation burden + significance for train set
        mutation_pathways_train, _ = process_mutation_pathways(
            M_matrix=data['mut'].loc[train_ids],
            subtype_meta=subtype_meta_train,
            pathway_dicts=all_pathway_dicts,
            pathway_options=pathway_options,
            output_dir=None,
            subtype=subtype
        )

        # === Step 2: Compute full burden matrix on test set (no significance testing needed)
        _, mutation_full_test = process_mutation_pathways(
            M_matrix=data['mut'].loc[test_ids],
            subtype_meta=meta.loc[test_ids].copy(),
            pathway_dicts=all_pathway_dicts,
            pathway_options=pathway_options,
            output_dir=None,
            subtype=subtype
        )

        # === Step 3: Subset test burden matrix to significant features
        significant_mutation_features = mutation_pathways_train.columns
        mutation_pathways_test = mutation_full_test[significant_mutation_features].fillna(0)
      
        print("Mutation features:", significant_mutation_features)
        print("Non-zero features in X_train_final:", (mutation_pathways_train != 0).sum().sum())
        print("Non-zero features in X_test_proc:", (mutation_pathways_test != 0).sum().sum())

        # === Step 2: Run 1st round significance filtering for RNA GSVA pathways ===
        rna_results = run_significance_tests(
            gsva_scores_rna_all.loc[train_ids], subtype_meta_train,
            group_col='Disease_Status', group1='recurrent', group2='non-recurrent', use_mannwhitney=True
        )
        sig_rna = rna_results[rna_results['p_value'] <= 0.01]['feature']
        if sig_rna.empty:
            sig_rna = rna_results.nsmallest(5, 'p_value')['feature']
        rna_pathways_initial = gsva_scores_rna_all.loc[train_ids, sig_rna]

        # === Step 3: Run 1st round significance filtering for CNV GSVA pathways ===
        cnv_results = run_significance_tests(
            gsva_scores_cnv_all.loc[train_ids], subtype_meta_train,
            group_col='Disease_Status', group1='recurrent', group2='non-recurrent', use_mannwhitney=True
        )
        sig_cnv = cnv_results[cnv_results['p_value'] <= 0.01]['feature']
        if sig_cnv.empty:
            sig_cnv = cnv_results.nsmallest(5, 'p_value')['feature']
        cnv_pathways_initial = gsva_scores_cnv_all.loc[train_ids, sig_cnv]

        # === Step 4: Run DESeq2-style filtering for raw RNA ===
        subtype_meta_train['Disease_Status'] = pd.Categorical(
            subtype_meta_train['Disease_Status'], categories=["non-recurrent", "recurrent"]
        )
        de_results = process_deseq_results(
            rna_expression=data['rna_raw'].loc[train_ids],
            metadata=subtype_meta_train[['Disease_Status']],
            design_factors=['Disease_Status'],
            contrast=["Disease_Status", "recurrent", "non-recurrent"],
            pvalue_threshold=0.01, padj_threshold=0.25, logfold_threshold=1.5,
            base_mean_threshold=50, gene_sum_threshold=100
        )
        rna_log2 = np.log2(data['rna'] + 1)
        de_genes = [g for g in de_results.index if g in rna_log2.columns]
        de_expr_train_initial = rna_log2.loc[train_ids, de_genes]
        de_expr_test = rna_log2.loc[test_ids, de_genes]

        # === Step 5: Merge features into blocks ===
        X_train_full, X_train_blocks = merge_feature_blocks(
            subtype_meta=subtype_meta_train,
            mutation_df=mutation_pathways_train,
            rna_df=rna_pathways_initial,
            cnv_df=cnv_pathways_initial,
            tf_df=data['tf'].loc[train_ids],
            tmb_df=data['tmb'].loc[train_ids],
            hoadley_df=data['hoadley'].loc[train_ids],
            cibersort_df=data['cibersort'].loc[train_ids],
            pathogenic_df=data['pathogenic_count'].loc[train_ids],
            cnv_pct_df=data['cnv_pct_altered'].loc[train_ids],
            clinical_df=data['clinical'].loc[train_ids],
            de_gene_df=de_expr_train_initial,
            return_separate_blocks=True
        )

        X_test_full, X_test_blocks = merge_feature_blocks(
            subtype_meta=subtype_meta_test,
            mutation_df=mutation_pathways_test,
            rna_df=gsva_scores_rna_all.loc[test_ids, sig_rna],
            cnv_df=gsva_scores_cnv_all.loc[test_ids, sig_cnv],
            tf_df=data['tf'].loc[test_ids],
            tmb_df=data['tmb'].loc[test_ids],
            hoadley_df=data['hoadley'].loc[test_ids],
            cibersort_df=data['cibersort'].loc[test_ids],
            pathogenic_df=data['pathogenic_count'].loc[test_ids],
            cnv_pct_df=data['cnv_pct_altered'].loc[test_ids],
            clinical_df=data['clinical'].loc[test_ids],
            de_gene_df=de_expr_test,
            return_separate_blocks=True
        )

        # ✅ Filter to selected blocks (both times after merge)
        if selected_blocks is not None:
            X_train_blocks = {k: v for k, v in X_train_blocks.items() if k in selected_blocks}
            X_test_blocks  = {k: v for k, v in X_test_blocks.items()  if k in selected_blocks}
            print(f"🔎 Using selected blocks only: {list(X_train_blocks.keys())}")

        # === Step 6: Preprocess each block (recurrence labels) ===
        X_train_proc_blocks = {}
        X_test_proc_blocks = {}
        for block_name, block_info in X_train_blocks.items():
            print(f"\n🧪 Preprocessing Block: {block_name}")
            X_block_train = block_info['data']
            X_block_test  = X_test_blocks[block_name]['data']
            process_type  = block_info['process']

            X_train_proc, X_test_proc = preprocess_block(
                X_block_train, process_type, block_name,
                X_test=X_block_test, y_train=y_train,
                output_path=fold_output_dir, fold=fold
            )
            X_train_proc_blocks[block_name] = X_train_proc
            X_test_proc_blocks[block_name]  = X_test_proc

        # === STEP 7: Concatenate all processed training blocks ===
        print("🔗 Concatenating all preprocessed training blocks...")
        X_train_concat = pd.concat(X_train_proc_blocks.values(), axis=1).fillna(0)
        X_test_concat  = pd.concat(X_test_proc_blocks.values(),  axis=1).fillna(0)

        # ── RSF: train-only fit, predict train & test risks; bin by train tertiles ──
        train_scores, test_scores = fit_rsf_and_predict(X_train_concat, subtype_meta_train, X_test_concat)


# can update to change where high / low is 

        low_cut  = np.percentile(train_scores, 40)
        high_cut = np.percentile(train_scores, 60)

        test_low_cut  = np.percentile(train_scores, 40)
        test_high_cut = np.percentile(train_scores, 60)


        y_train_mc = pd.Series(
            np.where(train_scores <= low_cut, 0, np.where(train_scores > high_cut, 2, 1)),
            index=X_train_concat.index
        )
        y_test_mc  = pd.Series(
            np.where(test_scores  <= test_low_cut, 0, np.where(test_scores  > test_high_cut, 2, 1)),
            index=X_test_concat.index
        )



        print("Train classes:", dict(zip(*np.unique(y_train_mc, return_counts=True))))
        print("Test  classes:", dict(zip(*np.unique(y_test_mc,  return_counts=True))))

        # ── Feature selection using TRAIN extremes only (high vs low); apply to all ──
        ext_mask = y_train_mc.isin([0, 2])
        ext_ids  = y_train_mc.index[ext_mask]
        risk_group = y_train_mc.loc[ext_ids].map({0: 'low', 2: 'high'})

        # RNA GSVA selection
        rna_results = run_significance_tests(
            gsva_scores_rna_all.loc[ext_ids],
            pd.DataFrame({'RiskClass': risk_group}, index=ext_ids),
            group_col='RiskClass', group1='high', group2='low', use_mannwhitney=True
        )
        sig_rna = rna_results[rna_results['p_value'] <= 0.01]['feature']
        if sig_rna.empty:
            sig_rna = rna_results.nsmallest(5, 'p_value')['feature']
        rna_pathways_train = gsva_scores_rna_all.loc[y_train_mc.index, sig_rna]
        rna_pathways_test  = gsva_scores_rna_all.loc[y_test_mc.index,  sig_rna]

        # CNV GSVA selection
        cnv_results = run_significance_tests(
            gsva_scores_cnv_all.loc[ext_ids],
            pd.DataFrame({'RiskClass': risk_group}, index=ext_ids),
            group_col='RiskClass', group1='high', group2='low', use_mannwhitney=True
        )
        sig_cnv = cnv_results[cnv_results['p_value'] <= 0.01]['feature']
        if sig_cnv.empty:
            sig_cnv = cnv_results.nsmallest(5, 'p_value')['feature']
        cnv_pathways_train = gsva_scores_cnv_all.loc[y_train_mc.index, sig_cnv]
        cnv_pathways_test  = gsva_scores_cnv_all.loc[y_test_mc.index,  sig_cnv]

        # DE-style raw RNA on extremes only; apply genes to all train/test
        sub_meta_ext = pd.DataFrame({'RiskClass': risk_group}, index=ext_ids)
        de_results = process_deseq_results(
            rna_expression=data['rna_raw'].loc[ext_ids],
            metadata=sub_meta_ext,
            design_factors=['RiskClass'],
            contrast=["RiskClass", "high", "low"],
            pvalue_threshold=0.01, padj_threshold=0.25, logfold_threshold=1.5,
            base_mean_threshold=50, gene_sum_threshold=100
        )
        de_genes = [g for g in de_results.index if g in rna_log2.columns]
        de_expr_train = rna_log2.loc[y_train_mc.index, de_genes]
        de_expr_test  = rna_log2.loc[y_test_mc.index,  de_genes]

        # Mutations: recompute TRAIN with the selected set; align TEST to selected cols
        mutation_pathways_train_sel, _ = process_mutation_pathways(
            M_matrix=data['mut'].loc[y_train_mc.index],
            subtype_meta=subtype_meta_train.loc[y_train_mc.index],
            pathway_dicts=all_pathway_dicts,
            pathway_options=pathway_options,
            output_dir=None,
            subtype=subtype
        )
        mut_sig_cols = mutation_pathways_train_sel.columns
        mutation_pathways_train = mutation_pathways_train_sel
        mutation_pathways_test  = mutation_full_test[mut_sig_cols].reindex(index=y_test_mc.index).fillna(0)

        # Re-merge blocks with selected features
        X_train_full, X_train_blocks = merge_feature_blocks(
            subtype_meta=subtype_meta_train.loc[y_train_mc.index],
            mutation_df=mutation_pathways_train,
            rna_df=rna_pathways_train,
            cnv_df=cnv_pathways_train,
            tf_df=data['tf'].loc[y_train_mc.index],
            tmb_df=data['tmb'].loc[y_train_mc.index],
            hoadley_df=data['hoadley'].loc[y_train_mc.index],
            cibersort_df=data['cibersort'].loc[y_train_mc.index],
            pathogenic_df=data['pathogenic_count'].loc[y_train_mc.index],
            cnv_pct_df=data['cnv_pct_altered'].loc[y_train_mc.index],
            clinical_df=data['clinical'].loc[y_train_mc.index],
            de_gene_df=de_expr_train,
            return_separate_blocks=True
        )
        X_test_full, X_test_blocks = merge_feature_blocks(
            subtype_meta=subtype_meta_test.loc[y_test_mc.index],
            mutation_df=mutation_pathways_test,
            rna_df=rna_pathways_test,
            cnv_df=cnv_pathways_test,
            tf_df=data['tf'].loc[y_test_mc.index],
            tmb_df=data['tmb'].loc[y_test_mc.index],
            hoadley_df=data['hoadley'].loc[y_test_mc.index],
            cibersort_df=data['cibersort'].loc[y_test_mc.index],
            pathogenic_df=data['pathogenic_count'].loc[y_test_mc.index],
            cnv_pct_df=data['cnv_pct_altered'].loc[y_test_mc.index],
            clinical_df=data['clinical'].loc[y_test_mc.index],
            de_gene_df=de_expr_test,
            return_separate_blocks=True
        )

        if selected_blocks is not None:
            X_train_blocks = {k: v for k, v in X_train_blocks.items() if k in selected_blocks}
            X_test_blocks  = {k: v for k, v in X_test_blocks.items()  if k in selected_blocks}

        # ── Train multiclass per block ──
        for block_name, block_info in X_train_blocks.items():
            print(f"\n🧪 Block: {block_name}")
            X_block_train = block_info["data"]
            X_block_test  = X_test_blocks[block_name]["data"]
            process_type  = block_info["process"]

            # Final preprocessing with risk multiclass labels available
            X_train_proc, X_test_proc = preprocess_block(
                X_block_train, process_type, block_name,
                y_train=y_train_mc.loc[X_block_train.index], X_test=X_block_test,
                output_path=fold_output_dir, fold=fold
            )
            # Align columns
            X_test_proc = X_test_proc.reindex(columns=X_train_proc.columns, fill_value=0)

            # Inner-CV tuning (multiclass scoring)
            safe_cv = get_safe_cv(y_train_mc.loc[X_train_proc.index], desired_folds=inner_folds)
            best_model, model_type, best_score, best_params = tune_model_multiclass(
                X_train_proc, y_train_mc.loc[X_train_proc.index],
                model_grid, cv=safe_cv
            )

            # Predict proba for 3 classes
            if hasattr(best_model, "predict_proba"):
                proba_test  = best_model.predict_proba(X_test_proc)
                proba_train = best_model.predict_proba(X_train_proc)
            else:
                proba_test  = softmax(best_model.decision_function(X_test_proc))
                proba_train = softmax(best_model.decision_function(X_train_proc))

            # Ground truth (multiclass)
            y_test_block = y_test_mc.loc[X_test_proc.index].to_numpy()
            y_pred_labels = np.argmax(proba_test, axis=1)

            # 1) Multiclass ROC (OvR) — saved per fold/block
            model_classes = getattr(best_model, "classes_", np.arange(proba_test.shape[1]))
            plot_multiclass_roc(
                y_true=y_test_block,
                proba=proba_test,
                model_classes=model_classes,
                title=f"{block_name} • Fold {fold+1} (OvR ROC)",
                outpath=os.path.join(roc_dir, f"ovr_roc_{block_name}_fold{fold+1}.png")
            )

            # 2) Collapse to binary + binary ROC vs true recurrence (clinical endpoint)
            y_true_bin = label_encoder.transform(y.loc[X_test_proc.index])  # 0/1
            present_bin = np.unique(y_true_bin)
            if present_bin.size == 2:
                p_bin_ignore = collapse_proba_to_binary(proba_test, rule="ignore_mid")
                auc_bin_ignore = plot_binary_roc(
                    y_true_bin, p_bin_ignore,
                    title=f"{block_name} • Fold {fold+1} Binary ROC (ignore_mid)",
                    outpath=os.path.join(roc_dir, f"binary_roc_ignore_{block_name}_fold{fold+1}.png")
                )

                p_bin_w05 = collapse_proba_to_binary(proba_test, rule="weighted", alpha=0.5)
                auc_bin_w05 = plot_binary_roc(
                    y_true_bin, p_bin_w05,
                    title=f"{block_name} • Fold {fold+1} Binary ROC (weighted α=0.5)",
                    outpath=os.path.join(roc_dir, f"binary_roc_weighted05_{block_name}_fold{fold+1}.png")
                )
            else:
                auc_bin_ignore = np.nan
                auc_bin_w05 = np.nan

            # 3) Multiclass metrics
            auc_mc = safe_multiclass_auc(y_test_block, proba_test, model_classes=model_classes)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_test_block, y_pred_labels, average='macro', zero_division=0
            )
            acc = accuracy_score(y_test_block, y_pred_labels)

            performance_metrics.append({
                'block_name': block_name, 'fold': fold + 1,
                'auc_macro_ovo': auc_mc, 'accuracy': acc,
                'precision_macro': prec, 'recall_macro': rec, 'f1_macro': f1,
                'auc_binary_ignore_mid': auc_bin_ignore,
                'auc_binary_weighted_0.5': auc_bin_w05,
                'n_test_samples': len(y_test_block),
                'model_name': type(best_model).__name__,
                'params': str(best_model.get_params() if hasattr(best_model, "get_params") else {})
            })



            # --- Save OOF preds (test fold) WITH clinical labels ---
            y_true_clin_series = y.loc[X_test_proc.index]                      # strings, e.g., recurrent/non-recurrent
            y_true_clin_bin    = label_encoder.transform(y_true_clin_series)    # 0/1 via your encoder

            oof_predictions = pd.concat([oof_predictions, pd.DataFrame({
                'sample_id': X_test_proc.index,
                'true_label_mc': y_test_block,
                'pred_proba_0': proba_test[:,0],
                'pred_proba_1': proba_test[:,1],
                'pred_proba_2': proba_test[:,2],
                'block_name': block_name,
                'fold': fold,
                # --- NEW: carry clinical labels alongside multiclass risk label
                'clinical_label': y_true_clin_series.values,
                'clinical_label_bin': y_true_clin_bin
            })], axis=0)

            # --- Save TRAIN preds (optional) WITH clinical labels (aligned to train index) ---
            y_train_clin_series = y.loc[X_train_proc.index]
            y_train_clin_bin    = label_encoder.transform(y_train_clin_series)

            training_predictions = pd.concat([training_predictions, pd.DataFrame({
                'sample_id': X_train_proc.index,
                'true_label_mc': y_train_mc.loc[X_train_proc.index].values,
                'pred_proba_0': proba_train[:,0],
                'pred_proba_1': proba_train[:,1],
                'pred_proba_2': proba_train[:,2],
                'block_name': block_name,
                'set': 'train',
                'fold': fold,
                # --- NEW
                'clinical_label': y_train_clin_series.values,
                'clinical_label_bin': y_train_clin_bin
            })], axis=0)


            # Save model + info
            joblib.dump(best_model, os.path.join(cv_models_dir, f"{block_name}_fold{fold + 1}.pkl"))
            with open(os.path.join(cv_models_dir, f"{block_name}_fold{fold + 1}_info.txt"), 'w') as f:
                f.write(f"Model Type: {model_type}\n")
                f.write(f"Best Score: {best_score}\n")
                f.write(f"Best Params: {best_params}\n")

            # (Optional) SHAP / Coefs can be re-enabled here with multiclass-safe guards

    # ───────────────────────────────
    # Save outputs
    # ───────────────────────────────
    oof_predictions.to_csv(os.path.join(fold_output_dir, "oof_predictions_mc.csv"), index=False)
    training_predictions.to_csv(os.path.join(fold_output_dir, "training_predictions_mc.csv"), index=False)

        # =========================
    # NEW: alignment summaries
    # =========================
    align_dir = os.path.join(fold_output_dir, "alignment")
    os.makedirs(align_dir, exist_ok=True)

    # (A) Global alignment of RSF tertiles vs clinical labels
    #     Use a single row per sample to avoid block-duplication
    first_rows = (oof_predictions[['sample_id','true_label_mc','clinical_label','clinical_label_bin']]
                .drop_duplicates('sample_id')
                .set_index('sample_id'))

    ctab_global = pd.crosstab(first_rows['true_label_mc'], first_rows['clinical_label'], dropna=False)
    ctab_global.to_csv(os.path.join(align_dir, "risk_tertile_vs_clinical_crosstab.csv"))
    (ctab_global.div(ctab_global.sum(axis=1).replace(0, np.nan), axis=0)
    ).to_csv(os.path.join(align_dir, "risk_tertile_vs_clinical_crosstab_rowpct.csv"))

    tertile_event = (first_rows.groupby('true_label_mc')['clinical_label_bin']
                    .agg(n='count', event_rate='mean')
                    .reindex([0,1,2], fill_value=0))
    tertile_event.to_csv(os.path.join(align_dir, "tertile_event_rates.csv"))

    # (B) Per-block alignment of *predicted class* vs clinical label
    rows = []
    for blk in sorted(oof_predictions['block_name'].unique()):
        sub = oof_predictions[oof_predictions['block_name'] == blk].copy()
        if sub.empty: 
            continue

        P = sub[['pred_proba_0','pred_proba_1','pred_proba_2']].to_numpy()
        sub['pred_class'] = P.argmax(axis=1)

        # crosstabs (counts + row%)
        ct = pd.crosstab(sub['pred_class'], sub['clinical_label'], dropna=False)
        ct.to_csv(os.path.join(align_dir, f"crosstab_predclass_vs_clinical_{blk}.csv"))
        (ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0)
        ).to_csv(os.path.join(align_dir, f"crosstab_predclass_vs_clinical_rowpct_{blk}.csv"))

        # event rates by predicted class
        ev = (sub.groupby('pred_class')['clinical_label_bin']
            .mean().reindex([0,1,2], fill_value=np.nan))

        # binary AUC using the "ignore_mid" collapse (only if both classes present)
        yb = sub['clinical_label_bin'].to_numpy()
        if np.unique(yb).size == 2:
            p_rec = P[:, 2]; p_non = P[:, 0]
            scores_ignore_mid = p_rec / (p_rec + p_non + 1e-9)
            auc_b = roc_auc_score(yb, scores_ignore_mid)
        else:
            auc_b = np.nan

        rows.append({
            "block_name": blk,
            "n_samples": int(len(sub)),
            "event_rate_pred0": float(ev.loc[0]) if not np.isnan(ev.loc[0]) else np.nan,
            "event_rate_pred1": float(ev.loc[1]) if not np.isnan(ev.loc[1]) else np.nan,
            "event_rate_pred2": float(ev.loc[2]) if not np.isnan(ev.loc[2]) else np.nan,
            "binary_auc_ignore_mid": float(auc_b) if auc_b == auc_b else np.nan
        })

    pd.DataFrame(rows).to_csv(os.path.join(align_dir, "alignment_by_block_summary.csv"), index=False)


    oof_path = os.path.join(fold_output_dir, "oof_predictions_mc.csv")
    oof_df = pd.read_csv(oof_path)
    ext_outdir = os.path.join(fold_output_dir, "extremes_eval")
    evaluate_high_low_from_oof(oof_df, outdir=ext_outdir, ks=(0.05, 0.10, 0.20), make_plots=True)
    print(f"🧪 Extremes evaluation saved under: {ext_outdir}")

    
    perf_df = pd.DataFrame(performance_metrics)
    perf_df.to_csv(os.path.join(fold_output_dir, "performance_summary_mc.csv"), index=False)

        # === Summary ROC plots across all folds (one line per block) ===
    ovr_path = os.path.join(roc_dir, "ovr_micro_roc_blocks.png")
    plot_summary_ovr_micro_rocs(oof_predictions, outpath=ovr_path)

    bin_ignore_path  = os.path.join(roc_dir, "binary_roc_blocks_ignore_mid.png")
    bin_weighted_path= os.path.join(roc_dir, "binary_roc_blocks_weighted05.png")
    plot_summary_binary_rocs(
        oof_df=oof_predictions,
        y_series=y,                      # original recurrence labels (index = sample_id)
        label_encoder=label_encoder,
        outpath_ignore=bin_ignore_path,
        outpath_weighted=bin_weighted_path,
        alpha=0.5
    )
    print(f"📈 Saved averaged ROC plots:\n  • {ovr_path}\n  • {bin_ignore_path}\n  • {bin_weighted_path}")


    # NEW: block-wise and overall summaries with CI
    summarize_metrics(perf_df, fold_output_dir)

    print(f"📁 Saved OOF / training predictions and metrics to: {fold_output_dir}")

    # Meta-features for stacking: pivot per block probs (3 per block)
    if not oof_predictions.empty:
        oof_wide = (oof_predictions
                    .pivot_table(index='sample_id',
                                 columns='block_name',
                                 values=['pred_proba_0','pred_proba_1','pred_proba_2'],
                                 aggfunc='mean'))
        # Flatten columns
        oof_wide.columns = [f"{blk}_{col.split('_')[-1]}"
                            for col, blk in zip(oof_wide.columns.get_level_values(0),
                                                oof_wide.columns.get_level_values(1))]
        # Align labels (one row per sample)
        y_meta_series = (oof_predictions[['sample_id','true_label_mc']]
                         .drop_duplicates(subset='sample_id')
                         .set_index('sample_id').loc[oof_wide.index, 'true_label_mc'])
        X_meta = oof_wide.fillna(0).values
        y_meta = y_meta_series.values
    else:
        X_meta = np.array([])
        y_meta = np.array([])

    return X_meta, y_meta, best_models_by_block





def tune_model(X, y, model_grid, cv=5):
    from sklearn.model_selection import GridSearchCV

    best_model = None
    best_name = None
    best_auc = -float("inf")
    best_params = None

    for model_name, (model, param_grid) in model_grid.items():
        try:
            print(f"🔍 Tuning {model_name}...")
            gs = GridSearchCV(model, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1)
            gs.fit(X, y)
        except Exception as e:
            print(f"⚠️ GridSearchCV failed for {model_name} with n_jobs=-1: {e}")
            print("🔁 Retrying with n_jobs=1")
            gs = GridSearchCV(model, param_grid, scoring='roc_auc', cv=cv, n_jobs=1)
            gs.fit(X, y)

        mean_auc = gs.best_score_
        print(f"📊 {model_name} Mean CV AUC: {mean_auc:.4f}")
        print(f"✅ Best params: {gs.best_params_}")

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_model = gs.best_estimator_
            best_name = model_name
            best_params = gs.best_params_

    return best_model, best_name, best_auc, best_params


def evaluate_model(model, X_test, y_test, mean_fpr=None):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'y_pred': y_pred,
        'y_prob': y_pred_proba
    }

    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        metrics['auc'] = auc(fpr, tpr)
        metrics['fpr_raw'] = fpr
        metrics['tpr_raw'] = tpr

        if mean_fpr is not None:
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[-1] = 1.0
            metrics['tpr'] = interp_tpr

    return metrics



def extract_logistic_coefficients(model, X_train, y_train, fold, model_name):
    coef_df = pd.DataFrame({
        'Fold': fold + 1,
        'Feature': X_train.columns,
        'Coefficient': model.coef_[0],
        'Coefficient_Sign': np.sign(model.coef_[0]),
        'Model': model_name
    })
    try:
        X_const = sm.add_constant(X_train)
        sm_model = sm.Logit(y_train, X_const).fit(disp=False)
        coef_df['p-value'] = sm_model.pvalues[1:].values
    except Exception:
        coef_df['p-value'] = np.nan
    return coef_df


def extract_rf_importance(model, X_train, fold, model_name):
    return pd.DataFrame({
        'Fold': fold + 1,
        'Feature': X_train.columns,
        'RF_Importance': model.feature_importances_,
        'Model': model_name
    })


import shap
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def compute_shap(model, X_train, X_test, model_name, fold):
    try:
        if isinstance(model, LogisticRegression):
            # Use LinearExplainer directly
            explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
            shap_vals = explainer(X_test)
        else:
            # Let SHAP auto-select (likely TreeExplainer for trees)
            explainer = shap.Explainer(model, X_test)
            shap_vals = explainer(X_test)

        # Handle binary vs multiclass outputs
        if hasattr(shap_vals, 'values') and shap_vals.values.ndim == 3:
            shap_values_used = shap_vals.values[:, :, 1]
        else:
            shap_values_used = shap_vals.values

        shap_df = pd.DataFrame({
            'Fold': fold + 1,
            'Feature': X_test.columns,
            'Mean_ABS_SHAP': np.abs(shap_values_used).mean(axis=0),
            'Model': model_name
        })
        return shap_df, shap_vals

    except Exception as e:
        print(f"❌ SHAP computation failed for {model_name} (fold {fold + 1}): {e}")
        return None, None


def rank_features_from_best_model_fallback(best_model_name, all_importances, block_name='meta',output_path=None):
    """
    Rank features from the best-performing model using SHAP if available,
    otherwise fall back to model-specific feature importance (coefs, RF importance).
    """
    # # Step 1: Identify best model by metric
    # best_model_row = metrics_df.sort_values(by=metric, ascending=False).iloc[0]
    # best_model_name = best_model_row['Classifier']
    # print(f"🏆 Best model by {metric}: {best_model_name}")

    # Step 2: Collect all importance info for that model
    # Step 2: Collect all importance info for that model
    model_importances = [
        df for df in all_importances if not df.empty and 'Model' in df.columns and df['Model'].iloc[0] == best_model_name
    ]

    if not model_importances:
        print(f"❌ No importance data found for model '{best_model_name}'. Skipping ranking.")
        return pd.DataFrame()

    importance_df = pd.concat(model_importances, axis=0)

    if importance_df.empty:
        print("❌ No importance data available for this model.")
        return pd.DataFrame()

    # Step 3: Rank by SHAP if available, else fallback
    if 'Mean_ABS_SHAP' in importance_df['Importance_Type'].values:
        print("📈 Ranking features using SHAP values.")
        ranked = (
            importance_df[importance_df['Importance_Type'] == 'Mean_ABS_SHAP']
            .groupby('Feature')['Mean_ABS_SHAP']
            .mean()
            .sort_values(ascending=False)
            .reset_index()
#            .rename(columns={'Mean_ABS_SHAP': 'Importance'})
        )
    elif 'Coefficient' in importance_df['Importance_Type'].values:
        print("📈 Ranking features using logistic regression coefficients.")
        ranked = (
            importance_df[importance_df['Importance_Type'] == 'Coefficient']
            .groupby('Feature')['Coefficient']
            .mean()
            .abs()
            .sort_values(ascending=False)
            .reset_index()
 #           .rename(columns={'Coefficient': 'Importance'})
        )
    elif 'RF_Importance' in importance_df['Importance_Type'].values:
        print("📈 Ranking features using random forest importances.")
        ranked = (
            importance_df[importance_df['Importance_Type'] == 'RF_Importance']
            .groupby('Feature')['RF_Importance']
            .mean()
            .sort_values(ascending=False)
            .reset_index()
 #           .rename(columns={'RF_Importance': 'Importance'})
        )
    elif 'SVM_Coef' in importance_df['Importance_Type'].values:
        print("📈 Ranking features using SVM coefficients.")
        ranked = (
            importance_df[importance_df['Importance_Type'] == 'SVM_Coef']
            .groupby('Feature')['SVM_Coef']
            .mean()
            .abs()
            .sort_values(ascending=False)
            .reset_index()
 #           .rename(columns={'SVM_Coef': 'Importance'})
        )
    else:
        print("❌ No recognized importance type found.")
        return pd.DataFrame()

    if output_path:
        out_file = os.path.join(output_path, f"{block_name}_ranked_features_best_model_{best_model_name.replace(' ', '_')}.csv")
        ranked.to_csv(out_file, index=False)
        print(f"💾 Saved ranked features to {out_file}")

    return ranked




# === Function: Identify Best Model ===
def identify_best_model(metrics_df, metric='auc'):
    """
    Identify the best-performing model based on a chosen evaluation metric.

    Parameters:
        metrics_df (pd.DataFrame): Output from save_metrics(), must include 'Classifier' column.
        metric (str): Metric to sort by (e.g. 'auc', 'f1')

    Returns:
        str: Best performing model name
    """
    if 'Classifier' not in metrics_df.columns:
        # If 'Classifier' is not present, assume it's in the index and reset
        metrics_df = metrics_df.reset_index().rename(columns={'index': 'Classifier'})
    
    if metric not in metrics_df.columns:
        raise ValueError(f"Metric '{metric}' not found in metrics_df. Available columns: {metrics_df.columns.tolist()}")

    best_model_row = metrics_df.sort_values(by=metric, ascending=False).iloc[0]
    return best_model_row['Classifier']


from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def encode_clinical_features(df):
    """
    Encode clinical features using predefined mappings.

    Args:
        df (pd.DataFrame): Clinical data with raw values.

    Returns:
        pd.DataFrame: Encoded clinical dataframe.
    """

    # Binary encodings
    binary_map = {
        'chemo_med': {'yes': 1, 'no chemo': 0},
        'hormtxt_evr': {'ever': 1, 'never': 0},
        'herceptin_med': {'ever': 1, 'never': 0},
        'er_lab': {'positive(alread 3-8)': 1, 'negative': 0},
        'pr_lab': {'positive(alread 3-8)': 1, 'negative(alread 0-2)': 0},
        'ki67_lab': {'positive(>14%)': 1, 'negative(<=14%)': 0},
        'ck56_lab': {'positive': 1, 'negative': 0},
        'egfr_lab': {'positive': 1, 'negative': 0},
        'lymphnode_invol': {'yes': 1, 'no': 0}
    }

    # Ordinal encodings
    ordinal_map = {
        'ajcc_css': {'I': 1, 'II': 2, 'III': 3},
        'her2_lab': {'negative(1-2)': 0, 'inderterminate(3)': 1, 'positive(4)': 2},
        'tumor_size': {'<=2cm': 0, '2.1-5.0cm': 1, '>5.0cm': 2},
        'lymphocyteresponse': {'Low': 0, 'Intermediate': 1, 'High': 2},
        'bmidx_medint_who': {'<25': 0, '25-<30': 1, '30+': 2}
        # 'meno_med_int_gr': {
        #     '(1)pre-meno': 0,
        #     '(2)peri-meno': 1,
        #     '(3)post-meno(natual meno/bilateral oopho)': 2
        # }
    }

    df_encoded = df.copy()

    # Apply binary mappings
    for col, mapping in binary_map.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(mapping)

    # Apply ordinal mappings
    for col, mapping in ordinal_map.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(mapping)

    # Convert to numeric (helpful for ensuring compatibility with ML models)
    df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce')

        # Fill NaNs in lab markers with 0 (assuming negative)
    labs_to_fill = ['ck56_lab', 'egfr_lab', 'her2_lab', 'ki67_lab', 'er_lab', 'pr_lab', 'lymphocyteresponse']
    for col in labs_to_fill:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].fillna(0)


    return df_encoded





def init_metric_dict(param_grids):
    """
    Initialize a nested dictionary for tracking model metrics.

    Args:
        param_grids (dict): Dictionary of model names and hyperparameters.

    Returns:
        dict: Dictionary with empty lists for each metric per classifier.
    """
    return {
        name: {
            'accuracies': [],
            'precisions': [],
            'recalls': [],
            'f1s': [],
            'aucs': [],
            'tprs': []
        }
        for name in param_grids
    }


#from model_utils import train_and_tune_model

def train_final_model(X, y, model_name, best_params):
    """
    Train the final model on the full training set using the best model and its parameters.

    Args:
        X (pd.DataFrame): Training features
        y (pd.Series or np.ndarray): Training labels
        model_name (str): Name of the model to train
        best_params (dict): Best hyperparameters found in inner CV

    Returns:
        Trained model
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    model_lookup = {
        "Logistic Regression L1": LogisticRegression,
        "Logistic Regression L2": LogisticRegression,
        "Random Forest": RandomForestClassifier,
        "SVM": SVC,
        "XGBoost": XGBClassifier,
        "Gradient Boosting": GradientBoostingClassifier,
        "LightGBM": LGBMClassifier,
        "CatBoost": CatBoostClassifier,
        "Naive Bayes": GaussianNB,
    }

    if model_name not in model_lookup:
        raise ValueError(f"Model {model_name} not recognized. Available models: {list(model_lookup.keys())}")


    print(X.describe())
    print((X == 0).mean())  # % of zero entries

    model_class = model_lookup[model_name]
    model = model_class(**best_params)
    model.fit(X, y)

    return model




def run_model_training_and_shap(
    X_train, X_test, y_train_enc, y_test_enc, sig_features,
    results, all_predictions, all_importances, param_grids,
    output_path, fold, mean_fpr=None, block_name=None
):
    import numpy as np
    import pandas as pd
    import shap
    import os
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    from sklearn.svm import SVC
    import statsmodels.api as sm
    from model_utils import evaluate_model
    from output_utils import save_top_genes_by_pc

    best_overall_model_name = None
    best_overall_score = -np.inf
    best_overall_params = None

    # Sanitize inputs
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.applymap(lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(np.atleast_1d(x)) == 1 else x)
        X_test = X_test.astype(np.float32)

    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.applymap(lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(np.atleast_1d(x)) == 1 else x)
        X_train = X_train.astype(np.float32)

    for model_name, param_grid in param_grids.items():
        print(f"🧠 Training: {model_name} [Block: {block_name}]")
        try:
            clf, best_params = train_and_tune_model(model_name, X_train, y_train_enc, param_grid)
            print(f"✅ Model training completed. [Block: {block_name}]")

            metrics = evaluate_model(clf, X_test, y_test_enc)
            if metrics is None:
                print(f"⚠️ Metrics not returned for {model_name} in fold {fold + 1}. Skipping.")
                continue

            print("📊 Evaluation completed.")
            metric_map = {
                'accuracy': 'accuracies',
                'precision': 'precisions',
                'recall': 'recalls',
                'f1': 'f1s',
                'auc': 'aucs'
            }

            for m in metric_map:
                if m in metrics:
                    results[model_name][metric_map[m]].append(metrics[m])

            if 'fpr_raw' in metrics and 'tpr_raw' in metrics and mean_fpr is not None:
                fpr = metrics['fpr_raw']
                tpr = metrics['tpr_raw']
                tpr_interp = np.interp(mean_fpr, fpr, tpr)
                tpr_interp[-1] = 1.0
                results[model_name]['tprs'].append(tpr_interp)

            if 'y_pred' in metrics and metrics['y_pred'] is not None:
                for i, idx in enumerate(X_test.index):
                    all_predictions.append({
                        'Classifier': model_name,
                        'Sample_Index': idx,
                        'True_Label': y_test_enc[i],
                        'Predicted_Label': metrics['y_pred'][i],
                        'Predicted_Probability': metrics['y_prob'][i] if metrics.get('y_prob') is not None else None
                    })

            model_importance_df = None
            print(f"🧠 Calculating SHAP values... [Block: {block_name}]")

            if isinstance(clf, LogisticRegression):
                coef_df = pd.DataFrame({
                    'Fold': fold + 1,
                    'Feature': X_train.columns,
                    'Coefficient': clf.coef_[0],
                    'Coefficient_Sign': np.sign(clf.coef_[0]),
                    'Model': model_name,
                    'Importance_Type': 'Coefficient',
                    'Block': block_name
                })
                try:
                    X_const = sm.add_constant(X_train)
                    sm_model = sm.Logit(y_train_enc, X_const).fit(disp=False)
                    coef_df['p-value'] = sm_model.pvalues[1:].values
                except Exception as e:
                    print(f"⚠️ Statsmodels failed: {e}")
                    coef_df['p-value'] = np.nan
                all_importances.append(coef_df)

                explainer = shap.Explainer(
                    lambda x: clf.predict_proba(pd.DataFrame(x, columns=X_train.columns))[:, 1],
                    X_train.values
                )
                shap_vals = explainer(X_test.values)
                model_importance_df = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Mean_ABS_SHAP': np.abs(shap_vals.values).mean(axis=0),
                    'Model': model_name,
                    'Fold': fold + 1,
                    'Importance_Type': 'Mean_ABS_SHAP',
                    'Block': block_name
                })

            elif isinstance(clf, (GradientBoostingClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier)):
                explainer = shap.Explainer(clf, X_test)
                shap_vals = explainer(X_test)
                shap_values_used = shap_vals.values[:, :, 1] if hasattr(shap_vals, 'values') and shap_vals.values.ndim == 3 else shap_vals.values
                model_importance_df = pd.DataFrame({
                    'Feature': X_test.columns,
                    'Mean_ABS_SHAP': np.abs(shap_values_used).mean(axis=0),
                    'Model': model_name,
                    'Fold': fold + 1,
                    'Importance_Type': 'Mean_ABS_SHAP',
                    'Block': block_name
                })

            elif isinstance(clf, RandomForestClassifier):
                rf_importance = pd.DataFrame({
                    'Fold': fold + 1,
                    'Feature': X_train.columns,
                    'RF_Importance': clf.feature_importances_,
                    'Model': model_name,
                    'Importance_Type': 'RF_Importance',
                    'Block': block_name
                })
                all_importances.append(rf_importance)

                try:
                    explainer = shap.Explainer(clf, X_test)
                    shap_vals = explainer(X_test)
                    model_importance_df = pd.DataFrame({
                        'Feature': X_test.columns,
                        'Mean_ABS_SHAP': np.abs(shap_vals.values).mean(axis=0),
                        'Model': model_name,
                        'Fold': fold + 1,
                        'Importance_Type': 'Mean_ABS_SHAP',
                        'Block': block_name
                    })
                except Exception as e:
                    print(f"⚠️ SHAP failed for Random Forest in fold {fold + 1} [Block: {block_name}]: {e}")

            elif isinstance(clf, SVC) and clf.kernel == 'linear':
                svm_coef = pd.DataFrame({
                    'Fold': fold + 1,
                    'Feature': X_train.columns,
                    'SVM_Coef': clf.coef_[0],
                    'Model': model_name,
                    'Importance_Type': 'SVM_Coef',
                    'Block': block_name
                })
                all_importances.append(svm_coef)
                if clf.probability:
                    explainer = shap.Explainer(
                        lambda x: clf.predict_proba(pd.DataFrame(x, columns=X_train.columns))[:, 1],
                        X_train.values
                    )
                    shap_vals = explainer(X_test.values)
                    model_importance_df = pd.DataFrame({
                        'Feature': X_train.columns,
                        'Mean_ABS_SHAP': np.abs(shap_vals.values).mean(axis=0),
                        'Model': model_name,
                        'Fold': fold + 1,
                        'Importance_Type': 'Mean_ABS_SHAP',
                        'Block': block_name
                    })

            else:
                print(f"⚠️ SHAP not implemented for {model_name} [Block: {block_name}]. Skipping SHAP.")

            if model_importance_df is not None:
                all_importances.append(model_importance_df)

                # Save to CSV
                if output_path:
                    fname = f"importances_fold{fold+1}_{block_name}_{model_name}_importance.csv"
                    filepath = os.path.join(output_path, fname)
                    model_importance_df.to_csv(filepath, index=False)
                    print(f"💾 Saved feature importances: {filepath}")


            # Track best model by AUC for this fold
            if 'auc' in metrics and metrics['auc'] > best_overall_score:
                best_overall_score = metrics['auc']
                best_overall_model_name = model_name
                best_overall_params = best_params

        except Exception as e:
            print(f"❌ Model {model_name} failed in fold {fold + 1} [Block: {block_name}]: {e}")

    # === Summarize metrics across models for this fold ===
    metrics_summary = []
    for model_name, metric_dict in results.items():
        summary = {'Model': model_name}
        for metric_name, values in metric_dict.items():
            if isinstance(values, list) and values:
                if metric_name == 'tprs':
                    summary[f'mean_{metric_name}'] = np.mean(values, axis=0)
                else:
                    summary[f'mean_{metric_name}'] = np.mean(values)
        summary['Fold'] = fold + 1
        summary['Block'] = block_name
        metrics_summary.append(summary)

    metrics_df = pd.DataFrame(metrics_summary)
    return metrics_df, best_overall_model_name, best_overall_params




### risk function - for 2 step process 


def label_from_oof_risk(oof_risk, strategy="tertiles"):
    """
    oof_risk: pd.Series of OOF risk (higher = worse)
    Returns labels (0=low,1=mid,2=high) and the cutpoints used.
    """
    import numpy as np, pandas as pd
    r = oof_risk.values
    if strategy == "tertiles":
        low_cut  = np.percentile(r, 33.333)
        high_cut = np.percentile(r, 66.667)
    else:
        raise ValueError("Only 'tertiles' implemented here.")

    y_mc = pd.Series(1, index=oof_risk.index)             # mid by default
    y_mc[oof_risk <= low_cut] = 0                          # low
    y_mc[oof_risk >  high_cut] = 2                         # high
    return y_mc.astype(int), (float(low_cut), float(high_cut))

def rsf_oof_risk_scores_blocks(
    data, meta, y,                       # like your run_cv_... signature pieces
    all_pathway_dicts, pathway_options,
    gsva_scores_rna_all, gsva_scores_cnv_all,
    subtype,
    target_col='Disease_Status',         # for stratification in the outer CV
    selected_blocks=None,
    outer_folds=5,
    fold_output_dir=None,
    random_state=42,
    rsf_param_grid=None                  # optional RSF grid override
):
    """
    Stage-1: RSF OOF risk scoring *with your per-fold data processing pipeline*.

    For each outer fold:
      1) sig pathways (mut, GSVA RNA/CNV) & DE genes on TRAIN (rec vs non-rec)
      2) merge all blocks and preprocess per block using y_train (rec vs non-rec)
      3) concat processed blocks -> X_train_concat / X_test_concat
      4) fit RSF on TRAIN (grid by c-index), predict risk on TRAIN & TEST
      5) store OOF risk for TEST fold

    After all folds:
      - compute OOF tertiles (global, from OOF only)
      - compute OOF ROC-AUC / PR-AUC / c-index
      - save CSVs & plots (if fold_output_dir provided)

    Returns
    -------
    oof_risk     : pd.Series (index=sample_id), higher = worse
    risk_labels  : pd.Series in {0,1,2} from OOF tertiles (0=low,1=mid,2=high)
    thresholds   : {'low_cut': float, 'high_cut': float} from OOF
    metrics      : dict with c_index, roc_auc, pr_auc, auc_extremes_only
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
    from sklearn.exceptions import UndefinedMetricWarning
    from sksurv.util import Surv
    from sksurv.metrics import concordance_index_censored
    import warnings
    try:
        from sklearn.exceptions import UndefinedMetricWarning
    except Exception:
        # Fallback in case sklearn.exceptions isn't available in this env
        class UndefinedMetricWarning(Warning):
            pass

    # ───────────────────────────────
    # Output dirs
    # ───────────────────────────────
    out_dir = None
    if fold_output_dir:
        out_dir = os.path.join(fold_output_dir, f"{subtype}_stage1_rsf")
        os.makedirs(out_dir, exist_ok=True)
        for d in ["plots", "tables", "fold_logs"]:
            os.makedirs(os.path.join(out_dir, d), exist_ok=True)

    # ───────────────────────────────
    # Helper: RSF fit (grid by c-index) and predict
    # ───────────────────────────────
    def fit_rsf_and_predict(X_train_concat, subtype_meta_train, X_test_concat, random_state=42):
        """
        Fit RSF on TRAIN only (grid by c-index), return train & test risk scores.
        subtype_meta_train must contain 'time' and 'event'.
        """
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
        from sksurv.metrics import concordance_index_censored

        assert {'time', 'event'}.issubset(subtype_meta_train.columns), "meta needs 'time' and 'event'."

        y_struct = Surv.from_dataframe(event='event', time='time',
                                       data=subtype_meta_train.loc[X_train_concat.index])

        rsf = RandomSurvivalForest(random_state=random_state)

        param_grid = rsf_param_grid or {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
            "max_features": ["sqrt", "log2"],
        }

        def cindex_scorer(estimator, X, y_structured):
            risk = estimator.predict(X)  # higher = worse
            e = y_structured["event"]
            t = y_structured["time"]
            return concordance_index_censored(e, t, risk)[0]

        grid = GridSearchCV(
            estimator=rsf,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            scoring=cindex_scorer
        )
        print("🔍 Grid-searching RSF by c-index...")
        grid.fit(X_train_concat, y_struct)
        best_rsf = grid.best_estimator_
        print(f"✅ Best RSF params: {grid.best_params_} | best c-index: {grid.best_score_:.3f}")

        train_scores = np.asarray(best_rsf.predict(X_train_concat))
        test_scores  = np.asarray(best_rsf.predict(X_test_concat))
        return train_scores, test_scores

    # ───────────────────────────────
    # Prep
    # ───────────────────────────────
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    # clean indices
    meta = meta.copy()
    meta.index = meta.index.astype(str).str.strip()
    y = y.copy()
    y.index = y.index.astype(str).str.strip()

    # stratification by original binary label (Disease_Status)
    assert target_col in meta.columns, f"{target_col} not found in meta"
    y_strat = meta[target_col].astype(str).values

    # rna log2 once to reuse
    rna_log2_all = np.log2(data['rna'] + 1)

    # Outer CV
    skf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)
    oof_risk = pd.Series(index=meta.index, dtype=float)

    # ───────────────────────────────
    # Outer folds
    # ───────────────────────────────
    for fold, (train_idx, test_idx) in enumerate(skf.split(meta, meta[target_col]), start=1):
        print(f"\n🔁 Stage-1 RSF • Outer Fold {fold}/{outer_folds}")

        # Split metadata and labels into train/test
        train_ids = meta.iloc[train_idx].index.astype(str).str.strip()
        test_ids  = meta.iloc[test_idx].index.astype(str).str.strip()
        subtype_meta_train = meta.loc[train_ids].copy()
        subtype_meta_test  = meta.loc[test_ids].copy()
        y_train = y.loc[train_ids]
        # y_test  = y.loc[test_ids]  # not needed for RSF

        # === Step 1: Mutation pathways (train significance, test subset) ===
        mutation_pathways_train, _ = process_mutation_pathways(
            M_matrix=data['mut'].loc[train_ids],
            subtype_meta=subtype_meta_train,
            pathway_dicts=all_pathway_dicts,
            pathway_options=pathway_options,
            output_dir=None,
            subtype=subtype
        )
        _, mutation_full_test = process_mutation_pathways(
            M_matrix=data['mut'].loc[test_ids],
            subtype_meta=subtype_meta_test,
            pathway_dicts=all_pathway_dicts,
            pathway_options=pathway_options,
            output_dir=None,
            subtype=subtype
        )
        significant_mutation_features = mutation_pathways_train.columns
        mutation_pathways_test = mutation_full_test[significant_mutation_features].fillna(0)

        # === Step 2: GSVA RNA/CNV significance on TRAIN (rec vs non-rec) ===
        rna_results = run_significance_tests(
            gsva_scores_rna_all.loc[train_ids],
            subtype_meta_train,
            group_col='Disease_Status', group1='recurrent', group2='non-recurrent',
            use_mannwhitney=True
        )
        # coerce p_value to numeric to avoid 'object' dtype traps
        rna_results['p_value'] = pd.to_numeric(rna_results['p_value'], errors='coerce')
        sig_rna = rna_results[rna_results['p_value'] <= 0.01]['feature']
        if sig_rna.empty:
            sig_rna = rna_results.nsmallest(5, 'p_value')['feature']
        rna_pathways_train = gsva_scores_rna_all.loc[train_ids, sig_rna]
        rna_pathways_test  = gsva_scores_rna_all.loc[test_ids,  sig_rna]

        cnv_results = run_significance_tests(
            gsva_scores_cnv_all.loc[train_ids],
            subtype_meta_train,
            group_col='Disease_Status', group1='recurrent', group2='non-recurrent',
            use_mannwhitney=True
        )
        cnv_results['p_value'] = pd.to_numeric(cnv_results['p_value'], errors='coerce')
        sig_cnv = cnv_results[cnv_results['p_value'] <= 0.01]['feature']
        if sig_cnv.empty:
            sig_cnv = cnv_results.nsmallest(5, 'p_value')['feature']
        cnv_pathways_train = gsva_scores_cnv_all.loc[train_ids, sig_cnv]
        cnv_pathways_test  = gsva_scores_cnv_all.loc[test_ids,  sig_cnv]

        # === Step 3: DE-style filtering for raw RNA on TRAIN ===
        # ensure category order
        subtype_meta_train['Disease_Status'] = pd.Categorical(
            subtype_meta_train['Disease_Status'], categories=["non-recurrent", "recurrent"]
        )
        de_results = process_deseq_results(
            rna_expression=data['rna_raw'].loc[train_ids],
            metadata=subtype_meta_train[['Disease_Status']],
            design_factors=['Disease_Status'],
            contrast=["Disease_Status", "recurrent", "non-recurrent"],
            pvalue_threshold=0.01, padj_threshold=0.25, logfold_threshold=1.5,
            base_mean_threshold=50, gene_sum_threshold=100
        )
        de_genes = [g for g in de_results.index if g in rna_log2_all.columns]
        de_expr_train = rna_log2_all.loc[train_ids, de_genes]
        de_expr_test  = rna_log2_all.loc[test_ids,  de_genes]

        # === Step 4: Merge features into blocks (TRAIN/TEST) ===
        X_train_full, X_train_blocks = merge_feature_blocks(
            subtype_meta=subtype_meta_train,
            mutation_df=mutation_pathways_train,
            rna_df=rna_pathways_train,
            cnv_df=cnv_pathways_train,
            tf_df=data['tf'].loc[train_ids],
            tmb_df=data['tmb'].loc[train_ids],
            hoadley_df=data['hoadley'].loc[train_ids],
            cibersort_df=data['cibersort'].loc[train_ids],
            pathogenic_df=data['pathogenic_count'].loc[train_ids],
            cnv_pct_df=data['cnv_pct_altered'].loc[train_ids],
            clinical_df=data['clinical'].loc[train_ids],
            de_gene_df=de_expr_train,
            return_separate_blocks=True
        )
        X_test_full, X_test_blocks = merge_feature_blocks(
            subtype_meta=subtype_meta_test,
            mutation_df=mutation_pathways_test,
            rna_df=rna_pathways_test,
            cnv_df=cnv_pathways_test,
            tf_df=data['tf'].loc[test_ids],
            tmb_df=data['tmb'].loc[test_ids],
            hoadley_df=data['hoadley'].loc[test_ids],
            cibersort_df=data['cibersort'].loc[test_ids],
            pathogenic_df=data['pathogenic_count'].loc[test_ids],
            cnv_pct_df=data['cnv_pct_altered'].loc[test_ids],
            clinical_df=data['clinical'].loc[test_ids],
            de_gene_df=de_expr_test,
            return_separate_blocks=True
        )

        # optional block filter
        if selected_blocks is not None:
            X_train_blocks = {k: v for k, v in X_train_blocks.items() if k in selected_blocks}
            X_test_blocks  = {k: v for k, v in X_test_blocks.items()  if k in selected_blocks}
            print(f"🔎 Using selected blocks only: {list(X_train_blocks.keys())}")

        # === Step 5: Per-block preprocessing (using rec/non-rec labels) ===
        X_train_proc_blocks = {}
        X_test_proc_blocks  = {}
        for block_name, block_info in X_train_blocks.items():
            print(f"🧪 Preprocessing Block: {block_name}")
            X_block_train = block_info['data']
            X_block_test  = X_test_blocks[block_name]['data']
            process_type  = block_info['process']

            X_train_proc, X_test_proc = preprocess_block(
                X_block_train, process_type, block_name,
                X_test=X_block_test, y_train=y_train,  # original labels
                output_path=out_dir, fold=fold
            )
            X_train_proc_blocks[block_name] = X_train_proc
            X_test_proc_blocks[block_name]  = X_test_proc

        # === Step 6: Concatenate processed blocks ===
        print("🔗 Concatenating all preprocessed blocks...")
        X_train_concat = pd.concat(X_train_proc_blocks.values(), axis=1).fillna(0)
        X_test_concat  = pd.concat(X_test_proc_blocks.values(),  axis=1).fillna(0)

        # === Step 7: RSF fit on TRAIN, predict TRAIN/TEST risk ===
        train_scores, test_scores = fit_rsf_and_predict(
            X_train_concat, subtype_meta_train, X_test_concat, random_state=random_state
        )

        # store OOF on test fold
        oof_risk.loc[X_test_concat.index] = test_scores

        # optional: per-fold log
        if out_dir:
            pd.DataFrame({
                "sample_id": X_test_concat.index,
                "oof_risk": test_scores
            }).to_csv(os.path.join(out_dir, "fold_logs", f"fold{fold}_oof_risk.csv"), index=False)

    # ───────────────────────────────
    # Post: metrics, tertiles, plots
    # ───────────────────────────────
    if oof_risk.isna().any():
        missing = oof_risk[oof_risk.isna()].index.tolist()
        raise ValueError(f"OOF risk has NaNs for: {missing[:5]} ...")

    # Align event/time
    aligned_idx = oof_risk.index.intersection(meta.index)
    risk = oof_risk.loc[aligned_idx].astype(float).values
    event = meta.loc[aligned_idx, 'event'].astype(int).values
    time  = meta.loc[aligned_idx, 'time'].values

    # c-index (time-aware) + ROC/PR vs event
    c_index = concordance_index_censored(event.astype(bool), time, risk)[0]
    roc_auc = roc_auc_score(event, risk)
    pr_auc  = average_precision_score(event, risk)

    # tertiles from OOF
    low_cut, high_cut = np.percentile(risk, [33.333, 66.667])
    risk_labels = pd.Series(1, index=aligned_idx, dtype=int)
    risk_labels[risk <= low_cut]  = 0
    risk_labels[risk >  high_cut] = 2

    # extremes-only AUC (descriptive)
    ext_mask = risk_labels.isin([0, 2]).values
    if ext_mask.sum() >= 2 and np.unique(event[ext_mask]).size == 2:
        auc_ext = roc_auc_score(event[ext_mask], risk[ext_mask])
    else:
        auc_ext = np.nan

    metrics = {
        "c_index": float(c_index),
        "roc_auc": float(roc_auc),
        "pr_auc":  float(pr_auc),
        "roc_auc_extremes_only": float(auc_ext),
        "low_cut": float(low_cut),
        "high_cut": float(high_cut)
    }

    # save CSVs & plots
    if out_dir:
        # tables
        pd.DataFrame({"sample_id": aligned_idx, "oof_risk": risk, "risk_tertile": risk_labels.values,
                      "event": event}).to_csv(os.path.join(out_dir, "tables", "rsf_oof_risk_scores.csv"), index=False)
        pd.Series(metrics).to_csv(os.path.join(out_dir, "tables", "oof_metrics.csv"))

        # plots
        fpr, tpr, _ = roc_curve(event, risk)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
        plt.plot([0,1],[0,1], linestyle=":", color="gray")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("Stage-1 RSF OOF ROC")
        plt.legend(loc="lower right"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plots", "oof_roc.png"), dpi=160); plt.close()

        prec, rec, _ = precision_recall_curve(event, risk)
        plt.figure(figsize=(6,5))
        plt.plot(rec, prec, label=f"AP={pr_auc:.2f}")
        plt.hlines(y=event.mean(), xmin=0, xmax=1, linestyles=":", colors="gray", label=f"baseline={event.mean():.2f}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Stage-1 RSF OOF PR")
        plt.legend(loc="lower left"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plots", "oof_pr.png"), dpi=160); plt.close()

        plt.figure(figsize=(6,5))
        plt.hist(risk, bins=30, alpha=0.85)
        plt.axvline(low_cut, color="tab:orange", linestyle="--", label=f"low_cut={low_cut:.3f}")
        plt.axvline(high_cut, color="tab:red", linestyle="--", label=f"high_cut={high_cut:.3f}")
        plt.xlabel("OOF risk score (higher = worse)")
        plt.ylabel("Count")
        plt.title("OOF Risk Distribution with Tertiles")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plots", "oof_risk_hist.png"), dpi=160); plt.close()

    thresholds = {"low_cut": float(low_cut), "high_cut": float(high_cut)}
    return oof_risk, risk_labels, thresholds, metrics
