#!/usr/bin/env python3
# validation_processing.simple.py



"""
External validation feature-processing script.

This script builds a validation-set feature matrix that matches the feature schema
used by the trained recurrence-risk model. It does not train models or perform feature
selection. Instead, it reconstructs RNA, CNV, mutation, immune/TME, clinical, and
other model-input features in an external cohort and aligns them to the final feature
list selected during training.

The main output is an ordered sample-by-feature matrix that can be passed directly to
the final model prediction script.

Overall workflow:

1. Load the feature-selection outputs from the training cohort, either from
   --training_fs_dir or from explicitly supplied training feature files.
2. Read the final training feature list, usually FINAL_features_only.csv, to define
   the exact feature names and order expected by the trained model.
3. Load and normalize validation-cohort input data, including expression, CNV,
   mutation, immune/TME scores, clinical data, and optional precomputed feature
   matrices.
4. Recompute only the feature families needed by the trained model whenever possible.
5. Prefix all generated features using the same block naming convention used during
   training:
   RNAblock__
   CNVblock__
   MUTblock__
   CLINICALblock__
6. Concatenate all available validation feature blocks into a combined matrix.
7. Fuzzy-match, reorder, and zero-fill features so the final validation matrix matches
   the training feature schema.
8. Save the combined pre-selection matrix, the final ordered feature matrix, mapping
   logs, and a manifest describing which feature blocks were generated.

Training feature inputs:
The script uses the training feature-selection outputs to determine which validation
features to rebuild. If --training_fs_dir is provided, the script will look for:

* RNAblock_selected_post_redundancy.csv
* CNVblock_selected_post_redundancy.csv
* MUTblock_selected_post_redundancy.csv
* FINAL_features_only.csv

These files can also be supplied individually using:

* --training_rna_feature_file
* --training_cnv_feature_file
* --training_mut_feature_file
* --training_final_feature_file

RNA feature construction:
The script can process validation expression data, map gene identifiers to symbols,
log-transform expression when needed, and compute RNA pathway scores using supplied
GMT files. It can also build direct gene-expression features, custom gene-set summary
features, precomputed RNA gene-set features, CIBERSORT features, TIDE features,
ESTIMATE scores, and EcoTyper features when those inputs are provided.

CNV feature construction:
The script can process validation copy-number matrices and generate CNV-derived
features matched to training-selected CNV features. This can include CNV pathway
scores, selected gene-level CNV features, percent-genome-altered features, and
arm-level or cytoband-level binary features when the required inputs are provided.

Mutation and pTMB feature construction:
The script can generate mutation-derived pathway features from validation mutation
matrices, include TMB/pathogenic mutation summaries, and compute pTMB features from
MAF/GMT/GTF inputs when needed. Mutation-derived features are written under the
MUTblock__ prefix to match the training schema.


Clinical feature construction:
If clinical data are supplied, the script processes and prefixes clinical covariates
as CLINICALblock__ features. Clinical feature names are also handled during fuzzy
feature alignment so common naming differences between the training and validation
datasets can be resolved when possible.

Sample ID handling:
The script includes flexible sample-ID normalization and alignment. It supports direct
matching, normalized string matching, and TCGA vial-aware matching (-01 at end of TCGA samples in some datatypes). TCGA vial letters
can be stripped per feature block or after block concatenation, depending on the
--tcga_vial_strip_mode setting.

Feature alignment:
After all validation feature blocks are generated, the script concatenates them into
a combined pre-selection matrix. It then uses the final training feature list to:

* select the features expected by the model
* preserve the exact training feature order
* fuzzy-match compatible feature names when formatting differs
* zero-fill missing features that cannot be generated in the validation cohort
* optionally write a feature-mapping log

Outputs:
The script writes outputs to --outdir using the specified --prefix. Typical outputs
include:

* <prefix>__combined_pre.csv
  All generated validation features before final feature-list alignment.
* <prefix>__combined_features.csv
  Final validation matrix aligned to the training feature schema.
* <prefix>__selected_ordered_features.csv
  Same ordered model-input matrix, saved for clarity and downstream prediction.
* feature-mapping log, when --mapping_log is provided.
* pathway/custom-gene-set matching tables when applicable.
* manifest file summarizing sample counts, feature counts, training feature sources,
  and which feature blocks were included.

Downstream use:
The final ordered validation matrix should be used as the input prediction matrix for
the final model prediction workflow. This script is therefore the bridge between raw
external validation cohort data and the trained model’s required input feature space.
"""

import os
import re
import json
import gzip
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

from gseapy import gsva as _gsva



HARDCODED_RNA_GENESET_CSV = "./data/RNAseq/genescores/custom_genesets.gmt"

# =========================================================
# Basic helpers
# =========================================================



def _exists(path: Optional[str]) -> bool:
    return bool(path) and os.path.exists(str(path))


def _norm_ids(idx) -> pd.Index:
    s = pd.Index(idx).astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.replace(r"^X(?=\d+$)", "", regex=True)
    return s


def _slug(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[()]", "", s)
    s = re.sub(r"[^0-9a-z]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def _write_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)


def _read_names_txt(path: Optional[str]) -> List[str]:
    if not _exists(path):
        print(f"[SKIP] names file not found: {path}")
        return []
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def _read_table_any(path: Optional[str]) -> pd.DataFrame:
    if not _exists(path):
        print(f"[SKIP] table not found: {path}")
        return pd.DataFrame()

    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)

    try:
        df = pd.read_csv(path, sep=None, engine="python", index_col=0)
        if df.shape[1] > 0:
            return df
    except Exception:
        pass

    for sep in ("\t", ",", " "):
        try:
            df = pd.read_csv(path, sep=sep, engine="python", index_col=0)
            if df.shape[1] > 0:
                return df
        except Exception:
            continue

    try:
        df = pd.read_csv(path, sep=None, engine="python")
        if df.shape[1] >= 2:
            df = df.set_index(df.columns[0])
            return df
    except Exception:
        pass

    print(f"[WARN] Could not parse table (no data columns detected): {path}")
    return pd.DataFrame()


def _read_table_auto(path: str) -> pd.DataFrame:
    if not _exists(path):
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path, sep=None, engine="python", low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, sep="\t", low_memory=False)
        except Exception:
            return pd.read_csv(path, low_memory=False)


def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _collapse_dup_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        out = _to_numeric(df)
        return out.loc[:, out.notna().any(axis=0)]
    num = _to_numeric(df)
    out = num.T.groupby(level=0).mean().T
    return out.loc[:, out.notna().any(axis=0)]


# =========================================================
# Training FS-dir helpers
# =========================================================


def load_precomputed_rna_geneset_block(
    path: Optional[str],
    target_index: Optional[pd.Index] = None,
    tcga_vial_strip_mode: str = "per_block",
    keep_cols: Optional[List[str]] = None,
    drop_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    if not _exists(path):
        print(f"[SKIP] precomputed RNA gene-set table not found: {path}")
        return pd.DataFrame(index=target_index if target_index is not None else pd.Index([], name="sample_id"))

    df = _read_table_any(path)
    if df.empty:
        print(f"[SKIP] precomputed RNA gene-set table unreadable/empty: {path}")
        return pd.DataFrame(index=target_index if target_index is not None else pd.Index([], name="sample_id"))

    df.index = _norm_ids(df.index)
    df = _collapse_dup_columns(df)

    if keep_cols:
        keep_set = {str(c) for c in keep_cols}
        df = df[[c for c in df.columns if str(c) in keep_set]]

    if drop_cols:
        drop_set = {str(c) for c in drop_cols}
        df = df[[c for c in df.columns if str(c) not in drop_set]]

    new_cols = []
    for c in df.columns.astype(str):
        if c.startswith("RNAblock__"):
            new_cols.append(c)
        else:
            new_cols.append(f"RNAblock__{c}")
    df.columns = new_cols

    df = _maybe_strip_vials(df, tcga_vial_strip_mode, "RNA precomputed gene-set table")

    if target_index is not None:
        df = align_index_smart(df, target_index, "rna_precomputed_genesets")

    return df

def _resolve_training_feature_files(
    fs_dir: Optional[str],
    rna_file: Optional[str],
    cnv_file: Optional[str],
    mut_file: Optional[str],
    final_file: Optional[str],
) -> dict:
    out = {
        "rna": rna_file,
        "cnv": cnv_file,
        "mut": mut_file,
        "final": final_file,
    }

    if not fs_dir or not os.path.isdir(fs_dir):
        return out

    defaults = {
        "rna": os.path.join(fs_dir, "RNAblock_selected_post_redundancy.csv"),
        "cnv": os.path.join(fs_dir, "CNVblock_selected_post_redundancy.csv"),
        "mut": os.path.join(fs_dir, "MUTblock_selected_post_redundancy.csv"),
        "final": os.path.join(fs_dir, "FINAL_features_only.csv"),
    }

    for k, pth in defaults.items():
        if not out[k] and os.path.exists(pth):
            out[k] = pth

    return out


def _load_feature_names_from_training_file(path: Optional[str]) -> List[str]:
    if not _exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if lines and all(("," not in x and "\t" not in x) for x in lines[:10]):
            if lines[0].lower() in {"feature", "features"}:
                lines = lines[1:]
            if lines:
                return lines
    except Exception:
        pass

    try:
        df = pd.read_csv(path, sep=None, engine="python", comment="#")
    except Exception:
        try:
            df = pd.read_csv(path, comment="#")
        except Exception:
            return []

    if df is None or df.empty:
        return []

    low = {str(c).lower(): c for c in df.columns}
    for key in ["feature", "features"]:
        if key in low:
            return df[low[key]].astype(str).str.strip().tolist()

    if df.shape[1] == 1:
        return df.iloc[:, 0].astype(str).str.strip().tolist()

    cols = [str(c).strip() for c in df.columns]
    first = cols[0].lower()
    if first in {"sample_id", "sample", "id", "patient_id", "index", "unnamed: 0"}:
        cols = cols[1:]
    return cols


def _extract_db_pathway_tail(s: str) -> Optional[str]:
    """
    Keep only the true pathway token beginning at the LAST occurrence of:
      REACTOME_, HALLMARK_, KEGG_, GOBP_
    This fixes names like:
      reactome_copynumber_REACTOME_SIGNALING_BY_EGFR_IN_CANCER
    by returning:
      REACTOME_SIGNALING_BY_EGFR_IN_CANCER
    """
    if s is None:
        return None
    x = str(s).strip()
    x = re.sub(r"[^A-Za-z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    x_up = x.upper()

    matches = list(re.finditer(r"(REACTOME_|HALLMARK_|KEGG_|GOBP_)", x_up))
    if not matches:
        return None
    m = matches[-1]
    return x_up[m.start():]


def _canon_pathway_name(s: str) -> str:
    hit = _extract_db_pathway_tail(s)
    if hit is not None:
        return hit
    s = str(s).strip()
    s = s.replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_").upper()
    return s


def _looks_like_pathway_tail(tail: str) -> bool:
    return _extract_db_pathway_tail(tail) is not None


def _infer_selected_pathways_from_training_features(
    feature_names: List[str],
    block_prefix: str,
    exclude_mut_special: bool = True,
) -> List[str]:
    keep = []
    seen = set()

    for f in feature_names:
        f = str(f).strip()
        if not f.startswith(block_prefix):
            continue

        tail = f[len(block_prefix):]

        if block_prefix == "MUTblock__" and exclude_mut_special:
            if tail in {"TMB", "pathogenic_mutation"}:
                continue
            if tail.startswith("pTMB__"):
                continue

        if _looks_like_pathway_tail(tail):
            canon = _canon_pathway_name(tail)
            if canon not in seen:
                keep.append(canon)
                seen.add(canon)

    return keep


def _subset_gmt_by_selected_pathway_features(
    feature_names: List[str],
    block_prefix: str,
    gmt_dict: Dict[str, List[str]],
    exclude_mut_special: bool = True,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    selected_pathways = _infer_selected_pathways_from_training_features(
        feature_names=feature_names,
        block_prefix=block_prefix,
        exclude_mut_special=exclude_mut_special,
    )

    gmt_canon_to_orig = {}
    for gmt_name in gmt_dict.keys():
        gmt_canon_to_orig.setdefault(_canon_pathway_name(gmt_name), gmt_name)

    matched = {}
    rows = []

    for req in selected_pathways:
        gmt_hit = gmt_canon_to_orig.get(req)
        if gmt_hit is not None:
            matched[gmt_hit] = gmt_dict[gmt_hit]
            rows.append({
                "requested_feature_tail": req,
                "matched_gmt_name": gmt_hit,
                "status": "matched",
            })
        else:
            rows.append({
                "requested_feature_tail": req,
                "matched_gmt_name": None,
                "status": "missing",
            })

    return matched, pd.DataFrame(rows)


def _infer_selected_direct_gene_features_from_training_features(
    feature_names: List[str],
    block_prefix: str,
    matrix_columns,
) -> List[str]:
    matrix_map = {str(c).upper(): str(c) for c in pd.Index(matrix_columns).astype(str)}

    keep = []
    seen = set()

    for f in feature_names:
        f = str(f).strip()
        if not f.startswith(block_prefix):
            continue

        tail = f[len(block_prefix):].strip()

        if _looks_like_pathway_tail(tail):
            continue

        bad_prefixes = (
            "TIDE__", "ESTIMATE__", "ECOTYPER__", "CELLSTATE_", "ECOTYPE_",
            "ARM__", "BAND__", "GS_", "chr", "CHR"
        )
        if tail.startswith(bad_prefixes):
            continue

        if tail.upper() in matrix_map and tail not in seen:
            keep.append(tail)
            seen.add(tail)

    return keep


def _infer_selected_custom_gene_set_families_from_training_features(
    feature_names: List[str],
    block_prefix: str = "RNAblock__",
) -> List[str]:
    keep = []
    seen = set()

    pat = re.compile(
        rf"^{re.escape(block_prefix)}GS_(?P<name>.+)__(median_log2|geo_mean_log2|zscore_mean)$",
        flags=re.I,
    )

    for f in feature_names:
        f = str(f).strip()
        m = pat.match(f)
        if not m:
            continue
        nm = str(m.group("name")).strip()
        if nm and nm not in seen:
            keep.append(nm)
            seen.add(nm)

    return keep


def _subset_custom_gene_sets_by_selected_families(
    selected_families: List[str],
    gene_sets: Dict[str, List[str]],
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    slug_to_raw = {}
    for raw_name in gene_sets.keys():
        slug_to_raw.setdefault(_slug(raw_name), raw_name)

    matched = {}
    rows = []

    for fam in selected_families:
        fam_slug = _slug(fam)
        raw = slug_to_raw.get(fam_slug)
        if raw is not None:
            matched[raw] = gene_sets[raw]
            rows.append({
                "selected_family": fam,
                "matched_gene_set_name": raw,
                "status": "matched",
            })
        else:
            rows.append({
                "selected_family": fam,
                "matched_gene_set_name": None,
                "status": "missing",
            })

    return matched, pd.DataFrame(rows)


def _compute_gene_set_scores(
    X_log2: pd.DataFrame,
    gene_sets: Dict[str, List[str]],
    mode: str = "median_log2",
    min_overlap: int = 1
) -> pd.DataFrame:
    if X_log2 is None or X_log2.empty or not gene_sets:
        return pd.DataFrame(index=X_log2.index if isinstance(X_log2, pd.DataFrame) else None)

    cols_upper = pd.Index([str(c).upper() for c in X_log2.columns])
    pos = {g: i for i, g in enumerate(cols_upper)}

    out = {}
    n_samples = X_log2.shape[0]

    for raw_name, gene_list in gene_sets.items():
        name = _slug(str(raw_name))
        gs = [str(g).upper() for g in (gene_list or []) if isinstance(g, str) and g.strip()]
        idxs = [pos[g] for g in gs if g in pos]

        if len(idxs) < int(min_overlap):
            out[f"GS_{name}__median_log2"] = np.full(n_samples, np.nan, dtype=float)
            out[f"GS_{name}__geo_mean_log2"] = np.full(n_samples, np.nan, dtype=float)
            out[f"GS_{name}__zscore_mean"] = np.full(n_samples, np.nan, dtype=float)
            continue

        sub = X_log2.iloc[:, idxs]
        A = sub.values.astype(float)

        med_log2 = np.nanmedian(A, axis=1)
        geo_mean_log2 = np.nanmean(A, axis=1)

        mu = np.nanmean(A, axis=0)
        sd = np.nanstd(A, axis=0)
        sd[sd == 0] = np.nan
        Z = (A - mu) / sd
        zmean = np.nanmean(Z, axis=1)

        out[f"GS_{name}__median_log2"] = med_log2
        out[f"GS_{name}__geo_mean_log2"] = geo_mean_log2
        out[f"GS_{name}__zscore_mean"] = zmean

    return pd.DataFrame(out, index=X_log2.index)


def _infer_selected_ptmb_features_from_training_features(
    feature_names: List[str],
    block_prefix: str = "MUTblock__",
) -> List[str]:
    keep = []
    seen = set()
    pat = re.compile(rf"^{re.escape(block_prefix)}pTMB__(.+)$", flags=re.I)

    for f in feature_names:
        f = str(f).strip()
        m = pat.match(f)
        if not m:
            continue
        nm = str(m.group(1)).strip()
        if nm and nm not in seen:
            keep.append(nm)
            seen.add(nm)
    return keep


def _subset_gmt_by_selected_ptmb_features(
    feature_names: List[str],
    gmt_dict: Dict[str, List[str]],
    block_prefix: str = "MUTblock__",
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    selected_ptmb = _infer_selected_ptmb_features_from_training_features(
        feature_names=feature_names,
        block_prefix=block_prefix,
    )

    slug_to_raw = {}
    for raw_name in gmt_dict.keys():
        slug_to_raw.setdefault(_slug(raw_name), raw_name)

    matched = {}
    rows = []

    for req in selected_ptmb:
        raw = slug_to_raw.get(_slug(req))
        if raw is not None:
            matched[raw] = gmt_dict[raw]
            rows.append({
                "requested_ptmb_feature_tail": req,
                "matched_gmt_name": raw,
                "status": "matched",
            })
        else:
            rows.append({
                "requested_ptmb_feature_tail": req,
                "matched_gmt_name": None,
                "status": "missing",
            })

    return matched, pd.DataFrame(rows)


# =========================================================
# TCGA vial helpers
# =========================================================



def normalize_index_to_tcga_sample_level(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    if df is None or df.empty:
        return df
    idx_new = pd.Index([_strip_tcga_vial_letter(x) for x in df.index], name=df.index.name or "sample_id")
    df = df.copy()
    df.index = idx_new
    if df.index.duplicated().any():
        if how == "first":
            df = df[~df.index.duplicated(keep="first")]
        elif how == "median":
            df = df.groupby(level=0).median(numeric_only=True)
        else:
            df = df.groupby(level=0).mean(numeric_only=True)
    return df


def _maybe_strip_vials(df: pd.DataFrame, mode: str, where: str, how: str = "mean") -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if mode == "per_block":
        print(f"[TCGA] strip vial on {where} …")
        return normalize_index_to_tcga_sample_level(df, how=how)
    return df

def _strip_tcga_vial_letter(s: str) -> str:
    if s is None or pd.isna(s):
        return s

    s = str(s).strip().strip('"').strip("'")
    s = re.sub(r"\.0$", "", s)
    s = re.sub(r"^X(?=\d+$)", "", s)

    parts = s.split("-")
    if len(parts) >= 4:
        comp = parts[3]
        m = re.match(r"^(\d{2})[A-Z]($|-.+)", comp, flags=re.I)
        if m:
            parts[3] = m.group(1)
            return "-".join(parts)

    out = re.sub(r"-(\d{2})[A-Z](?=$|-)", r"-\1", s, flags=re.I)

    # very last fallback for already de-delimited strings ending in a vial letter
    if out == s and "-" not in s and re.search(r"\d{2}[A-Z]$", s, flags=re.I):
        out = re.sub(r"(\d{2})[A-Z]$", r"\1", s, flags=re.I)

    return out


def _clean_match_id(s: str) -> str:
    if s is None or pd.isna(s):
        return ""
    s = str(s).strip().strip('"').strip("'")
    s = re.sub(r"\.0$", "", s)
    s = re.sub(r"^X(?=\d+$)", "", s)
    return s


def _norm_match_general(s: str) -> str:
    s = _clean_match_id(s)
    s = re.sub(r"[^A-Za-z0-9]", "", s)
    return s.upper()


def _norm_match_tcga_sample(s: str) -> str:
    s = _clean_match_id(s)
    s = _strip_tcga_vial_letter(s)
    s = re.sub(r"[^A-Za-z0-9]", "", str(s))
    return s.upper()


def align_index_smart(
    df: pd.DataFrame,
    target_index: Optional[pd.Index],
    label: str,
    try_tcga_vial_strip: bool = True
) -> pd.DataFrame:
    if df.empty:
        return df
    if target_index is None:
        print(f"[{label}] no target index provided; leaving as-is.")
        return df

    tgt = pd.Index([str(x) for x in target_index], name=df.index.name)

    min_good = max(1, int(0.5 * len(tgt)))

    # 1) direct
    direct = df.reindex(tgt)
    nmatch1 = int(direct.notna().any(axis=1).sum())
    if nmatch1 >= min_good:
        print(f"[{label}] direct match: {nmatch1}/{len(tgt)}")
        return direct

    best_df = direct
    best_n = nmatch1
    best_tag = "direct"

    # 2) general normalized match
    inv_norm = {}
    for idx in df.index.astype(str):
        inv_norm.setdefault(_norm_match_general(idx), idx)

    mapped = [inv_norm.get(_norm_match_general(s), None) for s in tgt]
    norm_out = df.reindex(mapped)
    norm_out.index = tgt
    nmatch2 = int(norm_out.notna().any(axis=1).sum())
    if nmatch2 >= min_good:
        print(f"[{label}] normalized match: {nmatch2}/{len(tgt)}")
        return norm_out
    if nmatch2 > best_n:
        best_df = norm_out
        best_n = nmatch2
        best_tag = "normalized"

    # 3) TCGA sample-level aware match
    if try_tcga_vial_strip:
        inv_tcga = {}
        for idx in df.index.astype(str):
            inv_tcga.setdefault(_norm_match_tcga_sample(idx), idx)

        mapped = [inv_tcga.get(_norm_match_tcga_sample(s), None) for s in tgt]
        tcga_out = df.reindex(mapped)
        tcga_out.index = tgt
        nmatch3 = int(tcga_out.notna().any(axis=1).sum())

        if nmatch3 >= min_good:
            print(f"[{label}] TCGA vial-aware match: {nmatch3}/{len(tgt)}")
            return tcga_out
        if nmatch3 > best_n:
            best_df = tcga_out
            best_n = nmatch3
            best_tag = "TCGA vial-aware"

        print(
            f"[{label}] alignment fallback: "
            f"direct={nmatch1}/{len(tgt)}, normalized={nmatch2}/{len(tgt)}, "
            f"tcga={nmatch3}/{len(tgt)}; using best={best_tag}"
        )
        return best_df

    print(f"[{label}] alignment fallback: direct={nmatch1}/{len(tgt)}, normalized={nmatch2}/{len(tgt)}")
    return best_df


# =========================================================
# GMT / gene set readers
# =========================================================

def _read_gmt_single(path: Optional[str]) -> Dict[str, List[str]]:
    if not _exists(path):
        print(f"[SKIP] GMT not found: {path}")
        return {}
    d: Dict[str, List[str]] = {}
    with open(path) as f:
        for ln in f:
            if not ln.strip():
                continue
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 3:
                parts = ln.split()
                if len(parts) < 3:
                    continue
            name = parts[0]
            genes = [g.strip() for g in parts[2:] if g.strip()]
            d.setdefault(name, [])
            d[name].extend(genes)
    for k, v in d.items():
        d[k] = list(dict.fromkeys(v))
    return d


def _read_gmt_many(paths: Optional[List[str]]) -> Dict[str, List[str]]:
    if not paths:
        return {}
    merged: Dict[str, set] = {}
    for p in paths:
        g = _read_gmt_single(p)
        for name, genes in g.items():
            merged.setdefault(name, set()).update(genes)
    return {k: sorted(list(v)) for k, v in merged.items()}


def _read_gene_sets_csv(path: Optional[str]) -> Dict[str, List[str]]:
    if path and str(path).lower().endswith((".gmt", ".gmt.gz")):
        return _read_gmt_single(path)
    if not _exists(path):
        print(f"[SKIP] hardcoded gene-set CSV not found: {path}")
        return {}
    try:
        df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    except Exception:
        df = pd.read_csv(path, dtype=str)
    if df is None or df.empty:
        return {}

    df = df.applymap(lambda x: str(x).strip() if pd.notna(x) else x)
    low = {c.lower(): c for c in df.columns}

    set_keys = [k for k in ["set", "pathway", "name", "geneset", "term"] if k in low]
    gene_keys = [k for k in ["gene", "symbol", "genes"] if k in low]
    if set_keys and gene_keys:
        sc = low[set_keys[0]]
        gc = low[gene_keys[0]]
        if gc.lower() == "genes":
            rows = []
            for _, r in df[[sc, gc]].dropna(subset=[sc, gc]).iterrows():
                for g in str(r[gc]).replace(";", ",").split(","):
                    g = g.strip()
                    if g:
                        rows.append((r[sc], g))
            df2 = pd.DataFrame(rows, columns=[sc, gc]) if rows else pd.DataFrame(columns=[sc, gc])
        else:
            df2 = df[[sc, gc]].dropna(subset=[sc, gc]).copy()

        d: Dict[str, List[str]] = {}
        for s, sub in df2.groupby(sc):
            genes = [g for g in sub[gc].tolist() if g and g.lower() != "nan"]
            d[s] = list(dict.fromkeys(genes))
        return d

    if df.shape[1] >= 2:
        first = df.columns[0]
        d: Dict[str, List[str]] = {}
        for _, row in df.iterrows():
            s = str(row[first]).strip()
            if not s or s.lower() == "nan":
                continue
            genes = []
            for val in row.iloc[1:].tolist():
                if pd.isna(val):
                    continue
                for g in str(val).replace(";", ",").split(","):
                    g = g.strip()
                    if g:
                        genes.append(g)
            if genes:
                d[s] = list(dict.fromkeys(genes))
        if d:
            return d

    d: Dict[str, List[str]] = {}
    for c in df.columns:
        genes = [g for g in df[c].tolist() if pd.notna(g) and str(g).strip() and str(g).lower() != "nan"]
        if genes:
            d[str(c)] = list(dict.fromkeys([str(g).strip() for g in genes]))
    return d


# =========================================================
# ID map / expression coercion
# =========================================================

_attr_re_cache = {}


def _extract_attr(attr: str, key: str) -> Optional[str]:
    if key not in _attr_re_cache:
        _attr_re_cache[key] = re.compile(rf'{re.escape(key)}\s+"([^"]+)"')
    m = _attr_re_cache[key].search(attr)
    return m.group(1) if m else None


def _open_text(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def _read_gene_map_from_gtf(gtf_path: str, strip_version: bool = True) -> pd.DataFrame:
    rows = []
    with _open_text(gtf_path) as f:
        for ln in f:
            if not ln or ln.startswith("#"):
                continue
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            if parts[2] != "gene":
                continue
            chrom = parts[0]
            try:
                start = int(parts[3])
                end = int(parts[4])
            except Exception:
                continue
            attr = parts[8]
            gid = _extract_attr(attr, "gene_id")
            gnm = _extract_attr(attr, "gene_name") or _extract_attr(attr, "gene_symbol")
            if gid is None:
                continue
            if strip_version:
                gid = re.sub(r"\.\d+$", "", gid)
            chrom = str(chrom).replace("chr", "").replace("CHR", "")
            rows.append((gid, gnm, chrom, start, end))

    df = pd.DataFrame(rows, columns=["ensembl_id", "symbol", "chrom", "start", "end"]).dropna(subset=["ensembl_id"])
    df["entrez_id"] = pd.Series(dtype="string")
    df = df.astype({"ensembl_id": "string", "symbol": "string", "entrez_id": "string", "chrom": "string"})
    return df[["ensembl_id", "entrez_id", "symbol", "chrom", "start", "end"]]


def _load_id_map(path: Optional[str], fmt: str = "auto", strip_version: bool = True) -> pd.DataFrame:
    if not _exists(path):
        print(f"[SKIP] id_map not found: {path}")
        return pd.DataFrame(columns=["ensembl_id", "entrez_id", "symbol", "chrom", "start", "end"])

    is_gtf = fmt == "gtf" or path.lower().endswith((".gtf", ".gff", ".gff3", ".gtf.gz", ".gff.gz", ".gff3.gz"))
    if fmt == "gtf" or is_gtf:
        return _read_gene_map_from_gtf(path, strip_version=strip_version)[["ensembl_id", "entrez_id", "symbol", "chrom", "start", "end"]]

    m = _read_table_any(path)
    if m.empty:
        print("[WARN] ID map empty/unreadable.")
        return pd.DataFrame(columns=["ensembl_id", "entrez_id", "symbol", "chrom", "start", "end"])

    cols = {c.lower(): c for c in m.columns}
    ens = next((cols[c] for c in ["ensembl_id", "gene_id", "ensembl_gene_id", "ensembl"] if c in cols), None)
    sym = next((cols[c] for c in ["symbol", "gene_name", "hgnc_symbol", "gene"] if c in cols), None)
    etz = next((cols[c] for c in ["entrez_id", "entrez_gene_id", "entrezgene", "entrez", "geneid"] if c in cols), None)
    chrom_col = next((cols[c] for c in ["chrom", "chr", "seqname", "seqnames", "chromosome_name"] if c in cols), None)
    start_col = next((cols[c] for c in ["start", "txstart", "gene_start", "chromstart"] if c in cols), None)
    end_col = next((cols[c] for c in ["end", "txend", "gene_end", "chromend"] if c in cols), None)

    if not sym and not ens and not etz:
        print("[WARN] ID map lacks recognizable columns.")
        return pd.DataFrame(columns=["ensembl_id", "entrez_id", "symbol", "chrom", "start", "end"])

    out = pd.DataFrame()
    out["ensembl_id"] = m[ens].astype(str).str.replace(r"\.\d+$", "", regex=True) if ens else pd.Series(dtype="string")
    out["entrez_id"] = m[etz].astype(str) if etz else pd.Series(dtype="string")
    out["symbol"] = m[sym].astype(str) if sym else pd.Series(dtype="string")
    out["chrom"] = m[chrom_col].astype(str) if chrom_col is not None else pd.Series(dtype="string")
    out["start"] = pd.to_numeric(m[start_col], errors="coerce") if start_col is not None else pd.Series(dtype="float")
    out["end"] = pd.to_numeric(m[end_col], errors="coerce") if end_col is not None else pd.Series(dtype="float")

    out = out.dropna(how="all").drop_duplicates()
    return out[["ensembl_id", "entrez_id", "symbol", "chrom", "start", "end"]]


_ENS_RE = re.compile(r"^ENSG\d+(?:\.\d+)?$", re.I)
_ENTREZ_RE = re.compile(r"^\d+$")


def _detect_id_type(ids) -> str:
    arr = pd.Index(ids).astype(str)
    if arr.size == 0:
        return "symbol"
    ens_rate = np.fromiter((bool(_ENS_RE.match(x)) for x in arr), dtype=bool, count=arr.size).mean()
    etz_rate = np.fromiter((bool(_ENTREZ_RE.match(x)) for x in arr), dtype=bool, count=arr.size).mean()
    if max(ens_rate, etz_rate) < 0.5:
        return "symbol"
    return "ensembl" if ens_rate >= etz_rate else "entrez"


def _strip_version(s: str) -> str:
    return re.sub(r"\.\d+$", "", str(s))


def _build_mappers(id_map: pd.DataFrame):
    want_cols = {"ensembl_id", "entrez_id", "symbol"}
    if id_map is None or id_map.empty or not want_cols.issubset(set(id_map.columns)):
        return {}, {}, {}
    df = id_map.dropna(subset=["symbol"]).astype(str).copy()
    df["ensembl_id"] = df["ensembl_id"].map(_strip_version)
    ens2sym = {e: s for e, s in zip(df["ensembl_id"], df["symbol"]) if e and e != "nan"}
    etz2sym = {e: s for e, s in zip(df["entrez_id"], df["symbol"]) if e and e != "nan"}
    sym2ens = {}
    for e, s in zip(df["ensembl_id"], df["symbol"]):
        if e and s and s not in sym2ens:
            sym2ens[s] = e
    return ens2sym, etz2sym, sym2ens


def coerce_expr_to_symbols(expr_df: pd.DataFrame, expr_orientation: str, id_map: pd.DataFrame,
                           strip_version: bool = True) -> pd.DataFrame:
    df = expr_df.copy()

    if expr_orientation == "auto":
        expr_orientation = "genes_by_samples" if df.shape[0] >= df.shape[1] else "samples_by_genes"
    if expr_orientation == "genes_by_samples":
        df = df.T

    df.columns = df.columns.astype(str)
    src_type = _detect_id_type(df.columns)
    if src_type == "ensembl" and strip_version:
        df.columns = df.columns.map(_strip_version)

    if id_map.empty:
        print("[WARN] No ID map; expression kept in its original ID space.")
        return _collapse_dup_columns(_to_numeric(df))

    ens2sym, etz2sym, _ = _build_mappers(id_map)
    if src_type == "ensembl":
        df.columns = pd.Index([ens2sym.get(c, c) for c in df.columns], name="gene")
    elif src_type == "entrez":
        df.columns = pd.Index([etz2sym.get(c, c) for c in df.columns], name="gene")
    else:
        df.columns = pd.Index(df.columns, name="gene")

    return _collapse_dup_columns(_to_numeric(df))


def subset_existing_matrix_by_genes_ci(matrix_df: pd.DataFrame, genes: List[str]) -> pd.DataFrame:
    if matrix_df is None or matrix_df.empty or not genes:
        return pd.DataFrame(index=matrix_df.index if isinstance(matrix_df, pd.DataFrame) else None)

    colmap = {str(c).upper(): c for c in matrix_df.columns}
    req = [str(g).strip() for g in genes if pd.notna(g)]
    req_u = [g.upper() for g in req]

    present = [g for g in req_u if g in colmap]
    missing = [g for g in req_u if g not in colmap]

    selected_cols = [colmap[g] for g in present]
    subset = matrix_df[selected_cols].copy()
    subset.columns = [req[req_u.index(g)] for g in present]

    if missing:
        print(f"[subset] {len(missing)} requested features not found in matrix. Examples: {missing[:10]}")
    print(f"[subset] matched {len(present)}/{len(req)} requested features.")
    return subset


# =========================================================
# GSVA / medians
# =========================================================

def gsva_scores(data_symbol_log2p1: pd.DataFrame, gene_sets: Dict[str, List[str]]) -> pd.DataFrame:
    if data_symbol_log2p1.shape[1] == 0 or not gene_sets:
        return pd.DataFrame(index=data_symbol_log2p1.index)
    dm = data_symbol_log2p1.copy()
    dm.columns = pd.Index([c.upper() for c in dm.columns])
    gs = {k: [g.upper() for g in v] for k, v in gene_sets.items()}
    expr = dm.T
    res = _gsva(expr, gs, method="gsva", outdir=None, min_size=1, max_size=100000, verbose=False)
    wide = res.res2d.pivot(index="Term", columns="Name", values="ES").apply(pd.to_numeric, errors="coerce")
    return wide.T.reindex(dm.index)


def medians_by_sets(expr_sym_log: pd.DataFrame, gene_sets: Dict[str, List[str]]) -> pd.DataFrame:
    if expr_sym_log.empty or not gene_sets:
        return pd.DataFrame(index=expr_sym_log.index)
    expr_cols_norm = pd.Index([str(c).upper() for c in expr_sym_log.columns])
    colpos = {g: i for i, g in enumerate(expr_cols_norm)}
    X = expr_sym_log.to_numpy(dtype=float, copy=False)
    n = X.shape[0]
    out_data, out_cols = [], []
    for k, genes in gene_sets.items():
        gs = [str(g).upper() for g in genes if pd.notna(g)]
        idxs = [colpos.get(g) for g in gs if g in colpos]
        if not idxs:
            out_data.append(np.full(n, np.nan))
            out_cols.append(k)
            continue
        sub = X[:, idxs]
        out_data.append(np.nanmedian(sub, axis=1))
        out_cols.append(k)
    return pd.DataFrame(np.column_stack(out_data), index=expr_sym_log.index, columns=out_cols)


# =========================================================
# CIBERSORT
# =========================================================

def _find_sample_col(df: pd.DataFrame) -> str:
    for cand in ["Mixture", "sample_id", "SampleID", "Sample", "ID", "PATIENT_ID", "case_submitter_id"]:
        if cand in df.columns:
            return cand
    return df.columns[0]


def _add_cibersort_aggregates(cell: pd.DataFrame, denom_eps: float = 1e-3, log_eps: float = 1e-3) -> pd.DataFrame:
    cell = cell.copy()

    def get(col: str) -> pd.Series:
        return cell[col] if col in cell.columns else pd.Series(0.0, index=cell.index)

    total_t = (
        get("T cells CD8")
        + get("T cells CD4 naive")
        + get("T cells CD4 memory resting")
        + get("T cells CD4 memory activated")
        + get("T cells follicular helper")
        + get("T cells regulatory (Tregs)")
        + get("T cells gamma delta")
    )
    total_b = get("B cells naive") + get("B cells memory") + get("Plasma cells")
    total_nk = get("NK cells resting") + get("NK cells activated")
    total_mac = get("Macrophages M0") + get("Macrophages M1") + get("Macrophages M2")
    total_dc = get("Dendritic cells resting") + get("Dendritic cells activated")
    total_mast = get("Mast cells resting") + get("Mast cells activated")
    total_myeloid = get("Monocytes") + total_mac + total_dc + total_mast + get("Eosinophils") + get("Neutrophils")
    total_lymph = total_t + total_b + total_nk
    total_adaptive = total_t + total_b
    total_innate = total_myeloid + total_nk

    cell["Total_T_Cells"] = total_t
    cell["Total_B_Cells"] = total_b
    cell["Total_NK_Cells"] = total_nk
    cell["Total_Macrophages"] = total_mac
    cell["Total_Dendritic_Cells"] = total_dc
    cell["Total_Lymphocytes"] = total_lymph
    cell["Total_Myeloid"] = total_myeloid
    cell["Total_Adaptive"] = total_adaptive
    cell["Total_Innate"] = total_innate

    def ratio(num: pd.Series, den: pd.Series) -> pd.Series:
        return num / den.clip(lower=denom_eps)

    cell["Adaptive_to_Innate"] = ratio(total_adaptive, total_innate)
    cell["CD8_to_CD4"] = ratio(
        get("T cells CD8"),
        get("T cells CD4 naive") + get("T cells CD4 memory resting") + get("T cells CD4 memory activated"),
    )
    cell["CD8_to_Treg"] = ratio(get("T cells CD8"), get("T cells regulatory (Tregs)"))
    cell["M1_to_M2"] = ratio(get("Macrophages M1"), get("Macrophages M2"))
    cell["Lymphocyte_to_Myeloid"] = ratio(total_lymph, total_myeloid)
    cell["CD8_to_Neutrophil"] = ratio(get("T cells CD8"), get("Neutrophils"))

    for col in [
        "Adaptive_to_Innate", "CD8_to_CD4", "CD8_to_Treg",
        "M1_to_M2", "Lymphocyte_to_Myeloid", "CD8_to_Neutrophil"
    ]:
        cell[f"{col}_log2"] = np.log2(cell[col] + log_eps)

    return cell


def read_cibersort_abs(path: Optional[str]) -> pd.DataFrame:
    if not _exists(path):
        print(f"[SKIP] CIBERSORTx ABS not found: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep="\t")

    sid = _find_sample_col(df)
    df[sid] = df[sid].astype(str)
    df = df.set_index(sid)

    drop = [c for c in ["P-value", "Correlation", "RMSE"] if c in df.columns]
    abs_col = next((c for c in df.columns if c.lower().startswith("absolute score")), None)
    if abs_col:
        drop.append(abs_col)

    cell = (
        df.drop(columns=drop, errors="ignore")
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    cell = _add_cibersort_aggregates(cell)
    cell.index = _norm_ids(cell.index)
    return cell


# =========================================================
# Smart alignment
# =========================================================


def align_index_smart(df: pd.DataFrame, target_index: Optional[pd.Index], label: str,
                      try_tcga_vial_strip: bool = True) -> pd.DataFrame:
    if df.empty:
        return df
    if target_index is None:
        print(f"[{label}] no target index provided; leaving as-is.")
        return df

    tgt = pd.Index([str(x) for x in target_index], name=df.index.name)

    direct = df.reindex(tgt)
    nmatch = direct.notna().any(axis=1).sum()
    if nmatch >= max(1, int(0.5 * len(tgt))):
        print(f"[{label}] direct match: {nmatch}/{len(tgt)}")
        return direct

    def _norm_match(s: str) -> str:
        s = str(s).strip().strip('"').strip("'")
        s = re.sub(r"\.0$", "", s)
        s = re.sub(r"^X(?=\d+$)", "", s)
        # crucial: treat MB.0362 and MB-0362 the same
        s = re.sub(r"[^A-Za-z0-9]", "", s).upper()
        return s

    inv_norm = {}
    for idx in df.index.astype(str):
        inv_norm.setdefault(_norm_match(idx), idx)

    mapped = [inv_norm.get(_norm_match(s), None) for s in tgt]
    out = df.reindex(mapped)
    out.index = tgt
    nmatch2 = out.notna().any(axis=1).sum()
    if nmatch2 >= max(1, int(0.5 * len(tgt))):
        print(f"[{label}] normalized match: {nmatch2}/{len(tgt)}")
        return out

    if try_tcga_vial_strip:
        def _strip_vial_barcode(s: str) -> str:
            s = _norm_match(s)
            parts = s.split("-")
            if len(parts) >= 4 and re.match(r"^\d{2}[A-Z]$", parts[3]):
                parts[3] = parts[3][:2]
                return "-".join(parts)
            return re.sub(r"-(\d{2})[A-Z](?=$|-)", r"-\1", s)

        inv_tcga = {}
        for idx in df.index.astype(str):
            inv_tcga.setdefault(_strip_vial_barcode(idx), idx)

        mapped = [inv_tcga.get(_strip_vial_barcode(s), None) for s in tgt]
        out2 = df.reindex(mapped)
        out2.index = tgt
        nmatch3 = out2.notna().any(axis=1).sum()
        print(f"[{label}] TCGA vial-aware match: {nmatch3}/{len(tgt)}")
        return out2

    print(f"[{label}] alignment: {nmatch}/{len(tgt)} (direct), {nmatch2}/{len(tgt)} (normalized)")
    return out

# =========================================================
# TIDE / ESTIMATE / EcoTyper
# =========================================================

def _canon_sample_id(s) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s

def _norm(s: str) -> str:
    return _canon_sample_id(s)

def _normalize_colname(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s
def _clean_ecotyper_token(x: str) -> str:
    return str(x).strip().strip('"').strip("'")



def load_tide_scores(path: Optional[str], target_index: Optional[pd.Index]) -> pd.DataFrame:
    if not _exists(path):
        print(f"[SKIP] TIDE table not found: {path}")
        return pd.DataFrame(index=target_index or pd.Index([], name="sample_id"))

    try:
        df = pd.read_csv(path, sep="\t", dtype=str)
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python", dtype=str)

    if df.empty:
        return pd.DataFrame(index=target_index or pd.Index([], name="sample_id"))

    id_col = None
    for cand in ["_patient_id", "Patient", "sample_id", "Sample", "ID", df.columns[0]]:
        if cand in df.columns:
            id_col = cand
            break

    df[id_col] = _norm_ids(df[id_col])
    df = df.set_index(id_col)

    drop_cols = {
        "Patient", "No benefits", "Responder",
        "_status_std", "_subtype", "Surv_Time", "Surv_Event",
        "Chemo", "Hormone", "age_dx"
    }
    drop_cols |= {c for c in df.columns if str(c).startswith("_")}
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    out = df.apply(lambda s: pd.to_numeric(s, errors="coerce"))
    out.index = _norm_ids(out.index)

    # strip BEFORE alignment
    out = normalize_index_to_tcga_sample_level(out, how="mean")

    if target_index is not None:
        out = align_index_smart(out, target_index, "tide", try_tcga_vial_strip=True)

    out = out.fillna(0.0)

    # preserve original names
    out.columns = [f"RNAblock__TIDE__{str(c).strip()}" for c in out.columns]

    # optional compatibility alias only if old training used MSI Expr Sig
    if "RNAblock__TIDE__MSI Score" in out.columns and "RNAblock__TIDE__MSI Expr Sig" not in out.columns:
        out["RNAblock__TIDE__MSI Expr Sig"] = out["RNAblock__TIDE__MSI Score"]

    return out



def load_estimate_scores(path: Optional[str], target_index: Optional[pd.Index]) -> pd.DataFrame:
    if not _exists(path):
        print(f"[SKIP] ESTIMATE table not found: {path}")
        return pd.DataFrame(index=target_index or pd.Index([], name="sample_id"))

    df = _read_table_any(path)
    if df.empty:
        return pd.DataFrame(index=target_index or pd.Index([], name="sample_id"))

    idx_name = str(df.index.name).strip().lower() if df.index.name is not None else ""
    idx_looks_like_id = (
        idx_name in {"sample", "sample_id", "patient", "patient_id", "id", "_patient_id", "mixture"}
        or df.index.dtype == object
    )

    if "patient_id" in df.columns and df["patient_id"].notna().any():
        df["patient_id"] = _norm_ids(df["patient_id"])
        df = df.set_index("patient_id")
    elif "_merge_sample_id_" in df.columns and df["_merge_sample_id_"].notna().any():
        df["_merge_sample_id_"] = _norm_ids(df["_merge_sample_id_"])
        df = df.set_index("_merge_sample_id_")
    elif "Sample" in df.columns and df["Sample"].notna().any():
        df["Sample"] = _norm_ids(df["Sample"])
        df = df.set_index("Sample")
    elif "sample_id" in df.columns and df["sample_id"].notna().any():
        df["sample_id"] = _norm_ids(df["sample_id"])
        df = df.set_index("sample_id")
    elif "ID" in df.columns and df["ID"].notna().any():
        df["ID"] = _norm_ids(df["ID"])
        df = df.set_index("ID")
    elif idx_looks_like_id:
        df = df.copy()
        df.index = _norm_ids(df.index)
    else:
        c0 = df.columns[0]
        df[c0] = _norm_ids(df[c0])
        df = df.set_index(c0)

    preferred = ["StromalScore", "ImmuneScore", "ESTIMATEScore", "TumorPurity"]
    keep_cols = [c for c in preferred if c in df.columns]

    if not keep_cols:
        tmp = df.apply(lambda s: pd.to_numeric(s, errors="coerce"))
        keep_cols = tmp.columns[tmp.notna().any(axis=0)].tolist()

    out = df[keep_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    out.index = _norm_ids(out.index)

    if out.index.duplicated().any():
        out = out.groupby(out.index).mean()

    # strip before alignment so TCGA sample-level matching can succeed
    out = normalize_index_to_tcga_sample_level(out, how="mean")

    if target_index is not None:
        out = align_index_smart(out, target_index, "estimate", try_tcga_vial_strip=True)

    out = out.fillna(0.0)
    out.columns = [f"RNAblock__ESTIMATE__{_normalize_colname(c)}" for c in out.columns]
    return out

def _load_ecotyper_tree(ecotyper_base_dir: str):
    out = {
        "cell_states_abundance": pd.DataFrame(),
        "cell_states_assignment": pd.DataFrame(),
        "ecotypes_abundance": pd.DataFrame(),
        "ecotypes_assignment": pd.DataFrame(),
    }

    if not ecotyper_base_dir or not os.path.exists(ecotyper_base_dir):
        print(f"[WARN] EcoTyper base dir missing: {ecotyper_base_dir}")
        return out

    cell_abunds = []
    cell_assigns = []
    eco_abunds = []
    eco_assigns = []

    for name in sorted(os.listdir(ecotyper_base_dir)):
        d = os.path.join(ecotyper_base_dir, name)
        if not os.path.isdir(d):
            continue

        lo = name.lower()
        is_ecotypes_dir = ("ecotype" in lo) or ("ecotypes" in lo)

        abund_path = None
        states_path = None
        types_path = None

        for fn in sorted(os.listdir(d)):
            flo = fn.lower()
            if not (flo.endswith(".txt") or flo.endswith(".tsv") or flo.endswith(".csv")):
                continue

            full = os.path.join(d, fn)

            if "abund" in flo:
                abund_path = full

            if (
                "state_assign" in flo
                or "state_assignment" in flo
                or flo == "state_assignments.txt"
            ):
                states_path = full

            if ("type" in flo or "ecotype" in flo) and ("abund" not in flo):
                types_path = full

        print(f"[ECOTYPER DEBUG] dir={name} | abund={abund_path} | states={states_path} | types={types_path}")

        if is_ecotypes_dir:
            if abund_path is not None:
                eco_abunds.append(
                    _read_ecotyper_abundance_final(
                        abund_path,
                        feature_prefix="ECOTYPES__",
                    )
                )

            if types_path is not None:
                eco_assigns.append(
                    _read_ecotyper_assignment_final(
                        types_path,
                        feature_prefix="ECOTYPE_ASSIGN__",
                    )
                )

        else:
            raw_name = str(name).strip()

            if abund_path is not None:
                cell_abunds.append(
                    _read_ecotyper_abundance_final(
                        abund_path,
                        feature_prefix=f"CELLSTATE_{raw_name}__",
                    )
                )

            if states_path is not None:
                cell_assigns.append(
                    _read_ecotyper_assignment_final(
                        states_path,
                        feature_prefix=f"CELLSTATE_ASSIGN__{raw_name}__",
                    )
                )

    if cell_abunds:
        out["cell_states_abundance"] = pd.concat(cell_abunds, axis=1)
    if cell_assigns:
        out["cell_states_assignment"] = pd.concat(cell_assigns, axis=1)
    if eco_abunds:
        out["ecotypes_abundance"] = pd.concat(eco_abunds, axis=1)
    if eco_assigns:
        out["ecotypes_assignment"] = pd.concat(eco_assigns, axis=1)

    for k, df in out.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.index = _norm_ids(df.index)
            df = df.groupby(df.index).max()
            df = df.loc[:, ~df.columns.duplicated()]
            out[k] = df

    print("[ECOTYPER LOAD] base:", ecotyper_base_dir)
    print("[ECOTYPER LOAD] cell_states_abundance:", out["cell_states_abundance"].shape)
    print("[ECOTYPER LOAD] cell_states_assignment:", out["cell_states_assignment"].shape)
    print("[ECOTYPER LOAD] ecotypes_abundance:", out["ecotypes_abundance"].shape)
    print("[ECOTYPER LOAD] ecotypes_assignment:", out["ecotypes_assignment"].shape)

    return out



def _sample_overlap_score(vals, target_index: Optional[pd.Index]) -> int:
    if target_index is None or len(target_index) == 0:
        return 0
    a = set(_norm_ids(pd.Index(vals).astype(str)))
    b = set(_norm_ids(pd.Index(target_index).astype(str)))
    return len(a & b)


def _looks_like_sample_ids(vals) -> float:
    vals = pd.Index(vals).astype(str)
    vals = vals[vals.notna()]
    if len(vals) == 0:
        return 0.0

    vals = vals[: min(50, len(vals))]

    pats = [
        r"^MB[_-]\d+",
        r"^TCGA-[A-Z0-9-]+",
        r"^[A-Z]{1,4}[_-]?[A-Z0-9]+$",
    ]

    hits = 0
    for v in vals:
        s = str(v).strip()
        if any(re.match(p, s, flags=re.I) for p in pats):
            hits += 1
    return hits / max(1, len(vals))


def _looks_like_ecotyper_feature_names(vals) -> float:
    vals = pd.Index(vals).astype(str)
    vals = vals[vals.notna()]
    if len(vals) == 0:
        return 0.0

    vals = vals[: min(50, len(vals))]

    feature_tokens = [
        "ecotype", "state", "cell", "b_cell", "t_cell", "myeloid",
        "luminal", "basal", "endothelial", "fibroblast", "plasma",
        "macrophage", "dendritic", "neutrophil", "mast"
    ]

    hits = 0
    for v in vals:
        s = str(v).strip().lower()
        if any(tok in s for tok in feature_tokens):
            hits += 1
    return hits / max(1, len(vals))


def _read_ecotyper_abundance_final(path: str, feature_prefix: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    df = None
    for kw in [
        {"sep": "\t", "index_col": 0},
        {"sep": None, "engine": "python", "index_col": 0},
        {"sep": ",", "index_col": 0},
    ]:
        try:
            df = pd.read_csv(path, **kw)
            if df is not None and not df.empty:
                break
        except Exception:
            continue

    if df is None or df.empty:
        return pd.DataFrame()

    df.columns = _norm_ids(df.columns)
    states = [_clean_ecotyper_token(x) for x in df.index]

    out = df.T
    out.columns = [f"{feature_prefix}{st}" for st in states]
    out.index = _norm_ids(out.index)
    out = out.apply(lambda s: pd.to_numeric(s, errors="coerce"))

    if out.index.duplicated().any():
        out = out.groupby(out.index).mean()

    return out

def _read_ecotyper_assignment_final(path: str, feature_prefix: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    if not lines:
        return pd.DataFrame()

    first = lines[0].strip().strip('"').strip("'").lower()
    skiprows = 1 if first in {"states", "state", "types", "type", "ecotypes", "ecotype"} else 0

    try:
        df = pd.read_csv(path, sep="\t", dtype=str, skiprows=skiprows)
    except Exception:
        try:
            df = pd.read_csv(path, sep=None, engine="python", dtype=str, skiprows=skiprows)
        except Exception:
            return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df.columns = [str(c).strip().strip('"').strip("'") for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).astype(str).str.lower().str.startswith("unnamed")].copy()
    if df.empty:
        return pd.DataFrame()

    sample_col = next(
        (c for c in ["ID", "sample", "sample_id", "patient_id", "Patient", "_patient_id", "Mixture"] if c in df.columns),
        df.columns[0]
    )

    cols = [c for c in df.columns if c != sample_col]
    assign_col = None
    for c in ["State", "Ecotype", "MaxEcotype"]:
        if c in cols:
            assign_col = c
            break

    if assign_col is None:
        bad_like = {
            "AssignmentP", "AssignmentQ", "AssignedToEcotypeStates",
            "Tissue", "Histology", "Type", "OS_Time", "OS_Status",
            "InitialState", "InitialEcotype"
        }
        cand = [c for c in cols if c not in bad_like and not str(c).lower().startswith("initial")]
        for key in ["ecotype", "state", "cluster", "label", "class"]:
            hit = next((c for c in cand if key in str(c).lower()), None)
            if hit is not None:
                assign_col = hit
                break

    if assign_col is None:
        return pd.DataFrame()

    sub = df[[sample_col, assign_col]].copy()
    sub[sample_col] = _norm_ids(sub[sample_col])
    sub[assign_col] = sub[assign_col].map(_clean_ecotyper_token)

    sub = sub[
        sub[sample_col].notna()
        & (sub[sample_col] != "")
        & sub[assign_col].notna()
        & (sub[assign_col] != "")
        & (~sub[assign_col].str.lower().isin({"nan", "na", "none", "missing", "unassigned", "false"}))
    ].copy()

    if sub.empty:
        return pd.DataFrame()

    sub["feature"] = sub[assign_col].map(lambda x: f"{feature_prefix}{x}")
    out = pd.crosstab(sub[sample_col], sub["feature"])
    out[out > 0] = 1
    out.index = _norm_ids(out.index)
    return out.groupby(out.index).max().astype(float)




def load_ecotyper_scores(base_dir: Optional[str], target_index: Optional[pd.Index]) -> pd.DataFrame:
    if not base_dir or not os.path.exists(base_dir):
        print(f"[SKIP] EcoTyper base dir not found: {base_dir}")
        return pd.DataFrame(index=target_index or pd.Index([], name="sample_id"))

    eco = _load_ecotyper_tree(base_dir)

    key_to_tag = {
        "cell_states_abundance": "ECOTYPER__CELLSTATE_ABUND__",
        "cell_states_assignment": "ECOTYPER__CELLSTATE_ASSIGN__",
        "ecotypes_abundance": "ECOTYPER__ECOTYPE_ABUND__",
        "ecotypes_assignment": "ECOTYPER__ECOTYPE_ASSIGN__",
    }

    parts = []
    for key, tag in key_to_tag.items():
        df = eco.get(key, pd.DataFrame())
        if df is None or df.empty:
            continue

        df = df.copy()
        df.index = _norm_ids(df.index)

        # strip before alignment; use max for binary assignment blocks, mean for abundance
        how = "max" if "assignment" in key else "mean"
        df = normalize_index_to_tcga_sample_level(df, how=how)

        if target_index is not None:
            df = align_index_smart(df, target_index, f"ecotyper:{key}", try_tcga_vial_strip=True)

        df.columns = [f"RNAblock__{tag}{c}" for c in df.columns.astype(str)]
        parts.append(df)

    if not parts:
        return pd.DataFrame(index=target_index or pd.Index([], name="sample_id"))

    out = pd.concat(parts, axis=1)
    out.index = _norm_ids(out.index)
    out = out.groupby(out.index).max()
    out = out.loc[:, ~out.columns.duplicated()]
    return out.fillna(0.0)

# =========================================================
# Mutation / TMB / pTMB
# =========================================================

def mutation_pathway_matrix(mut_df: pd.DataFrame, pathways: Dict[str, List[str]],
                            uppercase: bool = True, as_binary: bool = True) -> pd.DataFrame:
    if mut_df.empty or not pathways:
        return pd.DataFrame(index=mut_df.index)
    X = mut_df.copy().apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = (X > 0).astype(int) if as_binary else X
    if uppercase:
        X.columns = pd.Index([str(c).upper() for c in X.columns], name=X.columns.name)
        pathways = {k: [g.upper() for g in v] for k, v in pathways.items()}
    out = {}
    gene_universe = set(X.columns)
    for p, genes in pathways.items():
        pres = [g for g in genes if g in gene_universe]
        if not pres:
            continue
        s = X[pres].sum(axis=1)
        out[p] = (s > 0).astype(int) if as_binary else s
    return pd.DataFrame(out, index=X.index)


def load_tmb_summary(path: Optional[str], sample_col: str, tmb_col: str, pathogenic_col: str) -> pd.DataFrame:
    if not _exists(path):
        print(f"[SKIP] TMB table not found: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep="\t")
    if sample_col not in df.columns:
        for c in ["sample", "Sample", "SampleID", "Mixture", "sample_id", "ID", "PATIENT_ID"]:
            if c in df.columns:
                sample_col = c
                break
    if sample_col not in df.columns:
        print("[SKIP] TMB table missing sample column.")
        return pd.DataFrame()

    keep = [c for c in [tmb_col, pathogenic_col] if c in df.columns]
    if not keep:
        return pd.DataFrame(index=pd.Index([], name="sample_id"))

    m = df[[sample_col] + keep].copy()
    m[sample_col] = m[sample_col].astype(str)
    m = m.set_index(sample_col)
    m.index = _norm_ids(m.index)
    for c in keep:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    ren = {}
    if tmb_col in keep:
        ren[tmb_col] = "MUTblock__TMB"
    if pathogenic_col in keep:
        ren[pathogenic_col] = "MUTblock__pathogenic_mutation"
    return m.rename(columns=ren)[list(ren.values())]


def _clean_sample_id_for_merge(x) -> str:
    if pd.isna(x):
        return ""
    return str(_norm_ids([x])[0])


def _clean_gene_symbol(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def _infer_maf_cols(maf: pd.DataFrame, maf_sample_col: Optional[str], maf_gene_col: Optional[str]) -> Tuple[str, str]:
    sample_col = maf_sample_col
    gene_col = maf_gene_col

    if not sample_col or sample_col not in maf.columns:
        for c in ["Tumor_Sample_Barcode", "sample_id", "Sample", "ID", "patient_id", "PATIENT_ID", "sample"]:
            if c in maf.columns:
                sample_col = c
                break

    if not gene_col or gene_col not in maf.columns:
        for c in ["Hugo_Symbol", "Gene", "gene", "SYMBOL", "Gene_Symbol"]:
            if c in maf.columns:
                gene_col = c
                break

    if sample_col not in maf.columns or gene_col not in maf.columns:
        raise KeyError(
            f"[PTMB] Could not infer MAF sample/gene columns. "
            f"sample_col={sample_col}, gene_col={gene_col}, columns={list(maf.columns)[:50]}"
        )
    return sample_col, gene_col


def load_cds_lengths_from_gtf(
    gtf_path: str,
    cache_csv: Optional[str] = None,
    prefer_gene_name: bool = True,
) -> Dict[str, float]:
    if cache_csv and os.path.exists(cache_csv):
        try:
            cache = pd.read_csv(cache_csv)
            if {"gene", "cds_len_bp"}.issubset(cache.columns):
                return {
                    str(r["gene"]).strip().upper(): float(r["cds_len_bp"])
                    for _, r in cache.iterrows()
                    if pd.notna(r["gene"]) and pd.notna(r["cds_len_bp"])
                }
        except Exception:
            pass

    if not gtf_path or not os.path.exists(gtf_path):
        raise FileNotFoundError(f"[PTMB] GTF not found: {gtf_path}")

    gene_to_intervals = defaultdict(list)

    with _open_text(gtf_path) as f:
        for ln in f:
            if not ln or ln.startswith("#"):
                continue
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            feature = parts[2]
            if feature != "CDS":
                continue

            try:
                start = int(parts[3])
                end = int(parts[4])
            except Exception:
                continue

            attr = parts[8]
            gene_id = _extract_attr(attr, "gene_id")
            gene_name = _extract_attr(attr, "gene_name") or _extract_attr(attr, "gene_symbol")
            gene_key = gene_name if (prefer_gene_name and gene_name) else gene_id
            if not gene_key:
                continue

            gene_key = str(gene_key).strip().upper()
            gene_to_intervals[gene_key].append((start, end))

    cds_len = {}
    for gene, ivals in gene_to_intervals.items():
        ivals = sorted(ivals, key=lambda x: (x[0], x[1]))
        merged = []
        for s, e in ivals:
            if not merged or s > merged[-1][1] + 1:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        total = sum((e - s + 1) for s, e in merged)
        if total > 0:
            cds_len[gene] = float(total)

    if cache_csv:
        try:
            os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
            pd.DataFrame({
                "gene": list(cds_len.keys()),
                "cds_len_bp": list(cds_len.values()),
            }).to_csv(cache_csv, index=False)
        except Exception as e:
            print(f"[PTMB] Warning: could not write CDS cache {cache_csv}: {e}")

    return cds_len


def compute_ptmb_matrix_from_maf(
    maf_path: str,
    sample_ids_target: pd.Index,
    gene_sets: Dict[str, List[str]],
    cds_len_bp: Dict[str, float],
    maf_sample_col: Optional[str] = None,
    maf_gene_col: Optional[str] = None,
    denom_policy: str = "all_genes_with_len",
    min_genes_with_len: int = 1,
    out_prefix: str = "MUTblock__pTMB__",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not maf_path or not os.path.exists(maf_path):
        raise FileNotFoundError(f"[PTMB] maf_path not found: {maf_path}")

    if not gene_sets:
        raise ValueError("[PTMB] gene_sets is empty (nothing to compute).")

    denom_policy = str(denom_policy).strip().lower()
    if denom_policy not in {"all_genes_with_len", "all_genes", "mutated_genes_with_len"}:
        raise ValueError(
            "[PTMB] denom_policy must be one of: "
            "all_genes_with_len, all_genes, mutated_genes_with_len"
        )

    maf = _read_table_auto(maf_path)
    maf.columns = pd.Index([str(c).strip() for c in maf.columns])

    sample_col, gene_col = _infer_maf_cols(maf, maf_sample_col, maf_gene_col)

    # Handle duplicate column names safely by taking the first matching column
    sample_idx = np.where(maf.columns == sample_col)[0]
    gene_idx = np.where(maf.columns == gene_col)[0]

    if len(sample_idx) == 0:
        raise KeyError(f"[PTMB] sample column not found: {sample_col}")
    if len(gene_idx) == 0:
        raise KeyError(f"[PTMB] gene column not found: {gene_col}")

    if len(sample_idx) > 1:
        print(f"[PTMB] Warning: sample column '{sample_col}' appears {len(sample_idx)} times; using first occurrence.")
    if len(gene_idx) > 1:
        print(f"[PTMB] Warning: gene column '{gene_col}' appears {len(gene_idx)} times; using first occurrence.")

    sample_series = maf.iloc[:, sample_idx[0]]
    gene_series = maf.iloc[:, gene_idx[0]]

    maf = pd.DataFrame({
        "_sample_raw": sample_series,
        "_gene_raw": gene_series
    }).dropna()

    maf["_sample"] = maf["_sample_raw"].map(_clean_sample_id_for_merge)
    maf["_gene"] = maf["_gene_raw"].map(_clean_gene_symbol)

    maf = maf[(maf["_sample"].astype(str) != "") & (maf["_gene"].astype(str) != "")]
    target_set = set(pd.Index(sample_ids_target).astype(str))
    maf = maf[maf["_sample"].isin(target_set)]

    cnt = maf.groupby(["_sample", "_gene"]).size().rename("mut_count").reset_index()

    cds_clean = {
        str(k).strip().upper(): float(v)
        for k, v in cds_len_bp.items()
        if pd.notna(k) and pd.notna(v)
    }

    cnt["cds_len_bp"] = cnt["_gene"].map(cds_clean)
    cnt["cds_len_bp"] = pd.to_numeric(cnt["cds_len_bp"], errors="coerce")
    cnt = cnt.dropna(subset=["cds_len_bp"])
    cnt = cnt[cnt["cds_len_bp"] > 0]
    cnt["rate"] = cnt["mut_count"].astype(float) / cnt["cds_len_bp"].astype(float)

    mutated_genes_with_len = set(cnt["_gene"].astype(str).unique().tolist())

    ptmb_cols = {}
    info_rows = []
    all_samples = pd.Index(sample_ids_target).astype(str)

    for pw_name, genes_raw in gene_sets.items():
        genes_clean = [_clean_gene_symbol(g) for g in (genes_raw or []) if str(g).strip()]
        genes_clean = [g for g in genes_clean if g]
        genes_unique = sorted(list(dict.fromkeys(genes_clean)))

        genes_with_len = [
            g for g in genes_unique
            if g in cds_clean and np.isfinite(cds_clean[g]) and float(cds_clean[g]) > 0
        ]

        n_raw = int(len(genes_unique))
        n_len = int(len(genes_with_len))

        if denom_policy == "all_genes":
            Ni = n_raw
        elif denom_policy == "mutated_genes_with_len":
            Ni = int(len([g for g in genes_with_len if g in mutated_genes_with_len]))
        else:
            Ni = n_len

        if Ni < int(min_genes_with_len) or n_len == 0:
            values = pd.Series(np.nan, index=all_samples, dtype=float)
        else:
            sub = cnt[cnt["_gene"].isin(genes_with_len)]
            numer = sub.groupby("_sample")["rate"].sum()
            values = numer.reindex(all_samples).fillna(0.0) / float(Ni)

        col = out_prefix + _slug(pw_name)
        ptmb_cols[col] = values.values.astype(float)

        info_rows.append({
            "pathway": str(pw_name),
            "ptmb_feature": col,
            "n_genes_raw_unique": n_raw,
            "n_genes_with_len": n_len,
            "Ni_used": int(Ni),
            "denom_policy": denom_policy,
        })

    ptmb_df = pd.DataFrame(ptmb_cols, index=all_samples)
    ptmb_info = (
        pd.DataFrame(info_rows)
        .sort_values(["n_genes_with_len", "Ni_used"], ascending=[False, False])
        .reset_index(drop=True)
    )

    return ptmb_df, ptmb_info

# =========================================================
# CNV-derived features
# =========================================================

def expand_tcga_patient_ids_to_sample_index(
    df: pd.DataFrame,
    target_index: pd.Index,
    prefer_sample_type: str = "01",
    agg: str = "mean",
) -> pd.DataFrame:
    """
    df index: patient-level TCGA IDs, e.g. TCGA-C8-A1HN
    target_index: sample-level IDs, e.g. TCGA-C8-A1HN-01
    returns df reindexed to target_index
    """
    if df is None or df.empty:
        return pd.DataFrame(index=target_index)

    out = df.copy()
    out.index = pd.Index(out.index).astype(str).str.strip()

    tgt = pd.Index(target_index).astype(str)

    # patient -> list of target sample IDs
    patient_to_targets = {}
    for sid in tgt:
        parts = sid.split("-")
        if len(parts) >= 3:
            patient = "-".join(parts[:3])
            patient_to_targets.setdefault(patient, []).append(sid)

    expanded = []
    for pid in out.index:
        vals = out.loc[[pid]].copy()

        matches = patient_to_targets.get(pid, [])
        if not matches:
            continue

        # prefer requested sample type if multiple target samples exist
        preferred = [m for m in matches if len(m.split("-")) >= 4 and m.split("-")[3].startswith(prefer_sample_type)]
        chosen = preferred if preferred else matches

        for sid in chosen:
            row = vals.iloc[0].copy()
            row.name = sid
            expanded.append(row)

    if not expanded:
        return pd.DataFrame(index=tgt, columns=out.columns)

    ex = pd.DataFrame(expanded)

    if ex.index.duplicated().any():
        if agg == "first":
            ex = ex[~ex.index.duplicated(keep="first")]
        elif agg == "max":
            ex = ex.groupby(level=0).max(numeric_only=True)
        else:
            ex = ex.groupby(level=0).mean(numeric_only=True)

    ex = ex.reindex(tgt)
    return ex


def _load_cytoband(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not _exists(path):
        print(f"[SKIP] cytoband file not found: {path}")
        return None
    try:
        cb = pd.read_csv(path, sep=None, engine="python", header=None, names=["chrom", "start", "end", "name", "stain"])
    except Exception:
        cb = pd.read_csv(path, sep="\t", header=None, names=["chrom", "start", "end", "name", "stain"])
    cb["chrom"] = cb["chrom"].astype(str).str.replace("^chr", "", regex=True)
    return cb


def _centromeres_from_cytoband(cb: pd.DataFrame) -> Dict[str, Tuple[int, int]]:
    cents = {}
    for chrom, grp in cb.groupby("chrom"):
        acen = grp.loc[grp["stain"].str.lower() == "acen"]
        if acen.empty:
            continue
        cents[chrom] = (int(acen["start"].min()), int(acen["end"].max()))
    return cents


def _build_gene_coord_table_for_symbols(id_map: pd.DataFrame) -> pd.DataFrame:
    if id_map is None or id_map.empty:
        return pd.DataFrame(columns=["chrom", "start", "end", "length"])
    df = id_map.copy()
    have = df.dropna(subset=["symbol", "chrom", "start", "end"]).copy()
    if have.empty:
        return pd.DataFrame(columns=["chrom", "start", "end", "length"])
    have["symbol"] = have["symbol"].astype(str).str.upper()
    have["chrom"] = have["chrom"].astype(str).str.replace("^chr", "", regex=True)
    have["start"] = pd.to_numeric(have["start"], errors="coerce")
    have["end"] = pd.to_numeric(have["end"], errors="coerce")
    have = have.dropna(subset=["start", "end"])
    have["length"] = (have["end"] - have["start"]).clip(lower=1)
    have = have.drop_duplicates(subset=["symbol"])
    return have.set_index("symbol")[["chrom", "start", "end", "length"]]


def _assign_arm_series(chrom: pd.Series, pos: pd.Series, centromeres: Dict[str, Tuple[int, int]]) -> pd.Series:
    out = []
    for c, p in zip(chrom.astype(str), pd.to_numeric(pos, errors="coerce")):
        cen = centromeres.get(c)
        if cen is None or pd.isna(p):
            out.append(pd.NA)
            continue
        cstart, cend = cen
        if p < cstart:
            out.append("p")
        elif p > cend:
            out.append("q")
        else:
            out.append(pd.NA)
    return pd.Series(out, index=chrom.index, dtype="string")


def _assign_band_label_series(chrom: pd.Series, pos: pd.Series, cb: pd.DataFrame) -> pd.Series:
    by_chr = {str(k): g.sort_values(["start", "end"]).copy() for k, g in cb.groupby("chrom")}
    out = []
    for c, p in zip(chrom.astype(str), pd.to_numeric(pos, errors="coerce")):
        grp = by_chr.get(str(c))
        if grp is None or pd.isna(p):
            out.append(pd.NA)
            continue
        hit = grp[(grp["start"] <= p) & (grp["end"] >= p)]
        if hit.empty:
            out.append(pd.NA)
        else:
            band = str(hit.iloc[0]["name"]).strip()
            if not re.match(r"^(?:chr)?(?:\d+|X|Y)", band, flags=re.I):
                band = f"{c}{band}"
            out.append(band)
    return pd.Series(out, index=chrom.index, dtype="string")


def _derive_pga_from_cnv_gene_matrix(
    cn_gene_df: pd.DataFrame,    # centered CNV (0 neutral)
    id_map: pd.DataFrame,
    cytoband_file: Optional[str],
    delta: float = 0.30,
    min_genes: int = 30,
    weight_mode: str = "length",
) -> Tuple[pd.Series, pd.DataFrame]:
    if cn_gene_df is None or cn_gene_df.empty:
        return pd.Series(dtype=float), pd.DataFrame()

    coords = _build_gene_coord_table_for_symbols(id_map)
    if coords.empty:
        print("[PGA-from-CNV] No gene coordinates in id_map; cannot compute PGA from gene matrix.")
        return pd.Series(dtype=float), pd.DataFrame()

    gene_cols = pd.Index([str(c).upper() for c in cn_gene_df.columns])
    have = coords.reindex(gene_cols).dropna(subset=["chrom", "start", "end"])
    if have.empty:
        print("[PGA-from-CNV] None of the CNV genes had coordinates; skipping.")
        return pd.Series(dtype=float), pd.DataFrame()

    if weight_mode == "length":
        w = have["length"].astype(float).values
    else:
        w = np.ones(have.shape[0], dtype=float)

    X = cn_gene_df.loc[:, have.index].to_numpy(dtype=float, copy=False)
    amp = (X > float(delta))
    dele = (X < -float(delta))
    alt = amp | dele



    wsum = w.sum()
    genome_pga = (alt * w).sum(axis=1) / wsum * 100.0
    genome_pga = pd.Series(genome_pga, index=cn_gene_df.index, name="CNVblock__Percent_Genome_Altered")

    arm_df = pd.DataFrame(index=cn_gene_df.index)
    cb = _load_cytoband(cytoband_file) if cytoband_file else None
    if cb is None:
        return genome_pga, arm_df

    cents = _centromeres_from_cytoband(cb)
    mids = (have["start"].astype(float) + have["end"].astype(float)) / 2.0
    arms = _assign_arm_series(have["chrom"], mids, cents)
    valid = arms.dropna()
    if valid.empty:
        return genome_pga, arm_df

    arm_ids = pd.MultiIndex.from_frame(pd.DataFrame({
        "chrom": have.loc[valid.index, "chrom"].astype(str).str.upper(),
        "arm": valid.astype(str),
    }))
    arm_to_indices = defaultdict(list)
    for j, (chrom, arm) in enumerate(arm_ids):
        arm_to_indices[(chrom, arm)].append(j)

    for (chrom, arm), idxs in sorted(arm_to_indices.items()):
        if len(idxs) < min_genes:
            continue
        w_arm = w[idxs]
        wsum_arm = w_arm.sum()
        if wsum_arm <= 0:
            continue
        A = alt[:, idxs]
        G = amp[:, idxs]
        D = dele[:, idxs]
        pct_alt = (A * w_arm).sum(axis=1) / wsum_arm * 100.0
        pct_amp = (G * w_arm).sum(axis=1) / wsum_arm * 100.0
        pct_del = (D * w_arm).sum(axis=1) / wsum_arm * 100.0
        arm_key = f"CNVblock__chr{chrom}_{arm}_percent_"
        arm_df[arm_key + "altered"] = pct_alt
        arm_df[arm_key + "amplified"] = pct_amp
        arm_df[arm_key + "deleted"] = pct_del

    return genome_pga, arm_df


def derive_arm_binary_from_arm_percent(arm_pct_df: pd.DataFrame, pct_threshold: float = 30.0) -> pd.DataFrame:
    if arm_pct_df is None or arm_pct_df.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=arm_pct_df.index)
    pat = re.compile(r"^CNVblock__chr(?P<chrom>\d+|X|Y)_(?P<arm>p|q)_percent_(?P<kind>amplified|deleted)$", re.I)

    for c in arm_pct_df.columns:
        m = pat.match(str(c))
        if not m:
            continue
        chrom = m.group("chrom").upper()
        arm = m.group("arm").lower()
        kind = m.group("kind").lower()
        suffix = "gain" if kind == "amplified" else "loss"
        out[f"{chrom}{arm}_{suffix}"] = (
            pd.to_numeric(arm_pct_df[c], errors="coerce").fillna(0.0) >= float(pct_threshold)
        ).astype(int)

    return out


def derive_band_binary_from_cnv_genes(
    cn_gene_df: pd.DataFrame,   # centered CNV (0 neutral)
    id_map: pd.DataFrame,
    cytoband_file: Optional[str],
    delta: float = 0.30,
    min_genes: int = 10,
    pct_threshold: float = 30.0,
) -> pd.DataFrame:
    if cn_gene_df is None or cn_gene_df.empty:
        return pd.DataFrame()

    cb = _load_cytoband(cytoband_file)
    if cb is None or cb.empty:
        print("[BAND-from-CNV] No cytoband file; cannot derive band binaries.")
        return pd.DataFrame(index=cn_gene_df.index)

    coords = _build_gene_coord_table_for_symbols(id_map)
    if coords.empty:
        print("[BAND-from-CNV] No gene coordinates in id_map.")
        return pd.DataFrame(index=cn_gene_df.index)

    gene_cols = pd.Index([str(c).upper() for c in cn_gene_df.columns])
    have = coords.reindex(gene_cols).dropna(subset=["chrom", "start", "end"])
    if have.empty:
        print("[BAND-from-CNV] No CNV genes matched coordinates.")
        return pd.DataFrame(index=cn_gene_df.index)

    mids = (have["start"].astype(float) + have["end"].astype(float)) / 2.0
    bands = _assign_band_label_series(have["chrom"], mids, cb)
    valid = bands.dropna()
    if valid.empty:
        return pd.DataFrame(index=cn_gene_df.index)

    X = cn_gene_df.loc[:, have.index].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=False)

    band_to_indices = defaultdict(list)
    for j, band in enumerate(valid.astype(str).tolist()):
        band_to_indices[str(band)].append(j)

    out = pd.DataFrame(index=cn_gene_df.index)
    for band, idxs in sorted(band_to_indices.items()):
        if len(idxs) < int(min_genes):
            continue
        sub = X[:, idxs]
        pct_amp = (sub > float(delta)).sum(axis=1) / sub.shape[1] * 100.0
        pct_del = ((-sub) > float(delta)).sum(axis=1) / sub.shape[1] * 100.0
        out[f"{band}_gain"] = (pct_amp >= float(pct_threshold)).astype(int)
        out[f"{band}_loss"] = (pct_del >= float(pct_threshold)).astype(int)

    return out


def _parse_percentish(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    pct = s.str.endswith("%")
    s[pct] = s[pct].str.rstrip("%")
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def cnv_gsva(cnv_df: pd.DataFrame, mode: str, pathways: Dict[str, List[str]]) -> pd.DataFrame:
    if cnv_df.empty or not pathways:
        return pd.DataFrame(index=cnv_df.index)

    X = _to_numeric(cnv_df).fillna(0.0)
    X.columns = pd.Index([c.upper() for c in X.columns])

    if mode == "signed":
        Xc = X
    elif mode == "abs":
        Xc = X.abs()
    else:
        raise ValueError("--cn_gsva_mode must be 'signed' or 'abs'")

    return gsva_scores(Xc, pathways)


def load_pga_table(path: Optional[str], target_index: Optional[pd.Index]) -> pd.DataFrame:
    if not _exists(path):
        print(f"[SKIP] PGA table not found: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep="\t")
    if df.empty:
        return pd.DataFrame()

    sample_col = _find_sample_col(df)
    pga_col = None
    for cand in ["Percent_Genome_Altered", "percent_genome_altered", "fraction_genome_altered", "FGA", "PGA"]:
        if cand in df.columns:
            pga_col = cand
            break
    if pga_col is None:
        norm = lambda s: re.sub(r"[^\w]+", "_", s.lower()).strip("_")
        normmap = {norm(c): c for c in df.columns}
        for cand in ["percent_genome_altered", "fraction_genome_altered", "fga", "pga"]:
            if cand in normmap:
                pga_col = normmap[cand]
                break

    if pga_col is None:
        print("[SKIP] PGA metric column not found.")
        return pd.DataFrame(index=target_index or pd.Index([], name="sample_id"))

    m = df[[sample_col, pga_col]].copy()
    m[sample_col] = _norm_ids(m[sample_col].astype(str))
    m = m.set_index(sample_col)
    m[pga_col] = _parse_percentish(m[pga_col])
    out = m.rename(columns={pga_col: "CNVblock__Percent_Genome_Altered"})
    return align_index_smart(out, target_index, "pga") if target_index is not None else out


# =========================================================
# Clinical
# =========================================================


def load_clinical(path: Optional[str], header_row: int, sample_col: str,
                  target_index: Optional[pd.Index]) -> pd.DataFrame:
    if not _exists(path):
        print(f"[SKIP] clinical not found: {path}")
        return pd.DataFrame(index=target_index or pd.Index([], name="sample_id"))

    hdr = max(0, int(header_row) - 1)
    try:
        df = pd.read_csv(path, sep=None, engine="python", header=hdr, dtype=str)
    except Exception:
        df = pd.read_csv(path, sep="\t", header=hdr, dtype=str)

    if df is None or df.empty:
        return pd.DataFrame(index=target_index or pd.Index([], name="sample_id"))

    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    if str(sample_col).lower() == "auto":
        for cand in ["sample_id", "PATIENT_ID", "case_submitter_id", "SampleID", "Sample", "ID", "Mixture"]:
            if cand in df.columns:
                sample_col = cand
                break

    if sample_col not in df.columns:
        print("[SKIP] Clinical missing sample column.")
        return pd.DataFrame(index=target_index or pd.Index([], name="sample_id"))

    df[sample_col] = _norm_ids(df[sample_col])
    df = df.set_index(sample_col)
    df.index = _norm_ids(df.index)

    out = pd.DataFrame(index=df.index)
    tolow = lambda s: s.astype(str).str.strip().str.lower()

    def _stage_to_num(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().upper()
        s = re.sub(r"^STAGE\s*", "", s)
        s = re.sub(r"\s+", "", s)
        s = re.sub(r"[ABC]$", "", s)
        m = re.match(r"^(IV|III|II|I)\b", s)
        if m:
            return {"I": 1.0, "II": 2.0, "III": 3.0, "IV": 4.0}.get(m.group(1), np.nan)
        m = re.match(r"^([1-4])", s)
        if m:
            return float(m.group(1))
        return np.nan

    def _bin_posneg(sr):
        s = tolow(sr).replace({
            "true": "positive", "false": "negative",
            "1": "positive", "0": "negative",
            "present": "positive", "absent": "negative",
            "na": "", "n/a": "", "--": "", "'--": ""
        })
        return s.map({"positive": 1.0, "pos": 1.0, "negative": 0.0, "neg": 0.0})

    def _her2_threelevel(sr):
        s = tolow(sr).replace({
            "indeterminate": "equivocal", "borderline": "equivocal",
            "amplified": "positive", "non-amplified": "negative",
            "3+": "positive", "2+": "equivocal", "1+": "negative", "0+": "negative",
            "na": "", "n/a": "", "--": "", "'--": ""
        })
        return s.map({"negative": 0.0, "equivocal": 1.0, "positive": 2.0})

    colmap = {c.lower().strip(): c for c in df.columns}

    def getcol(*names):
        for n in names:
            c = colmap.get(n.lower().strip())
            if c is not None:
                return c
        return None

    def pick(names, regexes=()):
        c = getcol(*names)
        if c is not None:
            return c
        for rx in regexes:
            m = [c for c in df.columns if re.search(rx, c, flags=re.I)]
            if m:
                return m[0]
        return None

    age_col = pick(["age_dx", "AGE_AT_DIAGNOSIS", "Age", "age"])
    if age_col is not None:
        out["CLINICALblock__age_dx"] = pd.to_numeric(df[age_col], errors="coerce")

    stage_col = pick(
        ["ajcc_css", "AJCC_CSS", "TUMOR_STAGE", "Tumor_Stage", "tumor_stage", "AJCC", "Stage"],
        (r"\bajcc\b", r"\btumou?r[_\s-]*stage\b", r"\bstage\b")
    )
    if stage_col is not None:
        out["CLINICALblock__ajcc_css"] = pd.to_numeric(df[stage_col].map(_stage_to_num), errors="coerce")

    ln_count_col = pick(
        ["LYMPH_NODES_EXAMINED_POSITIVE", "lymph_nodes_examined_positive", "lymphnode_invol", "lymph_nodes_pos", "positive_nodes"],
        (r"lymph.*(examined|positive|pos)",)
    )
    if ln_count_col is not None:
        ln = pd.to_numeric(df[ln_count_col], errors="coerce")
        out["CLINICALblock__lymph_nodes_pos"] = ln
        out["CLINICALblock__lymph_nodes_pos_binary"] = (ln.fillna(0) >= 1).astype(float)
        out["CLINICALblock__lymphnode_invol"] = (ln.fillna(0) >= 1).astype(float)

    er_col = pick(["er_lab", "er_status", "ER_STATUS", "er status", "er_status_by_ihc"], (r"\ber[\s_-]*lab\b", r"\ber[\s_-]*status\b"))
    pr_col = pick(["pr_lab", "pr_status", "PR_STATUS", "pr status", "pr_status_by_ihc"], (r"\bpr[\s_-]*lab\b", r"\bpr[\s_-]*status\b"))
    her2col = pick(["her2_lab", "her2_status", "HER2_STATUS", "her2 status", "her2_status_by_ihc"], (r"\bher2[\s_-]*lab\b", r"\bher2[\s_-]*status\b"))

    if er_col is not None:
        out["CLINICALblock__er_lab"] = _bin_posneg(df[er_col])
    if pr_col is not None:
        out["CLINICALblock__pr_lab"] = _bin_posneg(df[pr_col])
    if her2col is not None:
        out["CLINICALblock__her2_lab"] = _her2_threelevel(df[her2col])

    if not out.empty:
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.replace([np.inf, -np.inf], np.nan)

    # collapse duplicate patient rows before any TCGA expansion
    if out.index.duplicated().any():
        dup_ids = out.index[out.index.duplicated()].unique().tolist()
        print(f"[clinical] duplicate IDs before TCGA expansion; collapsing by mean. Example: {dup_ids[:10]}")
        out = out.groupby(out.index).mean()

    # TCGA clinical often uses patient-level IDs (e.g. TCGA-XX-XXXX) while target_index is sample-level
    if (
        target_index is not None
        and not out.empty
        and out.index.astype(str).str.match(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$", case=False).mean() > 0.8
    ):
        out = expand_tcga_patient_ids_to_sample_index(
            out,
            target_index,
            prefer_sample_type="01",
            agg="mean",
        )

    # strip vial letters only after patient->sample expansion if needed
    out = _maybe_strip_vials(out, "per_block", "Clinical")

    if out.index.duplicated().any():
        dup_ids = out.index[out.index.duplicated()].unique().tolist()
        print(f"[clinical] duplicate IDs after TCGA expansion; collapsing by mean. Example: {dup_ids[:10]}")
        out = out.groupby(out.index).mean()

    if target_index is not None and not out.empty:
        out = align_index_smart(out, target_index, "clinical", try_tcga_vial_strip=True)

    if not out.empty:
        out = out.fillna(0.0)

    return out

# =========================================================
# Feature-list fuzzy matcher
# =========================================================

_CLINICAL_SYNONYMS = {
    "ajcc_css": {"ajcc", "ajcc_stage", "ajcc_css", "tumor_stage", "tumour_stage", "stage", "ajcc_tnm", "tnm_stage"},
    "er_status": {"er_status", "er", "er_lab"},
    "pr_status": {"pr_status", "pr", "pr_lab"},
    "her2_status": {"her2_status", "her2", "her2_lab"},
    "age_dx": {"age_dx", "age", "age_at_diagnosis", "age_at_dx"},
    "lymph_nodes_pos": {"lymph_nodes_pos", "lymph_nodes_examined_positive", "nodes_pos", "ln_pos", "positive_nodes"},
    "lymph_nodes_pos_binary": {"lymph_nodes_pos_binary", "ln_any", "nodes_any", "ln_binary", "lymphnode_any", "lymphnode_invol_binary"},
    "lymphnode_invol": {"lymphnode_invol", "lymphnodes_involved", "ln_involved", "ln_any_binary"},
    "tumor_size": {"tumor_size", "tumour_size", "tumor_size_mm", "tumour_size_mm"},
    "lymphocyteresponse": {"lymphocyteresponse", "lymphocyte_response", "tils", "tils_score"},
}
_CLINICAL_ALIAS2CANON = {alias.lower(): canon for canon, aliases in _CLINICAL_SYNONYMS.items() for alias in aliases}


def _split_block_prefix(name: str) -> Tuple[str, str]:
    s = str(name)
    s_low = s.lower()

    prefix_map = {
        "cnvblock__": "CNVblock__", "cnblock__": "CNVblock__", "cnv__": "CNVblock__", "cn__": "CNVblock__",
        "mutblock__": "MUTblock__", "mut__": "MUTblock__",
        "rnablock__": "RNAblock__", "rna__": "RNAblock__", "gsea__": "RNAblock__",
        "clinicalblock__": "CLINICALblock__", "clinical__": "CLINICALblock__", "clinical_": "CLINICALblock__",
    }
    for pre_low, canon in prefix_map.items():
        if s_low.startswith(pre_low):
            return canon, s[len(pre_low):]
    return "", s


def _canon_clinical_tail(tail: str) -> str:
    t = str(tail).strip().lower().replace("__", "_")
    t = re.sub(r"\s+", "_", t).strip("_")
    if t in _CLINICAL_ALIAS2CANON:
        return _CLINICAL_ALIAS2CANON[t]
    t = t.replace("tumour", "tumor")
    if t == "tumor_stage":
        return "ajcc_css"
    if t == "ajcc":
        return "ajcc_css"
    if t in {"age_at_diagnosis", "age_at_dx"}:
        return "age_dx"
    return t


def _canon_full(name: str) -> str:
    block, tail = _split_block_prefix(name)

    if block == "CLINICALblock__":
        return f"{block}{_canon_clinical_tail(tail)}"

    ptmb_match = re.match(r"^pTMB__(.+)$", str(tail), flags=re.I)
    if ptmb_match:
        return f"{block}pTMB__{_slug(ptmb_match.group(1))}"

    gs_match = re.match(r"^GS_(.+)__(median_log2|geo_mean_log2|zscore_mean)$", str(tail), flags=re.I)
    if gs_match:
        return f"{block}GS_{_slug(gs_match.group(1))}__{gs_match.group(2).lower()}"

    path_hit = _extract_db_pathway_tail(str(tail))
    if path_hit is not None:
        return f"{block}{path_hit}"

    s = str(tail).strip().replace("weighted_percent_", "percent_")
    arm_pat = re.compile(r"^(?:chr)?(?P<chr>\d+|x|y)_(?P<arm>p|q)_percent_(?P<kind>altered|amplified|deleted)$", re.I)
    m = arm_pat.match(s)
    if m:
        s = f"chr{m.group('chr').upper()}_{m.group('arm').lower()}_percent_{m.group('kind').lower()}"
    return f"{block}{s}"


def load_feature_list(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
        if lines and lines[0].lower() in {"feature", "features"}:
            lines = lines[1:]
        if lines and all(("," not in ln and "\t" not in ln) for ln in lines):
            return lines
    except Exception:
        pass

    for sep in ["\t", ",", ";", r"\s+"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", comment="#")
        except Exception:
            continue
        if df is None or df.empty:
            continue
        cols = {c.lower(): c for c in df.columns}
        if "feature" in cols:
            return df[cols["feature"]].astype(str).str.strip().tolist()
        if df.shape[1] == 1:
            return df.iloc[:, 0].astype(str).str.strip().tolist()

    raise ValueError(f"Could not parse feature list: {path}")


def select_and_order_features_fuzzy(combined: pd.DataFrame, feature_list: List[str],
                                    fill_missing_zero: bool = True, mapping_log: Optional[str] = None) -> pd.DataFrame:
    if not feature_list:
        return combined

    all_cols = combined.columns.astype(str)
    exact = {c: c for c in all_cols}
    canon_map, slug_map, tail_map, bare_tail_map = {}, {}, {}, {}

    for c in all_cols:
        canon_map.setdefault(_canon_full(c), c)
        block, tail = _split_block_prefix(c)
        tail_low = tail.strip().lower()
        tail_map.setdefault(tail_low, c)
        slug_map.setdefault(_slug(c), c)

        if block == "CLINICALblock__":
            canon_tail = _canon_clinical_tail(tail_low)
            canon_map.setdefault(f"{block}{canon_tail}", c)
            tail_map.setdefault(canon_tail, c)
            bare_tail_map.setdefault(canon_tail, c)
            for alias in _CLINICAL_SYNONYMS.get(canon_tail, []):
                canon_map.setdefault(f"{block}{alias.lower()}", c)
                tail_map.setdefault(alias.lower(), c)
                bare_tail_map.setdefault(alias.lower(), c)

    out_cols, mappings = [], []

    for f in feature_list:
        target = (
            exact.get(f)
            or canon_map.get(_canon_full(f))
            or tail_map.get(_split_block_prefix(f)[1].strip().lower())
            or slug_map.get(_slug(f))
        )

        if target is None:
            f_low = str(f).lower()
            if any(k in f_low for k in ("er", "pr", "her2", "ajcc", "stage", "age", "lymph", "tumor_size", "tumour_size", "tils")):
                target = bare_tail_map.get(_canon_clinical_tail(f_low))

        if target is not None:
            s = combined[target].copy()
            s.name = f
            out_cols.append(s)
            mappings.append({"requested": f, "matched": target, "status": "matched"})
        else:
            mappings.append({"requested": f, "matched": None, "status": "missing"})
            out_cols.append(
                pd.Series(0.0, index=combined.index, name=f)
                if fill_missing_zero else
                pd.Series(np.nan, index=combined.index, name=f)
            )

    if mapping_log:
        pd.DataFrame(mappings).to_csv(mapping_log, index=False)

    return pd.concat(out_cols, axis=1)


# =========================================================
# CLI
# =========================================================

def get_args():
    p = argparse.ArgumentParser(description="Validation feature builder with fuzzy feature alignment.")

    p.add_argument("--training_fs_dir", default=None)
    p.add_argument("--training_rna_feature_file", default=None)
    p.add_argument("--training_cnv_feature_file", default=None)
    p.add_argument("--training_mut_feature_file", default=None)
    p.add_argument("--training_final_feature_file", default=None)

    p.add_argument("--expr", default=None)
    p.add_argument("--expr_orientation", choices=["auto", "samples_by_genes", "genes_by_samples"], default="auto")
    p.add_argument("--id_map", default=None)
    p.add_argument("--id_map_format", choices=["auto", "table", "gtf"], default="auto")
    p.add_argument("--strip_version", action="store_true")
    p.add_argument("--expr_is_metabric", action="store_true")

    p.add_argument("--pathways_gmt", nargs="*", default=None,
                   help="GMT file(s) containing REACTOME/HALLMARK/KEGG/GOBP pathway definitions.")
    p.add_argument("--selected_pathways_file", default=None)
    p.add_argument("--genes_subset_files", nargs="*", default=None)
    p.add_argument("--gene_sets_gmt", nargs="*", default=None)
    p.add_argument("--medians_from_subset_files", action="store_true")
    p.add_argument("--custom_gene_sets_file", default=HARDCODED_RNA_GENESET_CSV)
    p.add_argument("--custom_gene_set_min_overlap", type=int, default=1)

    p.add_argument("--cibersort_abs", default=None)
    p.add_argument("--tide_table", default=None)
    p.add_argument("--estimate_table", default=None)
    p.add_argument("--ecotyper_base_dir", default=None)

    p.add_argument("--mut", default=None)
    p.add_argument("--mut_orientation", choices=["auto", "samples_by_genes", "genes_by_samples"], default="auto")
    p.add_argument("--mut_id_type", choices=["ensembl", "symbol", "entrez"], default="symbol")
    p.add_argument("--mut_as_counts", action="store_true")
    p.add_argument("--mut_selected_pathways_file", default=None)

    p.add_argument("--tmb_table", default=None)
    p.add_argument("--tmb_sample_col", default="sample_id")
    p.add_argument("--tmb_tmb_col", default="TMB")
    p.add_argument("--tmb_pathogenic_col", default="pathogenic_mutations")

    p.add_argument("--ptmb_maf_path", default=None)
    p.add_argument("--ptmb_maf_sample_col", default=None)
    p.add_argument("--ptmb_maf_gene_col", default=None)
    p.add_argument("--ptmb_gtf", default=None)
    p.add_argument("--ptmb_cds_cache_csv", default=None)
    p.add_argument("--ptmb_denom_policy", choices=["all_genes_with_len", "all_genes", "mutated_genes_with_len"], default="all_genes_with_len")
    p.add_argument("--ptmb_min_genes_with_len", type=int, default=1)

    p.add_argument("--cnv", default=None)
    p.add_argument("--cnv_orientation", choices=["auto", "samples_by_genes", "genes_by_samples"], default="auto")
    p.add_argument("--cnv_id_type", choices=["ensembl", "symbol", "entrez"], default="symbol")
    p.add_argument("--cn_gsva_mode", choices=["signed", "abs"], default="signed")
    p.add_argument("--cnv_selected_pathways_file", default=None)
    p.add_argument(
        "--cnv_units",
        choices=["absolute", "log2", "centered", "ploidy_norm"],
        default="centered",
    )
    p.add_argument("--pga_table", default=None)
    p.add_argument("--cytoband_file", default=None)
    p.add_argument("--arm_from_cnv_delta", type=float, default=0.30)
    p.add_argument("--arm_from_cnv_min_genes", type=int, default=40)
    p.add_argument("--arm_from_cnv_weight", choices=["length", "equal"], default="length")
    p.add_argument("--derive_arm_binary", action="store_true")
    p.add_argument("--derive_band_binary", action="store_true")
    p.add_argument("--arm_binary_pct_threshold", type=float, default=30.0)
    p.add_argument("--band_binary_pct_threshold", type=float, default=30.0)
    p.add_argument("--band_from_cnv_min_genes", type=int, default=10)

    p.add_argument("--clinical_data", default=None)
    p.add_argument("--clinical_header_row", type=int, default=1)
    p.add_argument("--clinical_sample_col", default="auto")

    p.add_argument("--feature_list", default=None)
    p.add_argument("--mapping_log", default=None)

    p.add_argument("--tcga_vial_strip_mode", choices=["off", "per_block", "after"], default="per_block")
    p.add_argument("--tcga_strip_vial_letter", action="store_true")

    p.add_argument("--outdir", required=True)
    p.add_argument("--prefix", default="features")
    p.add_argument("--rna_precomputed_geneset_csv", default=None)

    return p.parse_args()


# =========================================================
# MAIN
# =========================================================

def main():
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)

    if getattr(args, "tcga_strip_vial_letter", False) and args.tcga_vial_strip_mode == "off":
        args.tcga_vial_strip_mode = "per_block"

    training_files = _resolve_training_feature_files(
        fs_dir=args.training_fs_dir,
        rna_file=args.training_rna_feature_file,
        cnv_file=args.training_cnv_feature_file,
        mut_file=args.training_mut_feature_file,
        final_file=args.training_final_feature_file,
    )

    rna_training_features = _load_feature_names_from_training_file(training_files["rna"])
    cnv_training_features = _load_feature_names_from_training_file(training_files["cnv"])
    mut_training_features = _load_feature_names_from_training_file(training_files["mut"])

    id_map = (
        _load_id_map(args.id_map, fmt=args.id_map_format, strip_version=args.strip_version)
        if _exists(args.id_map)
        else pd.DataFrame(columns=["ensembl_id", "entrez_id", "symbol", "chrom", "start", "end"])
    )

    # 1) Expression
    expr_sym_log = pd.DataFrame()
    if _exists(args.expr):
        print("Step 1: Expression → SYMBOL …")
        expr = _read_table_any(args.expr)
        if not expr.empty:
            expr_sym = coerce_expr_to_symbols(expr, args.expr_orientation, id_map, strip_version=args.strip_version)
            if args.expr_is_metabric:
                expr_sym_log = expr_sym
            else:
                expr_sym_log = np.log2(expr_sym + 1.0)
            expr_sym_log = _maybe_strip_vials(expr_sym_log, args.tcga_vial_strip_mode, "RNA expression")

    # 2) RNA GSVA
    gsva_block = pd.DataFrame()
    inferred_rna_pathways = []
    all_pathways = _read_gmt_many(args.pathways_gmt)

    if not expr_sym_log.empty and all_pathways:
        print("Step 2: RNA GSVA…")
        if args.selected_pathways_file:
            requested = [x.strip() for x in _read_names_txt(args.selected_pathways_file)]
            tmp_features = [f"RNAblock__{x}" for x in requested]
            matched_rna_gmt, rna_match_tbl = _subset_gmt_by_selected_pathway_features(tmp_features, "RNAblock__", all_pathways)
        elif rna_training_features:
            matched_rna_gmt, rna_match_tbl = _subset_gmt_by_selected_pathway_features(rna_training_features, "RNAblock__", all_pathways)
        else:
            matched_rna_gmt = all_pathways
            rna_match_tbl = pd.DataFrame()

        inferred_rna_pathways = list(matched_rna_gmt.keys())
        if not rna_match_tbl.empty:
            _write_csv(rna_match_tbl, os.path.join(args.outdir, f"{args.prefix}__rna_pathway_gmt_matches.csv"))

        if matched_rna_gmt:
            gsva_block = gsva_scores(expr_sym_log, matched_rna_gmt).add_prefix("RNAblock__")
            gsva_block = _maybe_strip_vials(gsva_block, args.tcga_vial_strip_mode, "RNA GSVA")

    # 3) RNA direct genes
    subset_block = pd.DataFrame()
    inferred_rna_direct_genes = []
    manual_rna_genes = []

    if not expr_sym_log.empty:
        print("Step 3: RNA gene subset…")
        if rna_training_features:
            inferred_rna_direct_genes = _infer_selected_direct_gene_features_from_training_features(
                rna_training_features, "RNAblock__", expr_sym_log.columns
            )

        if args.genes_subset_files:
            manual_rna_genes = sorted(set().union(*[_read_names_txt(p) for p in args.genes_subset_files if _exists(p)]))

        genes_to_subset = []
        seen = set()
        for g in inferred_rna_direct_genes + manual_rna_genes:
            g = str(g).strip()
            if g and g not in seen:
                genes_to_subset.append(g)
                seen.add(g)

        if genes_to_subset:
            subset_block = subset_existing_matrix_by_genes_ci(expr_sym_log, genes_to_subset)
            subset_block = _maybe_strip_vials(subset_block, args.tcga_vial_strip_mode, "RNA subset")
            subset_block = subset_block.reindex(expr_sym_log.index).add_prefix("RNAblock__")

    # 4) RNA medians
    med_block = pd.DataFrame()
    if not expr_sym_log.empty:
        print("Step 4: RNA medians…")
        median_sets = {}
        if args.gene_sets_gmt:
            for gmt in args.gene_sets_gmt:
                median_sets.update(_read_gmt_single(gmt))
        if args.medians_from_subset_files and args.genes_subset_files:
            for p in args.genes_subset_files:
                if _exists(p):
                    median_sets[Path(p).stem] = _read_names_txt(p)

        if median_sets:
            med = medians_by_sets(expr_sym_log, median_sets)
            med_block = med.add_prefix("RNAblock__")
            med_block = _maybe_strip_vials(med_block, args.tcga_vial_strip_mode, "RNA medians")

    # 4b) Custom GS_*
    custom_gs_block = pd.DataFrame()
    inferred_custom_gs_families = []

    if not expr_sym_log.empty and rna_training_features:
        print("Step 4b: Custom RNA gene-set scores…")
        inferred_custom_gs_families = _infer_selected_custom_gene_set_families_from_training_features(
            rna_training_features,
            block_prefix="RNAblock__",
        )
        custom_gene_sets = _read_gene_sets_csv(args.custom_gene_sets_file)

        if custom_gene_sets and inferred_custom_gs_families:
            matched_custom_sets, custom_match_tbl = _subset_custom_gene_sets_by_selected_families(
                inferred_custom_gs_families,
                custom_gene_sets,
            )
            if not custom_match_tbl.empty:
                _write_csv(custom_match_tbl, os.path.join(args.outdir, f"{args.prefix}__custom_gs_matches.csv"))

            if matched_custom_sets:
                custom_scores = _compute_gene_set_scores(
                    X_log2=expr_sym_log,
                    gene_sets=matched_custom_sets,
                    min_overlap=int(args.custom_gene_set_min_overlap),
                )
                custom_gs_block = custom_scores.add_prefix("RNAblock__")
                custom_gs_block = _maybe_strip_vials(custom_gs_block, args.tcga_vial_strip_mode, "Custom RNA gene-set scores")

    # 4c) Precomputed RNA gene-set table
    precomputed_rna_gs_block = pd.DataFrame()
    if args.rna_precomputed_geneset_csv:
        print("Step 4c: Precomputed RNA gene-set table…")
        precomputed_rna_gs_block = load_precomputed_rna_geneset_block(
            path=args.rna_precomputed_geneset_csv,
            target_index=expr_sym_log.index if not expr_sym_log.empty else None,
            tcga_vial_strip_mode=args.tcga_vial_strip_mode,
            keep_cols=["oncotype_like_score"],
        )


    # 5) CIBERSORT / TIDE / ESTIMATE / EcoTyper
    cib_block = pd.DataFrame()
    tide_block = pd.DataFrame()
    estimate_block = pd.DataFrame()
    ecotyper_block = pd.DataFrame()

    if _exists(args.cibersort_abs):
        cib_raw = read_cibersort_abs(args.cibersort_abs)
        cib_raw = _maybe_strip_vials(cib_raw, args.tcga_vial_strip_mode, "CIBERSORT ABS")
        cib_raw = align_index_smart(cib_raw, expr_sym_log.index if not expr_sym_log.empty else None, "cibersort")
        cib_block = cib_raw.add_prefix("RNAblock__")

    if _exists(args.tide_table):
        tide_block = load_tide_scores(args.tide_table, expr_sym_log.index if not expr_sym_log.empty else None)
        tide_block = _maybe_strip_vials(tide_block, args.tcga_vial_strip_mode, "TIDE")

    if _exists(args.estimate_table):
        estimate_block = load_estimate_scores(args.estimate_table, expr_sym_log.index if not expr_sym_log.empty else None)
        estimate_block = _maybe_strip_vials(estimate_block, args.tcga_vial_strip_mode, "ESTIMATE")

    if _exists(args.ecotyper_base_dir):
        ecotyper_block = load_ecotyper_scores(args.ecotyper_base_dir, expr_sym_log.index if not expr_sym_log.empty else None)
        ecotyper_block = _maybe_strip_vials(ecotyper_block, args.tcga_vial_strip_mode, "EcoTyper")

    # 6) MUT + TMB + pTMB
    mut_block = pd.DataFrame()
    tmb_block = pd.DataFrame()
    ptmb_block = pd.DataFrame()
    inferred_mut_pathways = []
    inferred_ptmb_pathways = []

    if _exists(args.mut):
        print("Step 6: MUT pathways + TMB + pTMB…")
        mut_raw = _read_table_any(args.mut)
        if not mut_raw.empty:
            if args.mut_orientation == "auto":
                mut_df = mut_raw if mut_raw.shape[0] >= mut_raw.shape[1] else mut_raw.T
            elif args.mut_orientation == "genes_by_samples":
                mut_df = mut_raw.T
            else:
                mut_df = mut_raw

            mut_df.index = mut_df.index.astype(str)
            mut_df.columns = mut_df.columns.astype(str)

            if args.mut_id_type == "ensembl":
                tmp = mut_df.copy()
                tmp.columns = tmp.columns.map(_strip_version)
                ens2sym, etz2sym, _ = _build_mappers(id_map)
                tmp.columns = pd.Index([ens2sym.get(c, c) for c in tmp.columns])
                mut_sym = _to_numeric(tmp)
            elif args.mut_id_type == "entrez":
                tmp = mut_df.copy()
                _, etz2sym, _ = _build_mappers(id_map)
                tmp.columns = pd.Index([etz2sym.get(c, c) for c in tmp.columns])
                mut_sym = _to_numeric(tmp)
            else:
                mut_sym = _to_numeric(mut_df)

            mut_sym.columns = [str(c).upper() for c in mut_sym.columns]
            mut_sym = _maybe_strip_vials(mut_sym, args.tcga_vial_strip_mode, "MUT (gene table)")
            if not expr_sym_log.empty:
                mut_sym = mut_sym.reindex(expr_sym_log.index)

            mut_paths = _read_gmt_many(args.pathways_gmt)

            # Binary mutation pathway features
            if args.mut_selected_pathways_file:
                requested = [x.strip() for x in _read_names_txt(args.mut_selected_pathways_file)]
                tmp_features = [f"MUTblock__{x}" for x in requested]
                matched_mut_gmt, mut_match_tbl = _subset_gmt_by_selected_pathway_features(
                    tmp_features, "MUTblock__", mut_paths, exclude_mut_special=True
                )
            elif mut_training_features:
                matched_mut_gmt, mut_match_tbl = _subset_gmt_by_selected_pathway_features(
                    mut_training_features, "MUTblock__", mut_paths, exclude_mut_special=True
                )
            else:
                matched_mut_gmt, mut_match_tbl = {}, pd.DataFrame()

            inferred_mut_pathways = list(matched_mut_gmt.keys())
            if not mut_match_tbl.empty:
                _write_csv(mut_match_tbl, os.path.join(args.outdir, f"{args.prefix}__mut_pathway_gmt_matches.csv"))

            if matched_mut_gmt:
                mut_block = mutation_pathway_matrix(
                    mut_sym,
                    matched_mut_gmt,
                    uppercase=True,
                    as_binary=(not args.mut_as_counts),
                )
                mut_block = _maybe_strip_vials(mut_block, args.tcga_vial_strip_mode, "MUT (pathway matrix)")
                mut_block = mut_block.add_prefix("MUTblock__")

            # pTMB features
            if mut_training_features and _exists(args.ptmb_maf_path) and _exists(args.ptmb_gtf):
                matched_ptmb_gmt, ptmb_match_tbl = _subset_gmt_by_selected_ptmb_features(
                    mut_training_features,
                    mut_paths,
                    block_prefix="MUTblock__",
                )
                inferred_ptmb_pathways = list(matched_ptmb_gmt.keys())

                if not ptmb_match_tbl.empty:
                    _write_csv(ptmb_match_tbl, os.path.join(args.outdir, f"{args.prefix}__ptmb_pathway_gmt_matches.csv"))

                if matched_ptmb_gmt:
                    cds_len_bp = load_cds_lengths_from_gtf(
                        args.ptmb_gtf,
                        cache_csv=args.ptmb_cds_cache_csv,
                        prefer_gene_name=True,
                    )
                    ptmb_raw, ptmb_info = compute_ptmb_matrix_from_maf(
                        maf_path=args.ptmb_maf_path,
                        sample_ids_target=mut_sym.index,
                        gene_sets=matched_ptmb_gmt,
                        cds_len_bp=cds_len_bp,
                        maf_sample_col=args.ptmb_maf_sample_col,
                        maf_gene_col=args.ptmb_maf_gene_col,
                        denom_policy=args.ptmb_denom_policy,
                        min_genes_with_len=int(args.ptmb_min_genes_with_len),
                        out_prefix="MUTblock__pTMB__",
                    )
                    ptmb_raw = ptmb_raw.reindex(mut_sym.index).fillna(0.0)
                    ptmb_block = np.log10(ptmb_raw.clip(lower=0.0) + 1e-12)
                    _write_csv(ptmb_info, os.path.join(args.outdir, f"{args.prefix}__ptmb_info.csv"))

        if _exists(args.tmb_table):
            tmb_df = load_tmb_summary(args.tmb_table, args.tmb_sample_col, args.tmb_tmb_col, args.tmb_pathogenic_col)
            tmb_df = _maybe_strip_vials(tmb_df, args.tcga_vial_strip_mode, "TMB summary")
            if not expr_sym_log.empty:
                tmb_df = tmb_df.reindex(expr_sym_log.index)
            tmb_block = tmb_df

    # 7) CNV GSVA + direct genes + PGA + ARM/BAND
    cn_gene_df = pd.DataFrame()
    cn_gene_df_for_pga = pd.DataFrame()
    cn_block = pd.DataFrame()
    cnv_gene_subset_block = pd.DataFrame()
    pga_block = pd.DataFrame()
    arm_binary_block = pd.DataFrame()
    band_binary_block = pd.DataFrame()
    inferred_cnv_pathways = []
    inferred_cnv_direct_genes = []

    if _exists(args.cnv):
        print("Step 7: CNV GSVA + direct CNV genes + derived CNV features…")
        cn_raw = _read_table_any(args.cnv)
        if not cn_raw.empty:
            if args.cnv_orientation == "auto":
                cn_df = cn_raw if cn_raw.shape[0] >= cn_raw.shape[1] else cn_raw.T
            elif args.cnv_orientation == "genes_by_samples":
                cn_df = cn_raw.T
            else:
                cn_df = cn_raw

            if args.cnv_id_type == "ensembl":
                cn_df = coerce_expr_to_symbols(cn_df, "samples_by_genes", id_map, strip_version=args.strip_version)
            elif args.cnv_id_type == "entrez":
                cn_tmp = cn_df.copy()
                cn_tmp.columns = cn_tmp.columns.astype(str)
                cn_df = coerce_expr_to_symbols(cn_tmp, "samples_by_genes", id_map, strip_version=False)
            else:
                cn_df = _to_numeric(cn_df)

            cn_df.columns = [str(c).upper() for c in cn_df.columns]

            # normalize row ids
            cn_df.index = pd.Index(cn_df.index).astype(str).str.strip()

            # collapse duplicate sample/patient rows if present
            if cn_df.index.duplicated().any():
                dup_ids = cn_df.index[cn_df.index.duplicated()].unique().tolist()
                print(f"[CNV] duplicate row IDs before TCGA expansion; collapsing by mean. Example: {dup_ids[:10]}")
                cn_df = _to_numeric(cn_df)
                cn_df = cn_df.groupby(level=0).mean()

            # collapse duplicate gene columns if present
            if cn_df.columns.duplicated().any():
                dup_cols = cn_df.columns[cn_df.columns.duplicated()].unique().tolist()
                print(f"[CNV] duplicate gene columns before TCGA expansion; collapsing by mean. Example: {dup_cols[:20]}")
                cn_df = _to_numeric(cn_df)
                cn_df = cn_df.T.groupby(level=0).mean().T


            # new: TCGA patient-level CNV matrix -> sample-level target index
            if (
                not expr_sym_log.empty
                and cn_df.index.astype(str).str.match(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$", case=False).mean() > 0.8
            ):
                cn_df = expand_tcga_patient_ids_to_sample_index(
                    cn_df,
                    expr_sym_log.index,
                    prefer_sample_type="01",
                    agg="mean",
                )

            cn_df = _maybe_strip_vials(cn_df, args.tcga_vial_strip_mode, "CNV (gene table)")

            if cn_df.index.duplicated().any():
                dup_ids = cn_df.index[cn_df.index.duplicated()].unique().tolist()
                print(f"[CNV] duplicate sample IDs after TCGA expansion; collapsing by mean. Example: {dup_ids[:10]}")
                cn_df = _to_numeric(cn_df)
                cn_df = cn_df.groupby(level=0).mean()

            if not expr_sym_log.empty:
                cn_df = cn_df.reindex(expr_sym_log.index)

            cn_df = _to_numeric(cn_df).fillna(0.0)


            if args.cnv_units == "absolute":
                # only use if your input is true absolute CN with neutral ~2
                cn_gene_df = cn_df - 2.0
                cn_gene_df_for_pga = cn_df - 2.0

            elif args.cnv_units == "ploidy_norm":
                # training-style raw matrix: neutral ~1, convert to 0-centered
                cn_gene_df = cn_df - 1.0
                cn_gene_df_for_pga = cn_df - 1.0

            elif args.cnv_units == "centered":
                # already 0-centered
                cn_gene_df = cn_df.copy()
                cn_gene_df_for_pga = cn_df.copy()

            elif args.cnv_units == "log2":
                # use for direct genes / GSVA only; PGA from this is not comparable
                cn_gene_df = cn_df.copy()
                cn_gene_df_for_pga = pd.DataFrame(index=cn_df.index, columns=cn_df.columns, data=np.nan)

            else:
                raise ValueError(f"Unsupported --cnv_units: {args.cnv_units}")

            cn_sets_all = _read_gmt_many(args.pathways_gmt)
            if args.cnv_selected_pathways_file:
                requested = [x.strip() for x in _read_names_txt(args.cnv_selected_pathways_file)]
                tmp_features = [f"CNVblock__{x}" for x in requested]
                matched_cnv_gmt, cnv_match_tbl = _subset_gmt_by_selected_pathway_features(tmp_features, "CNVblock__", cn_sets_all)
            elif cnv_training_features:
                matched_cnv_gmt, cnv_match_tbl = _subset_gmt_by_selected_pathway_features(cnv_training_features, "CNVblock__", cn_sets_all)
            else:
                matched_cnv_gmt, cnv_match_tbl = {}, pd.DataFrame()

            inferred_cnv_pathways = list(matched_cnv_gmt.keys())
            if not cnv_match_tbl.empty:
                _write_csv(cnv_match_tbl, os.path.join(args.outdir, f"{args.prefix}__cnv_pathway_gmt_matches.csv"))

            if matched_cnv_gmt:
                cn_block = cnv_gsva(cn_gene_df, args.cn_gsva_mode, matched_cnv_gmt).add_prefix("CNVblock__")
                cn_block = _maybe_strip_vials(cn_block, args.tcga_vial_strip_mode, "CNV GSVA")

            if cnv_training_features:
                inferred_cnv_direct_genes = _infer_selected_direct_gene_features_from_training_features(
                    cnv_training_features, "CNVblock__", cn_gene_df.columns
                )
                if inferred_cnv_direct_genes:
                    cnv_gene_subset_block = subset_existing_matrix_by_genes_ci(cn_gene_df, inferred_cnv_direct_genes)
                    cnv_gene_subset_block = _maybe_strip_vials(cnv_gene_subset_block, args.tcga_vial_strip_mode, "CNV gene subset")
                    cnv_gene_subset_block = cnv_gene_subset_block.reindex(cn_gene_df.index).add_prefix("CNVblock__")

            pga_genome = load_pga_table(args.pga_table, expr_sym_log.index if not expr_sym_log.empty else None) if _exists(args.pga_table) else pd.DataFrame()
            arm_pct = pd.DataFrame()

            if not cn_gene_df_for_pga.empty and not cn_gene_df_for_pga.isna().all().all():
                genome_pga_series, arm_pct = _derive_pga_from_cnv_gene_matrix(
                    cn_gene_df=cn_gene_df_for_pga,
                    id_map=id_map,
                    cytoband_file=args.cytoband_file,
                    delta=float(args.arm_from_cnv_delta),
                    min_genes=int(args.arm_from_cnv_min_genes),
                    weight_mode=str(args.arm_from_cnv_weight),
                )

                if pga_genome.empty and not genome_pga_series.empty:
                    pga_genome = genome_pga_series.to_frame()
                    if not expr_sym_log.empty:
                        pga_genome = align_index_smart(pga_genome, expr_sym_log.index, "pga_genome_from_gene")

                if not arm_pct.empty and not expr_sym_log.empty:
                    arm_pct = align_index_smart(arm_pct, expr_sym_log.index, "pga_arm_from_gene")

            pga_parts = [x for x in [pga_genome, arm_pct] if isinstance(x, pd.DataFrame) and not x.empty]
            pga_block = pd.concat(pga_parts, axis=1) if pga_parts else pd.DataFrame()
            pga_block = _maybe_strip_vials(pga_block, args.tcga_vial_strip_mode, "PGA (combined)")

            if args.derive_arm_binary and not arm_pct.empty:
                arm_binary_raw = derive_arm_binary_from_arm_percent(arm_pct, pct_threshold=float(args.arm_binary_pct_threshold))
                arm_binary_block = arm_binary_raw.add_prefix("CNVblock__ARM__")
                arm_binary_block = _maybe_strip_vials(arm_binary_block, args.tcga_vial_strip_mode, "ARM binary")

            if args.derive_band_binary and not cn_gene_df.empty:
                band_binary_raw = derive_band_binary_from_cnv_genes(
                    cn_gene_df=cn_gene_df,
                    id_map=id_map,
                    cytoband_file=args.cytoband_file,
                    delta=float(args.arm_from_cnv_delta),
                    min_genes=int(args.band_from_cnv_min_genes),
                    pct_threshold=float(args.band_binary_pct_threshold),
                )
                if not expr_sym_log.empty:
                    band_binary_raw = align_index_smart(band_binary_raw, expr_sym_log.index, "band_binary_from_gene")
                band_binary_block = band_binary_raw.add_prefix("CNVblock__BAND__")
                band_binary_block = _maybe_strip_vials(band_binary_block, args.tcga_vial_strip_mode, "BAND binary")

    # 8) Clinical
    clinical_block = load_clinical(
        args.clinical_data,
        args.clinical_header_row,
        args.clinical_sample_col,
        expr_sym_log.index if not expr_sym_log.empty else None,
    ) if _exists(args.clinical_data) else pd.DataFrame()
    clinical_block = _maybe_strip_vials(clinical_block, args.tcga_vial_strip_mode, "Clinical")

    # 9) Concatenate
    parts = [
        b for b in [
            gsva_block,
            subset_block,
            med_block,
            custom_gs_block,
            precomputed_rna_gs_block,
            cib_block,
            tide_block,
            estimate_block,
            ecotyper_block,
            mut_block,
            ptmb_block,
            tmb_block,
            cn_block,
            cnv_gene_subset_block,
            pga_block,
            arm_binary_block,
            band_binary_block,
            clinical_block,
        ]
        if isinstance(b, pd.DataFrame) and not b.empty
    ]

    if not expr_sym_log.empty:
        can_idx = expr_sym_log.index
    else:
        idxs = [b.index for b in parts]
        if idxs:
            can_idx = idxs[0]
            for idx in idxs[1:]:
                can_idx = can_idx.union(idx)
            can_idx = pd.Index([str(x) for x in can_idx])
        else:
            can_idx = pd.Index([], name="sample_id")

    parts = [p.reindex(can_idx) for p in parts]
    combined_pre = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=can_idx)
    combined_pre = combined_pre.loc[:, ~combined_pre.columns.duplicated()]

    if args.tcga_vial_strip_mode == "after" and not combined_pre.empty:
        combined_pre = normalize_index_to_tcga_sample_level(combined_pre, how="mean")

    _write_csv(combined_pre, os.path.join(args.outdir, f"{args.prefix}__combined_pre.csv"))

    # 10) Final feature list
    feature_list_path = args.feature_list or training_files["final"]
    if not feature_list_path:
        raise ValueError("Need --feature_list or FINAL_features_only.csv via --training_fs_dir / --training_final_feature_file.")

    feat_list = load_feature_list(feature_list_path)
    combined_final = select_and_order_features_fuzzy(
        combined=combined_pre,
        feature_list=feat_list,
        fill_missing_zero=True,
        mapping_log=args.mapping_log,
    )

    _write_csv(combined_final, os.path.join(args.outdir, f"{args.prefix}__combined_features.csv"))
    _write_csv(combined_final, os.path.join(args.outdir, f"{args.prefix}__selected_ordered_features.csv"))

    manifest = {
        "n_samples": int(combined_final.shape[0]),
        "n_features": int(combined_final.shape[1]),
        "training_fs_dir": args.training_fs_dir,
        "training_feature_sources": training_files,
        "feature_list_path_used": feature_list_path,
        "blocks_included": {
            "rna_gsva": int(gsva_block.shape[1]) if not gsva_block.empty else 0,
            "rna_subset": int(subset_block.shape[1]) if not subset_block.empty else 0,
            "rna_medians": int(med_block.shape[1]) if not med_block.empty else 0,
            "rna_custom_gene_sets": int(custom_gs_block.shape[1]) if not custom_gs_block.empty else 0,
            "rna_precomputed_gene_sets": int(precomputed_rna_gs_block.shape[1]) if not precomputed_rna_gs_block.empty else 0,
            "cibersort_abs": int(cib_block.shape[1]) if not cib_block.empty else 0,
            "tide": int(tide_block.shape[1]) if not tide_block.empty else 0,
            "estimate": int(estimate_block.shape[1]) if not estimate_block.empty else 0,
            "ecotyper": int(ecotyper_block.shape[1]) if not ecotyper_block.empty else 0,
            "mut_pathways": int(mut_block.shape[1]) if not mut_block.empty else 0,
            "ptmb": int(ptmb_block.shape[1]) if not ptmb_block.empty else 0,
            "tmb": int(tmb_block.shape[1]) if not tmb_block.empty else 0,
            "cnv_gsva": int(cn_block.shape[1]) if not cn_block.empty else 0,
            "cnv_direct_genes": int(cnv_gene_subset_block.shape[1]) if not cnv_gene_subset_block.empty else 0,
            "pga_total": int(pga_block.shape[1]) if not pga_block.empty else 0,
            "arm_binary": int(arm_binary_block.shape[1]) if not arm_binary_block.empty else 0,
            "band_binary": int(band_binary_block.shape[1]) if not band_binary_block.empty else 0,
            "clinical": int(clinical_block.shape[1]) if not clinical_block.empty else 0,
        },
        "inferred_pathways": {
            "rna": inferred_rna_pathways,
            "cnv": inferred_cnv_pathways,
            "mut": inferred_mut_pathways,
            "ptmb": inferred_ptmb_pathways,
        },
        "inferred_direct_rna_genes": inferred_rna_direct_genes,
        "inferred_direct_cnv_genes": inferred_cnv_direct_genes,
        "inferred_custom_gs_families": inferred_custom_gs_families,
        "outputs": {
            "combined_pre": f"{args.prefix}__combined_pre.csv",
            "combined_features_final": f"{args.prefix}__combined_features.csv",
            "selected_ordered_features": f"{args.prefix}__selected_ordered_features.csv",
            "mapping_log": args.mapping_log,
        },
    }

    with open(os.path.join(args.outdir, f"{args.prefix}__manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("✅ Done.")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()


