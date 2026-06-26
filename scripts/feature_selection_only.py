#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
feature_selection_only.py

Feature-construction and supervised feature-selection workflow.

This script builds subtype-specific multi-omic candidate feature matrices and applies
the feature-selection workflow used before model training and cross-validation. It can
be run as a standalone feature-selection script or called from the training/CV scripts
through run_feature_selection().

Overall workflow:

1. Load subtype-specific input data and metadata.
2. Deduplicate and align samples across available data blocks.
3. Construct full candidate feature matrices for each feature block.
4. Apply label-free filtering steps, including variance filtering and optional
   protein-coding filtering for gene-level features.
5. Optionally restrict feature selection to a training subset, such as an outer-CV
   training fold, using sample_id_subset.
6. Apply optional label-correlation prefiltering within the selected sample subset.
7. Run supervised association testing between each candidate feature and recurrence
   status or outcome.
8. Combine statistical evidence across enabled tests.
9. Apply redundancy pruning to remove highly correlated selected features.
10. Save selected feature tables and block-specific matrices for downstream model
    training, cross-validation, or final model fitting.

Feature blocks:

* RNAblock__: RNA-derived features, including pathway scores, gene-level expression
  features, immune/TME features, custom gene sets, TIDE, and EcoTyper features when
  available.
* CNVblock__: copy-number-derived features, including arm-level, band-level,
  gene-level, pathway-level, tumor-fraction, and percent-altered features when
  available.
* MUTblock__: mutation-derived features, including binary mutated pathway features,
  TMB/pathogenic features, and optional pathway-TMB features.
* CLINICALblock__: clinical covariates and derived clinical features when available.

Mutation and pTMB feature sources:
Mutation-derived pathway features can be generated from raw mutation/SNV input or
loaded from precomputed sample-by-feature matrices. This allows the same feature
selection workflow to be used when raw variant files cannot be shared. Binary mutated
pathway features and pTMB features are incorporated into the MUTblock before feature
selection, so they pass through the same filtering, testing, and redundancy-pruning
steps as other mutation-derived features.

Filtering:
Variance filtering is performed without using outcome labels and is applied before
supervised testing. It can be run globally or within feature groups to preserve
representation across data types. Some feature families, such as selected immune/TME
or externally derived scores, can bypass variance filtering while still being tested
during supervised selection.

Supervised tests:
Depending on the command-line configuration, feature selection can use:

* Mann-Whitney U tests for recurrent vs non-recurrent group comparisons.
* Fisher exact tests for binary features.
* limma-based results for RNA features when available.
* Univariate Cox models for time-to-event associations.
* Multivariable Cox models adjusted for specified clinical covariates.

Each enabled test uses configurable p-value and FDR thresholds. Cox-based filtering
can also use hazard-ratio magnitude or direction filters when specified.

Subset-aware feature selection:
When sample_id_subset is provided, supervised testing and correlation-based filtering
are performed only within that subset. This is used by the cross-validation scripts to
perform leak-free fold-local feature selection. Label-free full-cohort preprocessing
artifacts may still be reused, but outcome-aware feature selection is restricted to
the training samples for the current fold.

Redundancy pruning:
After candidate features pass the enabled selection criteria, highly correlated
features are pruned using the configured correlation threshold. Block-specific
thresholds can be supplied when different feature families require different
redundancy-pruning behavior.

Outputs:
The script writes both full preprocessing artifacts and subset/fold-specific feature
selection outputs. Typical outputs include:

* preprocessing_full/* candidate matrices and filtering tables
* variance_filter_table_full.csv
* variance_keep_features.txt
* protein_coding_filter_summary.json, when applicable
* ALL_FEATURES_selection_table.csv
* MWU_results.csv, when MWU testing is enabled
* COX_uni_results.csv, when univariate Cox testing is enabled
* COX_multi_feature_results.csv and COX_multi_terms_long.csv, when multivariable
  Cox testing is enabled
* block-specific selected feature matrices:
  RNAblock_selected_post_redundancy.csv
  CNVblock_selected_post_redundancy.csv
  MUTblock_selected_post_redundancy.csv
  CLINICALblock_selected_post_redundancy.csv

Downstream use:
The selected block matrices produced by this script are consumed by the training,
CV, iterated-CV, and final-model scripts. The goal of this script is not to fit
models directly, but to provide reproducible, subset-aware, multi-omic feature
selection outputs for downstream recurrence-risk modeling.
"""


from config import *
from dataloading_utils import *
from dataprocessing_utils import *
from pca_utils import *
from output_utils import *

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests


# -------------------------
# Small helpers
# -------------------------

def _expand_paths(paths):
    out = []
    for p in paths or []:
        out.extend([x for x in str(p).split(",") if x.strip()])
    return out


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def _dump_json_safely(obj, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    try:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=str)
    except Exception as e:
        print(f"[WARN] Could not write JSON to {path}: {e}")

def _save_df(df: pd.DataFrame, path: str):
    d = os.path.dirname(path)
    if d:
        Path(d).mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)

def _slug(s: str) -> str:
    s = re.sub(r"[()]", "", str(s))
    s = re.sub(r"[^0-9A-Za-z]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def _bh_fdr_series(p: pd.Series) -> pd.Series:
    p = pd.Series(p, dtype=float)
    q = pd.Series(np.nan, index=p.index)
    msk = p.notna() & np.isfinite(p.values)
    if msk.any():
        _, qvals, _, _ = multipletests(p[msk].values, method="fdr_bh")
        q.loc[msk] = qvals
    return q

def _add_block_prefix(df, prefix: str):
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
    if hasattr(df, "to_frame"):
        return _add_block_prefix(df.to_frame(), prefix)
    return df

def _group_by_prefix(columns: List[str]) -> Dict[str, List[str]]:
    prefixes = ["RNAblock__", "CNVblock__", "MUTblock__", "CLINICALblock__"]
    groups = {p: [] for p in prefixes}
    other = []
    for c in columns:
        hit = False
        for p in prefixes:
            if c.startswith(p):
                groups[p].append(c); hit = True; break
        if not hit:
            other.append(c)
    if other:
        groups["OTHER__"] = other
    return {k: v for k, v in groups.items() if v}

def _dedup_index(df, how: str = "first", agg: str = "mean"):
    if not isinstance(df, pd.DataFrame):
        return df
    idx = pd.Index(df.index).astype(str).str.strip()
    idx = idx.astype(str).str.replace(r"\.0$", "", regex=True)
    dup_mask = idx.duplicated(keep=False)
    df = df.copy()
    df.index = idx
    if not dup_mask.any():
        return df

    print(f"[DEDUP] df: {dup_mask.sum()} duplicated rows across {idx[dup_mask].nunique()} ids")

    if how in ("first", "last"):
        keep = "first" if how == "first" else "last"
        return df[~df.index.duplicated(keep=keep)]
    elif how == "aggregate":
        num = df.select_dtypes(include="number")
        other = df.drop(columns=num.columns, errors="ignore")
        agg_num = num.groupby(level=0).agg(agg)
        if not other.empty:
            other_nodup = other[~other.index.duplicated(keep="first")]
            return agg_num.join(other_nodup, how="left")
        return agg_num
    else:
        raise ValueError("how must be 'first', 'last', or 'aggregate'")

def _safe_reindex(df, target_index, how_on_dups: str = "first"):
    df = _dedup_index(df, how=how_on_dups)
    tgt = pd.Index(target_index).astype(str).str.strip()
    tgt_unique = tgt[~tgt.duplicated(keep="first")]
    if len(tgt_unique) < len(tgt):
        print(f"[DEDUP] target_index: {len(tgt)} → {len(tgt_unique)} after removing duplicates")
    return df.reindex(tgt_unique)

def _to_num_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out

def _merge_on_feature_no_suffix(left: pd.DataFrame, right: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Prevent pandas from creating _x/_y columns by dropping overlapping cols in left before merge.
    """
    left = left.drop(columns=[c for c in cols if c in left.columns], errors="ignore")
    return left.merge(right[["feature"] + cols], on="feature", how="left")


# -------------------------
# NEW: helpers for TIDE / EcoTyper
# -------------------------

def _normalize_colname(c: str) -> str:
    c = str(c).strip()
    c = re.sub(r"\s+", "_", c)
    c = re.sub(r"[^\w]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c

def _coerce_df_to_numeric_features(
    df: pd.DataFrame,
    onehot_objects: bool = True,
    onehot_max_levels: int = 200,
    normalize_cols: bool = True,
) -> pd.DataFrame:
    """
    Convert a mixed-type DataFrame into numeric-only features.
    - Numeric cols: kept
    - Object/categorical cols: one-hot encoded if enabled and not too many levels
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(index=df.index if isinstance(df, pd.DataFrame) else None)

    out_parts = []
    # numeric-like part
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] > 0:
        out_parts.append(num)

    # object-like part
    if onehot_objects:
        obj_cols = [c for c in df.columns if c not in num.columns]
        for c in obj_cols:
            s = df[c]
            # if it's already numeric-ish but stored as object, try coercion first
            sn = pd.to_numeric(s, errors="coerce")
            if sn.notna().sum() >= max(1, int(0.9 * s.notna().sum())):
                out_parts.append(sn.rename(str(c)))
                continue

            # one-hot if reasonable
            s2 = s.astype(str).where(s.notna(), np.nan)
            # normalize values slightly to reduce accidental duplicates
            s2 = s2.str.strip()
            n_levels = int(pd.Series(s2.dropna().unique()).shape[0])
            if n_levels <= 1:
                # constant/uninformative
                continue
            if n_levels > int(onehot_max_levels):
                print(f"[WARN] Skipping one-hot for column '{c}' (n_levels={n_levels} > {onehot_max_levels}).")
                continue
            d = pd.get_dummies(s2, prefix=str(c), dummy_na=False, drop_first=False)
            out_parts.append(d)

    if not out_parts:
        return pd.DataFrame(index=df.index)

    out = pd.concat(out_parts, axis=1)

    # coerce all to numeric and sanitize
    out = _to_num_df(out)

    # drop all-NaN columns
    out = out.dropna(axis=1, how="all")

    if normalize_cols:
        out.columns = [str(x) for x in out.columns]
        out.columns = [_normalize_colname(c) for c in out.columns]

    return out

def _collect_ecotyper_tables(cleaned_data: dict) -> Dict[str, pd.DataFrame]:
    """
    Collect EcoTyper-related DataFrames from cleaned_data, but ABUNDANCE ONLY.

    Supports either a single 'ecotyper' key or multiple keys starting with 'ecotyper'.
    Explicitly excludes assignment / argmax / class label tables by key name.
    """
    if not cleaned_data:
        return {}

    def _is_assignment_like_key(k: str) -> bool:
        kl = str(k).lower()
        # anything that looks like assignment/argmax/class labels should be excluded
        bad_tokens = ["assign", "assignment", "argmax", "class", "label"]
        return any(t in kl for t in bad_tokens)

    # preferred ordering first (if present) — ABUNDANCE ONLY
    preferred = [
        "ecotyper_cell_states",          # abundance
        "ecotyper_ecotypes",             # abundance
        "ecotyper_cell_states_abundance",
        "ecotyper_ecotypes_abundance",
        "ecotyper",       
        "ecotyper_cell_assign" ,             # if you store a single combined abundance table
        "ecotyper_ecotype_assign" ,              # if you store a single combined abundance table
    ]

    found: Dict[str, pd.DataFrame] = {}
    keys_lower = {str(k).lower(): k for k in cleaned_data.keys()}

    # grab preferred keys first
    for k in preferred:
        if k in keys_lower:
            kk = keys_lower[k]
            if _is_assignment_like_key(kk):
                continue
            df = cleaned_data.get(kk)
            if isinstance(df, pd.DataFrame) and not df.empty:
                found[str(kk)] = df

    # add any other ecotyper* keys not already included (still excluding assignment-like)
    extra_keys = []
    for k, df in cleaned_data.items():
        kl = str(k).lower()
        if not kl.startswith("ecotyper"):
            continue
        if str(k) in found:
            continue
        if _is_assignment_like_key(k):
            continue
        if isinstance(df, pd.DataFrame) and not df.empty:
            extra_keys.append(str(k))

    for k in sorted(extra_keys):
        found[k] = cleaned_data[k]

    return found



def _prepare_ecotyper_block(cleaned_data: dict, ids: pd.Index) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Returns (ABUNDANCE ONLY):
      - ecotyper_full: concatenated (numeric) EcoTyper abundance features under RNAblock__
      - parts: dict[name -> df_part_prefixed] (already RNAblock__ prefixed)

    Note: assignment-like tables are excluded upstream by _collect_ecotyper_tables().
    """
    eco_tables = _collect_ecotyper_tables(cleaned_data)
    if not eco_tables:
        return pd.DataFrame(index=ids), {}

    parts: Dict[str, pd.DataFrame] = {}
    multi = len(eco_tables) > 1

    for name, df0 in eco_tables.items():
        df = _safe_reindex(df0, ids)

        # abundance should be numeric; do NOT one-hot object cols (avoids accidental assignment leakage)
        df = _coerce_df_to_numeric_features(
            df,
            onehot_objects=False,
            onehot_max_levels=0,
            normalize_cols=True
        )

        if df.shape[1] == 0:
            continue

        if multi:
            tag = _normalize_colname(str(name))
            df = df.copy()
            df.columns = [f"{tag}__{c}" for c in df.columns.astype(str)]

        df = _add_block_prefix(df.fillna(0.0), "RNAblock__")
        parts[str(name)] = df

    if not parts:
        return pd.DataFrame(index=ids), {}

    ecotyper_full = pd.concat(list(parts.values()), axis=1)
    ecotyper_full = _safe_reindex(ecotyper_full, ids)
    return ecotyper_full, parts




def _prepare_tide_block(cleaned_data: dict, ids: pd.Index) -> pd.DataFrame:
    tide_full = pd.DataFrame(index=ids)

    if "tide" in cleaned_data and isinstance(cleaned_data["tide"], pd.DataFrame) and not cleaned_data["tide"].empty:
        df = _safe_reindex(cleaned_data["tide"], ids)
        df = _coerce_df_to_numeric_features(
            df,
            onehot_objects=True,
            onehot_max_levels=200,
            normalize_cols=True,
        )

        # explicitly namespace TIDE columns so they cannot collide with genes
        df.columns = [f"TIDE__{str(c).strip()}" for c in df.columns]
        df = df.loc[:, ~df.columns.astype(str).str.lower().str.startswith("unnamed")]

        # collapse duplicate TIDE columns if any
        if df.columns.duplicated().any():
            dups = df.columns[df.columns.duplicated()].unique().tolist()
            print(f"[TIDE] duplicate columns detected after coercion; collapsing by max. Example: {dups[:20]}")
            df = _to_num_df(df).fillna(0.0)
            df = df.T.groupby(level=0).max().T

        tide_full = _add_block_prefix(df.fillna(0.0), "RNAblock__")

    return tide_full



# ----------------------------
# Variance filtering bypass (NEW)
# ----------------------------
# These feature families should NEVER be excluded by variance filtering,
# but they SHOULD still go through correlation prefilter + downstream tests.
VARIANCE_BYPASS_PREFIXES = tuple(p.lower() for p in [
    # CNV
    "CNVblock__TF",              # tumor fraction block
    "CNVblock__TumorFraction",   # (extra safety in case you used this naming)

    # MUT
    "MUTblock__TMB",
    "MUTblock__PATHOGENIC",      # any pathogenic count / derived features

    # RNA
    "RNAblock__HOADLEY",
    "RNAblock__CUSTOM",
    "RNAblock__CIBERSORT",
    "RNAblock__GS",
    "RNAblock__estimate",

    # Future additions (you said they should bypass variance too)
    "RNAblock__TIDE",
    "RNAblock__ECOTYPER",
])

def is_variance_bypassed(feature_name: str) -> bool:
    s = str(feature_name).strip().lower()
    return any(s.startswith(p) for p in VARIANCE_BYPASS_PREFIXES)



# -------------------------
# Pathway ID canonicalization (for MUTblock/GMT overlap)
# -------------------------

_CANON_TOKENS = {
    "rnablock", "cnvblock", "mutblock", "clinicalblock",
    "rna", "cnv", "mut", "ptmb", "ptmbblock", "ptmbscore", "ptmbscores",
    "tmb", "gsva", "pathways", "mutations", "mutation", "signature", "signatures",
    "cibersort", "hoadley", "custom", "de", "genes", "gene",
}

def clean_pathway_id(pid: str) -> str:
    """
    Canonicalize a pathway identifier so datatypes/prefixes don't break overlaps.

    Examples handled:
      - MUTblock__HALLMARK_PI3K_AKT...
      - MUTblock__MUT__HALLMARK_PI3K_AKT...
      - RNAblock__PATHWAYS__HALLMARK_PI3K_AKT...
      - CNVblock__GSVA__HALLMARK_PI3K_AKT...
      - X prefixes / numeric prefixes before first letter
    """
    if pid is None:
        return ""
    s = str(pid).strip()
    if not s or s.lower() in {"nan", "none"}:
        return ""

    # drop leading junk until first letter
    m = re.search(r"[A-Za-z]", s)
    if m:
        s = s[m.start():]

    # peel off repeated known "block__" prefixes
    for _ in range(6):
        s2 = re.sub(r"^(?:RNAblock__|CNVblock__|MUTblock__|CLINICALblock__)", "", s, flags=re.IGNORECASE)
        if s2 == s:
            break
        s = s2

    # peel off repeated leading tokens like MUT__/RNA__/PATHWAYS__/GSVA__/etc.
    for _ in range(8):
        tmp = s.lstrip("_- .")
        if "__" in tmp:
            head, tail = tmp.split("__", 1)
            if head.strip().lower() in _CANON_TOKENS:
                s = tail
                continue
        m2 = re.match(r"^([A-Za-z0-9]+)[_\-\.]+(.+)$", tmp)
        if m2 and m2.group(1).strip().lower() in _CANON_TOKENS:
            s = m2.group(2)
            continue
        break

    return s.strip().lstrip("_- .")

def canonical_pathway_slug(pid: str) -> str:
    return _slug(clean_pathway_id(pid))


# -------------------------
# GMT loader (PTMB gene sets)
# -------------------------

def load_gene_sets_from_gmt(gmt_path: str) -> Dict[str, List[str]]:
    """
    GMT format: pathway<TAB>description<TAB>gene1<TAB>gene2...
    Returns dict[pathway_name] -> list of genes (dedup preserving order).
    """
    if not gmt_path or not os.path.exists(gmt_path):
        raise FileNotFoundError(f"[PTMB] GMT not found: {gmt_path}")

    gene_sets: Dict[str, List[str]] = {}
    with open(gmt_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln.strip():
                continue
            parts = ln.split("\t")
            if len(parts) < 3:
                continue
            name = str(parts[0]).strip()
            genes = [g.strip() for g in parts[2:] if str(g).strip()]
            if not name or not genes:
                continue
            seen = set()
            uniq = []
            for g in genes:
                if g in seen:
                    continue
                seen.add(g)
                uniq.append(g)
            if name not in gene_sets:
                gene_sets[name] = uniq

    if not gene_sets:
        raise ValueError(f"[PTMB] No gene sets parsed from GMT: {gmt_path}")
    print(f"[PTMB] Loaded GMT gene sets: {gmt_path} (n_pathways={len(gene_sets)})")
    return gene_sets


# -------------------------
# PTMB helpers (MAF + GTF CDS lengths)
# -------------------------

def _read_table_auto(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return pd.read_csv(path, sep="\t")

def _clean_sample_id_for_merge(x) -> str:
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    if s.startswith("X") and s[1:].isdigit():
        s = s[1:]
    return s

def _clean_gene_symbol(x) -> str:
    s = str(x).strip()
    if not s:
        return ""
    s = s.split()[0]
    s = re.sub(r"\.0$", "", s)
    return s.upper()

def _gtf_get_attr(attr: str, key: str) -> Optional[str]:
    m = re.search(rf'{key}\s+"([^"]+)"', str(attr))
    return m.group(1) if m else None

def load_cds_lengths_from_gtf(
    gtf_path: str,
    cache_csv: Optional[str] = None,
    prefer_gene_name: bool = True,
) -> Dict[str, float]:
    """
    Build CDS length per gene (union of CDS intervals across transcripts).
    Returns dict: GENE_SYMBOL_UPPER -> cds_len_bp (float).

    If cache_csv is provided and exists, loads from cache.
    """
    if cache_csv and os.path.exists(cache_csv):
        df = pd.read_csv(cache_csv)
        if "gene" in df.columns and ("cds_len_bp" in df.columns or "cds_len" in df.columns):
            len_col = "cds_len_bp" if "cds_len_bp" in df.columns else "cds_len"
            lens = pd.to_numeric(df[len_col], errors="coerce")
            out = {
                _clean_gene_symbol(g): float(l)
                for g, l in zip(df["gene"].astype(str), lens)
                if str(g).strip() and pd.notna(l) and np.isfinite(l) and float(l) > 0
            }
            print(f"[PTMB] Loaded CDS lengths cache: {cache_csv} (n={len(out)})")
            return out

    if not gtf_path or not os.path.exists(gtf_path):
        raise FileNotFoundError(f"GTF not found: {gtf_path}")

    intervals: Dict[str, List[Tuple[int, int]]] = {}
    with open(gtf_path, "r", encoding="utf-8", errors="ignore") as f:
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
            gene_name = _gtf_get_attr(attr, "gene_name")
            gene_id = _gtf_get_attr(attr, "gene_id")
            gene = gene_name if (prefer_gene_name and gene_name) else (gene_id or gene_name)
            if not gene:
                continue
            g = _clean_gene_symbol(gene)
            if not g:
                continue
            intervals.setdefault(g, []).append((start, end))

    cds_len: Dict[str, float] = {}
    for g, ivals in intervals.items():
        if not ivals:
            continue
        ivals_sorted = sorted(ivals, key=lambda t: (t[0], t[1]))
        cur_s, cur_e = ivals_sorted[0]
        merged = []
        for s, e in ivals_sorted[1:]:
            if s <= cur_e + 1:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        length = sum((e - s + 1) for s, e in merged)
        cds_len[g] = float(length)

    if cache_csv:
        d = os.path.dirname(cache_csv)
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"gene": list(cds_len.keys()), "cds_len_bp": list(cds_len.values())}).to_csv(cache_csv, index=False)
        print(f"[PTMB] Wrote CDS lengths cache: {cache_csv} (n={len(cds_len)})")

    return cds_len

def _infer_maf_cols(df: pd.DataFrame, user_sample_col: Optional[str], user_gene_col: Optional[str]) -> Tuple[str, str]:
    cols = list(df.columns.astype(str))
    low = {c.lower(): c for c in cols}

    sample_col = None
    gene_col = None

    if user_sample_col and user_sample_col in df.columns:
        sample_col = user_sample_col
    if user_gene_col and user_gene_col in df.columns:
        gene_col = user_gene_col

    if sample_col is None:
        for cand in ["tumor_sample_barcode", "sample", "sample_id", "patient_id", "case_id"]:
            if cand in low:
                sample_col = low[cand]; break

    if gene_col is None:
        for cand in ["hugo_symbol", "gene", "gene_symbol", "symbol"]:
            if cand in low:
                gene_col = low[cand]; break

    if sample_col is None or gene_col is None:
        raise KeyError(
            f"[PTMB] Could not infer required MAF columns. "
            f"Found columns: {cols[:20]}... Provide --ptmb_maf_sample_col and --ptmb_maf_gene_col."
        )
    return sample_col, gene_col

def load_gene_sets_from_gmt_many(gmt_paths: List[str]) -> Dict[str, List[str]]:
    """
    Load multiple GMTs and UNION genes for pathways with the same name.
    Preserves gene order within each file; new genes append at the end.
    """
    merged: Dict[str, List[str]] = {}
    for gmt in (gmt_paths or []):
        gs = load_gene_sets_from_gmt(gmt)
        for pw, genes in gs.items():
            if pw not in merged:
                merged[pw] = list(genes)
            else:
                seen = set(merged[pw])
                for g in genes:
                    if g not in seen:
                        merged[pw].append(g)
                        seen.add(g)
    return merged

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
    """
    Returns:
      ptmb_df: index = sample_ids_target, columns = MUTblock__pTMB__<PATHWAY>
      ptmb_info: per-pathway table (sizes, Ni used, etc.)
    """
    if not maf_path or not os.path.exists(maf_path):
        raise FileNotFoundError(f"[PTMB] maf_path not found: {maf_path}")

    if not gene_sets:
        raise ValueError("[PTMB] gene_sets is empty (nothing to compute).")

    denom_policy = str(denom_policy).strip().lower()
    if denom_policy not in {"all_genes_with_len", "all_genes", "mutated_genes_with_len"}:
        raise ValueError("[PTMB] denom_policy must be one of: all_genes_with_len, all_genes, mutated_genes_with_len")

    maf = _read_table_auto(maf_path)
    sample_col, gene_col = _infer_maf_cols(maf, maf_sample_col, maf_gene_col)

    maf = maf[[sample_col, gene_col]].dropna()
    maf["_sample"] = maf[sample_col].map(_clean_sample_id_for_merge)
    maf["_gene"] = maf[gene_col].map(_clean_gene_symbol)

    maf = maf[(maf["_sample"].astype(str) != "") & (maf["_gene"].astype(str) != "")]
    target_set = set(pd.Index(sample_ids_target).astype(str))
    maf = maf[maf["_sample"].isin(target_set)]

    cnt = maf.groupby(["_sample", "_gene"]).size().rename("mut_count").reset_index()

    cnt["cds_len_bp"] = cnt["_gene"].map(cds_len_bp)
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
            if g in cds_len_bp and np.isfinite(cds_len_bp[g]) and float(cds_len_bp[g]) > 0
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
    ptmb_info = pd.DataFrame(info_rows).sort_values(
        ["n_genes_with_len", "Ni_used"], ascending=[False, False]
    ).reset_index(drop=True)
    return ptmb_df, ptmb_info


# -------------------------
# Precomputed MUT-feature matrix helpers
# -------------------------

def load_precomputed_feature_matrix(
    path: str,
    sample_ids_target: pd.Index,
    sample_col: Optional[str] = None,
    prefix: str = "",
    expected_binary: bool = False,
    transform: str = "none",
    table_name: str = "FEATURE_MATRIX",
) -> pd.DataFrame:
    """
    Load a precomputed sample x feature matrix and align it to the current cohort.

    Expected input:
      - rows = samples
      - columns = already-computed features
      - sample IDs either in `sample_col`, a common sample-id column, or the first/index column

    This is used for GitHub/data-sharing-safe mutation-derived features:
      - imported binary pathway mutation scores
      - imported precomputed pTMB scores

    It intentionally does NOT use raw SNV/MAF rows.
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"[{table_name}] File not found: {path}")

    df = _read_table_auto(path)
    if df is None or df.empty:
        raise ValueError(f"[{table_name}] Empty matrix: {path}")

    df.columns = df.columns.astype(str).str.strip()

    # Choose sample-id column.
    chosen_sample_col = None
    if sample_col:
        if sample_col not in df.columns:
            raise ValueError(
                f"[{table_name}] sample_col={sample_col!r} not found. "
                f"Columns include: {list(df.columns[:20])}"
            )
        chosen_sample_col = sample_col
    else:
        low = {str(c).strip().lower(): c for c in df.columns}
        for cand in ["sample_id", "sample", "patient_id", "case_id", "tumor_sample_barcode", "id"]:
            if cand in low:
                chosen_sample_col = low[cand]
                break

        # Common pandas index export: first column is unnamed.
        if chosen_sample_col is None and len(df.columns) > 0:
            first = str(df.columns[0]).strip()
            if first.lower().startswith("unnamed"):
                chosen_sample_col = df.columns[0]

        # Fallback: first column if it overlaps target sample IDs.
        if chosen_sample_col is None and len(df.columns) > 0:
            first_vals = df.iloc[:, 0].map(_clean_sample_id_for_merge)
            target_set = set(pd.Index(sample_ids_target).astype(str).str.strip())
            overlap = int(first_vals.isin(target_set).sum())
            if overlap > 0:
                chosen_sample_col = df.columns[0]

    if chosen_sample_col is None:
        raise ValueError(
            f"[{table_name}] Could not determine sample ID column. "
            "Pass --mut_binary_sample_col or --ptmb_sample_col."
        )

    idx = df[chosen_sample_col].map(_clean_sample_id_for_merge)
    df = df.drop(columns=[chosen_sample_col])
    df.index = idx.astype(str).str.strip()

    # Remove accidental unnamed columns left over from CSV exports.
    df = df.loc[:, ~df.columns.astype(str).str.lower().str.startswith("unnamed")]

    # Drop duplicate feature columns by averaging numeric values.
    if df.columns.duplicated().any():
        dups = df.columns[df.columns.duplicated()].unique().tolist()
        print(f"[{table_name}] duplicate feature columns detected; collapsing by mean. Example: {dups[:10]}")
        df = _to_num_df(df).T.groupby(level=0).mean().T

    # Deduplicate sample rows and align to current cohort.
    df = _dedup_index(df, how="aggregate", agg="mean")
    target = pd.Index(sample_ids_target).astype(str).str.strip()
    df = df.reindex(target)

    df = _to_num_df(df)

    # Optional transforms.
    transform = str(transform or "none").strip().lower()
    if expected_binary:
        # For binary pathway scores: anything > 0 is treated as 1.
        df = df.fillna(0.0)
        df = (df > 0).astype(float)
        transform = "none"
    elif transform == "log10":
        df = df.fillna(0.0).clip(lower=0.0)
        df = np.log10(df + 1e-12)
    elif transform == "log1p":
        df = df.fillna(0.0).clip(lower=0.0)
        df = np.log1p(df)
    elif transform == "none":
        df = df.fillna(0.0)
    else:
        raise ValueError(f"[{table_name}] Unsupported transform={transform!r}; use none, log10, or log1p.")

    # Add feature prefix unless already present.
    if prefix:
        df.columns = [
            str(c) if str(c).startswith(prefix) else f"{prefix}{_slug(c)}"
            for c in df.columns.astype(str)
        ]
    else:
        df.columns = [_slug(c) for c in df.columns.astype(str)]

    # Final duplicate cleanup after slugging/prefixing.
    if df.columns.duplicated().any():
        dups = df.columns[df.columns.duplicated()].unique().tolist()
        print(f"[{table_name}] duplicate columns after prefix/slug; collapsing by mean. Example: {dups[:10]}")
        df = _to_num_df(df).T.groupby(level=0).mean().T

    print(
        f"[{table_name}] Imported precomputed matrix: "
        f"samples={df.shape[0]}, features={df.shape[1]}, source={path}"
    )
    return df


# -------------------------
# Protein-coding GTF helpers
# -------------------------

def load_protein_coding_genes_from_gtf(gtf_path: str) -> set[str]:
    """
    Streaming parse: returns gene_name where gene_type/gene_biotype == protein_coding.
    Works with GENCODE/Ensembl GTFs.
    """
    if not gtf_path or not os.path.exists(gtf_path):
        raise FileNotFoundError(f"GTF not found: {gtf_path}")

    pc = set()
    with open(gtf_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if not ln or ln.startswith("#"):
                continue
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            feature = parts[2]
            if feature != "gene":
                continue
            attr = parts[8]
            gene_name = _gtf_get_attr(attr, "gene_name")
            gene_type = _gtf_get_attr(attr, "gene_type")
            if gene_type is None:
                gene_type = _gtf_get_attr(attr, "gene_biotype")
            if gene_name and gene_type == "protein_coding":
                pc.add(str(gene_name))
    return pc

def filter_gene_level_block_to_protein_coding(
    df: pd.DataFrame,
    prefix: str,
    pc_genes_upper: set[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Filters ONLY columns starting with `prefix` by checking the gene symbol after prefix.
    Returns (filtered_df, removed_cols).
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df, []

    keep = []
    removed = []
    for c in df.columns.astype(str):
        if not c.startswith(prefix):
            keep.append(c)
            continue
        gene = c[len(prefix):]
        if str(gene).upper() in pc_genes_upper:
            keep.append(c)
        else:
            removed.append(c)

    return df.loc[:, keep], removed


# -------------------------
# Robust Cox coercions
# -------------------------

_BIN_MAP = {
    "0": 0, "1": 1,
    "false": 0, "true": 1,
    "no": 0, "yes": 1,
    "never": 0, "ever": 1,
    "non-recurrent": 0, "nonrecurrent": 0,
    "recurrent": 1,
    "no chemo": 0,
    "none": 0,
}

def coerce_binary_like(series: pd.Series, case_label: str = "recurrent", ctrl_label: str = "non-recurrent") -> pd.Series:
    """
    Coerce an event/label-like series to 0/1 if possible.
    Priority:
      1) numeric 0/1
      2) mapping of common strings + provided case/ctrl labels
    """
    if series is None:
        return pd.Series(dtype=float)

    s = series.copy()
    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().sum() > 0:
        vals = set(pd.Series(sn.dropna().unique()).astype(float).tolist())
        if vals.issubset({0.0, 1.0}):
            return sn.astype(float)

    sl = s.astype(str).str.strip().str.lower()
    mapping = dict(_BIN_MAP)
    mapping[str(ctrl_label).strip().lower()] = 0
    mapping[str(case_label).strip().lower()] = 1
    mapped = sl.map(mapping)
    return pd.to_numeric(mapped, errors="coerce").astype(float)

def coerce_covariates_to_numeric(
    meta: pd.DataFrame,
    covariates: List[str],
) -> pd.DataFrame:
    """
    Convert covariates to numeric robustly:
      - Try numeric
      - Try binary map (yes/no, ever/never, no chemo, etc.)
      - Otherwise one-hot encode (drop_first=True)
    """
    if meta is None or meta.empty or not covariates:
        return pd.DataFrame(index=meta.index if isinstance(meta, pd.DataFrame) else None)

    out = pd.DataFrame(index=meta.index)
    for c in covariates:
        if c not in meta.columns:
            continue

        s = meta[c]
        sn = pd.to_numeric(s, errors="coerce")
        nonmissing = s.notna().sum()

        if nonmissing > 0 and sn.notna().sum() < max(1, int(0.2 * nonmissing)):
            sl = s.astype(str).str.strip().str.lower()
            mapped = sl.map(_BIN_MAP)
            if mapped.notna().sum() >= max(1, int(0.8 * nonmissing)):
                sn = mapped.astype(float)

        if nonmissing > 0 and sn.notna().sum() < max(1, int(0.8 * nonmissing)):
            sl = s.astype(str).str.strip()
            sl = sl.replace({"": np.nan, "nan": np.nan, "None": np.nan, "none": np.nan})
            d = pd.get_dummies(sl, prefix=c, dummy_na=False, drop_first=True)
            if d.shape[1] == 0:
                out[c] = sn
            else:
                out = out.join(d)
        else:
            out[c] = sn

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


# -------------------------
# Dedup/align
# -------------------------


def dedup_and_align_data(
    data: dict,
    status_col: str = "Disease_Status",
    prep_dir: str = None,
    raise_on_conflict: bool = True,
):
    if "meta" not in data or not hasattr(data["meta"], "copy"):
        raise KeyError("data['meta'] must be a pandas DataFrame.")

    binary_like_keys = {
        "arm_binary",
        "band_binary",
        "ecotyper_cell_assign",
        "ecotyper_ecotype_assign",
    }

    numeric_force_mean_keys = {
        "estimate",
        "tide",
        "ecotyper_cell_states",
        "ecotyper_ecotypes",
    }

    cleaned_data = {}
    for k, df in list(data.items()):
        if hasattr(df, "copy"):
            df = df.copy()
            if hasattr(df, "index"):
                df.index = (
                    pd.Index(df.index)
                    .astype(str)
                    .str.strip()
                    .str.replace(r"\.0$", "", regex=True)
                )
            cleaned_data[k] = df
        else:
            cleaned_data[k] = df

    meta = cleaned_data["meta"].rename(columns={"Disease Status": "Disease_Status"}).copy()
    if status_col not in meta.columns:
        raise KeyError(f"Target column {status_col!r} not in meta (columns: {list(meta.columns)})")

    meta.index = (
        pd.Index(meta.index)
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    vc = meta.index.value_counts()
    dup_ids = vc[vc > 1].index.tolist()

    audit_rows = []
    if dup_ids:
        for sid in dup_ids:
            rows = meta.loc[[sid]]
            labels = rows[status_col].astype(str).str.strip().str.lower().tolist()
            labset = sorted(set(labels))
            audit_rows.append({
                "sample_id": sid,
                "n_rows_meta": int(vc.loc[sid]),
                "labels_found": ",".join(labset),
            })
            if len(labset) > 1 and raise_on_conflict:
                raise ValueError(
                    f"[DEDUP] Conflicting labels for sample '{sid}' in meta: {labset}. "
                    "Fix or set raise_on_conflict=False to keep first occurrence."
                )
        meta = meta[~meta.index.duplicated(keep="first")].copy()
        print(f"[DEDUP] Collapsed {len(dup_ids)} duplicate sample_id(s) in meta.")

    dup_meta_audit_df = pd.DataFrame(audit_rows)
    if prep_dir:
        ensure_dir(prep_dir)
        if not dup_meta_audit_df.empty:
            dup_meta_audit_df.to_csv(os.path.join(prep_dir, "duplicates_meta_audit.csv"), index=False)

    def _collapse_by_index_explicit(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
        """
        Force duplicate collapse by grouping on an explicit temporary index column.
        This is more robust than relying on grouped index behavior alone.
        """
        tmp = df.copy()
        tmp["__tmp_idx__"] = (
            pd.Index(tmp.index)
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )

        if how == "max":
            tmp = tmp.drop(columns=[c for c in ["__tmp_idx__"] if c not in tmp.columns], errors="ignore")
            tmp["__tmp_idx__"] = (
                pd.Index(df.index)
                .astype(str)
                .str.strip()
                .str.replace(r"\.0$", "", regex=True)
            )

        if how == "binary_max":
            feat_cols = [c for c in tmp.columns if c != "__tmp_idx__"]
            tmp[feat_cols] = tmp[feat_cols].apply(lambda s: pd.to_numeric(s, errors="coerce")).fillna(0)
            tmp[feat_cols] = (tmp[feat_cols] > 0).astype(float)
            out = tmp.groupby("__tmp_idx__", as_index=True)[feat_cols].max()
            out.index.name = None
            return out

        if how == "numeric_mean":
            feat_cols = [c for c in tmp.columns if c != "__tmp_idx__"]
            tmp[feat_cols] = tmp[feat_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
            out = tmp.groupby("__tmp_idx__", as_index=True)[feat_cols].mean()
            out.index.name = None
            return out

        # mixed aggregation
        feat_cols = [c for c in tmp.columns if c != "__tmp_idx__"]
        num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(tmp[c])]
        non_num_cols = [c for c in feat_cols if c not in num_cols]

        parts = []
        if num_cols:
            parts.append(tmp.groupby("__tmp_idx__", as_index=True)[num_cols].mean())
        if non_num_cols:
            parts.append(tmp.groupby("__tmp_idx__", as_index=True)[non_num_cols].first())

        if not parts:
            out = pd.DataFrame(index=pd.Index([], dtype=object))
        elif len(parts) == 1:
            out = parts[0]
        else:
            out = pd.concat(parts, axis=1)

        out.index.name = None
        return out

    def _dedup_df_by_index(df: pd.DataFrame, key_name: str) -> pd.DataFrame:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return df

        df2 = df.copy()
        df2.index = (
            pd.Index(df2.index)
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )

        if not df2.index.duplicated().any():
            return df2

        if key_name in binary_like_keys:
            return _collapse_by_index_explicit(df2, how="binary_max")

        if key_name in numeric_force_mean_keys:
            return _collapse_by_index_explicit(df2, how="numeric_mean")

        num_cols = df2.select_dtypes(include=[np.number]).columns
        if len(num_cols) == df2.shape[1]:
            return _collapse_by_index_explicit(df2, how="numeric_mean")

        return _collapse_by_index_explicit(df2, how="mixed")

    for k, df in list(cleaned_data.items()):
        if k == "meta" or not isinstance(df, pd.DataFrame):
            continue

        before_n = df.shape[0]
        df2 = _dedup_df_by_index(df, k)
        after_n = df2.shape[0]

        if after_n < before_n:
            if k in binary_like_keys:
                print(f"[DEDUP:{k}] Collapsed {before_n - after_n} duplicate row(s) by index (max for binary).")
            elif k in numeric_force_mean_keys:
                print(f"[DEDUP:{k}] Collapsed {before_n - after_n} duplicate row(s) by index (mean for continuous).")
            else:
                print(f"[DEDUP:{k}] Collapsed {before_n - after_n} duplicate row(s) by index (aggregated).")

        # hard safety check
        if pd.Index(df2.index).duplicated().any():
            dup_left = pd.Index(df2.index)[pd.Index(df2.index).duplicated()].unique().tolist()
            raise ValueError(
                f"[DEDUP:{k}] duplicate labels remain after deduplication. "
                f"Example duplicate IDs: {dup_left[:10]}"
            )

        df2 = df2.reindex(meta.index)
        cleaned_data[k] = df2

    y_map = (
        meta[status_col]
        .astype(str).str.strip().str.lower()
        .map({
            "non-recurrent": 0,
            "nonrecurrent": 0,
            "0": 0,
            "recurrent": 1,
            "1": 1,
        })
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

# -------------------------
# Variance filtering (label-free)
# -------------------------

def compute_variance_filter_table(
    X: pd.DataFrame,
    var_min: float = 0.0,
    min_nonmissing: int = 10,
) -> pd.DataFrame:
    if X is None or not isinstance(X, pd.DataFrame) or X.shape[1] == 0:
        return pd.DataFrame(columns=["feature", "variance", "n_nonmissing", "passed_variance"])

    Xn = _to_num_df(X)
    n_nonmissing = Xn.notna().sum(axis=0).astype(int)
    var = Xn.var(axis=0, ddof=1, skipna=True)

    tbl = pd.DataFrame({
        "feature": var.index.astype(str),
        "variance": var.values.astype(float),
        "n_nonmissing": n_nonmissing.values.astype(int),
    })
    passed = np.isfinite(tbl["variance"].values) & (tbl["n_nonmissing"].values >= int(min_nonmissing)) & (tbl["variance"].values >= float(var_min))
    tbl["passed_variance"] = passed.astype(bool)
    return tbl.sort_values(["passed_variance", "variance"], ascending=[False, False]).reset_index(drop=True)

def compute_variance_filter_table_top_frac(
    X: pd.DataFrame,
    keep_top_frac: float,
    var_min: float = 0.0,
    min_nonmissing: int = 10,
) -> Tuple[pd.DataFrame, float]:
    if X is None or not isinstance(X, pd.DataFrame) or X.shape[1] == 0:
        return pd.DataFrame(columns=["feature", "variance", "n_nonmissing", "passed_variance"]), np.nan

    keep_top_frac = float(keep_top_frac)
    if not np.isfinite(keep_top_frac):
        keep_top_frac = 1.0
    keep_top_frac = max(0.0, min(1.0, keep_top_frac))

    Xn = _to_num_df(X)
    n_nonmissing = Xn.notna().sum(axis=0).astype(int)
    var = Xn.var(axis=0, ddof=1, skipna=True)

    tbl = pd.DataFrame({
        "feature": var.index.astype(str),
        "variance": var.values.astype(float),
        "n_nonmissing": n_nonmissing.values.astype(int),
    })

    base_ok = np.isfinite(tbl["variance"].values) & (tbl["n_nonmissing"].values >= int(min_nonmissing)) & (tbl["variance"].values >= float(var_min))

    eligible_var = tbl.loc[base_ok, "variance"].values.astype(float)
    if eligible_var.size == 0:
        thr = np.inf
        passed = np.zeros(len(tbl), dtype=bool)
    elif keep_top_frac >= 1.0:
        thr = float(np.nanmin(eligible_var)) if eligible_var.size else np.nan
        passed = base_ok
    elif keep_top_frac <= 0.0:
        thr = np.inf
        passed = np.zeros(len(tbl), dtype=bool)
    else:
        q = 1.0 - keep_top_frac
        thr = float(np.nanquantile(eligible_var, q))
        passed = base_ok & (tbl["variance"].values >= thr)

    tbl["passed_variance"] = passed.astype(bool)
    return tbl.sort_values(["passed_variance", "variance"], ascending=[False, False]).reset_index(drop=True), float(thr)

def compute_variance_filter_tables_by_group(
    X: pd.DataFrame,
    groups: Dict[str, List[str]],
    keep_top_frac: float,
    var_min: float = 0.0,
    min_nonmissing: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, float], set]:
    tables = []
    thresholds = {}
    keep = set()

    for gname, cols in (groups or {}).items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        Xt = X[cols]
        tbl, thr = compute_variance_filter_table_top_frac(
            Xt,
            keep_top_frac=keep_top_frac,
            var_min=var_min,
            min_nonmissing=min_nonmissing,
        )
        tbl["variance_group"] = gname
        tbl["variance_threshold"] = thr
        thresholds[gname] = thr
        tables.append(tbl)
        keep.update(tbl.loc[tbl["passed_variance"], "feature"].astype(str).tolist())

    combined = pd.concat(tables, axis=0, ignore_index=True) if tables else pd.DataFrame()
    return combined, thresholds, keep


# -------------------------
# Correlation prefilter (label-based)
# -------------------------

def corr_prefilter_table(
    X: pd.DataFrame,
    y_str: pd.Series,
    case_label: str = "recurrent",
    ctrl_label: str = "non-recurrent",
    abs_min: Optional[float] = None,
    # topk: Optional[int] = None,
    min_n: int = 10,
) -> pd.DataFrame:
    ys = y_str.astype(str).str.strip().str.lower()
    case = str(case_label).strip().lower()
    ctrl = str(ctrl_label).strip().lower()
    ybin = ys.map({ctrl: 0, case: 1})

    if ybin.isna().any():
        bad = sorted(set(ys[ybin.isna()].tolist()))
        raise ValueError(f"[CORR] Unrecognized labels for corr prefilter: {bad}")

    Xnum = _to_num_df(X)
    n_valid = Xnum.notna().sum(axis=0).astype(int)
    r = Xnum.corrwith(ybin)

    tbl = pd.DataFrame({
        "feature": r.index.astype(str),
        "corr_r": r.values.astype(float),
        "corr_abs": np.abs(r.values.astype(float)),
        "corr_n": n_valid.values.astype(int),
    })

    passed = np.isfinite(tbl["corr_abs"].values) & (tbl["corr_n"].values >= int(min_n))
    if abs_min is not None:
        passed &= (tbl["corr_abs"].values >= float(abs_min))

    # if topk is not None:
    #     k = int(topk)
    #     if k > 0:
    #         eligible = np.isfinite(tbl["corr_abs"].values) & (tbl["corr_n"].values >= int(min_n))
    #         scores = tbl["corr_abs"].values.copy()
    #         scores[~eligible] = -np.inf
    #         order = np.argsort(-scores)
    #         keep_idx = set(order[:min(k, int(eligible.sum()))].tolist())
    #         topk_mask = np.array([i in keep_idx for i in range(len(tbl))], dtype=bool)
    #         passed &= topk_mask

    tbl["passed_corr"] = passed.astype(bool)
    return tbl.sort_values(["passed_corr", "corr_abs"], ascending=[False, False]).reset_index(drop=True)


# -------------------------
# Tests
# -------------------------


def mwu_table(
    X: pd.DataFrame,
    y_str: pd.Series,
    case_label: str = "recurrent",
    ctrl_label: str = "non-recurrent",
) -> pd.DataFrame:
    Xn = _to_num_df(X)

    # Align y to X rows (avoids boolean mask misalignment after any reindex/dedup)
    ys = y_str.reindex(Xn.index).astype(str).str.strip().str.lower()
    case = str(case_label).strip().lower()
    ctrl = str(ctrl_label).strip().lower()

    case_mask = (ys == case)
    ctrl_mask = (ys == ctrl)
    if case_mask.sum() == 0 or ctrl_mask.sum() == 0:
        raise ValueError("Case/control groups missing for MWU.")

    # ✅ CRASH FIX: handle duplicate feature names (duplicate columns)
    if isinstance(Xn, pd.DataFrame) and Xn.columns.duplicated().any():
        dups = Xn.columns[Xn.columns.duplicated()].unique().tolist()
        print(f"[MWU] WARNING: {len(dups)} duplicate feature names in X; collapsing duplicates by mean. Example: {dups[:10]}")
        Xn = Xn.T.groupby(level=0).mean().T  # collapse duplicates -> unique columns

    rows = []
    for c in Xn.columns:
        a = Xn.loc[case_mask, c].dropna().to_numpy()
        b = Xn.loc[ctrl_mask, c].dropna().to_numpy()

        # force 1D (extra safety)
        a = np.asarray(a, dtype=float).ravel()
        b = np.asarray(b, dtype=float).ravel()

        if a.size == 0 or b.size == 0:
            rows.append({
                "feature": str(c),
                "mwu_p": np.nan,
                "mwu_case_median": np.nan,
                "mwu_ctrl_median": np.nan,
                "mwu_delta_median": np.nan
            })
            continue

        try:
            _, p = mannwhitneyu(a, b, alternative="two-sided")
            p_arr = np.asarray(p)
            p_val = float(p_arr) if p_arr.size == 1 else float(np.nanmin(p_arr))
        except Exception:
            p_val = np.nan

        m_case = float(np.nanmedian(a)) if a.size else np.nan
        m_ctrl = float(np.nanmedian(b)) if b.size else np.nan

        rows.append({
            "feature": str(c),
            "mwu_p": p_val if np.isfinite(p_val) else np.nan,
            "mwu_case_median": m_case,
            "mwu_ctrl_median": m_ctrl,
            "mwu_delta_median": (m_case - m_ctrl) if np.isfinite(m_case) and np.isfinite(m_ctrl) else np.nan
        })

    df = pd.DataFrame(rows)
    df["mwu_q"] = _bh_fdr_series(df["mwu_p"])
    return df.sort_values("mwu_p", na_position="last").reset_index(drop=True)


def cox_univariate_table(
    X: pd.DataFrame,
    meta: pd.DataFrame,
    time_col: str,
    event_col: str,
    min_n: int = 10,
    min_events: int = 1,
    case_label: str = "recurrent",
    ctrl_label: str = "non-recurrent",
) -> pd.DataFrame:
    """
    Univariate CoxPH per feature.

    CRASH FIXES:
      - Collapses duplicate feature-name columns in X (mean across duplicates).
      - Aligns meta to X index (meta.reindex(X.index)).
      - Uses a stable internal covariate name 'x' to avoid weird column-name edge cases.
    """
    from lifelines import CoxPHFitter

    Xn = _to_num_df(X)

    # ✅ CRASH FIX: handle duplicate feature names (duplicate columns)
    if isinstance(Xn, pd.DataFrame) and Xn.columns.duplicated().any():
        dups = Xn.columns[Xn.columns.duplicated()].unique().tolist()
        print(f"[COX-UNI] WARNING: {len(dups)} duplicate feature names in X; collapsing duplicates by mean. Example: {dups[:10]}")
        Xn = Xn.T.groupby(level=0).mean().T  # collapse duplicates -> unique columns

    # ✅ Align metadata to X rows (prevents index mismatch)
    meta2 = meta.reindex(Xn.index)

    T = pd.to_numeric(meta2[time_col], errors="coerce")
    E = coerce_binary_like(meta2[event_col], case_label=case_label, ctrl_label=ctrl_label)

    rows = []
    for c in Xn.columns:
        x = pd.to_numeric(Xn[c], errors="coerce")  # guaranteed 1D Series now

        # Use stable covariate name 'x' for lifelines
        df = pd.DataFrame({"T": T, "E": E, "x": x}).dropna()

        if (
            df.shape[0] < int(min_n)
            or df["T"].nunique() < 2
            or df["E"].sum() < int(min_events)
            or df["x"].nunique() < 2
        ):
            rows.append({"feature": c, "cox_uni_p": np.nan, "cox_uni_coef": np.nan, "cox_uni_hr": np.nan})
            continue

        try:
            cph = CoxPHFitter()
            cph.fit(df, duration_col="T", event_col="E", show_progress=False)
            summ = cph.summary.loc["x"]
            pval = float(summ["p"])
            coef = float(summ["coef"])
            hr = float(np.exp(coef))
            rows.append({"feature": c, "cox_uni_p": pval, "cox_uni_coef": coef, "cox_uni_hr": hr})
        except Exception:
            rows.append({"feature": c, "cox_uni_p": np.nan, "cox_uni_coef": np.nan, "cox_uni_hr": np.nan})

    out = pd.DataFrame(rows)
    out["cox_uni_q"] = _bh_fdr_series(out["cox_uni_p"])
    return out.sort_values("cox_uni_p", na_position="last").reset_index(drop=True)




def cox_multivariable_tables(
    X: pd.DataFrame,
    meta: pd.DataFrame,
    time_col: str,
    event_col: str,
    covariates: List[str],
    min_n: int = 10,
    min_events: int = 1,
    case_label: str = "recurrent",
    ctrl_label: str = "non-recurrent",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Multivariable CoxPH: one model per feature, adjusting for covariates.

    CRASH FIXES:
      - Collapses duplicate feature-name columns in X (mean across duplicates).
      - Aligns meta to X index (meta.reindex(X.index)).
      - Uses stable internal feature column name 'x' to avoid collisions & ensure summary lookup.

    NEW (robust behavior):
      - If no valid covariates are found, DO NOT raise.
        Return per-feature rows with NA adjusted-Cox stats and an empty terms table.
    """
    from lifelines import CoxPHFitter

    Xn = _to_num_df(X)

    # ✅ CRASH FIX: handle duplicate feature names (duplicate columns)
    if isinstance(Xn, pd.DataFrame) and Xn.columns.duplicated().any():
        dups = Xn.columns[Xn.columns.duplicated()].unique().tolist()
        print(f"[COX-MULTI] WARNING: {len(dups)} duplicate feature names in X; collapsing duplicates by mean. Example: {dups[:10]}")
        Xn = Xn.T.groupby(level=0).mean().T  # collapse duplicates -> unique columns

    meta2 = meta.copy()
    meta2.columns = meta2.columns.astype(str).str.strip()

    # match covariate names case-insensitively
    meta_col_map = {str(c).strip().lower(): c for c in meta2.columns}
    covariates_matched = []
    for c in (covariates or []):
        key = str(c).strip().lower()
        if key in meta_col_map:
            covariates_matched.append(meta_col_map[key])
    covariates = covariates_matched

    # ✅ DO NOT CRASH if no covariates: return NA tables for all features
    if len(covariates) == 0:
        print("[COX-MULTI] WARNING: No valid covariates found in meta for multivariable Cox. "
              "Returning NA adjusted-Cox results (continuing).")

        feat_tbl = pd.DataFrame({
            "feature": list(Xn.columns),
            "cox_multi_p": np.nan,
            "cox_multi_coef": np.nan,
            "cox_multi_hr": np.nan,
        })
        feat_tbl["cox_multi_q"] = np.nan

        terms_long = pd.DataFrame(columns=[
            "feature_tested", "term", "coef", "hr", "se", "z", "p_value"
        ])

        return (
            feat_tbl.sort_values("cox_multi_p", na_position="last").reset_index(drop=True),
            terms_long
        )

    # ✅ Align metadata to X rows
    meta2 = meta2.reindex(Xn.index)

    T = pd.to_numeric(meta2[time_col], errors="coerce")
    E = coerce_binary_like(meta2[event_col], case_label=case_label, ctrl_label=ctrl_label)
    C = coerce_covariates_to_numeric(meta2, covariates=covariates)  # expects meta-aligned index

    feat_rows = []
    long_rows = []

    for feat in Xn.columns:
        x = pd.to_numeric(Xn[feat], errors="coerce")  # guaranteed 1D Series now

        df = pd.DataFrame({"T": T, "E": E, "x": x})
        df = df.join(C)

        cols = ["T", "E", "x"] + list(C.columns)
        df = df[cols].dropna()

        if (
            df.shape[0] < int(min_n)
            or df["T"].nunique() < 2
            or df["E"].sum() < int(min_events)
            or df["x"].nunique() < 2
        ):
            feat_rows.append({"feature": feat, "cox_multi_p": np.nan, "cox_multi_coef": np.nan, "cox_multi_hr": np.nan})
            continue

        try:
            cph = CoxPHFitter()
            cph.fit(df, duration_col="T", event_col="E", show_progress=False)

            summ = cph.summary.copy()

            # long terms table (rename 'x' term to the actual feature for readability)
            for term, row in summ.iterrows():
                term_str = str(term)
                pretty_term = feat if term_str == "x" else term_str

                coef = float(row.get("coef", np.nan))
                p = float(row.get("p", np.nan))
                se = float(row.get("se(coef)", np.nan))
                z = float(row.get("z", np.nan))
                hr = float(np.exp(coef)) if np.isfinite(coef) else np.nan

                long_rows.append({
                    "feature_tested": feat,
                    "term": pretty_term,
                    "coef": coef,
                    "hr": hr,
                    "se": se,
                    "z": z,
                    "p_value": p
                })

            # per-feature row comes from the 'x' coefficient
            if "x" in summ.index:
                row = summ.loc["x"]
                pval = float(row.get("p", np.nan))
                coef = float(row.get("coef", np.nan))
                hr = float(np.exp(coef)) if np.isfinite(coef) else np.nan
            else:
                pval, coef, hr = np.nan, np.nan, np.nan

            feat_rows.append({"feature": feat, "cox_multi_p": pval, "cox_multi_coef": coef, "cox_multi_hr": hr})

        except Exception:
            feat_rows.append({"feature": feat, "cox_multi_p": np.nan, "cox_multi_coef": np.nan, "cox_multi_hr": np.nan})

    feat_tbl = pd.DataFrame(feat_rows)
    feat_tbl["cox_multi_q"] = _bh_fdr_series(feat_tbl["cox_multi_p"])

    terms_long = pd.DataFrame(long_rows)

    return (
        feat_tbl.sort_values("cox_multi_p", na_position="last").reset_index(drop=True),
        terms_long
    )



def _hr_direction_mask(hr: pd.Series, direction: str) -> pd.Series:
    direction = str(direction).lower()
    if direction == "gt1":
        return hr > 1.0
    if direction == "lt1":
        return hr < 1.0
    return pd.Series(True, index=hr.index)

def _hr_magnitude_mask(coef: pd.Series, abs_min: float) -> pd.Series:
    if abs_min <= 0:
        return pd.Series(True, index=coef.index)
    return coef.abs() >= float(abs_min)


# -------------------------
# Existing CNV helper (kept)
# -------------------------

def compute_signed_freq_and_tests(
    cn_df: pd.DataFrame,
    group: pd.Series,
    amp_threshold: float = 0.3,
    del_threshold: float = -0.3,
) -> pd.DataFrame:
    genes = list(cn_df.columns)
    g = group.astype(int).values

    mask_pos = (g == 1)
    mask_neg = (g == 0)
    n_pos = int(mask_pos.sum())
    n_neg = int(mask_neg.sum())

    rows = []
    for gene in genes:
        vals = cn_df[gene].values.astype(float)

        amp_mask = vals >= amp_threshold
        del_mask = vals <= del_threshold
        any_mask = amp_mask | del_mask

        n_any_pos = int((any_mask & mask_pos).sum())
        n_any_neg = int((any_mask & mask_neg).sum())
        n_amp_pos = int((amp_mask & mask_pos).sum())
        n_amp_neg = int((amp_mask & mask_neg).sum())
        n_del_pos = int((del_mask & mask_pos).sum())
        n_del_neg = int((del_mask & mask_neg).sum())

        freq_any_pos  = n_any_pos  / n_pos if n_pos > 0 else np.nan
        freq_any_neg  = n_any_neg  / n_neg if n_neg > 0 else np.nan
        freq_amp_pos  = n_amp_pos  / n_pos if n_pos > 0 else np.nan
        freq_amp_neg  = n_amp_neg  / n_neg if n_neg > 0 else np.nan
        freq_del_pos  = n_del_pos  / n_pos if n_pos > 0 else np.nan
        freq_del_neg  = n_del_neg  / n_neg if n_neg > 0 else np.nan

        try:
            OR_any, p_any = fisher_exact([[int((any_mask & mask_pos).sum()), int((any_mask & mask_neg).sum())],
                                         [int((~any_mask & mask_pos).sum()), int((~any_mask & mask_neg).sum())]], alternative="two-sided")
        except Exception:
            OR_any, p_any = np.nan, np.nan

        try:
            OR_amp, p_amp = fisher_exact([[int((amp_mask & mask_pos).sum()), int((amp_mask & mask_neg).sum())],
                                         [int((~amp_mask & mask_pos).sum()), int((~amp_mask & mask_neg).sum())]], alternative="two-sided")
        except Exception:
            OR_amp, p_amp = np.nan, np.nan

        try:
            OR_del, p_del = fisher_exact([[int((del_mask & mask_pos).sum()), int((del_mask & mask_neg).sum())],
                                         [int((~del_mask & mask_pos).sum()), int((~del_mask & mask_neg).sum())]], alternative="two-sided")
        except Exception:
            OR_del, p_del = np.nan, np.nan

        vals_pos = vals[mask_pos]
        vals_neg = vals[mask_neg]
        try:
            _, p_cont = mannwhitneyu(vals_pos, vals_neg, alternative="two-sided")
        except Exception:
            p_cont = np.nan

        median_pos  = float(np.median(vals_pos)) if len(vals_pos) > 0 else np.nan
        median_neg  = float(np.median(vals_neg)) if len(vals_neg) > 0 else np.nan
        delta_median = (median_pos - median_neg) if np.isfinite(median_pos) and np.isfinite(median_neg) else np.nan

        rows.append({
            "gene": gene,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_any_pos": n_any_pos,
            "n_any_neg": n_any_neg,
            "freq_any_pos": freq_any_pos,
            "freq_any_neg": freq_any_neg,
            "n_amp_pos": n_amp_pos,
            "n_amp_neg": n_amp_neg,
            "freq_amp_pos": freq_amp_pos,
            "freq_amp_neg": freq_amp_neg,
            "n_del_pos": n_del_pos,
            "n_del_neg": n_del_neg,
            "freq_del_pos": freq_del_pos,
            "freq_del_neg": freq_del_neg,
            "median_cn_dev_pos": median_pos,
            "median_cn_dev_neg": median_neg,
            "delta_median": delta_median,
            "p_any": p_any,
            "OR_any": OR_any,
            "p_amp": p_amp,
            "OR_amp": OR_amp,
            "p_del": p_del,
            "OR_del": OR_del,
            "p_cont": p_cont,
        })

    df_stats = pd.DataFrame(rows)

    for p_col in ["p_any", "p_amp", "p_del", "p_cont"]:
        pvals = df_stats[p_col].values.astype(float)
        mask = np.isfinite(pvals)
        fdr_col = p_col + "_fdr"
        df_stats[fdr_col] = np.nan
        if mask.sum() > 0:
            _, p_fdr, _, _ = multipletests(pvals[mask], method="fdr_bh")
            df_stats.loc[mask, fdr_col] = p_fdr

    return df_stats


# -------------------------
# Correlation pruning (same strategy)
# -------------------------


def prune_by_correlation(
    X: pd.DataFrame,
    pv: pd.DataFrame,
    corr_threshold: float = 0.90
):
    """
    Greedy correlation pruning:
      - keep the lowest-p feature in a correlated set (|r| >= corr_threshold)
      - drop the others

    SPECIAL RULE:
      - band vs band: do NOT prune
      - band vs non-band: pruning allowed
      - non-band vs non-band: pruning allowed

    Returns:
      kept_cols (List[str]),
      drops_df  (DataFrame with columns:
         feature, dropped_because_of, abs_r, p_self, p_other
      )
    """
    import numpy as np
    import pandas as pd

    def _is_band_feature(name: str) -> bool:
        s = str(name)
        return s.startswith("CNVblock__BAND__")

    Xn = _to_num_df(X)

    if isinstance(Xn, pd.DataFrame) and Xn.columns.duplicated().any():
        dups = Xn.columns[Xn.columns.duplicated()].unique().tolist()
        print(f"[PRUNE] WARNING: {len(dups)} duplicate feature names; collapsing duplicates by mean. Example: {dups[:10]}")
        Xn = Xn.T.groupby(level=0).mean().T

    Xn = Xn.dropna(axis=1, how="all")
    if Xn.shape[1] == 0:
        return [], pd.DataFrame(columns=["feature", "dropped_because_of", "abs_r", "p_self", "p_other"])

    pv = pv.copy()
    pv["feature"] = pv["feature"].astype(str)

    if "p_prune" in pv.columns:
        pcol = "p_prune"
    else:
        cand = [c for c in pv.columns if c != "feature"]
        pcol = None
        for c in cand:
            try:
                pd.to_numeric(pv[c], errors="coerce")
                pcol = c
                break
            except Exception:
                pass
        if pcol is None:
            raise ValueError("prune_by_correlation: pv must contain 'p_prune' or another numeric p-value column.")

    pv[pcol] = pd.to_numeric(pv[pcol], errors="coerce")

    features = [f for f in pv.sort_values(pcol, na_position="last")["feature"].tolist() if f in Xn.columns]
    if not features:
        return [], pd.DataFrame(columns=["feature", "dropped_because_of", "abs_r", "p_self", "p_other"])

    pmap = {f: (float(p) if np.isfinite(p) else np.inf) for f, p in zip(pv["feature"], pv[pcol])}

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = Xn[features].corr().abs()
    corr = corr.fillna(0.0)

    remaining = set(features)
    kept = []
    drops = []

    while remaining:
        f_keep = min(remaining, key=lambda f: pmap.get(f, np.inf))
        remaining.remove(f_keep)
        kept.append(f_keep)

        if not remaining:
            break

        ser = corr.loc[f_keep, list(remaining)]
        if isinstance(ser, pd.DataFrame):
            ser = ser.iloc[0]

        to_check = ser[ser >= float(corr_threshold)]

        for f_drop, r in to_check.items():
            # NEW RULE: do not prune band-vs-band
            if _is_band_feature(f_keep) and _is_band_feature(f_drop):
                continue

            drops.append({
                "feature": str(f_drop),
                "dropped_because_of": str(f_keep),
                "abs_r": float(r),
                "p_self": float(pmap.get(str(f_drop), np.inf)) if np.isfinite(pmap.get(str(f_drop), np.inf)) else np.nan,
                "p_other": float(pmap.get(str(f_keep), np.inf)) if np.isfinite(pmap.get(str(f_keep), np.inf)) else np.nan,
            })
            remaining.discard(f_drop)

    drops_df = pd.DataFrame(drops, columns=["feature", "dropped_because_of", "abs_r", "p_self", "p_other"])
    return kept, drops_df

def _jsonable(x):
    import numpy as np
    import pandas as pd
    from pathlib import Path

    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, pd.Index):
        return [str(v) for v in x.tolist()]
    if isinstance(x, set):
        return sorted([str(v) for v in x])
    if isinstance(x, tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    return str(x)

# -------------------------
# Main function
# -------------------------


def run_feature_selection(
    data_path: str,
    output_dir: str,
    subtype: str,
    status_col: str = "Disease_Status",
    pathway_options: Optional[List[str]] = None,
    RNAgene_bypass: bool = False,
    CNVgene_bypass: bool = False,
    # NEW blocks
    include_binary_arms: bool = True,
    include_binary_bands: bool = True,
    include_estimate_values: bool = True,

    # MUTblock toggles
    include_mutated_pathways: bool = True,

    # Binary mutated pathway source:
    #   snv    = current behavior, calculate from mutation/SNV matrix in data
    #   import = load a precomputed sample x binary-pathway matrix
    #   none   = skip binary pathway mutation features
    mut_pathway_source: str = "snv",
    mut_binary_matrix_csv: Optional[str] = None,
    mut_binary_sample_col: Optional[str] = None,
    mut_binary_prefix: str = "MUTblock__",

    # PTMB source:
    #   snv    = current behavior, calculate from MAF + GMT + GTF/CDS lengths
    #   import = load a precomputed sample x pTMB matrix
    #   none   = skip PTMB
    enable_ptmb: bool = False,
    ptmb_source: str = "none",
    ptmb_matrix_csv: Optional[str] = None,
    ptmb_sample_col: Optional[str] = None,
    ptmb_prefix: str = "MUTblock__pTMB__",
    ptmb_transform: str = "none",

    # PTMB SNV/MAF calculation inputs
    ptmb_maf_path: Optional[str] = None,
    ptmb_maf_sample_col: Optional[str] = None,
    ptmb_maf_gene_col: Optional[str] = None,
    ptmb_gtf_path: Optional[str] = None,
    ptmb_cds_cache_csv: Optional[str] = None,
    ptmb_denom_policy: str = "all_genes_with_len",
    ptmb_min_genes_with_len: int = 1,

    ptmb_gmt_path: Optional[List[str]] = None,
    protein_coding_gtf: Optional[str] = None,

    var_min: float = 0.0,
    var_min_nonmissing: int = 10,
    variance_keep_file: Optional[str] = None,
    var_keep_top_frac: Optional[float] = None,

    corr_prefilter: bool = False,
    corr_abs_min: Optional[float] = None,
    corr_min_n: int = 10,

    run_mwu: bool = True,
    run_cox_uni: bool = False,
    run_cox_multi: bool = False,

    mwu_p_thresh: float = 0.09,
    mwu_fdr_thresh: float = 0.10,
    cox_uni_p_thresh: float = 0.09,
    cox_uni_fdr_thresh: float = 0.10,
    cox_multi_p_thresh: float = 0.09,
    cox_multi_fdr_thresh: float = 0.10,

    cox_time_col: Optional[str] = None,
    cox_event_col: Optional[str] = None,
    cox_covariates: Optional[List[str]] = None,
    cox_hr_abs_min: float = 0.0,
    cox_hr_direction: str = "any",
    cox_min_n: int = 10,
    cox_min_events: int = 1,

    corr_threshold: float = 0.90,

    # NEW: which significance source to use for correlation pruning
    corr_prune_sig_mode: str = "hierarchical",

    sample_id_subset: Optional[Iterable[str]] = None,
    subset_run_name: Optional[str] = None,
    enable_subset_fs: bool = False,

    limma_pvalue_threshold: float = 0.05,
    limma_padj_threshold: float = 0.25,
    limma_logfc_threshold: float = 0.25,

    fs_min_features: int = 10,
    fs_topk_if_empty: int = 200,
    fs_min_per_block: int | dict = 0,

    custom_gene_sets_file: Optional[str] = None,
    custom_gene_set_mode: str = "median_log2",
    custom_gene_set_min_overlap: int = 1,

    mwu_case_label: str = "recurrent",
    mwu_ctrl_label: str = "non-recurrent",
):
    from pathlib import Path
    from scipy.stats import fisher_exact

    out_root = make_output_path(output_dir, subtype)
    fold_dir = os.path.join(out_root, subtype)
    full_dir = os.path.join(out_root, "preprocessing_full")
    ensure_dir(full_dir)

    # corr_prune_sig_mode = str(corr_prune_sig_mode).strip().lower()
    # if corr_prune_sig_mode == "mwu":
    #     corr_prune_sig_mode = "group"

    # valid_corr_prune_sig_modes = {"hierarchical", "cox_multi", "cox_uni", "group", "limma"}
    # if corr_prune_sig_mode not in valid_corr_prune_sig_modes:
    #     raise ValueError(
    #         f"Unsupported corr_prune_sig_mode={corr_prune_sig_mode!r}. "
    #         f"Choose from: {sorted(valid_corr_prune_sig_modes)}"
    #     )

    if corr_prune_sig_mode is None:
        corr_prune_sig_mode = "hierarchical"
    else:
        corr_prune_sig_mode = str(corr_prune_sig_mode).strip().lower()

    # treat "none" / blank / default-like values as "use the default behavior"
    if corr_prune_sig_mode in {"", "none", "default", "auto", "na", "nan"}:
        corr_prune_sig_mode = "hierarchical"

    if corr_prune_sig_mode == "mwu":
        corr_prune_sig_mode = "group"

    valid_corr_prune_sig_modes = {"hierarchical", "cox_multi", "cox_uni", "group", "limma"}
    if corr_prune_sig_mode not in valid_corr_prune_sig_modes:
        raise ValueError(
            f"Unsupported corr_prune_sig_mode={corr_prune_sig_mode!r}. "
            f"Choose from: {sorted(valid_corr_prune_sig_modes)}"
        )

    fs_dir = None
    if enable_subset_fs:
        if subset_run_name is None:
            if sample_id_subset is None:
                subset_run_name = "fullcohort"
            else:
                ids_tmp = pd.Index(sample_id_subset).astype(str).str.strip()
                tag = str(abs(hash(tuple(sorted(ids_tmp)))) % (10**10))
                subset_run_name = f"subset_{tag}"
        fs_dir = os.path.join(fold_dir, f"fs_fold_{subset_run_name}")
        ensure_dir(fs_dir)

    if pathway_options is None:
        pathway_options = []

    # Normalize mutation feature source flags.
    mut_pathway_source = str(mut_pathway_source or "snv").strip().lower()
    if not include_mutated_pathways:
        mut_pathway_source = "none"
    if mut_pathway_source not in {"snv", "import", "none"}:
        raise ValueError("mut_pathway_source must be one of: snv, import, none")
    include_mutated_pathways = (mut_pathway_source != "none")

    ptmb_source = str(ptmb_source or "none").strip().lower()
    # Backward compatibility: old callers used enable_ptmb=True without ptmb_source.
    if enable_ptmb and ptmb_source == "none":
        ptmb_source = "snv"
    if ptmb_source not in {"snv", "import", "none"}:
        raise ValueError("ptmb_source must be one of: snv, import, none")
    enable_ptmb = (ptmb_source != "none")

    mut_binary_prefix = str(mut_binary_prefix or "MUTblock__")
    ptmb_prefix = str(ptmb_prefix or "MUTblock__pTMB__")
    ptmb_transform = str(ptmb_transform or "none").strip().lower()


    run_input_params = {
        "data_path": data_path,
        "output_dir": output_dir,
        "subtype": subtype,
        "status_col": status_col,
        "pathway_options": pathway_options,

        "RNAgene_bypass": RNAgene_bypass,
        "CNVgene_bypass": CNVgene_bypass,

        "include_binary_arms": include_binary_arms,
        "include_binary_bands": include_binary_bands,
        "include_estimate_values": include_estimate_values,

        "include_mutated_pathways": include_mutated_pathways,
        "mut_pathway_source": mut_pathway_source,
        "mut_binary_matrix_csv": mut_binary_matrix_csv,
        "mut_binary_sample_col": mut_binary_sample_col,
        "mut_binary_prefix": mut_binary_prefix,

        "enable_ptmb": enable_ptmb,
        "ptmb_source": ptmb_source,
        "ptmb_matrix_csv": ptmb_matrix_csv,
        "ptmb_sample_col": ptmb_sample_col,
        "ptmb_prefix": ptmb_prefix,
        "ptmb_transform": ptmb_transform,
        "ptmb_maf_path": ptmb_maf_path,
        "ptmb_maf_sample_col": ptmb_maf_sample_col,
        "ptmb_maf_gene_col": ptmb_maf_gene_col,
        "ptmb_gtf_path": ptmb_gtf_path,
        "ptmb_cds_cache_csv": ptmb_cds_cache_csv,
        "ptmb_denom_policy": ptmb_denom_policy,
        "ptmb_min_genes_with_len": ptmb_min_genes_with_len,
        "ptmb_gmt_path": ptmb_gmt_path,

        "protein_coding_gtf": protein_coding_gtf,

        "var_min": var_min,
        "var_min_nonmissing": var_min_nonmissing,
        "variance_keep_file": variance_keep_file,
        "var_keep_top_frac": var_keep_top_frac,

        "corr_prefilter": corr_prefilter,
        "corr_abs_min": corr_abs_min,
        "corr_min_n": corr_min_n,
        "corr_threshold": corr_threshold,
        "corr_prune_sig_mode": corr_prune_sig_mode,

        "run_mwu": run_mwu,
        "run_cox_uni": run_cox_uni,
        "run_cox_multi": run_cox_multi,

        "mwu_p_thresh": mwu_p_thresh,
        "mwu_fdr_thresh": mwu_fdr_thresh,
        "cox_uni_p_thresh": cox_uni_p_thresh,
        "cox_uni_fdr_thresh": cox_uni_fdr_thresh,
        "cox_multi_p_thresh": cox_multi_p_thresh,
        "cox_multi_fdr_thresh": cox_multi_fdr_thresh,

        "cox_time_col": cox_time_col,
        "cox_event_col": cox_event_col,
        "cox_covariates": cox_covariates,
        "cox_hr_abs_min": cox_hr_abs_min,
        "cox_hr_direction": cox_hr_direction,
        "cox_min_n": cox_min_n,
        "cox_min_events": cox_min_events,

        "sample_id_subset": sample_id_subset,
        "subset_run_name": subset_run_name,
        "enable_subset_fs": enable_subset_fs,

        "limma_pvalue_threshold": limma_pvalue_threshold,
        "limma_padj_threshold": limma_padj_threshold,
        "limma_logfc_threshold": limma_logfc_threshold,

        "fs_min_features": fs_min_features,
        "fs_topk_if_empty": fs_topk_if_empty,
        "fs_min_per_block": fs_min_per_block,

        "custom_gene_sets_file": custom_gene_sets_file,
        "custom_gene_set_mode": custom_gene_set_mode,
        "custom_gene_set_min_overlap": custom_gene_set_min_overlap,

        "mwu_case_label": mwu_case_label,
        "mwu_ctrl_label": mwu_ctrl_label,
    }
    run_input_params = _jsonable(run_input_params)

    def _build_prefixed_block(
        df: Optional[pd.DataFrame],
        index: pd.Index,
        outer_prefix: str,
        inner_prefix: str,
        fill_value: Optional[float] = None,
    ) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame(index=index)
        out = _safe_reindex(df, index).copy()
        out.index = out.index.astype(str).str.strip()
        out.columns = [f"{inner_prefix}{str(c).strip()}" for c in out.columns]
        out = out.loc[:, ~out.columns.astype(str).str.lower().str.startswith("unnamed")]
        if fill_value is not None:
            out = out.fillna(fill_value)
        return _add_block_prefix(out, outer_prefix)

    def _concat_nonempty(
        frames: List[pd.DataFrame],
        name: str,
        index: Optional[pd.Index] = None,
    ) -> pd.DataFrame:
        keep = [df for df in frames if isinstance(df, pd.DataFrame) and df.shape[1] > 0]
        if not keep:
            return pd.DataFrame(index=index)

        out = pd.concat(keep, axis=1)
        out = out.loc[:, ~out.columns.astype(str).str.lower().str.startswith("unnamed")]

        if out.columns.duplicated().any():
            dups = out.columns[out.columns.duplicated()].unique().tolist()
            print(f"[{name}] duplicate columns detected; collapsing by max. Example: {dups[:20]}")
            out = _to_num_df(out).fillna(0.0)
            out = out.T.groupby(level=0).max().T

        return out

    def _nonempty_shape(df: pd.DataFrame) -> int:
        return int(df.shape[1]) if isinstance(df, pd.DataFrame) else 0

    def _is_binary_01_series(s) -> bool:
        if isinstance(s, pd.DataFrame):
            if s.shape[1] == 1:
                s = s.iloc[:, 0]
            else:
                s = s.apply(lambda col: pd.to_numeric(col, errors="coerce"))
                s = s.max(axis=1)

        x = pd.to_numeric(s, errors="coerce").dropna()
        if x.empty:
            return False
        vals = set(np.unique(x.astype(float)))
        return vals.issubset({0.0, 1.0})

    def _split_binary_features(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        Xn = _to_num_df(X)

        if Xn.columns.duplicated().any():
            dups = Xn.columns[Xn.columns.duplicated()].unique().tolist()
            print(f"[GROUP TEST] duplicate columns detected; collapsing by max. Example: {dups[:20]}")
            Xn = Xn.T.groupby(level=0).max().T

        binary_cols = []
        nonbinary_cols = []
        for c in Xn.columns:
            if _is_binary_01_series(Xn[c]):
                binary_cols.append(str(c))
            else:
                nonbinary_cols.append(str(c))
        return binary_cols, nonbinary_cols

    def _fisher_binary_table(
        X: pd.DataFrame,
        y: pd.Series,
        case_label: str = "recurrent",
        ctrl_label: str = "non-recurrent",
    ) -> pd.DataFrame:
        Xn = _to_num_df(X)
        if Xn.columns.duplicated().any():
            dups = Xn.columns[Xn.columns.duplicated()].unique().tolist()
            print(f"[FISHER] duplicate columns detected; collapsing by max. Example: {dups[:20]}")
            Xn = Xn.T.groupby(level=0).max().T

        yb = coerce_binary_like(y, case_label=case_label, ctrl_label=ctrl_label)

        rows = []
        for feat in Xn.columns:
            s = pd.to_numeric(Xn[feat], errors="coerce")
            mask = s.notna() & yb.notna()
            x = s[mask].astype(float)
            yy = yb[mask].astype(int)

            if x.empty or yy.empty:
                rows.append({"feature": feat, "group_p": np.nan, "group_test": "fisher"})
                continue

            vals = set(np.unique(x))
            if not vals.issubset({0.0, 1.0}) or len(np.unique(yy)) < 2 or len(np.unique(x)) < 2:
                rows.append({"feature": feat, "group_p": np.nan, "group_test": "fisher"})
                continue

            a = int(((yy == 1) & (x == 1)).sum())
            b = int(((yy == 0) & (x == 1)).sum())
            c = int(((yy == 1) & (x == 0)).sum())
            d = int(((yy == 0) & (x == 0)).sum())

            try:
                _, pval = fisher_exact([[a, b], [c, d]], alternative="two-sided")
            except Exception:
                pval = np.nan

            rows.append({
                "feature": feat,
                "group_p": float(pval) if np.isfinite(pval) else np.nan,
                "group_test": "fisher",
            })

        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(columns=["feature", "group_p", "group_q", "group_test"])
        out["group_q"] = _bh_fdr_series(out["group_p"])
        return out

    # --- load & normalize everything
    data, all_pathway_dicts = load_all_data(data_path, subtype, pathway_options)

    cleaned_data, meta, y_all_enc, all_ids_full, _ = dedup_and_align_data(
        data,
        status_col=status_col,
        prep_dir=os.path.join(fold_dir, "preprocessing_feature_selection_only")
    )

    binary_preserve_keys = {
        "arm_binary",
        "band_binary",
        "ecotyper_cell_assign",
        "ecotyper_ecotype_assign",
    }

    for k, df in list(cleaned_data.items()):
        if not isinstance(df, pd.DataFrame):
            continue

        df = df.copy()
        df.index = df.index.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

        if df.index.duplicated().any():
            if k in binary_preserve_keys:
                print(f"[POST-DEDUP:{k}] duplicate indices remain; collapsing by max.")
                df = _to_num_df(df).fillna(0.0)
                df = df.groupby(level=0).max()
            else:
                print(f"[POST-DEDUP:{k}] duplicate indices remain; collapsing by mean.")
                df = _dedup_index(df, how="aggregate", agg="mean")

        cleaned_data[k] = df

    meta = cleaned_data["meta"].copy()
    if status_col not in meta.columns:
        raise KeyError(f"Target column {status_col!r} not found in meta: {list(meta.columns)}")
    all_ids_full = meta.index[~meta.index.duplicated(keep="first")]

    # -------------------------------------------------
    # Protein-coding gene set (optional)
    # -------------------------------------------------
    pc_genes_upper = None
    if protein_coding_gtf:
        pc = load_protein_coding_genes_from_gtf(protein_coding_gtf)
        pc_genes_upper = {g.upper() for g in pc}
        _dump_json_safely(
            {
                "protein_coding_gtf": protein_coding_gtf,
                "n_protein_coding_genes": int(len(pc_genes_upper)),
            },
            os.path.join(full_dir, "protein_coding_filter_summary.json")
        )
        print(f"[PC] Loaded protein_coding genes from GTF: n={len(pc_genes_upper)}")

    # -------------------------------------------------
    # 1) FULL-COHORT PREPROCESSING (always saved)
    # -------------------------------------------------
    mut_full = pd.DataFrame(index=all_ids_full)
    mut_pathway_slugs_for_ptmb: Optional[set[str]] = None

    if mut_pathway_source == "snv":
        mut_src_full = _safe_reindex(cleaned_data["mut"], all_ids_full).fillna(0)
        mut_all_full, mut_aux = process_mutation_pathways(
            mut_src_full, meta, all_pathway_dicts, pathway_options, None, subtype
        )
        if isinstance(mut_all_full, pd.DataFrame) and mut_all_full.shape[1] > 0:
            mut_pathway_slugs_for_ptmb = set(canonical_pathway_slug(x) for x in mut_all_full.columns.astype(str).tolist())
        mut_full = _add_block_prefix(_safe_reindex(mut_all_full, all_ids_full).fillna(0), "MUTblock__")
        mut_full.to_csv(os.path.join(full_dir, "MUT_binary_pathways_full_snv_calculated.csv"))
        print(f"[MUT] Calculated binary mutated pathway features from SNV/mutation matrix: {mut_full.shape[1]}")
    elif mut_pathway_source == "import":
        if not mut_binary_matrix_csv:
            raise ValueError("[MUT] mut_pathway_source='import' requires --mut_binary_matrix_csv")
        mut_full = load_precomputed_feature_matrix(
            path=mut_binary_matrix_csv,
            sample_ids_target=pd.Index(all_ids_full).astype(str),
            sample_col=mut_binary_sample_col,
            prefix=mut_binary_prefix,
            expected_binary=True,
            transform="none",
            table_name="MUT_BINARY_IMPORT",
        )
        if isinstance(mut_full, pd.DataFrame) and mut_full.shape[1] > 0:
            mut_pathway_slugs_for_ptmb = set(canonical_pathway_slug(x) for x in mut_full.columns.astype(str).tolist())
        mut_full.to_csv(os.path.join(full_dir, "MUT_binary_pathways_full_imported.csv"))
        _dump_json_safely(
            {
                "mut_pathway_source": "import",
                "mut_binary_matrix_csv": mut_binary_matrix_csv,
                "mut_binary_sample_col": mut_binary_sample_col,
                "mut_binary_prefix": mut_binary_prefix,
                "n_mutated_pathway_features": int(mut_full.shape[1]),
            },
            os.path.join(full_dir, "mut_binary_import_config.json")
        )
        print(f"[MUT] Imported binary mutated pathway features: {mut_full.shape[1]}")
    else:
        print("[MUT] mut_pathway_source='none' → skipping binary mutated pathway features.")

    # PTMB: either calculate from SNV/MAF inputs or import a precomputed pTMB matrix.
    ptmb_full = pd.DataFrame(index=all_ids_full)
    ptmb_info_full = pd.DataFrame()
    if enable_ptmb and ptmb_source == "snv":
        if not ptmb_maf_path:
            raise ValueError("[PTMB] ptmb_source='snv' requires --ptmb_maf_path")
        if not ptmb_gmt_path:
            raise ValueError("[PTMB] ptmb_source='snv' requires --ptmb_gmt_path (one or more GMTs).")

        gene_sets = load_gene_sets_from_gmt_many(ptmb_gmt_path or [])

        if not ptmb_gtf_path:
            if protein_coding_gtf:
                ptmb_gtf_path = protein_coding_gtf
                print(f"[PTMB] ptmb_gtf_path not provided; using protein_coding_gtf: {ptmb_gtf_path}")
            else:
                raise ValueError("[PTMB] ptmb_source='snv' requires --ptmb_gtf_path (or set --protein_coding_gtf and omit --ptmb_gtf_path).")

        cds_cache = ptmb_cds_cache_csv or os.path.join(full_dir, "ptmb_cds_lengths_cache.csv")
        cds_len = load_cds_lengths_from_gtf(ptmb_gtf_path, cache_csv=cds_cache, prefer_gene_name=True)

        if mut_pathway_slugs_for_ptmb is not None and len(mut_pathway_slugs_for_ptmb) > 0:
            gene_sets = {pw: gs for pw, gs in gene_sets.items() if canonical_pathway_slug(pw) in mut_pathway_slugs_for_ptmb}

        if not gene_sets:
            raise ValueError("[PTMB] After optional restriction to MUTblock universe, gene_sets is empty.")

        ptmb_full_raw, ptmb_info_full = compute_ptmb_matrix_from_maf(
            maf_path=ptmb_maf_path,
            sample_ids_target=pd.Index(all_ids_full).astype(str),
            gene_sets=gene_sets,
            cds_len_bp=cds_len,
            maf_sample_col=ptmb_maf_sample_col,
            maf_gene_col=ptmb_maf_gene_col,
            denom_policy=ptmb_denom_policy,
            min_genes_with_len=int(ptmb_min_genes_with_len),
            out_prefix=ptmb_prefix,
        )

        ptmb_full_raw = _safe_reindex(ptmb_full_raw, all_ids_full)
        ptmb_full_raw.to_csv(os.path.join(full_dir, "PTMB_full_raw.csv"))

        # Preserve previous behavior for SNV-calculated PTMB: log10(raw + 1e-12).
        ptmb_log_eps = 1e-12
        ptmb_full_num = _to_num_df(ptmb_full_raw).fillna(0.0)
        ptmb_full_num = ptmb_full_num.clip(lower=0.0)
        ptmb_full_log10 = np.log10(ptmb_full_num + ptmb_log_eps)

        ptmb_full = ptmb_full_log10
        ptmb_full.to_csv(os.path.join(full_dir, "PTMB_full_log10.csv"))

        _dump_json_safely(
            {
                "ptmb_source": "snv",
                "ptmb_transform": "log10",
                "ptmb_log_eps": float(ptmb_log_eps),
            },
            os.path.join(full_dir, "ptmb_transform.json")
        )

        ptmb_info_full.to_csv(os.path.join(full_dir, "PTMB_pathway_info.csv"), index=False)
        _dump_json_safely(
            {
                "enable_ptmb": True,
                "ptmb_source": "snv",
                "ptmb_maf_path": ptmb_maf_path,
                "ptmb_maf_sample_col": ptmb_maf_sample_col,
                "ptmb_maf_gene_col": ptmb_maf_gene_col,
                "ptmb_gtf_path": ptmb_gtf_path,
                "ptmb_cds_cache_csv": cds_cache,
                "ptmb_denom_policy": ptmb_denom_policy,
                "ptmb_min_genes_with_len": int(ptmb_min_genes_with_len),
                "ptmb_gmt_path": ptmb_gmt_path,
                "ptmb_prefix": ptmb_prefix,
                "n_ptmb_features": int(ptmb_full.shape[1]),
                "restricted_to_mutblock_universe": bool(mut_pathway_slugs_for_ptmb is not None and len(mut_pathway_slugs_for_ptmb) > 0),
            },
            os.path.join(full_dir, "ptmb_config.json")
        )
        print(f"[PTMB] Computed PTMB features from SNV/MAF: {ptmb_full.shape[1]} (saved to preprocessing_full/)")
    elif enable_ptmb and ptmb_source == "import":
        if not ptmb_matrix_csv:
            raise ValueError("[PTMB] ptmb_source='import' requires --ptmb_matrix_csv")
        ptmb_full = load_precomputed_feature_matrix(
            path=ptmb_matrix_csv,
            sample_ids_target=pd.Index(all_ids_full).astype(str),
            sample_col=ptmb_sample_col,
            prefix=ptmb_prefix,
            expected_binary=False,
            transform=ptmb_transform,
            table_name="PTMB_IMPORT",
        )
        ptmb_full.to_csv(os.path.join(full_dir, "PTMB_full_imported.csv"))
        _dump_json_safely(
            {
                "enable_ptmb": True,
                "ptmb_source": "import",
                "ptmb_matrix_csv": ptmb_matrix_csv,
                "ptmb_sample_col": ptmb_sample_col,
                "ptmb_prefix": ptmb_prefix,
                "ptmb_transform": ptmb_transform,
                "n_ptmb_features": int(ptmb_full.shape[1]),
            },
            os.path.join(full_dir, "ptmb_config.json")
        )
        print(f"[PTMB] Imported precomputed PTMB features: {ptmb_full.shape[1]}")
    else:
        print("[PTMB] ptmb_source='none' → skipping PTMB features.")

    gsva_rna_full = get_gsea(
        data_matrix=cleaned_data["rna"],
        subtype_meta=cleaned_data["meta"],
        pathway_dicts=all_pathway_dicts,
        pathway_options=pathway_options,
        subtype=subtype,
        data_type="RNA",
    )
    gsva_rna_full.index = gsva_rna_full.index.astype(str)

    gsva_cnv_full = get_gsea(
        data_matrix=cleaned_data["cnv"],
        subtype_meta=cleaned_data["meta"],
        pathway_dicts=all_pathway_dicts,
        pathway_options=pathway_options,
        subtype=subtype,
        data_type="copynumber",
    )
    gsva_cnv_full.index = gsva_cnv_full.index.astype(str)

    # # CNV gene-level significance
    # cnv_gene_sig_full = pd.DataFrame(index=all_ids_full)
    # if "cnv" in cleaned_data and isinstance(cleaned_data["cnv"], pd.DataFrame) and not cleaned_data["cnv"].empty:
    #     cnv_gene_full = _to_num_df(_safe_reindex(cleaned_data["cnv"], all_ids_full)).fillna(0.0)
    #     cnv_stats = compute_signed_freq_and_tests(
    #         cn_df=cnv_gene_full,
    #         group=y_all_enc.reindex(all_ids_full),
    #         amp_threshold=0.3,
    #         del_threshold=-0.3,
    #     )
    #     cnv_stats.to_csv(os.path.join(full_dir, "CNV_gene_signed_freq_stats.csv"), index=False)

    #     keep_genes = cnv_stats["gene"].astype(str).tolist()
    #     if keep_genes:
    #         cnv_gene_sig_full = _add_block_prefix(cnv_gene_full[keep_genes].copy(), "CNVblock__")

            # if pc_genes_upper is not None and not cnv_gene_sig_full.empty:
            #     cnv_gene_sig_full, removed = filter_gene_level_block_to_protein_coding(
            #         cnv_gene_sig_full, prefix="CNVblock__", pc_genes_upper=pc_genes_upper
            #     )
            #     with open(os.path.join(full_dir, "protein_coding_removed_CNVblock_genes.txt"), "w") as f:
            #         for c in removed:
            #             f.write(c + "\n")
            #     print(f"[PC] CNV gene-level block: removed non-protein-coding cols: n={len(removed)}")

            # cnv_gene_sig_full.to_csv(os.path.join(full_dir, "CNVblock_genelevel_all.csv"))


    # CNV gene-level significance
    cnv_gene_sig_full = pd.DataFrame(index=all_ids_full)

    if CNVgene_bypass:
        print("[FS] CNVgene-bypass=True -> skipping CNV gene-level stats/significance block.")
    else:
        if "cnv" in cleaned_data and isinstance(cleaned_data["cnv"], pd.DataFrame) and not cleaned_data["cnv"].empty:
            cnv_gene_full = _to_num_df(_safe_reindex(cleaned_data["cnv"], all_ids_full)).fillna(0.0)

            cnv_stats = compute_signed_freq_and_tests(
                cn_df=cnv_gene_full,
                group=y_all_enc.reindex(all_ids_full),
                amp_threshold=0.3,
                del_threshold=-0.3,
            )
            cnv_stats.to_csv(os.path.join(full_dir, "CNV_gene_signed_freq_stats.csv"), index=False)

            keep_genes = cnv_stats["gene"].astype(str).tolist()
            if keep_genes:
                cnv_gene_sig_full = _add_block_prefix(cnv_gene_full[keep_genes].copy(), "CNVblock__")
                if pc_genes_upper is not None and not cnv_gene_sig_full.empty:
                    cnv_gene_sig_full, removed = filter_gene_level_block_to_protein_coding(
                        cnv_gene_sig_full, prefix="CNVblock__", pc_genes_upper=pc_genes_upper
                    )
                    with open(os.path.join(full_dir, "protein_coding_removed_CNVblock_genes.txt"), "w") as f:
                        for c in removed:
                            f.write(c + "\n")
                    print(f"[PC] CNV gene-level block: removed non-protein-coding cols: n={len(removed)}")

                cnv_gene_sig_full.to_csv(os.path.join(full_dir, "CNVblock_genelevel_all.csv"))

    rna_pw_full = _add_block_prefix(_safe_reindex(gsva_rna_full, all_ids_full), "RNAblock__")
    cnv_pw_full = _add_block_prefix(_safe_reindex(gsva_cnv_full, all_ids_full), "CNVblock__")

    cib_full        = _add_block_prefix(_safe_reindex(cleaned_data["cibersort"],        all_ids_full).fillna(0), "RNAblock__")
    pga_full        = _add_block_prefix(_safe_reindex(cleaned_data["cnv_pct_altered"],  all_ids_full).fillna(0), "CNVblock__")
    tf_full         = _add_block_prefix(_safe_reindex(cleaned_data["tf"],               all_ids_full).fillna(0), "CNVblock__")
    tmb_full        = _add_block_prefix(_safe_reindex(cleaned_data["tmb"],              all_ids_full).fillna(0), "MUTblock__")
    hoadley_full    = _add_block_prefix(_safe_reindex(cleaned_data["hoadley"],          all_ids_full).fillna(0), "RNAblock__")
    pathogenic_full = _add_block_prefix(_safe_reindex(cleaned_data["pathogenic_count"], all_ids_full).fillna(0), "MUTblock__")
    clinical_full   = _add_block_prefix(_safe_reindex(cleaned_data["clinical"],         all_ids_full),           "CLINICALblock__")

    # binary arm / binary band / ESTIMATE
    arm_binary_full = pd.DataFrame(index=all_ids_full)
    if include_binary_arms and "arm_binary" in cleaned_data:
        arm_binary_full = _build_prefixed_block(
            cleaned_data["arm_binary"],
            index=all_ids_full,
            outer_prefix="CNVblock__",
            inner_prefix="ARM__",
            fill_value=0.0,
        )

    band_binary_full = pd.DataFrame(index=all_ids_full)
    if include_binary_bands and "band_binary" in cleaned_data:
        band_binary_full = _build_prefixed_block(
            cleaned_data["band_binary"],
            index=all_ids_full,
            outer_prefix="CNVblock__",
            inner_prefix="BAND__",
            fill_value=0.0,
        )

    estimate_full = pd.DataFrame(index=all_ids_full)
    if include_estimate_values and "estimate" in cleaned_data:
        estimate_full = _build_prefixed_block(
            cleaned_data["estimate"],
            index=all_ids_full,
            outer_prefix="RNAblock__",
            inner_prefix="ESTIMATE__",
            fill_value=0.0,
        )

    # Explicitly carry all loaded TIDE / EcoTyper tables forward
    tide_full = _build_prefixed_block(
        cleaned_data.get("tide"),
        index=all_ids_full,
        outer_prefix="RNAblock__",
        inner_prefix="TIDE__",
        fill_value=0.0,
    )

    ecotyper_cell_states_full = _build_prefixed_block(
        cleaned_data.get("ecotyper_cell_states"),
        index=all_ids_full,
        outer_prefix="RNAblock__",
        inner_prefix="ECOTYPER__CELLSTATE_ABUND__",
        fill_value=0.0,
    )

    ecotyper_cell_assign_full = _build_prefixed_block(
        cleaned_data.get("ecotyper_cell_assign"),
        index=all_ids_full,
        outer_prefix="RNAblock__",
        inner_prefix="ECOTYPER__CELLSTATE_ASSIGN__",
        fill_value=0.0,
    )

    ecotyper_ecotypes_full = _build_prefixed_block(
        cleaned_data.get("ecotyper_ecotypes"),
        index=all_ids_full,
        outer_prefix="RNAblock__",
        inner_prefix="ECOTYPER__ECOTYPE_ABUND__",
        fill_value=0.0,
    )

    ecotyper_ecotype_assign_full = _build_prefixed_block(
        cleaned_data.get("ecotyper_ecotype_assign"),
        index=all_ids_full,
        outer_prefix="RNAblock__",
        inner_prefix="ECOTYPER__ECOTYPE_ASSIGN__",
        fill_value=0.0,
    )

    ecotyper_parts_full = {
        "cell_states_abundance": ecotyper_cell_states_full,
        "cell_states_assignment": ecotyper_cell_assign_full,
        "ecotypes_abundance": ecotyper_ecotypes_full,
        "ecotypes_assignment": ecotyper_ecotype_assign_full,
    }

    ecotyper_full = _concat_nonempty(
        [
            ecotyper_cell_states_full,
            ecotyper_cell_assign_full,
            ecotyper_ecotypes_full,
            ecotyper_ecotype_assign_full,
        ],
        name="ECOTYPER_FULL",
        index=all_ids_full,
    )

    print("[RUN FS] tide_full:", tide_full.shape)
    print("[RUN FS] ecotyper_cell_states_full:", ecotyper_cell_states_full.shape)
    print("[RUN FS] ecotyper_cell_assign_full:", ecotyper_cell_assign_full.shape)
    print("[RUN FS] ecotyper_ecotypes_full:", ecotyper_ecotypes_full.shape)
    print("[RUN FS] ecotyper_ecotype_assign_full:", ecotyper_ecotype_assign_full.shape)
    print("[RUN FS] ecotyper_full:", ecotyper_full.shape)

    rna_log2_full = np.log2(_safe_reindex(cleaned_data["rna"], all_ids_full).fillna(0) + 1)

    # ---- custom gene-set scores ----
    def _read_custom_gene_sets(path: str) -> Dict[str, List[str]]:
        if not path or not os.path.exists(path):
            return {}
        rows = []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for ln in f:
                    if not ln.strip() or ln.lstrip().startswith("#"):
                        continue
                    parts = ln.rstrip("\n").split("\t")
                    if len(parts) >= 3:
                        name = parts[0].strip()
                        genes = [g.strip() for g in parts[2:] if g.strip()]
                        if name and genes:
                            rows.append((name, genes))
        except Exception:
            rows = []
        if rows:
            return {n: sorted(list(dict.fromkeys(g))) for n, g in rows}

        try:
            df = pd.read_csv(path, sep=None, engine="python", dtype=str, low_memory=False)
            low = {c.lower(): c for c in df.columns}
            if "set" in low and ("gene" in low or "symbol" in low):
                set_col = low["set"]; gene_col = low.get("gene", low.get("symbol"))
                df = df[[set_col, gene_col]].dropna()
                d: Dict[str, List[str]] = {}
                for name, sub in df.groupby(set_col):
                    genes = [str(x).strip() for x in sub[gene_col] if str(x).strip()]
                    if genes:
                        d[str(name).strip()] = sorted(list(dict.fromkeys(genes)))
                if d:
                    return d
        except Exception:
            pass

        try:
            g = [ln.strip() for ln in open(path, "r", encoding="utf-8", errors="ignore") if ln.strip()]
            if g:
                return {Path(path).stem: sorted(list(dict.fromkeys(g)))}
        except Exception:
            pass
        return {}

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
                out[f"GS_{name}__median_log2"]   = np.full(n_samples, np.nan, dtype=float)
                out[f"GS_{name}__geo_mean_log2"] = np.full(n_samples, np.nan, dtype=float)
                out[f"GS_{name}__zscore_mean"]   = np.full(n_samples, np.nan, dtype=float)
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

            out[f"GS_{name}__median_log2"]   = med_log2
            out[f"GS_{name}__geo_mean_log2"] = geo_mean_log2
            out[f"GS_{name}__zscore_mean"]   = zmean

        return pd.DataFrame(out, index=X_log2.index)

    rna_custom_sets_full = pd.DataFrame(index=all_ids_full)
    if custom_gene_sets_file:
        gs_dict = _read_custom_gene_sets(custom_gene_sets_file)
        if gs_dict:
            custom_scores = _compute_gene_set_scores(
                X_log2=rna_log2_full,
                gene_sets=gs_dict,
                mode=custom_gene_set_mode,
                min_overlap=int(custom_gene_set_min_overlap),
            )

            rna_custom_sets_full = _add_block_prefix(custom_scores, "RNAblock__")
            rna_custom_sets_full.to_csv(os.path.join(full_dir, "RNA_custom_sets_full.csv"))

    # full candidate blocks
    rna_candidates_full = _concat_nonempty(
        [
            rna_pw_full,
            hoadley_full,
            cib_full,
            rna_custom_sets_full,
            tide_full,
            ecotyper_cell_states_full,
            ecotyper_cell_assign_full,
            ecotyper_ecotypes_full,
            ecotyper_ecotype_assign_full,
            estimate_full,
        ],
        name="RNA_CANDIDATES_FULL",
        index=all_ids_full,
    )

    cnv_candidates_full = _concat_nonempty(
        [
            cnv_pw_full,
            pga_full,
            tf_full,
            cnv_gene_sig_full,
            arm_binary_full,
            band_binary_full,
        ],
        name="CNV_CANDIDATES_FULL",
        index=all_ids_full,
    )

    mut_candidates_full = _concat_nonempty(
        [
            mut_full,
            ptmb_full,
            tmb_full,
            pathogenic_full,
        ],
        name="MUT_CANDIDATES_FULL",
        index=all_ids_full,
    )

    # save full preprocessing artifacts
    rna_candidates_full.to_csv(os.path.join(full_dir, "RNAblock_candidates_full.csv"))
    cnv_candidates_full.to_csv(os.path.join(full_dir, "CNVblock_candidates_full.csv"))
    mut_candidates_full.to_csv(os.path.join(full_dir, "MUTblock_candidates_full.csv"))
    clinical_full.to_csv(os.path.join(full_dir, "CLINICALblock_candidates_full.csv"))

    cib_full.to_csv(os.path.join(full_dir, "CIBERSORT_full.csv"))
    pga_full.to_csv(os.path.join(full_dir, "CNV_pct_altered_full.csv"))
    tf_full.to_csv(os.path.join(full_dir, "TF_full.csv"))
    tmb_full.to_csv(os.path.join(full_dir, "TMB_full.csv"))
    hoadley_full.to_csv(os.path.join(full_dir, "Hoadley_full.csv"))
    pathogenic_full.to_csv(os.path.join(full_dir, "Pathogenic_count_full.csv"))
    rna_log2_full.to_csv(os.path.join(full_dir, "RNA_log2_full.csv"))
    gsva_rna_full.to_csv(os.path.join(full_dir, "all_RNA_gsva_scores.csv"))
    gsva_cnv_full.to_csv(os.path.join(full_dir, "all_CNV_gsva_scores.csv"))

    if not arm_binary_full.empty:
        arm_binary_full.to_csv(os.path.join(full_dir, "CNV_binary_arms_full.csv"))
    if not band_binary_full.empty:
        band_binary_full.to_csv(os.path.join(full_dir, "CNV_binary_bands_full.csv"))
    if not estimate_full.empty:
        estimate_full.to_csv(os.path.join(full_dir, "ESTIMATE_full.csv"))

    if not tide_full.empty:
        tide_full.to_csv(os.path.join(full_dir, "TIDE_full.csv"))

    if not ecotyper_cell_states_full.empty:
        ecotyper_cell_states_full.to_csv(os.path.join(full_dir, "ECOTYPER_cell_states_abundance_full.csv"))
    if not ecotyper_cell_assign_full.empty:
        ecotyper_cell_assign_full.to_csv(os.path.join(full_dir, "ECOTYPER_cell_states_assignment_full.csv"))
    if not ecotyper_ecotypes_full.empty:
        ecotyper_ecotypes_full.to_csv(os.path.join(full_dir, "ECOTYPER_ecotypes_abundance_full.csv"))
    if not ecotyper_ecotype_assign_full.empty:
        ecotyper_ecotype_assign_full.to_csv(os.path.join(full_dir, "ECOTYPER_ecotypes_assignment_full.csv"))

    if not ecotyper_full.empty:
        ecotyper_full.to_csv(os.path.join(full_dir, "ECOTYPER_full.csv"))
        if ecotyper_parts_full:
            for name, dfp in ecotyper_parts_full.items():
                if isinstance(dfp, pd.DataFrame) and not dfp.empty:
                    safe = _normalize_colname(name)
                    dfp.to_csv(os.path.join(full_dir, f"ECOTYPER_part__{safe}.csv"))

    # -------------------------------------------------
    # VARIANCE FILTER (computed once on full cohort, label-free)
    # -------------------------------------------------
    variance_bypass_set_full = set()
    for _df in [
        tf_full,
        tmb_full,
        pathogenic_full,
        hoadley_full,
        rna_custom_sets_full,
        cib_full,
        tide_full,
        ecotyper_cell_states_full,
        ecotyper_cell_assign_full,
        ecotyper_ecotypes_full,
        ecotyper_ecotype_assign_full,
        estimate_full,
    ]:
        if isinstance(_df, pd.DataFrame) and not _df.empty:
            variance_bypass_set_full |= set(_df.columns.astype(str).tolist())

    with open(os.path.join(full_dir, "variance_bypass_features.txt"), "w") as f:
        for c in sorted(variance_bypass_set_full):
            f.write(f"{c}\n")

    if variance_keep_file and os.path.exists(variance_keep_file):
        keep_feats = [ln.strip() for ln in open(variance_keep_file, "r") if ln.strip()]
        variance_keep_set_full = set(keep_feats)
        variance_keep_set_full |= variance_bypass_set_full

        print(f"[VAR] Loaded variance keep list from: {variance_keep_file} (n={len(variance_keep_set_full)})")
        var_tbl_full = pd.DataFrame({
            "feature": sorted(list(variance_keep_set_full)),
            "variance": np.nan,
            "n_nonmissing": np.nan,
            "passed_variance": True,
        })
        var_tbl_full["variance_bypassed"] = var_tbl_full["feature"].astype(str).isin(variance_bypass_set_full)

        var_thresholds_by_group = {}
        var_method = "keep_file"
        full_var_groups = {}
    else:
        X_var_full = _concat_nonempty(
            [
                rna_pw_full,
                cnv_pw_full,
                mut_full,
                ptmb_full,
                cib_full,
                pga_full,
                tf_full,
                tmb_full,
                hoadley_full,
                pathogenic_full,
                cnv_gene_sig_full,
                rna_custom_sets_full,
                tide_full,
                ecotyper_cell_states_full,
                ecotyper_cell_assign_full,
                ecotyper_ecotypes_full,
                ecotyper_ecotype_assign_full,
                estimate_full,
                arm_binary_full,
                band_binary_full,
            ],
            name="X_VAR_FULL",
            index=all_ids_full,
        )
        X_var_full = _safe_reindex(X_var_full, all_ids_full)

        rna_pw_cols       = list(rna_pw_full.columns.astype(str))
        rna_cib_cols      = list(cib_full.columns.astype(str))
        rna_hoadley_cols  = list(hoadley_full.columns.astype(str))
        rna_custom_cols   = list(rna_custom_sets_full.columns.astype(str))
        rna_tide_cols     = list(tide_full.columns.astype(str)) if isinstance(tide_full, pd.DataFrame) else []
        rna_ecotyper_cellstate_abund_cols = list(ecotyper_cell_states_full.columns.astype(str)) if isinstance(ecotyper_cell_states_full, pd.DataFrame) else []
        rna_ecotyper_cellstate_assign_cols = list(ecotyper_cell_assign_full.columns.astype(str)) if isinstance(ecotyper_cell_assign_full, pd.DataFrame) else []
        rna_ecotyper_ecotype_abund_cols = list(ecotyper_ecotypes_full.columns.astype(str)) if isinstance(ecotyper_ecotypes_full, pd.DataFrame) else []
        rna_ecotyper_ecotype_assign_cols = list(ecotyper_ecotype_assign_full.columns.astype(str)) if isinstance(ecotyper_ecotype_assign_full, pd.DataFrame) else []
        rna_estimate_cols = list(estimate_full.columns.astype(str))

        cnv_pw_cols    = list(cnv_pw_full.columns.astype(str))
        cnv_gene_cols  = list(cnv_gene_sig_full.columns.astype(str))
        cnv_pga_cols   = list(pga_full.columns.astype(str))
        cnv_tf_cols    = list(tf_full.columns.astype(str))
        cnv_arm_cols   = list(arm_binary_full.columns.astype(str))
        cnv_band_cols  = list(band_binary_full.columns.astype(str))

        mut_path_cols       = list(mut_full.columns.astype(str))
        mut_ptmb_cols       = list(ptmb_full.columns.astype(str))
        mut_tmb_cols        = list(tmb_full.columns.astype(str))
        mut_pathogenic_cols = list(pathogenic_full.columns.astype(str))

        full_var_groups = {
            "RNAblock__PATHWAYS":   rna_pw_cols,
            "RNAblock__CIBERSORT":  rna_cib_cols,
            "RNAblock__HOADLEY":    rna_hoadley_cols,
            "RNAblock__CUSTOM":     rna_custom_cols,
            "RNAblock__TIDE":       rna_tide_cols,
            "RNAblock__ECOTYPER_CELLSTATE_ABUND":  rna_ecotyper_cellstate_abund_cols,
            "RNAblock__ECOTYPER_CELLSTATE_ASSIGN": rna_ecotyper_cellstate_assign_cols,
            "RNAblock__ECOTYPER_ECOTYPE_ABUND":    rna_ecotyper_ecotype_abund_cols,
            "RNAblock__ECOTYPER_ECOTYPE_ASSIGN":   rna_ecotyper_ecotype_assign_cols,
            "RNAblock__ESTIMATE":   rna_estimate_cols,

            "CNVblock__PATHWAYS":    cnv_pw_cols,
            "CNVblock__GENES":       cnv_gene_cols,
            "CNVblock__PCT_ALTERED": cnv_pga_cols,
            "CNVblock__TF":          cnv_tf_cols,
            "CNVblock__ARMS":        cnv_arm_cols,
            "CNVblock__BANDS":       cnv_band_cols,

            "MUTblock__MUTATIONS":  mut_path_cols,
            "MUTblock__PTMB":       mut_ptmb_cols,
            "MUTblock__TMB":        mut_tmb_cols,
            "MUTblock__PATHOGENIC": mut_pathogenic_cols,
        }

        used = set()
        for _, cols in full_var_groups.items():
            used |= set(cols)

        remaining = [c for c in X_var_full.columns.astype(str).tolist()
                     if (c not in used) and (not c.startswith("CLINICALblock__"))]
        if remaining:
            full_var_groups["OTHER__"] = remaining

        if var_keep_top_frac is not None:
            var_method = "top_frac_by_group"
            var_tbl_full, var_thresholds_by_group, variance_keep_set_full = compute_variance_filter_tables_by_group(
                X=X_var_full,
                groups=full_var_groups,
                keep_top_frac=float(var_keep_top_frac),
                var_min=float(var_min),
                min_nonmissing=int(var_min_nonmissing),
            )

            variance_keep_set_full |= variance_bypass_set_full
            if isinstance(var_tbl_full, pd.DataFrame) and "feature" in var_tbl_full.columns:
                var_tbl_full["variance_bypassed"] = var_tbl_full["feature"].astype(str).isin(variance_bypass_set_full)
                var_tbl_full.loc[var_tbl_full["variance_bypassed"], "passed_variance"] = True

            var_tbl_full.to_csv(os.path.join(full_dir, "variance_filter_table_full__by_group.csv"), index=False)

            _dump_json_safely(
                {"var_keep_top_frac": float(var_keep_top_frac), "thresholds_by_group": var_thresholds_by_group},
                os.path.join(full_dir, "variance_thresholds_by_group.json")
            )

            if not var_tbl_full.empty and "variance_group" in var_tbl_full.columns:
                for gname, sub in var_tbl_full.groupby("variance_group"):
                    sub.to_csv(os.path.join(full_dir, f"variance_filter_table_full__{gname}.csv"), index=False)
        else:
            var_method = "absolute_var_min"
            var_tbl_full = compute_variance_filter_table(X_var_full, var_min=var_min, min_nonmissing=var_min_nonmissing)

            if isinstance(var_tbl_full, pd.DataFrame) and "feature" in var_tbl_full.columns:
                var_tbl_full["variance_bypassed"] = var_tbl_full["feature"].astype(str).isin(variance_bypass_set_full)
                var_tbl_full.loc[var_tbl_full["variance_bypassed"], "passed_variance"] = True

            var_tbl_full.to_csv(os.path.join(full_dir, "variance_filter_table_full.csv"), index=False)
            variance_keep_set_full = set(var_tbl_full.loc[var_tbl_full["passed_variance"], "feature"].astype(str).tolist())
            variance_keep_set_full |= variance_bypass_set_full

            var_thresholds_by_group = {}

        lowvar = sorted(set(X_var_full.columns.astype(str)) - variance_keep_set_full)

        with open(os.path.join(full_dir, "variance_keep_features.txt"), "w") as f:
            for x in sorted(variance_keep_set_full):
                f.write(x + "\n")
        with open(os.path.join(full_dir, "variance_lowvar_features.txt"), "w") as f:
            for x in lowvar:
                f.write(x + "\n")

        print(f"[VAR] Full-cohort variance keep: {len(variance_keep_set_full)} / {X_var_full.shape[1]}  (method={var_method})")

    full_config = {
        "input_parameters": run_input_params,
        "derived_run_metadata": {
            "n_full_samples": int(len(all_ids_full)),
            "var_method": var_method,
            "variance_groups_full": {k: len(v) for k, v in (full_var_groups or {}).items()},
            "var_thresholds_by_group": var_thresholds_by_group,
            "n_variance_bypass_features": int(len(variance_bypass_set_full)),

            "n_tide_features": int(tide_full.shape[1]) if isinstance(tide_full, pd.DataFrame) else 0,
            "n_ecotyper_features": int(ecotyper_full.shape[1]) if isinstance(ecotyper_full, pd.DataFrame) else 0,

            "n_ecotyper_cell_states_abundance_full": _nonempty_shape(ecotyper_cell_states_full),
            "n_ecotyper_cell_states_assignment_full": _nonempty_shape(ecotyper_cell_assign_full),
            "n_ecotyper_ecotypes_abundance_full": _nonempty_shape(ecotyper_ecotypes_full),
            "n_ecotyper_ecotypes_assignment_full": _nonempty_shape(ecotyper_ecotype_assign_full),

            "n_arm_binary_features_full": int(arm_binary_full.shape[1]) if isinstance(arm_binary_full, pd.DataFrame) else 0,
            "n_band_binary_features_full": int(band_binary_full.shape[1]) if isinstance(band_binary_full, pd.DataFrame) else 0,
            "n_estimate_features_full": int(estimate_full.shape[1]) if isinstance(estimate_full, pd.DataFrame) else 0,

            "n_rna_candidate_features_full": int(rna_candidates_full.shape[1]),
            "n_cnv_candidate_features_full": int(cnv_candidates_full.shape[1]),
            "n_mut_candidate_features_full": int(mut_candidates_full.shape[1]),
            "n_clinical_features_full": int(clinical_full.shape[1]),

            "n_ptmb_features": int(ptmb_full.shape[1]) if isinstance(ptmb_full, pd.DataFrame) else 0,
            "n_mutated_pathway_features": int(mut_full.shape[1]) if isinstance(mut_full, pd.DataFrame) else 0,
            "n_cnv_gene_features_full": int(cnv_gene_sig_full.shape[1]) if isinstance(cnv_gene_sig_full, pd.DataFrame) else 0,
            "n_rna_custom_features_full": int(rna_custom_sets_full.shape[1]) if isinstance(rna_custom_sets_full, pd.DataFrame) else 0,

            "protein_coding_filter_applied": bool(protein_coding_gtf),
            "ecotyper_tables_found": list(ecotyper_parts_full.keys()) if isinstance(ecotyper_parts_full, dict) else [],
        }
    }

    _dump_json_safely(full_config, os.path.join(full_dir, "preprocessing_full_config.json"))

    # -------------------------------------------------
    # 2) SUBSET Feature Selection (only if enabled)
    # -------------------------------------------------
    if not enable_subset_fs:
        print(f"✅ Preprocessing (full cohort) saved → {full_dir}")
#        return full_dir

    if sample_id_subset is None:
        subset_ids = meta.index.astype(str).str.strip().drop_duplicates()
        print(f"[FS] enable_subset_fs=True with no subset provided → using FULL cohort (n={len(subset_ids)})")
    else:
        subset_ids = pd.Index(sample_id_subset).astype(str).str.strip().drop_duplicates()
        subset_ids = meta.index.intersection(subset_ids)

    if len(subset_ids) == 0:
        raise ValueError("Provided subset is empty after alignment to meta.index.")

    y_subset = meta.loc[subset_ids, status_col].astype(str).str.strip().str.lower()
    if y_subset.isna().any() or (y_subset == "").any():
        bad = list(subset_ids[y_subset.isna() | (y_subset == "")])
        raise ValueError(f"{len(bad)} subset sample(s) missing {status_col}: e.g., {bad[:5]}")

    pd.DataFrame({"sample_id": subset_ids}).to_csv(os.path.join(fs_dir, "subset_ids.csv"), index=False)

    mut_sub        = mut_full.loc[subset_ids] if (isinstance(mut_full, pd.DataFrame) and not mut_full.empty) else pd.DataFrame(index=subset_ids)
    ptmb_sub       = ptmb_full.loc[subset_ids] if (isinstance(ptmb_full, pd.DataFrame) and not ptmb_full.empty) else pd.DataFrame(index=subset_ids)

    rna_pw_sub       = rna_pw_full.loc[subset_ids]
    cnv_pw_sub       = cnv_pw_full.loc[subset_ids]
    cib_sub          = cib_full.loc[subset_ids]
    pga_sub          = pga_full.loc[subset_ids]
    tf_sub           = tf_full.loc[subset_ids]
    cnv_gene_sig_sub = cnv_gene_sig_full.loc[subset_ids] if not cnv_gene_sig_full.empty else pd.DataFrame(index=subset_ids)

    arm_binary_sub = arm_binary_full.loc[subset_ids] if not arm_binary_full.empty else pd.DataFrame(index=subset_ids)
    band_binary_sub = band_binary_full.loc[subset_ids] if not band_binary_full.empty else pd.DataFrame(index=subset_ids)
    estimate_sub = estimate_full.loc[subset_ids] if not estimate_full.empty else pd.DataFrame(index=subset_ids)

    tmb_sub        = tmb_full.loc[subset_ids]
    hoadley_sub    = hoadley_full.loc[subset_ids]
    pathogenic_sub = pathogenic_full.loc[subset_ids]
    clinical_sub   = clinical_full.loc[subset_ids]
    rna_log2_sub   = rna_log2_full.loc[subset_ids]
    rna_custom_sub = rna_custom_sets_full.loc[subset_ids] if not rna_custom_sets_full.empty else pd.DataFrame(index=subset_ids)

    tide_sub = tide_full.loc[subset_ids] if (isinstance(tide_full, pd.DataFrame) and not tide_full.empty) else pd.DataFrame(index=subset_ids)

    ecotyper_cell_states_sub = ecotyper_cell_states_full.loc[subset_ids] if not ecotyper_cell_states_full.empty else pd.DataFrame(index=subset_ids)
    ecotyper_cell_assign_sub = ecotyper_cell_assign_full.loc[subset_ids] if not ecotyper_cell_assign_full.empty else pd.DataFrame(index=subset_ids)
    ecotyper_ecotypes_sub = ecotyper_ecotypes_full.loc[subset_ids] if not ecotyper_ecotypes_full.empty else pd.DataFrame(index=subset_ids)
    ecotyper_ecotype_assign_sub = ecotyper_ecotype_assign_full.loc[subset_ids] if not ecotyper_ecotype_assign_full.empty else pd.DataFrame(index=subset_ids)

    ecotyper_sub = _concat_nonempty(
        [
            ecotyper_cell_states_sub,
            ecotyper_cell_assign_sub,
            ecotyper_ecotypes_sub,
            ecotyper_ecotype_assign_sub,
        ],
        name="ECOTYPER_SUB",
        index=subset_ids,
    )

    # ========== limma DE on subset ==========

    # defaults so downstream code is safe even when bypassing
    de_genes = []
    limma_p_map = {}
    limma_q_map = {}
    de_expr_sub_prefixed = pd.DataFrame(index=subset_ids)

    # ========== limma DE on subset ==========
    if RNAgene_bypass:
        print("[FS] RNAgene-bypass=True -> skipping RNA gene-level stats/significance block.")
    else:
        subset_meta_for_limma = meta.loc[subset_ids].copy()
        subset_meta_for_limma["Disease_Status"] = pd.Categorical(
            subset_meta_for_limma["Disease_Status"], categories=["non-recurrent", "recurrent"]
        )

        _is_combined = str(subtype).lower() in {"all", "luminal"}
        limma_subtype_arg = None if _is_combined else subtype
        limma_subtype_col = None if _is_combined else "pam50_subtype"

        sig = process_limma_results(
            rna_expression=_safe_reindex(data["rna_raw"], subset_ids).fillna(0),
            metadata=subset_meta_for_limma,
            design_factors=["Disease_Status"],
            contrast=("Disease_Status", "recurrent", "non-recurrent"),
            pvalue_threshold=limma_pvalue_threshold,
            padj_threshold=limma_padj_threshold,
            logfold_threshold=limma_logfc_threshold,
            base_mean_threshold=50,
            gene_sum_threshold=50,
            id_col="index",
            subtype=limma_subtype_arg,
            subtype_col=limma_subtype_col,
            r_exe="/app/software/R/4.3.1-gfbf-2022b/bin/Rscript",
            output_dir=fs_dir
        )

        if isinstance(sig, pd.DataFrame) and not sig.empty and "gene" in sig.columns:
            de_genes = [g for g in sig["gene"].astype(str).tolist() if g in rna_log2_full.columns]
            sig_idx = sig.set_index("gene")
            low = {c.lower(): c for c in sig.columns}
            col_p = low.get("pvalue", None) or low.get("p_val", None) or low.get("p.value", None) or low.get("p", None)
            col_q = low.get("padj", None) or low.get("adj.p.val", None) or low.get("fdr", None) or low.get("qvalue", None)
            if col_p:
                limma_p_map = {f"RNAblock__{g}": float(sig_idx.at[g, col_p]) for g in de_genes if g in sig_idx.index}
            if col_q:
                limma_q_map = {f"RNAblock__{g}": float(sig_idx.at[g, col_q]) for g in de_genes if g in sig_idx.index}

            if pc_genes_upper is not None and de_genes:
                de_genes_pc = [g for g in de_genes if str(g).upper() in pc_genes_upper]
                removed_de = sorted(set(de_genes) - set(de_genes_pc))
                de_genes = de_genes_pc

                limma_p_map = {
                    k: v for k, v in limma_p_map.items()
                    if str(k).replace("RNAblock__", "").upper() in pc_genes_upper
                }
                limma_q_map = {
                    k: v for k, v in limma_q_map.items()
                    if str(k).replace("RNAblock__", "").upper() in pc_genes_upper
                }

                with open(os.path.join(fs_dir, "protein_coding_removed_RNA_DE_genes.txt"), "w") as f:
                    for g in removed_de:
                        f.write(str(g) + "\n")
                print(f"[PC] RNA DE genes: removed non-protein-coding genes: n={len(removed_de)}")

            if de_genes:
                de_expr_sub_prefixed = _add_block_prefix(rna_log2_sub[de_genes], "RNAblock__")

            sig.to_csv(os.path.join(fs_dir, "limma_DE_results.csv"), index=False)


    # ========== candidate blocks ==========
    rna_block_sub = _concat_nonempty(
        [
            rna_pw_sub,
            de_expr_sub_prefixed,
            hoadley_sub,
            cib_sub,
            rna_custom_sub,
            tide_sub,
            ecotyper_cell_states_sub,
            ecotyper_cell_assign_sub,
            ecotyper_ecotypes_sub,
            ecotyper_ecotype_assign_sub,
            estimate_sub,
        ],
        name="RNA_BLOCK_SUB",
        index=subset_ids,
    )

    cnv_block_sub = _concat_nonempty(
        [cnv_pw_sub, pga_sub, tf_sub, cnv_gene_sig_sub, arm_binary_sub, band_binary_sub],
        name="CNV_BLOCK_SUB",
        index=subset_ids,
    )

    mut_parts = []
    if include_mutated_pathways and isinstance(mut_sub, pd.DataFrame) and mut_sub.shape[1] > 0:
        mut_parts.append(mut_sub)
    if enable_ptmb and isinstance(ptmb_sub, pd.DataFrame) and ptmb_sub.shape[1] > 0:
        mut_parts.append(ptmb_sub)
    mut_parts.extend([tmb_sub, pathogenic_sub])
    mut_block_sub = _concat_nonempty(mut_parts, name="MUT_BLOCK_SUB", index=subset_ids) if mut_parts else pd.DataFrame(index=subset_ids)

    X_candidates_sub = _concat_nonempty(
        [rna_block_sub, cnv_block_sub, mut_block_sub],
        name="X_CANDIDATES_SUB",
        index=subset_ids,
    )
    X_candidates_sub = _safe_reindex(X_candidates_sub, subset_ids)
    all_features_nonclin = list(X_candidates_sub.columns.astype(str))

    # -------------------------
    # Apply variance filter
    # -------------------------
    var_tbl_subset_missing = pd.DataFrame(
        columns=["feature", "variance", "n_nonmissing", "passed_variance", "variance_source", "variance_group", "variance_threshold"]
    )

    missing = [f for f in all_features_nonclin if f not in variance_keep_set_full]
    passed_subset = set()

    if len(missing) > 0:
        if var_keep_top_frac is None:
            vt = compute_variance_filter_table(X_candidates_sub[missing], var_min=var_min, min_nonmissing=var_min_nonmissing)
            vt["variance_source"] = "subset"
            var_tbl_subset_missing = vt.copy()
            passed_subset = set(vt.loc[vt["passed_variance"], "feature"].astype(str).tolist())
        else:
            miss_groups: Dict[str, List[str]] = {}

            de_cols = set(de_expr_sub_prefixed.columns.astype(str)) if isinstance(de_expr_sub_prefixed, pd.DataFrame) else set()
            for f in missing:
                fs = str(f)
                if fs in de_cols:
                    miss_groups.setdefault("RNAblock__DE", []).append(fs)
                elif fs.startswith("RNAblock__ESTIMATE__"):
                    miss_groups.setdefault("RNAblock__ESTIMATE", []).append(fs)
                elif fs.startswith("RNAblock__TIDE__"):
                    miss_groups.setdefault("RNAblock__TIDE", []).append(fs)
                elif fs.startswith("RNAblock__ECOTYPER__CELLSTATE_ABUND__"):
                    miss_groups.setdefault("RNAblock__ECOTYPER_CELLSTATE_ABUND", []).append(fs)
                elif fs.startswith("RNAblock__ECOTYPER__CELLSTATE_ASSIGN__"):
                    miss_groups.setdefault("RNAblock__ECOTYPER_CELLSTATE_ASSIGN", []).append(fs)
                elif fs.startswith("RNAblock__ECOTYPER__ECOTYPE_ABUND__"):
                    miss_groups.setdefault("RNAblock__ECOTYPER_ECOTYPE_ABUND", []).append(fs)
                elif fs.startswith("RNAblock__ECOTYPER__ECOTYPE_ASSIGN__"):
                    miss_groups.setdefault("RNAblock__ECOTYPER_ECOTYPE_ASSIGN", []).append(fs)
                elif fs.startswith("RNAblock__"):
                    miss_groups.setdefault("RNAblock__OTHER", []).append(fs)
                elif fs.startswith("CNVblock__ARM__"):
                    miss_groups.setdefault("CNVblock__ARMS", []).append(fs)
                elif fs.startswith("CNVblock__BAND__"):
                    miss_groups.setdefault("CNVblock__BANDS", []).append(fs)
                elif fs.startswith("CNVblock__"):
                    miss_groups.setdefault("CNVblock__", []).append(fs)
                elif fs.startswith("MUTblock__pTMB__"):
                    miss_groups.setdefault("MUTblock__PTMB", []).append(fs)
                elif fs.startswith("MUTblock__"):
                    miss_groups.setdefault("MUTblock__", []).append(fs)
                else:
                    miss_groups.setdefault("OTHER__", []).append(fs)

            tables = []
            for gname, feats in miss_groups.items():
                feats = [c for c in feats if c in X_candidates_sub.columns]
                if not feats:
                    continue

                Xt = X_candidates_sub[feats]
                Xn = _to_num_df(Xt)
                n_nonmissing = Xn.notna().sum(axis=0).astype(int)
                var = Xn.var(axis=0, ddof=1, skipna=True)

                tbl = pd.DataFrame({
                    "feature": var.index.astype(str),
                    "variance": var.values.astype(float),
                    "n_nonmissing": n_nonmissing.values.astype(int),
                })
                tbl["variance_source"] = "subset"
                tbl["variance_group"] = gname

                base_ok = np.isfinite(tbl["variance"].values) & (tbl["n_nonmissing"].values >= int(var_min_nonmissing)) & (tbl["variance"].values >= float(var_min))

                thr = var_thresholds_by_group.get(gname, np.nan)
                tbl["variance_threshold"] = thr

                if np.isfinite(thr):
                    passed = base_ok & (tbl["variance"].values >= float(thr))
                else:
                    eligible_var = tbl.loc[base_ok, "variance"].values.astype(float)
                    if eligible_var.size == 0:
                        passed = np.zeros(len(tbl), dtype=bool)
                        thr2 = np.inf
                    else:
                        q = 1.0 - float(var_keep_top_frac)
                        thr2 = float(np.nanquantile(eligible_var, q))
                        passed = base_ok & (tbl["variance"].values >= thr2)
                    tbl["variance_threshold"] = thr2

                tbl["passed_variance"] = passed.astype(bool)
                tables.append(tbl)

            var_tbl_subset_missing = pd.concat(tables, axis=0, ignore_index=True) if tables else var_tbl_subset_missing
            passed_subset = set(var_tbl_subset_missing.loc[var_tbl_subset_missing["passed_variance"], "feature"].astype(str).tolist())

    variance_pass_set = set([f for f in all_features_nonclin if f in variance_keep_set_full]) | passed_subset

    feat_tbl = pd.DataFrame({"feature": all_features_nonclin})
    feat_tbl["block_prefix"] = feat_tbl["feature"].apply(
        lambda c: "RNAblock__" if c.startswith("RNAblock__")
        else "CNVblock__" if c.startswith("CNVblock__")
        else "MUTblock__" if c.startswith("MUTblock__")
        else "OTHER__"
    )
    feat_tbl["low_variance_filtered"] = ~feat_tbl["feature"].isin(variance_pass_set)
    feat_tbl["excluded_reason"] = ""
    feat_tbl.loc[feat_tbl["low_variance_filtered"], "excluded_reason"] = "variance_prefilter"

    var_full_map = {}
    if isinstance(var_tbl_full, pd.DataFrame) and "feature" in var_tbl_full.columns:
        for _, r in var_tbl_full.iterrows():
            var_full_map[str(r["feature"])] = (r.get("variance", np.nan), r.get("n_nonmissing", np.nan), True)

    feat_tbl["variance"] = np.nan
    feat_tbl["var_n_nonmissing"] = np.nan
    feat_tbl["variance_source"] = "full"

    for f in feat_tbl["feature"].astype(str).tolist():
        if f in var_full_map:
            v, n, _ = var_full_map[f]
            feat_tbl.loc[feat_tbl["feature"] == f, "variance"] = v
            feat_tbl.loc[feat_tbl["feature"] == f, "var_n_nonmissing"] = n
            feat_tbl.loc[feat_tbl["feature"] == f, "variance_source"] = "full"

    if not var_tbl_subset_missing.empty:
        tmp = var_tbl_subset_missing.rename(columns={"n_nonmissing": "var_n_nonmissing"})
        for col in ["variance", "var_n_nonmissing", "variance_source", "passed_variance", "variance_group", "variance_threshold"]:
            if col in feat_tbl.columns:
                feat_tbl = feat_tbl.drop(columns=[col], errors="ignore")
        feat_tbl = feat_tbl.merge(
            tmp[["feature", "variance", "var_n_nonmissing", "variance_source", "passed_variance", "variance_group", "variance_threshold"]],
            on="feature",
            how="left"
        )
        feat_tbl["variance_source"] = feat_tbl["variance_source"].fillna("full")

    # -------------------------
    # Label-based correlation prefilter
    # -------------------------
    eligible_for_corr = feat_tbl.loc[~feat_tbl["low_variance_filtered"], "feature"].astype(str).tolist()
    corr_pass_set = set(eligible_for_corr)

    if corr_prefilter and len(eligible_for_corr) > 0:
        corr_tbl = corr_prefilter_table(
            X=X_candidates_sub[eligible_for_corr],
            y_str=meta.loc[subset_ids, status_col],
            case_label=mwu_case_label,
            ctrl_label=mwu_ctrl_label,
            abs_min=corr_abs_min,
            min_n=int(corr_min_n),
        )
        corr_tbl.to_csv(os.path.join(fs_dir, "corr_prefilter_table.csv"), index=False)
        corr_pass_set = set(corr_tbl.loc[corr_tbl["passed_corr"], "feature"].astype(str).tolist())

        for col in ["corr_r", "corr_abs", "corr_n", "passed_corr"]:
            if col in feat_tbl.columns:
                feat_tbl = feat_tbl.drop(columns=[col])

        feat_tbl = feat_tbl.merge(
            corr_tbl[["feature", "corr_r", "corr_abs", "corr_n", "passed_corr"]],
            on="feature",
            how="left"
        )
        feat_tbl["passed_corr"] = feat_tbl["passed_corr"].fillna(False).astype(bool)

        mask_corr_excl = (~feat_tbl["feature"].isin(corr_pass_set)) & (~feat_tbl["low_variance_filtered"])
        feat_tbl.loc[mask_corr_excl, "excluded_reason"] = "corr_prefilter"
    else:
        feat_tbl["corr_r"] = np.nan
        feat_tbl["corr_abs"] = np.nan
        feat_tbl["corr_n"] = np.nan
        feat_tbl["passed_corr"] = ~feat_tbl["low_variance_filtered"]

    feat_tbl["tested"] = (~feat_tbl["low_variance_filtered"]) & (feat_tbl["passed_corr"])

    test_features = feat_tbl.loc[feat_tbl["tested"], "feature"].astype(str).tolist()
    X_test = X_candidates_sub[test_features] if len(test_features) > 0 else pd.DataFrame(index=subset_ids)

    for c in ["group_p", "group_q", "cox_uni_p", "cox_uni_q", "cox_uni_hr", "cox_uni_coef",
              "cox_multi_p", "cox_multi_q", "cox_multi_hr", "cox_multi_coef",
              "limma_p", "limma_q"]:
        feat_tbl[c] = np.nan
    feat_tbl["group_test"] = np.nan

    if limma_p_map:
        feat_tbl.loc[feat_tbl["feature"].isin(limma_p_map.keys()), "limma_p"] = feat_tbl["feature"].map(limma_p_map)
    if limma_q_map:
        feat_tbl.loc[feat_tbl["feature"].isin(limma_q_map.keys()), "limma_q"] = feat_tbl["feature"].map(limma_q_map)

    if run_mwu and X_test.shape[1] > 0:
        binary_cols, nonbinary_cols = _split_binary_features(X_test)
        res_parts = []

        if nonbinary_cols:
            mwu_res = mwu_table(
                X_test[nonbinary_cols],
                y_subset,
                case_label=mwu_case_label,
                ctrl_label=mwu_ctrl_label
            ).rename(columns={"mwu_p": "group_p", "mwu_q": "group_q"}).copy()
            mwu_res["group_test"] = "mwu"
            res_parts.append(mwu_res[["feature", "group_p", "group_q", "group_test"]])

        if binary_cols:
            fisher_res = _fisher_binary_table(
                X_test[binary_cols],
                y_subset,
                case_label=mwu_case_label,
                ctrl_label=mwu_ctrl_label
            ).copy()
            res_parts.append(fisher_res[["feature", "group_p", "group_q", "group_test"]])

        if res_parts:
            group_res = pd.concat(res_parts, axis=0, ignore_index=True)
            group_res = group_res.drop_duplicates(subset=["feature"], keep="first")
            group_res.to_csv(os.path.join(fs_dir, "group_comparison_results.csv"), index=False)

            feat_tbl = feat_tbl.drop(columns=["group_p", "group_q", "group_test"], errors="ignore")
            feat_tbl = feat_tbl.merge(
                group_res[["feature", "group_p", "group_q", "group_test"]],
                on="feature",
                how="left"
            )

    if run_cox_uni:
        if not cox_time_col or not cox_event_col:
            raise ValueError("Cox requested but --cox_time_col/--cox_event_col not provided.")
        if cox_time_col not in meta.columns or cox_event_col not in meta.columns:
            raise KeyError(f"Cox columns not found in meta: time={cox_time_col}, event={cox_event_col}")
        if X_test.shape[1] > 0:
            cox_uni = cox_univariate_table(
                X_test, meta.loc[subset_ids], time_col=cox_time_col, event_col=cox_event_col,
                min_n=cox_min_n, min_events=cox_min_events,
                case_label=mwu_case_label, ctrl_label=mwu_ctrl_label
            )
            cox_uni.to_csv(os.path.join(fs_dir, "COX_uni_results.csv"), index=False)
            feat_tbl = _merge_on_feature_no_suffix(feat_tbl, cox_uni, cols=["cox_uni_p", "cox_uni_q", "cox_uni_coef", "cox_uni_hr"])

    if run_cox_multi:
        if not cox_time_col or not cox_event_col:
            raise ValueError("Cox requested but --cox_time_col/--cox_event_col not provided.")
        if cox_time_col not in meta.columns or cox_event_col not in meta.columns:
            raise KeyError(f"Cox columns not found in meta: time={cox_time_col}, event={cox_event_col}")
        covs = cox_covariates or []
        if X_test.shape[1] > 0:
            cox_multi_feat, cox_multi_terms = cox_multivariable_tables(
                X_test, meta.loc[subset_ids], time_col=cox_time_col, event_col=cox_event_col,
                covariates=covs, min_n=cox_min_n, min_events=cox_min_events,
                case_label=mwu_case_label, ctrl_label=mwu_ctrl_label
            )
            cox_multi_feat.to_csv(os.path.join(fs_dir, "COX_multi_feature_results.csv"), index=False)
            cox_multi_terms.to_csv(os.path.join(fs_dir, "COX_multi_terms_long.csv"), index=False)
            feat_tbl = _merge_on_feature_no_suffix(feat_tbl, cox_multi_feat, cols=["cox_multi_p", "cox_multi_q", "cox_multi_coef", "cox_multi_hr"])

    feat_tbl["pass_group"] = False
    feat_tbl["pass_cox_uni"] = False
    feat_tbl["pass_cox_multi"] = False
    feat_tbl["pass_limma"] = False

    if run_mwu:
        feat_tbl["pass_group"] = (feat_tbl["group_p"] <= float(mwu_p_thresh)) & (feat_tbl["group_q"] <= float(mwu_fdr_thresh))
        feat_tbl["pass_group"] = feat_tbl["pass_group"].fillna(False)

    if run_cox_uni:
        hr_ok = _hr_direction_mask(feat_tbl["cox_uni_hr"].astype(float), cox_hr_direction)
        mag_ok = _hr_magnitude_mask(feat_tbl["cox_uni_coef"].astype(float), cox_hr_abs_min)
        feat_tbl["pass_cox_uni"] = (
            (feat_tbl["cox_uni_p"] <= float(cox_uni_p_thresh)) &
            (feat_tbl["cox_uni_q"] <= float(cox_uni_fdr_thresh)) &
            hr_ok & mag_ok
        )
        feat_tbl["pass_cox_uni"] = feat_tbl["pass_cox_uni"].fillna(False)

    if run_cox_multi:
        hr_ok = _hr_direction_mask(feat_tbl["cox_multi_hr"].astype(float), cox_hr_direction)
        mag_ok = _hr_magnitude_mask(feat_tbl["cox_multi_coef"].astype(float), cox_hr_abs_min)
        feat_tbl["pass_cox_multi"] = (
            (feat_tbl["cox_multi_p"] <= float(cox_multi_p_thresh)) &
            (feat_tbl["cox_multi_q"] <= float(cox_multi_fdr_thresh)) &
            hr_ok & mag_ok
        )
        feat_tbl["pass_cox_multi"] = feat_tbl["pass_cox_multi"].fillna(False)

    feat_tbl["pass_limma"] = feat_tbl["feature"].isin(limma_p_map.keys()) if limma_p_map else False

    feat_tbl.loc[feat_tbl["low_variance_filtered"], ["pass_group", "pass_cox_uni", "pass_cox_multi", "pass_limma"]] = False
    feat_tbl.loc[feat_tbl["excluded_reason"].eq("corr_prefilter"), ["pass_group", "pass_cox_uni", "pass_cox_multi", "pass_limma"]] = False

    feat_tbl["selected_any"] = feat_tbl["pass_group"] | feat_tbl["pass_cox_uni"] | feat_tbl["pass_cox_multi"] | feat_tbl["pass_limma"]

    clinical_cols = list(clinical_sub.columns.astype(str))
    clinical_tbl = pd.DataFrame({"feature": clinical_cols})
    clinical_tbl["block_prefix"] = "CLINICALblock__"
    clinical_tbl["low_variance_filtered"] = False
    clinical_tbl["variance"] = np.nan
    clinical_tbl["var_n_nonmissing"] = np.nan
    clinical_tbl["variance_source"] = "n/a"
    clinical_tbl["corr_r"] = np.nan
    clinical_tbl["corr_abs"] = np.nan
    clinical_tbl["corr_n"] = np.nan
    clinical_tbl["passed_corr"] = True
    clinical_tbl["tested"] = False
    clinical_tbl["excluded_reason"] = "clinical_unconditionally_kept"
    for c in ["group_p", "group_q", "cox_uni_p", "cox_uni_q", "cox_uni_hr", "cox_uni_coef",
              "cox_multi_p", "cox_multi_q", "cox_multi_hr", "cox_multi_coef", "limma_p", "limma_q"]:
        clinical_tbl[c] = np.nan
    clinical_tbl["group_test"] = np.nan
    for c in ["pass_group", "pass_cox_uni", "pass_cox_multi", "pass_limma"]:
        clinical_tbl[c] = False
    clinical_tbl["selected_any"] = True

    all_tbl = pd.concat([feat_tbl, clinical_tbl], axis=0, ignore_index=True)
    all_tbl.to_csv(os.path.join(fs_dir, "ALL_FEATURES_selection_table.csv"), index=False)

    # -------------------------------------------------
    # Correlation redundancy pruning (NON-CLINICAL ONLY)
    # -------------------------------------------------
    selected_nonclin = all_tbl[(all_tbl["block_prefix"] != "CLINICALblock__") & (all_tbl["selected_any"])]
    keep_cols_nonclin = selected_nonclin["feature"].astype(str).tolist()

    X_all_sub_numeric = _to_num_df(X_candidates_sub).fillna(0.0)
    X_pre = X_all_sub_numeric[keep_cols_nonclin] if keep_cols_nonclin else pd.DataFrame(index=subset_ids)

    # pcols = []
    # if run_cox_multi:
    #     pcols.append("cox_multi_p")

    # def _min_p(row):
    #     vals = [row.get(c, np.nan) for c in pcols]
    #     vals = [v for v in vals if np.isfinite(v)]
    #     return min(vals) if vals else np.nan

    # p_prune_tbl = all_tbl[all_tbl["feature"].isin(keep_cols_nonclin)].copy()
    # p_prune_tbl["p_prune"] = p_prune_tbl.apply(_min_p, axis=1)
    # p_prune_tbl = p_prune_tbl[["feature", "p_prune"]].dropna(subset=["p_prune"]).copy()

    def _prune_priority(row, mode: str = "hierarchical"):
        """
        Determine which p-value to use for correlation-pruning tie-breaking.

        Modes
        -----
        hierarchical : cox_multi_p -> cox_uni_p -> group_p
        cox_multi    : cox_multi_p only
        cox_uni      : cox_uni_p only
        group        : group_p only   (covers MWU or Fisher depending on feature type)
        limma        : limma_p only

        Returns
        -------
        Series with:
          p_prune
          p_prune_source
          p_prune_rank
        """
        mode = str(mode).strip().lower()
        if mode == "mwu":
            mode = "group"

        if mode == "hierarchical":
            candidates = [
                ("cox_multi_p", 1),
                ("cox_uni_p",   2),
                ("group_p",     3),
            ]
        elif mode == "cox_multi":
            candidates = [("cox_multi_p", 1)]
        elif mode == "cox_uni":
            candidates = [("cox_uni_p", 1)]
        elif mode == "group":
            candidates = [("group_p", 1)]
        elif mode == "limma":
            candidates = [("limma_p", 1)]
        else:
            raise ValueError(f"Unsupported corr_prune_sig_mode: {mode}")

        for col, rank in candidates:
            v = row.get(col, np.nan)
            if pd.notna(v) and np.isfinite(v):
                return pd.Series(
                    [float(v), str(col), int(rank)],
                    index=["p_prune", "p_prune_source", "p_prune_rank"]
                )

        return pd.Series(
            [np.nan, np.nan, np.nan],
            index=["p_prune", "p_prune_source", "p_prune_rank"]
        )


    p_prune_tbl = all_tbl[all_tbl["feature"].isin(keep_cols_nonclin)].copy()
    # p_prune_tbl[["p_prune", "p_prune_source", "p_prune_rank"]] = p_prune_tbl.apply(_prune_priority, axis=1)
    p_prune_tbl[["p_prune", "p_prune_source", "p_prune_rank"]] = p_prune_tbl.apply(
    lambda row: _prune_priority(row, mode=corr_prune_sig_mode),
    axis=1
    )
    p_prune_tbl = p_prune_tbl[["feature", "p_prune", "p_prune_source", "p_prune_rank"]].dropna(subset=["p_prune"]).copy()

    groups = _group_by_prefix(list(X_pre.columns))

    def _as_min_dict(x):
        if isinstance(x, dict):
            return {str(k): int(v) for k, v in x.items()}
        return {p: int(x) for p in ["RNAblock__", "CNVblock__", "MUTblock__", "CLINICALblock__"]}

    min_per = _as_min_dict(fs_min_per_block)

    kept_total = []
    drop_tables = []
    kept_per_prefix = {}

    for prefix, cols in groups.items():
        if prefix == "CLINICALblock__":
            continue
        if len(cols) == 0:
            kept_per_prefix[prefix] = []
            continue

        Xg = X_pre[cols]
        # pv = p_prune_tbl[p_prune_tbl["feature"].isin(cols)].copy()
        # kept_cols, drops_df = prune_by_correlation(Xg, pv, corr_threshold=corr_threshold)
        pv = p_prune_tbl[p_prune_tbl["feature"].isin(cols)].copy()

        if pv.empty:
            print(f"[PRUNE] {prefix}: no valid prune p-values; keeping all {len(cols)} features.")
            kept_cols = list(cols)
            drops_df = pd.DataFrame(columns=["feature", "dropped_because_of", "abs_r", "p_self", "p_other"])
        else:
            pv = pv.sort_values(["p_prune_rank", "p_prune"], ascending=[True, True])
            kept_cols, drops_df = prune_by_correlation(Xg, pv, corr_threshold=corr_threshold)

        want = int(min_per.get(prefix, 0))
        if want > 0 and len(kept_cols) < want:
            candidates = pv.dropna(subset=["p_prune"]).sort_values("p_prune")
            already = set(kept_cols)
            extras = []
            for f in candidates["feature"].astype(str).tolist():
                if f in already:
                    continue
                extras.append(f)
                if len(kept_cols) + len(extras) >= want:
                    break
            kept_cols = kept_cols + extras

        kept_per_prefix[prefix] = kept_cols
        kept_total.extend(kept_cols)

        if not drops_df.empty:
            drops_df["group_prefix"] = prefix
            drop_tables.append(drops_df)

        if kept_cols:
            block_df = None
            if prefix == "RNAblock__":
                block_df = rna_block_sub
            elif prefix == "CNVblock__":
                block_df = cnv_block_sub
            elif prefix == "MUTblock__":
                block_df = mut_block_sub

            if block_df is not None:
                _safe_reindex(block_df[kept_cols], subset_ids).to_csv(
                    os.path.join(fs_dir, f"{prefix.strip('_')}_selected_post_redundancy.csv")
                )

        if Xg.shape[1] >= 2:
            Xg.corr().to_csv(os.path.join(fs_dir, f"CORR_{prefix.strip('_')}.csv"))

    _safe_reindex(clinical_sub, subset_ids).to_csv(os.path.join(fs_dir, "CLINICALblock_selected_post_redundancy.csv"))
    kept_per_prefix["CLINICALblock__"] = clinical_cols
    kept_total.extend(clinical_cols)

    drops_all = pd.DataFrame(columns=["feature", "dropped_because_of", "abs_r", "p_self", "p_other", "group_prefix"])
    if drop_tables:
        drops_all = pd.concat(drop_tables, axis=0, ignore_index=True)
    drops_all.to_csv(os.path.join(fs_dir, "redundancy_drops.csv"), index=False)

    final_cols = sorted(set(kept_total))

    all_tbl["kept_after_corr"] = all_tbl["feature"].isin(final_cols)
    all_tbl["dropped_due_to_corr"] = all_tbl["feature"].isin(set(drops_all["feature"].astype(str).tolist())) if not drops_all.empty else False
    all_tbl.to_csv(os.path.join(fs_dir, "FINAL_feature_list.csv"), index=False)
    pd.Series(final_cols, name="feature").to_csv(os.path.join(fs_dir, "FINAL_features_only.csv"), index=False)

    fs_config = {
        "input_parameters": run_input_params,
        "derived_run_metadata": {
            "n_subset_samples": int(len(subset_ids)),
            "subset_run_name": subset_run_name if subset_run_name else "fullcohort",

            "var_method": var_method,
            "var_thresholds_by_group": var_thresholds_by_group,
            "n_variance_bypass_features": int(len(variance_bypass_set_full)),

            "n_tide_features_in_subset": int(tide_sub.shape[1]) if isinstance(tide_sub, pd.DataFrame) else 0,
            "n_ecotyper_features_in_subset": int(ecotyper_sub.shape[1]) if isinstance(ecotyper_sub, pd.DataFrame) else 0,

            "n_ecotyper_cell_states_abundance_in_subset": _nonempty_shape(ecotyper_cell_states_sub),
            "n_ecotyper_cell_states_assignment_in_subset": _nonempty_shape(ecotyper_cell_assign_sub),
            "n_ecotyper_ecotypes_abundance_in_subset": _nonempty_shape(ecotyper_ecotypes_sub),
            "n_ecotyper_ecotypes_assignment_in_subset": _nonempty_shape(ecotyper_ecotype_assign_sub),

            "n_arm_binary_features_in_subset": int(arm_binary_sub.shape[1]) if isinstance(arm_binary_sub, pd.DataFrame) else 0,
            "n_band_binary_features_in_subset": int(band_binary_sub.shape[1]) if isinstance(band_binary_sub, pd.DataFrame) else 0,
            "n_estimate_features_in_subset": int(estimate_sub.shape[1]) if isinstance(estimate_sub, pd.DataFrame) else 0,

            "n_rna_candidate_features_subset": int(rna_block_sub.shape[1]),
            "n_cnv_candidate_features_subset": int(cnv_block_sub.shape[1]),
            "n_mut_candidate_features_subset": int(mut_block_sub.shape[1]),
            "n_clinical_features_subset": int(clinical_sub.shape[1]),

            "n_tested_features_nonclinical": int(len(test_features)),
            "n_selected_any_nonclinical": int(selected_nonclin.shape[0]),
            "n_final_features_total": int(len(final_cols)),
            "n_final_features_rna": int(len(kept_per_prefix.get("RNAblock__", []))),
            "n_final_features_cnv": int(len(kept_per_prefix.get("CNVblock__", []))),
            "n_final_features_mut": int(len(kept_per_prefix.get("MUTblock__", []))),
            "n_final_features_clinical": int(len(kept_per_prefix.get("CLINICALblock__", []))),

            "n_redundancy_drops": int(drops_all.shape[0]) if isinstance(drops_all, pd.DataFrame) else 0,
        }
    }

    _dump_json_safely(fs_config, os.path.join(fs_dir, "feature_selection_config.json"))

    print(f"✅ Full preprocessing → {full_dir}")
    print(f"✅ Subset FS → {fs_dir}")
    
    return fs_dir



# -------------------------
# CLI
# -------------------------

def _split_list_arg(x: Optional[str]) -> Optional[List[str]]:
    if x is None:
        return None
    x = str(x).strip()
    if not x:
        return []
    return [p.strip() for p in x.split(",") if p.strip()]

def main(argv=None):
    class _ArgHelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                            argparse.RawTextHelpFormatter):
    
        pass

    parser = argparse.ArgumentParser(
        description=(
            "Feature-construction and feature-selection workflow.\n\n"
            "Builds subtype-specific RNA, CNV, MUT, CLINICAL, immune/TME, TIDE, "
            "EcoTyper, custom gene-set, and pTMB candidate features; applies "
            "label-free filters; runs supervised feature-selection tests; performs "
            "redundancy pruning; and writes selected block matrices for downstream "
            "training/CV/final-model scripts."
        ),
        allow_abbrev=False,
        formatter_class=_ArgHelpFormatter,
    )

    # -------------------------
    # Required inputs / run setup
    # -------------------------
    parser.add_argument(
        "--data_path",
        required=True,
        help=(
            "Path to the input data directory or project data root used by load_all_data(). "
            "This should contain the subtype-specific RNA, CNV, mutation, clinical, and "
            "metadata inputs expected by the data-loading utilities."
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help=(
            "Base output directory. The script will create subtype-specific preprocessing "
            "and feature-selection output folders under this location."
        ),
    )
    parser.add_argument(
        "--subtype",
        required=True,
        help=(
            "Subtype to process, such as Basal, LumA, or LumB. This is passed to the "
            "data-loading functions and used to name the output directory."
        ),
    )
    parser.add_argument(
        "--status_col",
        default="Disease_Status",
        help=(
            "Metadata column containing the recurrence label used for group comparisons "
            "and label-based filtering. Expected values are typically recurrent and "
            "non-recurrent, or binary 1/0 labels."
        ),
    )
    parser.add_argument(
        "--pathway_options",
        nargs="*",
        default=None,
        help=(
            "Pathway database/options to load for RNA/CNV/mutation pathway features. "
            "Pass as a space-separated list or as a single comma-separated string, "
            "depending on how your data-loading utilities parse this option.\n"
            "Example: --pathway_options hallmark kegg reactome"
        ),
    )

    # -------------------------
    # Mutation pathway features
    # -------------------------
    parser.add_argument(
        "--no_mutated_pathways",
        action="store_true",
        help=(
            "Backward-compatible shortcut to disable binary mutated pathway features. "
            "Equivalent to setting --mut_pathway_source none."
        ),
    )
    parser.add_argument(
        "--mut_pathway_source",
        default="snv",
        choices=["snv", "import", "none"],
        help=(
            "Source for binary mutated pathway features.\n"
            "  snv    = build pathway mutation features from loaded SNV/mutation data.\n"
            "  import = load a precomputed sample-by-pathway binary matrix.\n"
            "  none   = skip binary mutated pathway features."
        ),
    )
    parser.add_argument(
        "--mut_binary_matrix_csv",
        default=None,
        help=(
            "Path to a precomputed binary pathway mutation matrix used when "
            "--mut_pathway_source import. Rows should be samples and columns should be "
            "pathway or mutation-derived features. Values >0 are treated as mutated/present."
        ),
    )
    parser.add_argument(
        "--mut_binary_sample_col",
        default=None,
        help=(
            "Sample ID column in --mut_binary_matrix_csv. If omitted, the script will try "
            "to infer a sample column or use the first/index column."
        ),
    )
    parser.add_argument(
        "--mut_binary_prefix",
        default="MUTblock__",
        help=(
            "Prefix added to imported binary mutation pathway columns. Keep as MUTblock__ "
            "unless intentionally matching a different saved training schema."
        ),
    )

    # -------------------------
    # pTMB features
    # -------------------------
    parser.add_argument(
        "--enable_ptmb",
        action="store_true",
        help=(
            "Backward-compatible shortcut to enable pTMB. If --ptmb_source is none, this "
            "sets --ptmb_source snv, meaning pTMB will be calculated from MAF/GMT/GTF inputs."
        ),
    )
    parser.add_argument(
        "--ptmb_source",
        default="none",
        choices=["none", "snv", "import"],
        help=(
            "Source for pathway-TMB features.\n"
            "  none   = skip pTMB features.\n"
            "  snv    = calculate pTMB from a MAF/MAF-like table, pathway GMT, and GTF/CDS lengths.\n"
            "  import = load a precomputed sample-by-pathway pTMB matrix."
        ),
    )
    parser.add_argument(
        "--ptmb_matrix_csv",
        default=None,
        help=(
            "Path to a precomputed pTMB matrix used when --ptmb_source import. Rows should "
            "be samples and columns should be pathway-level pTMB features."
        ),
    )
    parser.add_argument(
        "--ptmb_sample_col",
        default=None,
        help=(
            "Sample ID column in --ptmb_matrix_csv. If omitted, the script will try to infer "
            "a sample column or use the first/index column."
        ),
    )
    parser.add_argument(
        "--ptmb_prefix",
        default="MUTblock__pTMB__",
        help=(
            "Prefix added to pTMB feature columns, whether imported or calculated. Keep as "
            "MUTblock__pTMB__ to match downstream training and prediction scripts."
        ),
    )
    parser.add_argument(
        "--ptmb_transform",
        default="none",
        choices=["none", "log10", "log1p"],
        help=(
            "Transform applied only to imported pTMB values. Use none to keep imported values "
            "unchanged; log1p for log(1+x); log10 for log10(x + small offset)."
        ),
    )
    parser.add_argument(
        "--ptmb_maf_path",
        default=None,
        help=(
            "MAF or MAF-like mutation table used when --ptmb_source snv. The file should "
            "contain sample IDs and gene symbols for mutation counting."
        ),
    )
    parser.add_argument(
        "--ptmb_maf_sample_col",
        default=None,
        help=(
            "Sample ID column in --ptmb_maf_path. If omitted, the script will try to infer "
            "a common MAF sample column such as Tumor_Sample_Barcode or sample_id."
        ),
    )
    parser.add_argument(
        "--ptmb_maf_gene_col",
        default=None,
        help=(
            "Gene-symbol column in --ptmb_maf_path. If omitted, the script will try to infer "
            "a common MAF gene column such as Hugo_Symbol, gene, or gene_symbol."
        ),
    )
    parser.add_argument(
        "--ptmb_cds_cache_csv",
        default=None,
        help=(
            "Optional cache file for CDS lengths derived from the GTF. If the file exists, "
            "CDS lengths are loaded from it; otherwise, the script computes CDS lengths and "
            "writes this cache for future runs."
        ),
    )
    parser.add_argument(
        "--ptmb_denom_policy",
        default="all_genes_with_len",
        choices=["all_genes_with_len", "all_genes", "mutated_genes_with_len"],
        help=(
            "Denominator used when normalizing pathway mutation burden.\n"
            "  all_genes_with_len      = divide by number of pathway genes with CDS length.\n"
            "  all_genes               = divide by all genes listed in the pathway.\n"
            "  mutated_genes_with_len  = divide by mutated pathway genes with CDS length."
        ),
    )
    parser.add_argument(
        "--ptmb_min_genes_with_len",
        type=int,
        default=1,
        help=(
            "Minimum number of pathway genes with valid CDS length required to calculate a "
            "pTMB value for that pathway. Pathways below this threshold are set to missing/zero."
        ),
    )
    parser.add_argument(
        "--ptmb_gmt_path",
        nargs="+",
        default=None,
        help=(
            "One or more GMT files defining pathway gene sets for pTMB calculation. Required "
            "when --ptmb_source snv. Multiple files can be supplied after the flag or passed "
            "as comma-separated paths if supported by the downstream parser."
        ),
    )
    parser.add_argument(
        "--ptmb_gtf_path",
        default=None,
        help=(
            "GTF used to compute CDS lengths for pTMB. If omitted, the script will reuse "
            "--protein_coding_gtf when available."
        ),
    )

    # -------------------------
    # Gene-level filtering
    # -------------------------
    parser.add_argument(
        "--protein_coding_gtf",
        default=None,
        help=(
            "Optional GTF used to identify protein-coding genes. When provided, gene-level "
            "RNA and CNV features can be filtered to protein-coding genes only. Pathway and "
            "signature features are not affected."
        ),
    )

    # -------------------------
    # Label-free variance filtering
    # -------------------------
    parser.add_argument(
        "--var_min",
        type=float,
        default=0.0,
        help=(
            "Minimum variance required for a feature to pass the label-free variance filter. "
            "This filter is applied before supervised testing and does not use recurrence labels."
        ),
    )
    parser.add_argument(
        "--var_min_nonmissing",
        type=int,
        default=10,
        help=(
            "Minimum number of non-missing samples required for variance filtering. Features "
            "with fewer usable samples are removed before supervised testing."
        ),
    )
    parser.add_argument(
        "--variance_keep_file",
        default=None,
        help=(
            "Optional file containing a precomputed list of features to keep after variance "
            "filtering. Use this to reuse a fixed variance-filter feature set across runs."
        ),
    )
    parser.add_argument(
        "--var_keep_top_frac",
        type=float,
        default=None,
        help=(
            "Optional top fraction of highest-variance features to keep within feature groups. "
            "For example, 0.60 keeps the top 60% most variable features per group. If omitted, "
            "the script uses the absolute --var_min threshold instead."
        ),
    )

    # -------------------------
    # Optional label-correlation prefilter
    # -------------------------
    parser.add_argument(
        "--corr_prefilter",
        action="store_true",
        help=(
            "Enable label-based correlation prefiltering before supervised testing. This is "
            "performed within the current subset/training fold when subset FS is used."
        ),
    )
    parser.add_argument(
        "--corr_abs_min",
        type=float,
        default=None,
        help=(
            "Minimum absolute correlation with the recurrence label required to pass the "
            "correlation prefilter. Only used when --corr_prefilter is enabled."
        ),
    )
    parser.add_argument(
        "--corr_min_n",
        type=int,
        default=10,
        help=(
            "Minimum number of non-missing samples required to calculate feature-label "
            "correlation during the optional correlation prefilter."
        ),
    )
    parser.add_argument(
        "--corr_prune_sig_mode",
        default="hierarchical",
        choices=["hierarchical", "cox_multi", "cox_uni", "group", "mwu", "limma"],
        help=(
            "Significance source used to choose the best representative feature during "
            "feature-feature correlation pruning.\n"
            "  hierarchical = cox_multi -> cox_uni -> group.\n"
            "  cox_multi    = prefer multivariable Cox p-values.\n"
            "  cox_uni      = prefer univariate Cox p-values.\n"
            "  group/mwu    = prefer recurrence group-comparison p-values.\n"
            "  limma        = prefer limma p-values for RNA features."
        ),
    )

    # -------------------------
    # Supervised testing toggles
    # -------------------------
    parser.add_argument(
        "--run_mwu",
        action="store_true",
        help=(
            "Run Mann-Whitney U testing for continuous features comparing recurrent vs "
            "non-recurrent samples. Binary features may use Fisher-style group testing "
            "elsewhere in the workflow."
        ),
    )
    parser.add_argument(
        "--no_run_mwu",
        action="store_true",
        help=(
            "Disable MWU/group-comparison testing even if another calling script would "
            "normally enable it. Useful for Cox-only feature selection."
        ),
    )
    parser.add_argument(
        "--run_cox_uni",
        action="store_true",
        help=(
            "Run univariate Cox proportional hazards models for each feature using "
            "--cox_time_col and --cox_event_col."
        ),
    )
    parser.add_argument(
        "--run_cox_multi",
        action="store_true",
        help=(
            "Run one multivariable Cox model per feature, adjusting for the covariates "
            "listed in --cox_covariates."
        ),
    )

    # -------------------------
    # Test thresholds
    # -------------------------
    parser.add_argument(
        "--mwu_p_thresh",
        type=float,
        default=0.05,
        help="Nominal p-value threshold for MWU/group-comparison feature selection.",
    )
    parser.add_argument(
        "--mwu_fdr_thresh",
        type=float,
        default=1,
        help=(
            "BH-FDR threshold for MWU/group-comparison feature selection. Use 1 to avoid "
            "filtering by FDR while still recording q-values."
        ),
    )
    parser.add_argument(
        "--cox_uni_p_thresh",
        type=float,
        default=0.05,
        help="Nominal p-value threshold for univariate Cox feature selection.",
    )
    parser.add_argument(
        "--cox_uni_fdr_thresh",
        type=float,
        default=1,
        help=(
            "BH-FDR threshold for univariate Cox feature selection. Use 1 to avoid "
            "filtering by FDR while still recording q-values."
        ),
    )
    parser.add_argument(
        "--cox_multi_p_thresh",
        type=float,
        default=0.05,
        help="Nominal p-value threshold for multivariable Cox feature selection.",
    )
    parser.add_argument(
        "--cox_multi_fdr_thresh",
        type=float,
        default=1,
        help=(
            "BH-FDR threshold for multivariable Cox feature selection. Use 1 to avoid "
            "filtering by FDR while still recording q-values."
        ),
    )

    # -------------------------
    # Cox configuration
    # -------------------------
    parser.add_argument(
        "--cox_time_col",
        default=None,
        help=(
            "Metadata column containing time-to-event values for Cox models, usually "
            "recurrence-free survival time or overall survival time."
        ),
    )
    parser.add_argument(
        "--cox_event_col",
        default=None,
        help=(
            "Metadata column containing event status for Cox models. Values should be "
            "binary or coercible to 0/1, where 1 indicates event/recurrence/death."
        ),
    )
    parser.add_argument(
        "--cox_covariates",
        nargs="*",
        default=None,
        help=(
            "Clinical covariates to include in multivariable Cox models. Provide as a "
            "space-separated list.\n"
            "Example: --cox_covariates age_dx chemo hormone"
        ),
    )
    parser.add_argument(
        "--cox_hr_abs_min",
        type=float,
        default=0.0,
        help=(
            "Minimum absolute log-hazard ratio magnitude required for Cox-selected "
            "features. Use 0.0 to disable HR-magnitude filtering."
        ),
    )
    parser.add_argument(
        "--cox_hr_direction",
        default="any",
        choices=["any", "gt1", "lt1"],
        help=(
            "Direction filter for Cox-selected features.\n"
            "  any = keep risk-increasing or risk-decreasing features.\n"
            "  gt1 = keep only features with HR > 1.\n"
            "  lt1 = keep only features with HR < 1."
        ),
    )
    parser.add_argument(
        "--cox_min_n",
        type=int,
        default=10,
        help=(
            "Minimum number of usable samples required to fit each per-feature Cox model."
        ),
    )
    parser.add_argument(
        "--cox_min_events",
        type=int,
        default=1,
        help=(
            "Minimum number of events required to fit each per-feature Cox model."
        ),
    )

    # -------------------------
    # Feature-feature redundancy pruning
    # -------------------------
    parser.add_argument(
        "--corr_threshold",
        type=float,
        default=0.90,
        help=(
            "Feature-feature absolute correlation threshold used for redundancy pruning "
            "after supervised testing. Highly correlated selected features are collapsed "
            "to a smaller representative set."
        ),
    )

    # -------------------------
    # Subset/fold-local feature selection
    # -------------------------
    parser.add_argument(
        "--enable_subset_fs",
        action="store_true",
        help=(
            "Enable subset-aware feature selection. When used with --sample_id_subset_file, "
            "supervised testing is restricted to that subset. This is how CV scripts perform "
            "fold-local, leak-free feature selection."
        ),
    )
    parser.add_argument(
        "--subset_run_name",
        default=None,
        help=(
            "Name for the subset/fold-specific feature-selection output directory. Useful "
            "for distinguishing CV folds, training subsets, or final full-cohort runs."
        ),
    )
    parser.add_argument(
        "--sample_id_subset_file",
        default=None,
        help=(
            "Optional file containing one sample_id per line for subset feature selection. "
            "If omitted and --enable_subset_fs is enabled, the full cohort is used."
        ),
    )

    # -------------------------
    # RNA limma thresholds
    # -------------------------
    parser.add_argument(
        "--limma_pvalue_threshold",
        type=float,
        default=0.05,
        help=(
            "Nominal limma p-value threshold used when incorporating RNA differential "
            "expression evidence into feature selection."
        ),
    )
    parser.add_argument(
        "--limma_padj_threshold",
        type=float,
        default=0.25,
        help=(
            "Adjusted limma p-value/FDR threshold used for RNA differential expression "
            "feature selection."
        ),
    )
    parser.add_argument(
        "--limma_logfc_threshold",
        type=float,
        default=0.25,
        help=(
            "Minimum absolute limma log2 fold-change threshold for RNA gene-level "
            "differential expression features."
        ),
    )

    # -------------------------
    # Custom gene sets
    # -------------------------
    parser.add_argument(
        "--custom_gene_sets_file",
        default=None,
        help=(
            "Optional custom gene-set file, usually GMT format, used to calculate custom "
            "RNA gene-set features."
        ),
    )
    parser.add_argument(
        "--custom_gene_set_mode",
        default="median_log2",
        choices=["median_log2", "zscore_mean"],
        help=(
            "Backward-compatibility flag. The mode value is retained for older commands, "
            "but current custom gene-set scoring computes multiple summaries internally, "
            "including median_log2, geo_mean_log2, and zscore_mean when available."
        ),
    )
    parser.add_argument(
        "--custom_gene_set_min_overlap",
        type=int,
        default=1,
        help=(
            "Minimum number of genes from a custom gene set that must overlap the RNA "
            "expression matrix to calculate a custom gene-set score."
        ),
    )

    # -------------------------
    # Case/control label names
    # -------------------------
    parser.add_argument(
        "--mwu_case_label",
        default="recurrent",
        help=(
            "Label value in --status_col treated as the case group for MWU/group testing."
        ),
    )
    parser.add_argument(
        "--mwu_ctrl_label",
        default="non-recurrent",
        help=(
            "Label value in --status_col treated as the control group for MWU/group testing."
        ),
    )

    # -------------------------
    # Gene-level bypass flags
    # -------------------------
    parser.add_argument(
        "--RNAgene-bypass",
        action="store_true",
        dest="RNAgene_bypass",
        help=(
            "Bypass RNA gene-level significance/statistical calculations and exclude RNA "
            "gene-level features. RNA pathway/signature/immune-derived features can still "
            "be used when available."
        ),
    )
    parser.add_argument(
        "--CNVgene-bypass",
        action="store_true",
        dest="CNVgene_bypass",
        help=(
            "Bypass CNV gene-level significance/statistical calculations and exclude CNV "
            "gene-level features. CNV arm, band, pathway, and other CNV-derived features "
            "can still be used when available."
        ),
    )

    args = parser.parse_args(argv)
    args.ptmb_gmt_path = _expand_paths(args.ptmb_gmt_path)

    pathway_options = args.pathway_options
    if pathway_options is not None and len(pathway_options) == 1 and isinstance(pathway_options[0], str) and ("," in pathway_options[0]):
        pathway_options = _split_list_arg(pathway_options[0])

    run_mwu = True
    if args.run_mwu:
        run_mwu = True
    if args.no_run_mwu:
        run_mwu = False

    sample_id_subset = None
    if args.enable_subset_fs and args.sample_id_subset_file:
        with open(args.sample_id_subset_file, "r") as f:
            sample_id_subset = [ln.strip() for ln in f if ln.strip()]

    # Normalize mutation-source shortcuts and validate source-specific inputs.
    if args.no_mutated_pathways:
        args.mut_pathway_source = "none"

    if args.enable_ptmb and args.ptmb_source == "none":
        args.ptmb_source = "snv"
    if args.ptmb_source != "none":
        args.enable_ptmb = True

    if args.mut_pathway_source == "import" and not args.mut_binary_matrix_csv:
        raise SystemExit("[ERROR] --mut_pathway_source import requires --mut_binary_matrix_csv.")

    if args.ptmb_source == "import" and not args.ptmb_matrix_csv:
        raise SystemExit("[ERROR] --ptmb_source import requires --ptmb_matrix_csv.")

    if args.ptmb_source == "snv":
        if not args.ptmb_gmt_path:
            raise SystemExit("[ERROR] --ptmb_source snv requires --ptmb_gmt_path.")
        if not args.ptmb_maf_path:
            raise SystemExit("[ERROR] --ptmb_source snv requires --ptmb_maf_path.")

    run_feature_selection(
        data_path=args.data_path,
        output_dir=args.output_dir,
        subtype=args.subtype,
        status_col=args.status_col,
        pathway_options=pathway_options,

        include_mutated_pathways=(args.mut_pathway_source != "none"),
        mut_pathway_source=args.mut_pathway_source,
        mut_binary_matrix_csv=args.mut_binary_matrix_csv,
        mut_binary_sample_col=args.mut_binary_sample_col,
        mut_binary_prefix=args.mut_binary_prefix,
        RNAgene_bypass=args.RNAgene_bypass,
        CNVgene_bypass=args.CNVgene_bypass,
        enable_ptmb=args.enable_ptmb,
        ptmb_source=args.ptmb_source,
        ptmb_matrix_csv=args.ptmb_matrix_csv,
        ptmb_sample_col=args.ptmb_sample_col,
        ptmb_prefix=args.ptmb_prefix,
        ptmb_transform=args.ptmb_transform,
        ptmb_maf_path=args.ptmb_maf_path,
        ptmb_maf_sample_col=args.ptmb_maf_sample_col,
        ptmb_maf_gene_col=args.ptmb_maf_gene_col,
        ptmb_gtf_path=args.ptmb_gtf_path,
        ptmb_cds_cache_csv=args.ptmb_cds_cache_csv,
        ptmb_denom_policy=args.ptmb_denom_policy,
        ptmb_min_genes_with_len=args.ptmb_min_genes_with_len,
        ptmb_gmt_path=args.ptmb_gmt_path,

        protein_coding_gtf=args.protein_coding_gtf,

        var_min=args.var_min,
        var_min_nonmissing=args.var_min_nonmissing,
        variance_keep_file=args.variance_keep_file,
        var_keep_top_frac=args.var_keep_top_frac,

        corr_prefilter=args.corr_prefilter,
        corr_abs_min=args.corr_abs_min,
     #   corr_topk=args.corr_topk,
        corr_min_n=args.corr_min_n,
        corr_prune_sig_mode=args.corr_prune_sig_mode,

        run_mwu=run_mwu,
        run_cox_uni=args.run_cox_uni,
        run_cox_multi=args.run_cox_multi,

        mwu_p_thresh=args.mwu_p_thresh,
        mwu_fdr_thresh=args.mwu_fdr_thresh,
        cox_uni_p_thresh=args.cox_uni_p_thresh,
        cox_uni_fdr_thresh=args.cox_uni_fdr_thresh,
        cox_multi_p_thresh=args.cox_multi_p_thresh,
        cox_multi_fdr_thresh=args.cox_multi_fdr_thresh,

        cox_time_col=args.cox_time_col,
        cox_event_col=args.cox_event_col,
        cox_covariates=args.cox_covariates,
        cox_hr_abs_min=args.cox_hr_abs_min,
        cox_hr_direction=args.cox_hr_direction,
        cox_min_n=args.cox_min_n,
        cox_min_events=args.cox_min_events,

        corr_threshold=args.corr_threshold,

        sample_id_subset=sample_id_subset,
        subset_run_name=args.subset_run_name,
        enable_subset_fs=args.enable_subset_fs,

        limma_pvalue_threshold=args.limma_pvalue_threshold,
        limma_padj_threshold=args.limma_padj_threshold,
        limma_logfc_threshold=args.limma_logfc_threshold,

        custom_gene_sets_file=args.custom_gene_sets_file,
        custom_gene_set_mode=args.custom_gene_set_mode,
        custom_gene_set_min_overlap=args.custom_gene_set_min_overlap,

        mwu_case_label=args.mwu_case_label,
        mwu_ctrl_label=args.mwu_ctrl_label,
    )

if __name__ == "__main__":
    main()


