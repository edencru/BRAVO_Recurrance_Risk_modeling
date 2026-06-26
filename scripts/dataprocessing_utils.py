import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict 
import sys
# Force it to use your user-level version
import typing_extensions
#from typing_extensions import ReadOnly
try:
    from typing_extensions import ReadOnly
except ImportError:
    class ReadOnly:
        def __class_getitem__(cls, item):
            return item

print("✅ Successfully loaded typing_extensions from:", typing_extensions.__file__)

from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
sys.path.append('./scripts')
from pathwayanalysis_utils import *


import os
import re
import numpy as np
import pandas as pd

def _detect_id_type(index_like) -> str:
    """Return 'ensembl' if ≥50% look like ENSG..., else 'symbol'."""
    idx = pd.Index(index_like).astype(str)
    if idx.size == 0: return "symbol"
    flags = idx.str.match(r"^ENSG\d+(?:\.\d+)?$", na=False)
    return "ensembl" if flags.mean() >= 0.5 else "symbol"

def _strip_version(s: str) -> str:
    return re.sub(r"\.\d+$", "", str(s))

def _build_maps(id_map: pd.DataFrame):
    """Expect a 2-col table with headers including 'ensembl' and 'symbol' (any case/alias)."""
    if id_map is None or id_map.empty:
        return {}, {}
    cols = {c.lower(): c for c in id_map.columns}
    ens = next((cols[c] for c in ["ensembl_id","gene_id","ensembl","ensemblgeneid","ensembl_gene_id"] if c in cols), None)
    sym = next((cols[c] for c in ["symbol","gene_name","hgnc_symbol","gene","hgnc"] if c in cols), None)
    if ens is None or sym is None:
        return {}, {}
    df = id_map[[ens, sym]].dropna().astype(str).copy()
    df[ens] = df[ens].map(_strip_version)
    ens2sym = dict(zip(df[ens], df[sym]))
    sym2ens = {}
    for e, s in zip(df[ens], df[sym]):
        sym2ens.setdefault(s, e)
    return ens2sym, sym2ens

def subset_log2tpm_with_limma_sig(
    tpm_df: pd.DataFrame,              # samples × genes (TPM)
    limma_csv: str,                    # limma results file
    id_map: pd.DataFrame | None = None,
    p_adj_thresh: float = 0.25,
    logfc_abs_min: float = 0.5
):
    """
    Returns:
      de_expr_log2:  (samples × selected_genes) log2(TPM+1)
      limma_genes:   Index of genes read from limma file (post-mapping to TPM ID space if needed)
      overlap:       List of genes actually used (present in TPM columns)
    """
    # 1) Read limma results
    lim = pd.read_csv(limma_csv, sep=None, engine="python")
    lim_cols = {c.lower(): c for c in lim.columns}

    # Find p-adj and logFC columns (common names)
    padj_col  = next((lim_cols[c] for c in ["adj.p.val","fdr","padj","adj_p_val","adjpval"] if c in lim_cols), None)
    logfc_col = next((lim_cols[c] for c in ["logfc","log2foldchange","log2_fc","log2fold"] if c in lim_cols), None)
    if padj_col is None or logfc_col is None:
        raise ValueError("Could not find adjusted P and logFC columns in limma results.")

    # Find gene identifier column (fallback to index)
    gene_col = next((lim_cols[c] for c in ["gene","symbol","hgnc_symbol","genesymbol","hgnc","ensembl","ensembl_id","gene_id"]
                     if c in lim_cols), None)
    if gene_col is None:
        lim.index = lim.index.astype(str)
        gene_series = lim.index.to_series()
    else:
        gene_series = lim[gene_col].astype(str)

    # 2) Filter by thresholds
    lim_filt = lim[(lim[padj_col] <= p_adj_thresh) & (lim[logfc_col].abs() >= logfc_abs_min)].copy()
    if lim_filt.empty:
        # Return empty selection but valid shapes
        return pd.DataFrame(index=tpm_df.index), gene_series.iloc[0:0], []

    # 3) Map gene IDs to TPM’s ID space if needed
    tpm_id_type   = _detect_id_type(tpm_df.columns)
    lim_id_type   = _detect_id_type(gene_series)
    ens2sym, sym2ens = _build_maps(id_map)

    lim_genes = gene_series.copy()
    if lim_id_type != tpm_id_type:
        if lim_id_type == "ensembl" and tpm_id_type == "symbol":
            lim_genes = lim_genes.map(lambda x: ens2sym.get(_strip_version(x), np.nan))
        elif lim_id_type == "symbol" and tpm_id_type == "ensembl":
            lim_genes = lim_genes.map(lambda x: sym2ens.get(x, np.nan))
        lim_filt = lim_filt.assign(_mapped=lim_genes).dropna(subset=["_mapped"])
        lim_genes = lim_filt["_mapped"].astype(str)
    else:
        lim_genes = lim_filt[gene_col] if gene_col else lim_filt.index.astype(str)

    # 4) Intersect with TPM columns and produce log2(TPM+1)
    lim_genes_unique = pd.Index(lim_genes.astype(str).tolist()).dropna().unique()
    overlap = [g for g in lim_genes_unique if g in tpm_df.columns]

    if not overlap:
        # No overlap—return empties to avoid crashes downstream
        return pd.DataFrame(index=tpm_df.index), lim_genes_unique, []

    de_expr_log2 = np.log2(tpm_df.loc[:, overlap].fillna(0.0) + 1.0)

    return de_expr_log2, lim_genes_unique, overlap

def process_limma_results(
    rna_expression,
    metadata,
    design_factors,
    contrast,
    pvalue_threshold=0.05,
    padj_threshold=0.25,
    logfold_threshold=0.5,
    base_mean_threshold=50,
    gene_sum_threshold=100,
    *,
    id_col="index",            # set to "index" if IDs are in the index; or use a column name
    subtype=None,
    subtype_col=None,
    output_dir=".",
    r_exe="Rscript"
):
    """
    Run limma-voom in R and return a DESeq2-like result table.
    Expects rna_expression: rows = samples, cols = genes (raw counts).
    Guarantees a 'sample_id' column for the R side, even if IDs are in the index.
    """
    import os, subprocess, textwrap
    import numpy as np
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    # ------------ build metadata with a stable 'sample_id' column -------------
    md = metadata.copy()

    def _resolve_sample_id(df, id_col):
        # If user says IDs are in index
        if id_col == "index":
            return df.index.astype(str)
        # If user provided a column name and it exists
        if id_col in df.columns:
            return df[id_col].astype(str)
        # Fallbacks: common names or a literal 'index' column created from CSVs
        for cand in ("sample_id", "patient_id", "SampleID", "case_submitter_id", "barcode", "ID", "id", "index"):
            if cand in df.columns:
                return df[cand].astype(str)
        # Last resort: index
        return df.index.astype(str)

    md["sample_id"] = _resolve_sample_id(md, id_col).str.strip()

    # ------------ handle group column name variants (space vs underscore) -----
    if not (isinstance(contrast, (list, tuple)) and len(contrast) == 3):
        raise ValueError("contrast must be a 3-tuple/list: (factor, levelA, levelB)")
    group_col_in, group1, group2 = map(str, contrast)
    group_candidates = [group_col_in, group_col_in.replace("_", " "), group_col_in.replace(" ", "_")]
    group_col = next((g for g in group_candidates if g in md.columns), None)
    if group_col is None:
        raise ValueError(f"[metadata] group_col '{group_col_in}' not found. Available: {list(md.columns)}")

    # covariates = other design_factors that exist
    covars = [c for c in design_factors if c != group_col and c in md.columns]

    # ------------ optional subtype filter -------------------------------------
    if subtype is not None and subtype_col is not None and subtype_col in md.columns:
        before = md.shape[0]
        md = md.loc[md[subtype_col].astype(str) == str(subtype)].copy()
        print(f"[subtype] filtering metadata where {subtype_col} == '{subtype}' → kept {md.shape[0]} / {before} rows")

    # ------------ counts matrix: rows=samples, cols=genes ---------------------
    X = rna_expression.copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X.index = X.index.astype(str)

    # ------------ align by 'sample_id' (no set_index/reset_index) ------------
    overlap = X.index.intersection(md["sample_id"])
    print(f"[align] overlap: counts({len(X)}) ∩ meta({len(md)}) = {len(overlap)}")
    if len(overlap) < 3:
        raise RuntimeError("[align] <3 overlapping samples: counts and meta do not match by IDs")

    X = X.loc[overlap]
    md = md.loc[md["sample_id"].isin(overlap)].copy()

    # columns to write to meta CSV
    keep_cols = ["sample_id", group_col] + covars
    if subtype is not None and subtype_col is not None and subtype_col in md.columns:
        keep_cols.append(subtype_col)
    missing_keep = [c for c in keep_cols if c not in md.columns]
    if missing_keep:
        raise ValueError(f"[metadata] required columns missing for limma meta file: {missing_keep}. "
                         f"Available: {list(md.columns)}")

    # ------------ write persistent inputs for R -------------------------------
    prefix    = f"limma_{(subtype or 'all')}".replace(" ", "_")
    counts_p  = os.path.join(output_dir, f"{prefix}__counts.csv")
    meta_p    = os.path.join(output_dir, f"{prefix}__meta.csv")
    out_p     = os.path.join(output_dir, f"{prefix}__results.csv")

    X_out = X.copy()
    X_out.insert(0, "sample_id", X_out.index.astype(str))  # first col = sample_id
    X_out.to_csv(counts_p, index=False)
    md.loc[:, keep_cols].to_csv(meta_p, index=False)
    print(f"[io] wrote counts → {counts_p}")
    print(f"[io] wrote meta   → {meta_p}")

    # ------------ R bridge (unchanged core logic) -----------------------------
    r_code = textwrap.dedent(r'''
        #!/usr/bin/env Rscript
        suppressPackageStartupMessages({
        library(edgeR)
        library(limma)
        library(optparse)
        })

        opt_list <- list(
        make_option("--counts", type="character"),
        make_option("--meta",   type="character"),
        make_option("--group_col", type="character"),
        make_option("--group1", type="character"),
        make_option("--group2", type="character"),
        make_option("--covars", type="character", default=""),
        make_option("--subtype_col", type="character", default=""),
        make_option("--subtype",     type="character", default=""),
        make_option("--gene_sum", type="double", default=50),
        make_option("--min_count", type="double", default=10),
        make_option("--min_samp_prop", type="double", default=0.1),
        make_option("--padj_thresh", type="double", default=0.25),
        make_option("--logfc_thresh", type="double", default=0.5),
        make_option("--out", type="character")
        )
        opt <- parse_args(OptionParser(option_list=opt_list))
        stop2 <- function(msg) { write(sprintf("ERROR: %s", msg), stderr()); quit(status=1) }

        norm_group <- function(x) {
        s <- tolower(trimws(as.character(x)))
        s <- gsub("\\s+(case|control)\\s*$", "", s)         # strip trailing 'case'/'control'
        s <- gsub("[^a-z0-9 _-]+", " ", s, perl=TRUE)       # keep letters, digits, space, _, -
        s <- gsub("\\s+", " ", s)
        trimws(s)
        }

        # ---- metadata ----
        meta <- tryCatch(read.csv(opt$meta, check.names=FALSE, stringsAsFactors=FALSE), error=function(e) NULL)
        if (is.null(meta)) stop2("failed to read metadata")
        if (!("sample_id" %in% colnames(meta))) {
        if (!is.null(rownames(meta)) && all(nzchar(rownames(meta)))) {
            meta$sample_id <- rownames(meta)
        } else {
            stop2("metadata must contain 'sample_id' column or have non-empty rownames")
        }
        }
        meta$sample_id <- trimws(as.character(meta$sample_id))
        dup_m <- duplicated(meta$sample_id)
        if (any(dup_m)) {
        message(sprintf("[meta] dropping %d duplicate rows for 'sample_id' (keeping first)", sum(dup_m)))
        meta <- meta[!dup_m, , drop=FALSE]
        }

        # ---- counts ----
        cts <- tryCatch(read.csv(opt$counts, check.names=FALSE, stringsAsFactors=FALSE), error=function(e) NULL)
        if (is.null(cts)) stop2("failed to read counts")
        if (!("sample_id" %in% colnames(cts))) stop2("counts must contain 'sample_id' column")
        cts$sample_id <- trimws(as.character(cts$sample_id))
        dup_c <- duplicated(cts$sample_id)
        if (any(dup_c)) {
        message(sprintf("[counts] dropping %d duplicate rows for 'sample_id' (keeping first)", sum(dup_c)))
        cts <- cts[!dup_c, , drop=FALSE]
        }

        # orientation: expect samples x genes; transpose if necessary
        meta_ids <- meta$sample_id
        ov_cols <- sum(colnames(cts) %in% meta_ids)
        ov_rows <- sum(cts$sample_id %in% meta_ids)
        if (ov_cols > ov_rows) {
        message("[counts] detected genes×samples; transposing")
        rn <- cts$sample_id
        cts$sample_id <- NULL
        mat <- t(as.matrix(cts))
        cts <- data.frame(sample_id=rownames(mat), mat, check.names=FALSE)
        } else {
        message("[counts] detected samples×genes; using as-is")
        }

        # align by sample_id
        common <- intersect(meta$sample_id, cts$sample_id)
        message(sprintf("[align] overlap: counts(%d) ∩ meta(%d) = %d", nrow(cts), nrow(meta), length(common)))
        if (length(common) < 3) stop2("too few overlapping samples between counts and meta")
        meta <- meta[match(common, meta$sample_id), , drop=FALSE]
        cts  <- cts [match(common, cts$sample_id),  , drop=FALSE]

        # optional subtype filter
        if (nzchar(opt$subtype) && nzchar(opt$subtype_col) && (opt$subtype_col %in% colnames(meta))) {
        keep <- which(as.character(meta[[opt$subtype_col]]) == opt$subtype)
        message(sprintf("[subtype] filtering metadata where %s == '%s' → kept %d / %d rows",
                        opt$subtype_col, opt$subtype, length(keep), nrow(meta)))
        meta <- meta[keep, , drop=FALSE]
        cts  <- cts [match(meta$sample_id, cts$sample_id), , drop=FALSE]
        if (nrow(meta) < 3) stop2("too few samples after subtype filtering")
        }

        # group + covariates
        if (!(opt$group_col %in% colnames(meta))) stop2(sprintf("group_col '%s' not in metadata", opt$group_col))
        g_raw <- norm_group(meta[[opt$group_col]])
        g1 <- norm_group(opt$group1); g2 <- norm_group(opt$group2)
        mask <- (g_raw %in% c(g1, g2))
        meta <- meta[mask, , drop=FALSE]
        cts  <- cts [match(meta$sample_id, cts$sample_id), , drop=FALSE]
        g_raw <- g_raw[mask]
        if (nrow(meta) < 3) stop2("too few samples after group filtering")
        group <- factor(g_raw, levels=c(g2, g1))  # baseline = group2

        covars <- trimws(unlist(strsplit(opt$covars, ",")))
        covars <- covars[nzchar(covars)]
        cov_df <- data.frame()
        if (length(covars) > 0) {
        keep_cov <- covars[covars %in% colnames(meta)]
        if (length(keep_cov) > 0) {
            cov_df <- meta[, keep_cov, drop=FALSE]
            drop_cols <- names(which(sapply(cov_df, function(x) length(unique(na.omit(x))) <= 1)))
            if (length(drop_cols) > 0) cov_df <- cov_df[, setdiff(colnames(cov_df), drop_cols), drop=FALSE]
            for (nm in colnames(cov_df)) {
            x <- cov_df[[nm]]
            if (!is.numeric(x)) cov_df[[nm]] <- factor(x)
            }
        }
        }

        # numeric counts matrix
        mat <- cts
        rownames(mat) <- mat$sample_id
        mat$sample_id <- NULL
        mat <- as.matrix(apply(mat, 2, function(x) suppressWarnings(as.numeric(x))))
        rownames(mat) <- meta$sample_id

        # gene filtering
        min_samples <- max(1, floor(nrow(mat) * opt$min_samp_prop))
        gs_keep <- which(colSums(mat, na.rm=TRUE) >= opt$gene_sum)
        mc_keep <- which(colSums(mat >= opt$min_count, na.rm=TRUE) >= min_samples)
        keep_idx <- intersect(gs_keep, mc_keep)
        message(sprintf("[filter] min_samples=%d | gene_sum≥%g → %d genes | count≥%g in ≥%d samples → %d genes | both → %d genes",
                min_samples, opt$gene_sum, length(gs_keep), opt$min_count, min_samples, length(mc_keep), length(keep_idx)))
        if (length(keep_idx) == 0) stop2("no genes pass filtering (gene_sum/min_count)")
        mat <- mat[, keep_idx, drop=FALSE]

        # edgeR + voom + limma
        y <- DGEList(counts=t(mat))          # genes x samples
        y <- calcNormFactors(y, method="TMM")

        # DESeq-like baseMean
        eff_lib <- y$samples$lib.size * y$samples$norm.factors
        norm_counts <- t(t(y$counts) / eff_lib) * mean(eff_lib)
        baseMean <- rowMeans(norm_counts)

        if (ncol(cov_df) > 0) {
        design <- model.matrix(~ 0 + group + ., data=cbind(data.frame(group=group), cov_df))
        } else {
        design <- model.matrix(~ 0 + group, data=data.frame(group=group))
        }
        colnames(design) <- make.names(colnames(design))

        v   <- voom(y, design, plot=FALSE)
        fit <- lmFit(v, design)

        g1n <- paste0("group", make.names(g1))
        g2n <- paste0("group", make.names(g2))
        ct  <- makeContrasts(contrasts=paste0(g1n, " - ", g2n), levels=colnames(design))
        fit2 <- eBayes(contrasts.fit(fit, ct))

        tab <- topTable(fit2, number=Inf, sort.by="P")
        res <- data.frame(
        gene           = rownames(tab),
        log2FoldChange = tab$logFC,
        pvalue         = tab$P.Value,
        padj           = tab$adj.P.Val,
        AveExpr        = tab$AveExpr,
        t              = tab$t,
        B              = tab$B,
        stringsAsFactors = FALSE,
        row.names = NULL
        )
        bm <- data.frame(gene=rownames(y$counts), baseMean=as.numeric(baseMean), stringsAsFactors=FALSE)
        res <- merge(res, bm, by="gene", all.x=TRUE)

        full_path <- opt$out
        sig_path  <- sub("\\.csv$", "_sig.csv", full_path)
        write.csv(res, full_path, row.names=FALSE)
        sig <- subset(res, padj < opt$padj_thresh & abs(log2FoldChange) > opt$logfc_thresh)
        write.csv(sig, sig_path, row.names=FALSE)
        message(sprintf("[out] wrote full → %s", full_path))
        message(sprintf("[out] wrote sig  → %s (n=%d)", sig_path, nrow(sig)))
                    
    ''')
    r_path = os.path.join(output_dir, f"{prefix}__bridge.R")
    with open(r_path, "w") as f:
        f.write(r_code)

    covar_str = ",".join(covars) if covars else ""
    cmd = [
        r_exe, "--vanilla", r_path,
        "--counts", counts_p,
        "--meta",   meta_p,
        "--group_col", group_col,
        "--group1", group1,
        "--group2", group2,
        "--covars", covar_str,
        "--gene_sum", str(float(gene_sum_threshold)),
        "--min_count", "10",
        "--min_samp_prop", "0.1",
        "--out", out_p,
    ]
    if subtype is not None and subtype_col is not None and subtype_col in metadata.columns:
        cmd += ["--subtype_col", subtype_col, "--subtype", str(subtype)]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"[R/limma] failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

    # ------------ read & filter like your DESeq step --------------------------
    res = pd.read_csv(out_p)
    need = {"gene","log2FoldChange","pvalue","padj","baseMean"}
    missing = need - set(res.columns)
    if missing:
        raise RuntimeError(f"limma output missing columns: {missing}")

    sig = res.loc[
        (res["pvalue"] < pvalue_threshold) &
        (res["padj"]   < padj_threshold) &
        (res["log2FoldChange"].abs() > logfold_threshold) &
        (res["baseMean"].abs() > base_mean_threshold)
    ].copy()

    # persist both
    res.to_csv(os.path.join(output_dir, f"{prefix}__results_full.csv"), index=False)
    sig.to_csv(os.path.join(output_dir, f"{prefix}__results_sig.csv"),  index=False)
    print(f"[out] wrote full → {os.path.join(output_dir, f'{prefix}__results_full.csv')}")
    print(f"[out] wrote sig  → {os.path.join(output_dir, f'{prefix}__results_sig.csv')} (n={len(sig)})")

    return sig




def process_deseq_results(
    rna_expression,
    metadata,
    design_factors,
    contrast,
    pvalue_threshold=0.05,
    padj_threshold=0.25,
    logfold_threshold=0.5,
    base_mean_threshold=50,
    gene_sum_threshold=100
):
    """
    Processes RNA expression data using DESeq2 and returns filtered results based on specified thresholds.

    Parameters:
    - rna_expression (DataFrame): A pandas DataFrame with raw RNA expression counts.
    - metadata (DataFrame): A pandas DataFrame containing metadata for the samples.
    - design_factors (list of str): List of column names in metadata to use as design factors.
    - pvalue_threshold (float): Threshold for filtering results by p-value.
    - padj_threshold (float): Threshold for filtering results by adjusted p-value.
    - logfold_threshold (float): Threshold for filtering results by absolute log2 fold change.
    - base_mean_threshold (float): Threshold for filtering results by base mean.
    - gene_sum_threshold (int): Minimum sum of counts across all samples for a gene to be retained.

    Returns:
    - DataFrame: Filtered results DataFrame.
    """
    # Step 1: Transpose and filter RNA expression data
    rna_expression = rna_expression.apply(pd.to_numeric, errors='coerce')#.transpose()
    
    n_samples = rna_expression.shape[0]
    min_samples = int(n_samples * 0.1)  # or 0.2 if you want stricter

    genes_to_keep = rna_expression.columns[(rna_expression >= gene_sum_threshold).sum(axis=0) >= min_samples]

    
    #genes_to_keep = rna_expression.columns[rna_expression.sum(axis=0) >= gene_sum_threshold]
    rna_expression = rna_expression[genes_to_keep]

    # Step 2: Initialize DESeq2 dataset and perform analysis
    inference = DefaultInference(n_cpus=8)
    dds = DeseqDataSet(
        counts=rna_expression,
        metadata=metadata,
        design_factors=design_factors,
        refit_cooks=True,
        inference=inference
    )
    dds.deseq2()

    print("Metadata columns:", metadata.columns)
    print("Design factors:", design_factors)    
    ## Step 3: Retrieve and filter results
    stat_res = DeseqStats(dds, contrast=contrast, inference=inference)
    stat_res.summary()
    results = stat_res.results_df
    results = results[results['pvalue'] < pvalue_threshold]
    results = results[results['padj'] < padj_threshold]
    results = results[abs(results['log2FoldChange']) > logfold_threshold]
    results = results[abs(results['baseMean']) > base_mean_threshold]

    return results


def process_mutation_pathways(
    M_matrix, subtype_meta, pathway_dicts, pathway_options,
    output_dir=None, subtype="", threshold=0.05
):
    from pathwayanalysis_utils import (
        run_pathway_mutation_enrichment,
        get_pathway_mutation_matrix,
        plot_significant_pathway_mutation_proportions
    )
    import os
    import matplotlib.pyplot as plt

    all_final_burdens = []
    all_burden_matrices = []

    for pathway in pathway_options:
        print(f"\n🔬 Mutation enrichment for: {pathway}")
        mutation_pathway_dict = pathway_dicts['mutation'][pathway]

        # Run enrichment
        greater_df = run_pathway_mutation_enrichment(
            M_matrix, mutation_pathway_dict, subtype_meta, 'Disease_Status', 'greater'
        )
        less_df = run_pathway_mutation_enrichment(
            M_matrix, mutation_pathway_dict, subtype_meta, 'Disease_Status', 'less'
        )
        results_df = pd.concat([greater_df, less_df], ignore_index=True)

        # Get burden matrix and store full (with MUT_ prefix)
        burden_matrix = get_pathway_mutation_matrix(
            M_matrix, M_matrix.columns, mutation_pathway_dict
        )
        burden_matrix_full = burden_matrix.add_prefix("MUT_")
        all_burden_matrices.append(burden_matrix_full)

        # Add Disease_Status column for plotting
        burden_matrix['Disease_Status'] = subtype_meta['Disease_Status']

        if output_dir:
            results_df.to_csv(
                os.path.join(output_dir, f'{pathway}_mutation_enrichment_{subtype}.csv'),
                index=False
            )
            plot_significant_pathway_mutation_proportions(
                pathway_burden=burden_matrix,
                results_df=results_df,
                output_path=output_dir,
                prefix=f'{pathway}_{subtype}',
                group_col='Disease_Status',
                alpha=threshold
            )
            plt.close()

        # Get significant or top 10 fallback
        sig_pathways = results_df[results_df['p_value'] <= threshold]['Pathway']
        if sig_pathways.empty:
            sig_pathways = results_df.nsmallest(10, 'p_value')['Pathway']
            print(f"⚠️ No significant mutation pathways for {pathway}")
            print(results_df)
        else:
            print(sig_pathways)

        # Extract significant burden matrix
        filtered = burden_matrix[sig_pathways]
        filtered = filtered.add_prefix("MUT_")
        all_final_burdens.append(filtered)

    if all_final_burdens:
        combined = pd.concat(all_final_burdens, axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        combined = combined.reindex(subtype_meta.index, fill_value=0)

        full_burden = pd.concat(all_burden_matrices, axis=1)
        full_burden = full_burden.loc[:, ~full_burden.columns.duplicated()]
        full_burden = full_burden.reindex(subtype_meta.index, fill_value=0)

        if output_dir:
            combined.to_csv(
                os.path.join(output_dir, f'combined_significant_mutation_pathway_burden_{subtype}.csv')
            )
            print(f"✅ Saved: {combined.shape}")
        else:
            print(f"✅ Generated mutation pathway burden (not saved): {combined.shape}")

        return combined, full_burden

def process_cnv_pathways(CN_matrix, subtype_meta, pathway_dicts, pathway_options, output_dir, subtype):
    from pathwayanalysis_utils import run_gsva_and_test_by_group, plot_top_gsva_differences

    #CN_matrix = CN_matrix.astype(float) - 1.0  # 1 -> 0, positive= gain, negative= loss
    CN_matrix = CN_matrix.astype(float)# input matrix is now discrete -2,-1,0,1,2

    CN_matrix = CN_matrix.loc[subtype_meta.index]

    all_final_burdens = []

    for pathway in pathway_options:
        print(f"\n📊 CNV GSVA for: {pathway}")
        cnv_dict = pathway_dicts['copynumber'][pathway]

        scores, results = run_gsva_and_test_by_group(CN_matrix.T, cnv_dict, subtype_meta,
                                                     group_col='Disease_Status', group1='recurrent', group2='non-recurrent')

        scores.to_csv(os.path.join(output_dir, f'{pathway}_copynumber_gsva_all_scores_{subtype}.csv'))
        results.to_csv(os.path.join(output_dir, f'{pathway}_copynumber_gsva_all_group_comparison_{subtype}.csv'))

        plot_top_gsva_differences(
            gsva_scores_df=scores,
            metadata=subtype_meta,
            test_results_df=results,
            top_n=10,
            group_col='Disease_Status',
            group_colors={'recurrent': 'firebrick', 'non-recurrent': 'mediumseagreen'},
            output_dir=output_dir,
            prefix=f'{pathway}_CNV_{subtype}'
        )
        plt.close()

        sig_pathways = results[results['pval'] < 0.02]['Pathway']
        if sig_pathways.empty:
            print(f"⚠️ No significant CNV pathways for {pathway}")
            continue

        filtered = scores.T[sig_pathways].add_prefix("CNV_")
        all_final_burdens.append(filtered)

    if all_final_burdens:
        combined = pd.concat(all_final_burdens, axis=1).loc[:, ~pd.concat(all_final_burdens, axis=1).columns.duplicated()]
        combined = combined.reindex(subtype_meta.index, fill_value=0)
        combined.to_csv(os.path.join(output_dir, f'combined_significant_copynumber_pathway_burden_{subtype}.csv'))
        print(f"✅ Saved: {combined.shape}")
        return combined
    else:
        print("⚠️ No significant CNV pathways found.")
        return pd.DataFrame(index=subtype_meta.index)


def process_rna_pathways(RNA_matrix, subtype_meta, pathway_dicts, pathway_options, output_dir, subtype):
    from pathwayanalysis_utils import run_gsva_and_test_by_group, plot_top_gsva_differences

    RNA_matrix = np.log2(RNA_matrix + 1)
    RNA_matrix = RNA_matrix.loc[subtype_meta.index]

    all_final_burdens = []

    for pathway in pathway_options:
        print(f"\n🧬 RNA GSVA for: {pathway}")
        rna_dict = pathway_dicts['RNA'][pathway]

        scores, results = run_gsva_and_test_by_group(RNA_matrix.T, rna_dict, subtype_meta,
                                                     group_col='Disease_Status', group1='recurrent', group2='non-recurrent')

        scores.to_csv(os.path.join(output_dir, f'{pathway}_RNA_gsva_all_scores_{subtype}.csv'))
        results.to_csv(os.path.join(output_dir, f'{pathway}_RNA_gsva_all_group_comparison_{subtype}.csv'))

        plot_top_gsva_differences(
            gsva_scores_df=scores,
            metadata=subtype_meta,
            test_results_df=results,
            top_n=10,
            group_col='Disease_Status',
            group_colors={'recurrent': 'firebrick', 'non-recurrent': 'mediumseagreen'},
            output_dir=output_dir,
            prefix=f'{pathway}_RNA_{subtype}'
        )
        plt.close()

        sig_pathways = results[results['pval'] < 0.02]['Pathway']
        if sig_pathways.empty:
            print(f"⚠️ No significant RNA pathways for {pathway}")
            continue

        filtered = scores.T[sig_pathways].add_prefix("RNA_")
        all_final_burdens.append(filtered)

    if all_final_burdens:
        combined = pd.concat(all_final_burdens, axis=1).loc[:, ~pd.concat(all_final_burdens, axis=1).columns.duplicated()]
        combined = combined.reindex(subtype_meta.index, fill_value=0)
        combined.to_csv(os.path.join(output_dir, f'combined_significant_RNA_pathway_burden_{subtype}.csv'))
        print(f"✅ Saved: {combined.shape}")
        return combined
    else:
        print("⚠️ No significant RNA pathways found.")
        return pd.DataFrame(index=subtype_meta.index)



def merge_feature_blocks(
    subtype_meta,
    mutation_df=None,
    rna_df=None,          # GSVA (RNA)
    cnv_df=None,          # GSVA (CNV)
    tf_df=None,           # TF / ploidy CNV features
    tmb_df=None,
    hoadley_df=None,
    cibersort_df=None,
    pathogenic_df=None,
    cnv_pct_df=None,      # percent altered
    clinical_df=None,
    de_gene_df=None,      # differential genes (log2 TPM+1 subset)
    return_separate_blocks=False,
    *,
    block_as_subtype=False,
    subtype=None,
):
    """
    Build feature blocks. If `block_as_subtype=True` and subtype is known,
    use a subtype-specific recipe.

    LumA:
      - RNAblock         = de_genes + rna + cibersort + hoadley (as_is)
      - RNAblock_PCA     = same as RNAblock (pca)
      - CNVblock         = cnv (as_is)                      <-- CHANGED
      - CNVblock_PCA     = cnv_pct_altered + tf (pca)       <-- CHANGED
      - MUTblock         = mutation + tmb + pathogenic (as_is)
      - MUTblock_PCA     = same as MUTblock (pca)
      - ALLblock / ALLblock_PCA (prefixed sub-block columns)

    LumB:
      - RNAblock         = de_genes + cibersort + hoadley (as_is)  <-- CHANGED
      - RNAblock_PCA     = rna only (pca)                          <-- CHANGED
      - CNVblock         = cnv + cnv_pct_altered + tf (as_is)
      - CNVblock_PCA     = same as CNVblock (pca)
      - MUTblock / MUTblock_PCA
      - CLINICALblock / CLINICALblock_PCA
      - ALLblock / ALLblock_PCA (includes clinical)

    Basal:
      - RNAblock         = rna + de_genes + hoadley (as_is)        <-- CHANGED
      - RNAblock_PCA     = cibersort only (pca)                    <-- CHANGED
      - CNVblock / CNVblock_PCA     = cnv + cnv_pct_altered + tf
      - MUTblock / MUTblock_PCA
      - CLINICALblock / CLINICALblock_PCA
      - ALLblock / ALLblock_PCA (includes clinical)

    Default (when block_as_subtype=False or subtype unrecognized):
      - Legacy multi-block behavior (your previous grouping).

    Returns
    -------
    merged_df : pd.DataFrame
      For subtype recipes, this is ALLblock (non-PCA). For legacy path, your previous merged set.
    grouped_blocks : dict (optional)
      { block_name: {"data": DataFrame, "process": str}, ... } if requested.
    """
    import pandas as pd

    index = subtype_meta.index.astype(str)

    def align_df(df):
        if df is not None:
            df = df.copy()
            df.index = df.index.astype(str)
            return df.reindex(index).fillna(0)
        return pd.DataFrame(index=index)

    # align all inputs
    blocks = {
        "mutation": align_df(mutation_df),
        "rna": align_df(rna_df),                       # GSVA (RNA)
        "cnv": align_df(cnv_df),                       # GSVA (CNV)
        "tf": align_df(tf_df),
        "tmb": align_df(tmb_df),
        "hoadley": align_df(hoadley_df),
        "cibersort": align_df(cibersort_df),
        "pathogenic": align_df(pathogenic_df),
        "cnv_pct_altered": align_df(cnv_pct_df),
        "clinical": align_df(clinical_df),
        "de_genes": align_df(de_gene_df),
    }

    def _cat(names):
        """Concat selected aligned blocks by column; de-duplicate columns."""
        parts = [blocks[n] for n in names if n in blocks and not blocks[n].empty]
        if not parts:
            return pd.DataFrame(index=index)
        out = pd.concat(parts, axis=1)
        return out.loc[:, ~out.columns.duplicated()]

    def _non_empty(df: pd.DataFrame) -> bool:
        return df is not None and df.shape[1] > 0

    def _prefixed(df, prefix):
        """Prefix column names to keep sources traceable and unique in ALLblock."""
        if df is None or df.empty:
            return df
        out = df.copy()
        out.columns = pd.Index([f"{prefix}__{c}" for c in out.columns], name="feature")
        return out

    # ---------- LumA ----------
    if block_as_subtype and (subtype == "LumA"):
        grouped_blocks = {}

        RNAblock = _cat(["de_genes", "rna", "cibersort", "hoadley"])
  #      RNAblock = _cat(["de_genes", "cibersort", "hoadley"])
        CNVblock = _cat(["cnv", "cnv_pct_altered", "tf"])                                 # CHANGED
        MUTblock = _cat(["mutation", "tmb", "pathogenic"])

        def _add_block(name, df, process):
            if _non_empty(df):
                grouped_blocks[name] = {"data": df, "process": process}

        # as-is
        # _add_block("RNAblock", RNAblock, "as_is")
        # _add_block("CNVblock", CNVblock, "as_is")
        # _add_block("MUTblock", MUTblock, "as_is")

        _add_block("RNAblock", RNAblock, "as_is")
        _add_block("CNVblock", CNVblock, "as_is")
        _add_block("MUTblock", MUTblock, "as_is")

        # PCA twins (note CNVblock_PCA is from cnv_pct_altered + tf)
        _add_block("RNAblock_PCA", RNAblock, "pca")
        _add_block("CNVblock_PCA", CNVblock, "pca")  # CHANGED
        _add_block("MUTblock_PCA", MUTblock, "pca")

        # ALLblock (prefixed)
        parts = []
        for k in ("RNAblock", "CNVblock", "MUTblock"):
            if k in grouped_blocks:
                dfk = _prefixed(grouped_blocks[k]["data"], k)
                if _non_empty(dfk):
                    parts.append(dfk)
        ALLblock = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=index)

        _add_block("ALLblock", ALLblock, "as_is")
        _add_block("ALLblock_PCA", ALLblock, "pca")

        merged = ALLblock
        return (merged, grouped_blocks) if return_separate_blocks else merged

    # ---------- LumB ----------
    if block_as_subtype and (subtype == "LumB"):
        grouped_blocks = {}

        RNAblock = _cat(["de_genes", "rna", "cibersort", "hoadley"])
        CNVblock = _cat(["cnv", "cnv_pct_altered", "tf"])                                 # CHANGED
        MUTblock = _cat(["mutation", "tmb", "pathogenic"])
 #       CLINICALblock = _cat(["clinical"])

        # RNAblock = _cat(["de_genes", "cibersort", "hoadley"])
        # CNVblock = _cat(["cnv"])                                 # CHANGED
        # MUTblock = _cat(["mutation"])
 #       CLINICALblock = _cat(["clinical"])


        def _add_block(name, df, process):
            if _non_empty(df):
                grouped_blocks[name] = {"data": df, "process": process}

        # as-is
        _add_block("RNAblock", RNAblock, "as_is")
        _add_block("CNVblock", CNVblock, "as_is")
        _add_block("MUTblock", MUTblock, "as_is")
 #       _add_block("CLINICALblock", CLINICALblock, "clinical")

        # PCA twins (RNAblock_PCA is 'rna' only)
        _add_block("RNAblock_PCA", RNAblock, "pca")
        _add_block("CNVblock_PCA", CNVblock, "pca")  # CHANGED
        _add_block("MUTblock_PCA", MUTblock, "pca")
  #      _add_block("CLINICALblock_PCA", CLINICALblock, "pca")

        # ALLblock (prefixed)
        parts = []
        #for k in ("RNAblock", "CNVblock", "MUTblock", "CLINICALblock"):
        for k in ("RNAblock", "CNVblock", "MUTblock"):

            if k in grouped_blocks:
                dfk = _prefixed(grouped_blocks[k]["data"], k)
                if _non_empty(dfk):
                    parts.append(dfk)
        ALLblock = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=index)

        _add_block("ALLblock", ALLblock, "as_is")
        _add_block("ALLblock_PCA", ALLblock, "pca")

        merged = ALLblock
        return (merged, grouped_blocks) if return_separate_blocks else merged

    # ---------- Basal ----------
    if block_as_subtype and (subtype == "Basal"):
        grouped_blocks = {}

        RNAblock = _cat(["de_genes", "rna", "cibersort", "hoadley"])
        CNVblock = _cat(["cnv", "cnv_pct_altered", "tf"])                                 # CHANGED
        MUTblock = _cat(["mutation", "tmb", "pathogenic"])
        # CLINICALblock = _cat(["clinical"])

        # RNAblock = _cat(["de_genes", "cibersort", "hoadley"])
        # CNVblock = _cat(["cnv"])                                 # CHANGED
        # MUTblock = _cat(["mutation"])
    #    CLINICALblock = _cat(["clinical"])

        def _add_block(name, df, process):
            if _non_empty(df):
                grouped_blocks[name] = {"data": df, "process": process}

        # as-is
        _add_block("RNAblock", RNAblock, "as_is")
        _add_block("CNVblock", CNVblock, "as_is")
        _add_block("MUTblock", MUTblock, "as_is")
   #     _add_block("CLINICALblock", CLINICALblock, "clinical")

        # PCA twins (RNAblock_PCA is cibersort only)
        _add_block("RNAblock_PCA", RNAblock, "pca")
        _add_block("CNVblock_PCA", CNVblock, "pca")  # CHANGED
        _add_block("MUTblock_PCA", MUTblock, "pca")
   #     _add_block("CLINICALblock_PCA", CLINICALblock, "pca")

        # ALLblock (prefixed)
        parts = []
  #      for k in ("RNAblock", "CNVblock", "MUTblock", "CLINICALblock"):
        for k in ("RNAblock", "CNVblock", "MUTblock"):
            if k in grouped_blocks:
                dfk = _prefixed(grouped_blocks[k]["data"], k)
                if _non_empty(dfk):
                    parts.append(dfk)
        ALLblock = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=index)

        _add_block("ALLblock", ALLblock, "as_is")
        _add_block("ALLblock_PCA", ALLblock, "pca")

        merged = ALLblock
        return (merged, grouped_blocks) if return_separate_blocks else merged

    # ---------- Default / legacy behavior ----------
    rna_gsva_block = blocks["rna"]
    rna_block = _cat(["hoadley", "cibersort"])
    rna_DE_block = blocks["de_genes"]
    mutation_block = blocks["mutation"]
    TMB_block = _cat(["tmb", "pathogenic"])
    cnv_gsva_block = blocks["cnv"]
    cnv_block = _cat(["tf", "cnv_pct_altered"])
    clinical_block = blocks["clinical"]
    cibersort_block = blocks["cibersort"]
    cnv_pct_altered_block = blocks["cnv_pct_altered"]

    merged_default = pd.concat(
        [
            rna_gsva_block, rna_block, rna_DE_block,
            mutation_block, TMB_block,
            cnv_gsva_block, cnv_block,
            clinical_block, cibersort_block, cnv_pct_altered_block,
        ],
        axis=1
    ).loc[:, lambda df: ~df.columns.duplicated()]

    grouped_default = {
        "rna_gsva_PCA": {"data": rna_gsva_block, "process": "pca"},
        "rna_gsva":     {"data": rna_gsva_block, "process": "as_is"},
        "rna":          {"data": rna_block,      "process": "sig_lax"},
        "rna_DE_PCA":   {"data": rna_DE_block,   "process": "pca"},
        "rna_DE":       {"data": rna_DE_block,   "process": "as_is"},
        "TMB":          {"data": TMB_block,      "process": "as_is"},
        "cnv_gsva_PCA": {"data": cnv_gsva_block, "process": "pca"},
        "cnv_gsva":     {"data": cnv_gsva_block, "process": "as_is"},
        "cnv":          {"data": cnv_block,      "process": "sig_lax"},
        "mutationPCA":  {"data": mutation_block, "process": "pca"},
        "mutation":     {"data": mutation_block, "process": "as_is"},
        "cibersortPCA": {"data": cibersort_block,"process": "pca"},
        "cnv_pct_alteredPCA": {"data": cnv_pct_altered_block, "process": "pca"},
    }
    if not clinical_block.empty:
        grouped_default["clinical"] = {"data": clinical_block, "process": "clinical"}

    return (merged_default, grouped_default) if return_separate_blocks else merged_default




from scipy.stats import ttest_ind, mannwhitneyu
import pandas as pd

def run_significance_tests(data_df, meta_df, group_col, group1, group2, use_mannwhitney=False):
    # Align sample indices
    common_samples = data_df.index.intersection(meta_df.index)
    data_df = data_df.loc[common_samples]
    meta_df = meta_df.loc[common_samples]

    # Print group sizes
    print(f"👥 Group sizes for {group_col}:")
    print(meta_df[group_col].value_counts(dropna=False))
    print("Group1 label:", group1)
    print("Group2 label:", group2)

    results = []

    for col in data_df.columns:
        vals1 = data_df.loc[meta_df[group_col] == group1, col].dropna()
        vals2 = data_df.loc[meta_df[group_col] == group2, col].dropna()

        if len(vals1) < 2 or len(vals2) < 2:
            print(f"⚠️ Skipping {col}: insufficient samples (group1={len(vals1)}, group2={len(vals2)})")
            continue

        try:
            if use_mannwhitney:
                stat, p = mannwhitneyu(vals1, vals2, alternative='two-sided')
            else:
                stat, p = ttest_ind(vals1, vals2, equal_var=False)

            results.append({
                'feature': col,
                'p_value': p,
                f'{group1}_mean': vals1.mean(),
                f'{group2}_mean': vals2.mean()
            })
        except Exception as e:
            print(f"⚠️ Skipping feature {col} due to error: {e}")
            continue

    if not results:
        print("❌ No features tested — check sample grouping, labels, or data.")
        return pd.DataFrame(columns=["feature", "p_value"])

    return pd.DataFrame(results).sort_values('p_value')




def get_gsea(data_matrix, subtype_meta, pathway_dicts, pathway_options,
             output_dir=None, subtype="", data_type="copynumber"):
    """
    Runs GSVA for each pathway option and concatenates all scores into one dataframe.
    """

    import os
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import numpy as np
    from pathwayanalysis_utils import run_gsva_and_test_by_group, plot_top_gsva_differences

    # ----- transform by data_type -----
    if data_type.lower() == "rna":
        # Only do this if input RNA is not already log2-transformed
        data_matrix = np.log2(data_matrix + 1)
    elif data_type.lower() == "copynumber":
        # Ploidy-normalized CN (~2 = neutral) -> center so 0 = neutral
        #data_matrix = data_matrix - 1.0 
        ## have already ploidy normalized -1 and thresholded for discrete values (-2, -1, 0 , 1, 2)
        data_matrix = data_matrix
        # NOTE: no log2 here

    # Align to samples in subtype_meta
    data_matrix = data_matrix.loc[subtype_meta.index]
    all_scores = []

    def deduplicate_pathway_dict(pathway_dict):
        seen = defaultdict(int)
        new_dict = {}
        for name, genes in pathway_dict.items():
            key = name
            while key in new_dict:
                seen[name] += 1
                key = f"{name}_{seen[name]}"
            new_dict[key] = genes
        return new_dict

    for pathway in pathway_options:
        print(f"\n📊 {data_type.upper()} GSVA for: {pathway}")

        raw_pathway_dict = pathway_dicts[data_type][pathway]
        pathway_dict = deduplicate_pathway_dict(raw_pathway_dict)

        scores, results = run_gsva_and_test_by_group(
            data_matrix.T,
            pathway_dict,
            subtype_meta,
            group_col='Disease_Status',
            group1='recurrent',
            group2='non-recurrent'
        )

        # Save only if output_dir is specified
        if output_dir:
            scores.to_csv(os.path.join(output_dir, f'{pathway}_{data_type}_gsva_all_scores_{subtype}.csv'))
            results.to_csv(os.path.join(output_dir, f'{pathway}_{data_type}_gsva_group_comparison_{subtype}.csv'))

            plot_top_gsva_differences(
                gsva_scores_df=scores,
                metadata=subtype_meta,
                test_results_df=results,
                top_n=10,
                group_col='Disease_Status',
                group_colors={'recurrent': 'firebrick', 'non-recurrent': 'mediumseagreen'},
                output_dir=output_dir,
                prefix=f'{pathway}_{data_type}_{subtype}'
            )
            plt.close()

        scores = scores.T.add_prefix(f"{pathway}_{data_type}_")
        all_scores.append(scores)

    if all_scores:
        combined = pd.concat(all_scores, axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        combined = combined.reindex(subtype_meta.index)
        if output_dir:
            combined.to_csv(os.path.join(output_dir, f'all_{data_type}_gsva_scores_{subtype}.csv'))
            print(f"✅ Saved concatenated {data_type} GSVA scores: {combined.shape}")
        else:
            print(f"✅ Generated {data_type} GSVA scores (not saved): {combined.shape}")
        return combined
    else:
        print(f"⚠️ No GSVA scores generated for {data_type}.")
        return pd.DataFrame(index=subtype_meta.index)
