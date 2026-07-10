import pandas as pd
import os

def get_pathway_mutation_matrix(df, gene_cols, pathway_dict):
    results = {}
    for pathway, genes in pathway_dict.items():
        genes_present = [g for g in genes if g in gene_cols]
        if not genes_present:
            continue
        # Binary: 1 if any gene in pathway is mutated
        results[pathway] = (df[genes_present].astype(int).sum(axis=1) > 0).astype(int)
    return pd.DataFrame(results)

# Ensure pathway names are unique
from collections import Counter

def make_unique(names):
    counts = Counter()
    result = []
    for name in names:
        counts[name] += 1
        if counts[name] == 1:
            result.append(name)
        else:
            result.append(f"{name}_{counts[name]}")
    return result


def run_pathway_mutation_enrichment(M_matrix, pathway_dict, meta, group_col='group', test_type='two-sided'):
    """
    Perform pathway-level mutation enrichment using Fisher's exact test.
    
    Parameters:
        M_matrix (pd.DataFrame): Binary mutation matrix (genes as columns, index = patient_id).
        pathway_dict (dict): Dictionary {pathway_name: list of genes}.
        meta (pd.DataFrame): Metadata with group assignment, indexed by patient_id.
        group_col (str): Column in meta to compare (e.g., 'group').

    Returns:
        pd.DataFrame: Pathway enrichment results (p-value, FDR, odds ratio).
    """
    gene_cols = M_matrix.columns

    # Compute pathway burden
    pathway_burden = {}
    for pathway, genes in pathway_dict.items():
        genes_present = [g for g in genes if g in gene_cols]
        if not genes_present:
            continue
        pathway_burden[pathway] = (M_matrix[genes_present].sum(axis=1) > 0).astype(int)

    pathway_burden = pd.DataFrame(pathway_burden, index=M_matrix.index)

    # Merge group info
    pathway_burden[group_col] = meta[group_col]

    # Run Fisher's tests
    results = []
    for pathway in pathway_burden.columns.drop(group_col):
        contingency = pd.crosstab(pathway_burden[pathway], pathway_burden[group_col])
        if contingency.shape == (2, 2):
            odds, p = fisher_exact(contingency, alternative=test_type) ## test_type='two-sided' alternative='two-sided' alternative='less' alternative='greater'
            results.append({'Pathway': pathway, 'OddsRatio': odds, 'p_value': p})

    # Multiple testing correction
    results_df = pd.DataFrame(results).sort_values('p_value')
    if not results_df.empty:
        pvals = results_df['p_value'].values
        rejected, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
        results_df['fdr'] = pvals_corrected
        results_df['significant'] = rejected
    else:
        results_df['fdr'] = []
        results_df['significant'] = []

    return results_df

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import os

def plot_significant_pathway_mutation_proportions(
    pathway_burden,
    results_df,
    output_path,
    prefix,
    group_col='group',
    alpha=0.05,
    top_n=10
):
    """
    Save a horizontal bar plot showing the proportion of patients with mutations
    in each significant pathway, grouped by condition.
    
    Parameters:
        pathway_burden (pd.DataFrame): Pathway mutation burden per sample.
        results_df (pd.DataFrame): Enrichment test results (must include 'p_value' and 'Pathway').
        output_path (str): Directory to save the plot.
        prefix (str): Prefix for filename.
        group_col (str): Column to group by (e.g., 'Disease_Status').
        alpha (float): p-value cutoff.
        top_n (int): Number of top pathways to plot.
    """
    sig_pathways = results_df[results_df['p_value'] < alpha]['Pathway']

    if sig_pathways.empty:
        print(f"No pathways with p-value < {alpha}. Skipping plot.")
        return

    # Subset data
    subset = pathway_burden[sig_pathways.tolist() + [group_col]]

    # Melt and compute groupwise proportions
    melted = subset.melt(id_vars=group_col, var_name='Pathway', value_name='Mutation')
    prop_df = (
        melted.groupby(['Pathway', group_col])['Mutation']
        .mean()
        .reset_index()
        .pivot(index='Pathway', columns=group_col, values='Mutation')
        .fillna(0)
    )

    # Enforce column order
    group_order = ['recurrent', 'non-recurrent']
    for group in group_order:
        if group not in prop_df.columns:
            prop_df[group] = 0
    prop_df = prop_df[group_order]

    # Sort by total mutation burden and select top
    prop_df['total'] = prop_df.sum(axis=1)
    prop_df = prop_df.sort_values('total', ascending=False).drop(columns='total').head(top_n)

    # Plot
    colors = ['firebrick', 'mediumseagreen']
    ax = prop_df.plot(kind='barh', figsize=(10, 5), color=colors, edgecolor='black')
    plt.xlabel('Proportion of Patients with Mutation')
    plt.ylabel('Pathway')
    plt.title(f'Significant Mutation Burden by Pathway (p < {alpha})')
    plt.xlim(0, 1)
    plt.gca().invert_yaxis()
    plt.legend(group_order, title=group_col)
    plt.tight_layout()

    # Save
    out_file = os.path.join(output_path, f'{prefix}_mutation_pathway_plot.pdf')
    plt.savefig(out_file)
    plt.close()


def get_pathway_cnv_matrix(CN_matrix, pathway_dict, amp_thresh=1.5, del_thresh=0.5):
    gene_cols = CN_matrix.columns
    results = {}
    
    for pathway, genes in pathway_dict.items():
        genes_present = [g for g in genes if g in gene_cols]
        if not genes_present:
            continue
        # Define altered if any gene in pathway has CN > amp_thresh or CN < del_thresh
        altered = ((CN_matrix[genes_present] > amp_thresh) | (CN_matrix[genes_present] < del_thresh)).any(axis=1)
        results[pathway] = altered.astype(int)
    
    return pd.DataFrame(results, index=CN_matrix.index)

from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import pandas as pd

from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

def run_pathway_cnv_enrichment(pathway_burden, group_col='Disease_Status', alpha=0.05):
    if group_col not in pathway_burden.columns:
        raise KeyError(f"'{group_col}' not found in pathway_burden columns.")

    pathway_cols = [col for col in pathway_burden.columns if col != group_col]
    stat_results = []

    for pathway in pathway_cols:
        contingency = pd.crosstab(pathway_burden[pathway], pathway_burden[group_col])
        if contingency.shape == (2, 2):
            odds, p = fisher_exact(contingency)
            stat_results.append({'Pathway': pathway, 'OddsRatio': odds, 'p_value': p})

    results_df = pd.DataFrame(stat_results)
    
    if results_df.empty:
        print("⚠️ No valid 2x2 contingency tables found.")
        return pd.DataFrame(columns=['Pathway', 'OddsRatio', 'p_value', 'fdr', 'significant'])

    # FDR correction
    pvals = results_df['p_value'].values
    rejected, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
    results_df['fdr'] = pvals_corrected
    results_df['significant'] = rejected

    return results_df.sort_values('p_value')

import matplotlib.pyplot as plt

def plot_significant_pathway_cnv_proportions(pathway_burden, results_df, group_col='group', alpha=0.05):
    # 🔍 Filter by raw p-value, not FDR
    sig_pathways = results_df[results_df['p_value'] < alpha]['Pathway']
    
    if sig_pathways.empty:
        print("⚠️ No pathways with p < alpha.")
        return

    # 🧮 Calculate proportion of altered patients in each group
    prop_data = (
        pathway_burden[[group_col] + list(sig_pathways)]
        .melt(id_vars=group_col, var_name='Pathway', value_name='Altered')
    )
    
    prop_df = (
        prop_data.groupby(['Pathway', group_col])['Altered']
        .mean()
        .unstack(fill_value=0)
        .loc[sig_pathways]
    )

    # 🎨 Plot
    colors = {'recurrent': 'firebrick', 'non-recurrent': 'mediumseagreen'}
    ax = prop_df.plot(
        kind='barh',
        figsize=(10, max(5, len(prop_df) * 0.4)),
        color=[colors.get(c, 'gray') for c in prop_df.columns],
        edgecolor='black'
    )
    plt.xlabel('Proportion of Patients with Altered CN')
    plt.ylabel('Pathway')
    plt.title(f'Significant CN Burden by Pathway (p < {alpha})')
    plt.xlim(0, 1)
    plt.gca().invert_yaxis()
    #plt.legend(prop_df.columns, title=group_col)
    plt.tight_layout()
    plt.show()

# Function to compute CN burden per pathway
def get_continuous_pathway_cnv_matrix(CN_matrix, pathway_dict):
    results = {}
    for pathway, genes in pathway_dict.items():
        genes_present = [g for g in genes if g in CN_matrix.columns]
        if not genes_present:
            continue
        # Mean absolute deviation from neutral copy number (1.0)
        #results[pathway] = (CN_matrix[genes_present] - 1).abs().mean(axis=1)
        #results[pathway] = (CN_matrix[genes_present] - 1).abs().median(axis=1)
        results[pathway] = (CN_matrix[genes_present] - 1).median(axis=1)

        #results[pathway] = ((CN_matrix[genes_present] - 1 ).abs().sum(axis=1))



    return pd.DataFrame(results)

from scipy.stats import mannwhitneyu
import pandas as pd
from statsmodels.stats.multitest import multipletests

def run_mannwhitney_pathway_cnv_enrichment(pathway_burden, metadata_df, group_col='Disease_Status', alpha=0.05):
    stat_results = []

    # Merge group info
    merged = pathway_burden.merge(metadata_df[[group_col]], left_index=True, right_index=True)

    for pathway in pathway_burden.columns:
        group1 = merged[merged[group_col] == 'recurrent'][pathway].dropna()
        group2 = merged[merged[group_col] == 'non-recurrent'][pathway].dropna()

        if len(group1) >= 5 and len(group2) >= 5:
            stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
            stat_results.append({'Pathway': pathway, 'U_stat': stat, 'p_value': p_value})

    results_df = pd.DataFrame(stat_results).sort_values('p_value')

    # FDR correction
    pvals = results_df['p_value'].values
    rejected, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
    results_df['fdr'] = pvals_corrected
    results_df['significant'] = rejected

    return results_df

import seaborn as sns
import matplotlib.pyplot as plt

def plot_continuous_pathway_cnv(results_df, pathway_burden, metadata_df, group_col='Disease_Status', alpha=0.05, top_n=20):
    sig_pathways = results_df[results_df['p_value'] < alpha]['Pathway'][:top_n]

    if sig_pathways.empty:
        print("No pathways pass significance threshold.")
        return

    merged = pathway_burden.merge(metadata_df[[group_col]], left_index=True, right_index=True)

    melted = merged[sig_pathways.tolist() + [group_col]].melt(id_vars=group_col, var_name='Pathway', value_name='CN_Burden')

    plt.figure(figsize=(10, max(4, len(sig_pathways)*0.6)))
    sns.boxplot(data=melted, x='CN_Burden', y='Pathway', hue=group_col, palette={'recurrent': 'firebrick', 'non-recurrent': 'mediumseagreen'})
    sns.stripplot(data=melted, x='CN_Burden', y='Pathway', hue=group_col, dodge=True, color='black', alpha=0.4)
    plt.title('CN Burden in Significant Pathways')
    plt.tight_layout()
    plt.legend(title=group_col)
    plt.show()


import pandas as pd
from scipy.stats import mannwhitneyu
import gseapy as gp

from scipy.stats import mannwhitneyu
import pandas as pd

# def rank_genes_by_cnv_difference(CN_matrix, metadata, group_col='group', group1='recurrent', group2='non-recurrent'):
#     group_labels = metadata[group_col]
#     group1_ids = group_labels[group_labels == group1].index
#     group2_ids = group_labels[group_labels == group2].index

#     scores = {}
#     for gene in CN_matrix.columns:
#         g1_vals = CN_matrix.loc[group1_ids, gene].dropna() - 1  # Subtract 1 for neutrality
#         g2_vals = CN_matrix.loc[group2_ids, gene].dropna() - 1

#         if len(g1_vals) > 1 and len(g2_vals) > 1:
#             stat, p = mannwhitneyu(g1_vals, g2_vals, alternative='two-sided')
#             diff = g1_vals.mean() - g2_vals.mean()
#             scores[gene] = diff

#     ranked_series = pd.Series(scores).sort_values(ascending=False)
#     return ranked_series


from scipy.stats import mannwhitneyu
import numpy as np
import pandas as pd

def rank_genes_by_cnv_difference(CN_matrix, metadata, group_col='group', group1='recurrent', group2='non-recurrent'):
    group_labels = metadata[group_col]
    group1_ids = group_labels[group_labels == group1].index
    group2_ids = group_labels[group_labels == group2].index

    scores = {}
    for gene in CN_matrix.columns:
        g1_vals = CN_matrix.loc[group1_ids, gene].dropna() - 1
        g2_vals = CN_matrix.loc[group2_ids, gene].dropna() - 1

        if len(g1_vals) > 1 and len(g2_vals) > 1:
            diff = g1_vals.mean() - g2_vals.mean()
            _, pval = mannwhitneyu(g1_vals, g2_vals, alternative='two-sided')

            # Add small number to prevent log(0)
            score = diff * -np.log10(pval + 1e-10)
            scores[gene] = score

    return pd.Series(scores).sort_values(ascending=False)




def run_preranked_cn_gsea(ranked_series, pathway_dict, outdir="gsea_cn_results"):
    # Convert pathway dict to GMT-like dataframe
    gmt_data = [[pathway, "custom"] + genes for pathway, genes in pathway_dict.items()]
    gmt_df = pd.DataFrame(gmt_data)

    # Save ranked list to file
    rnk_path = f"{outdir}/cn_ranks.rnk"
    ranked_series.to_csv(rnk_path, sep='\t', header=False)

    # Run GSEA
    prerank_res = gp.prerank(
        rnk=ranked_series,
        gene_sets=pathway_dict,
        processes=4,
        permutation_num=1000,
        outdir=outdir,
        format='png',
        seed=42
    )

    return prerank_res.res2d.sort_values("FDR q-val", ascending=True)


import matplotlib.pyplot as plt

def plot_gsea_cnv_results(results_df, alpha=0.05, top_n=20, group_col='group', data_type='CNV'):
    """
    Plot GSEA results for top enriched pathways.

    Parameters:
    - results_df: DataFrame from GSEA output (must contain 'NES', 'NOM p-val', 'Term')
    - alpha: significance threshold (default: 0.05)
    - top_n: number of top up/down pathways to display
    - group_col: grouping column (unused in plot, retained for compatibility)
    - data_type: 'CNV' or 'RNA' (affects title text only)
    """
    # Filter for significant results
    sig_df = results_df[results_df['NOM p-val'] < alpha].copy()

    if sig_df.empty:
        print("⚠️ No pathways pass the significance threshold.")
        return

    # Get top up- and down-regulated pathways
    top_up = sig_df[sig_df['NES'] > 0].sort_values('NES', ascending=False).head(top_n)
    top_down = sig_df[sig_df['NES'] < 0].sort_values('NES').head(top_n)
    sig_df = pd.concat([top_up, top_down])

    # Color coding
    colors = ['firebrick' if nes > 0 else 'mediumseagreen' for nes in sig_df['NES']]

    # Plot
    plt.figure(figsize=(10, max(6, 0.4 * len(sig_df))))
    plt.barh(sig_df['Term'], sig_df['NES'], color=colors, edgecolor='black')
    plt.xlabel('Enrichment Score (NES)')
    plt.title(f'Top {top_n} Pathways with Significant {data_type} Enrichment (p < {alpha})')
    plt.axvline(0, color='black', linestyle='--')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()





import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_combined_pathway_median_cn(
    CN_matrix,
    pathway_dict,
    metadata,
    group_col='Disease_Status',
    selected_pathways=None,
    palette={'recurrent': 'firebrick', 'non-recurrent': 'mediumseagreen'},
    figsize=(10, 6)
):
    """
    Plot all pathway median CNs in a single horizontal plot, grouped by metadata.

    Parameters:
        CN_matrix (pd.DataFrame): Sample x Gene CN matrix (normalized by ploidy).
        pathway_dict (dict): {pathway_name: [genes]}.
        metadata (pd.DataFrame): Sample metadata, index must match CN_matrix.
        group_col (str): Column in metadata for grouping.
        selected_pathways (list): Optional list of pathway names to include.
        palette (dict): Group -> color.
        figsize (tuple): Plot size.
    """
    plot_rows = []
    
    for pathway, genes in pathway_dict.items():
        if selected_pathways and pathway not in selected_pathways:
            continue

        genes_in_data = [g for g in genes if g in CN_matrix.columns]
        if not genes_in_data:
            continue

        median_cn = (CN_matrix[genes_in_data]-1).median(axis=1, skipna=True)
        for sample_id, value in median_cn.items():
            if sample_id in metadata.index:
                group = metadata.loc[sample_id, group_col]
                plot_rows.append({'Sample': sample_id, 'Pathway': pathway, 'Median_CN': value, group_col: group})
    
    if not plot_rows:
        print("⚠️ No data to plot. Check gene names and pathway selection.")
        return

    plot_df = pd.DataFrame(plot_rows)
    
    plt.figure(figsize=figsize)
    sns.boxplot(data=plot_df, x='Median_CN', y='Pathway', hue=group_col, palette=palette, fliersize=0)
    sns.stripplot(data=plot_df, x='Median_CN', y='Pathway', hue=group_col,
                  dodge=True, palette='dark:.3', size=3, alpha=0.5, jitter=True)
    plt.axvline(0, linestyle='--', color='gray', label='Neutral (1.0)')
    plt.xlabel('Median Absolute CN (normalized)')
    plt.ylabel('Pathway')
    plt.title('Median Absolute CN per Pathway')
    plt.legend(title=group_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


from scipy.stats import mannwhitneyu
import pandas as pd

import pandas as pd
from scipy.stats import ttest_ind

from scipy.stats import ttest_ind
import numpy as np
import pandas as pd

def rank_genes_by_tpm_difference_and_significance(
    TPM_matrix, metadata, group_col='group', group1='recurrent', group2='non-recurrent'
):
    group1_ids = metadata[metadata[group_col] == group1].index
    group2_ids = metadata[metadata[group_col] == group2].index

    results = []

    for gene in TPM_matrix.index:
        g1_vals = TPM_matrix.loc[gene, group1_ids].dropna()
        g2_vals = TPM_matrix.loc[gene, group2_ids].dropna()

        if len(g1_vals) > 1 and len(g2_vals) > 1:
            mean_diff = g1_vals.mean() - g2_vals.mean()
            _, pval = ttest_ind(g1_vals, g2_vals, equal_var=False)  # Welch's t-test
            score = mean_diff * -np.log10(pval + 1e-10)  # Composite score
            results.append({
                "gene": gene,
                "mean_diff": mean_diff,
                "pval": pval,
                "score": score
            })

    result_df = pd.DataFrame(results).set_index("gene")
    result_df["fdr"] = pd.Series(result_df["pval"]).rank(method='average') / len(result_df)

    # Sort by composite score
    result_df = result_df.sort_values(by="score", ascending=False)

    return result_df



import gseapy as gp

def run_preranked_tpm_gsea(ranked_series, pathway_dict, outdir="gsea_rna_results"):
    prerank_res = gp.prerank(
        rnk=ranked_series,
        gene_sets=pathway_dict,
        processes=4,
        permutation_num=1000,
        outdir=outdir,
        seed=42
    )
    return prerank_res.res2d.sort_values("FDR q-val", ascending=True)


import matplotlib.pyplot as plt

def plot_gsea_nes(gsea_df, fdr_threshold=0.05):
    sig = gsea_df[gsea_df['NOM p-val'] < fdr_threshold]
    sig = sig.sort_values('NES')
    colors = ['mediumseagreen' if x < 0 else 'firebrick' for x in sig['NES']]
    
    plt.figure(figsize=(10, max(4, 0.3 * len(sig))))
    plt.barh(sig['Term'], sig['NES'], color=colors, edgecolor='black')
    plt.xlabel('Normalized Enrichment Score (NES)')
    plt.title(f'GSEA Enrichment (FDR < {fdr_threshold})')
    plt.axvline(0, color='black', linestyle='--')
    plt.tight_layout()
    plt.show()


from gseapy import gsva

def compute_gsva_scores(data_matrix, significant_pathways, pathway_dict):
    selected_genesets = {k: pathway_dict[k] for k in significant_pathways if k in pathway_dict}
    if not selected_genesets:
        print("⚠️ No matching pathways found in the dictionary for GSVA computation.")
        return pd.DataFrame()

    # Subset to genes present in the matrix
    genes_to_keep = set(g for genes in selected_genesets.values() for g in genes)
    genes_to_keep = data_matrix.index.intersection(genes_to_keep)
    data_subset = data_matrix.loc[genes_to_keep]

    # Run GSVA
    result = gsva(data=data_subset, gene_sets=selected_genesets, method='gsva', 
                  sample_norm_method='none', verbose=False)

    # Return scores pivoted as pathways x samples
    gsva_scores_df = result.res2d.pivot(index='Term', columns='Name', values='ES')
    return gsva_scores_df


import seaborn as sns
import matplotlib.pyplot as plt

def plot_gsva_by_group(gsva_scores, metadata, group_col='Disease_Status', top_n=10, title='GSVA Scores by Group', save_path=None):
    """
    Plot boxplots with dots of GSVA scores grouped by a metadata column.

    Parameters:
        gsva_scores (pd.DataFrame): Pathways x Samples matrix of GSVA scores.
        metadata (pd.DataFrame): Sample metadata with group_col.
        group_col (str): Column in metadata to group by.
        top_n (int): Number of top pathways to plot.
        title (str): Plot title.
        save_path (str): If provided, saves the plot to this file path.
    """
    # Transpose so sample IDs are the index
    gsva_scores = gsva_scores.T  # samples x pathways

    # Merge scores with metadata
    gsva_scores = gsva_scores.merge(metadata[[group_col]], left_index=True, right_index=True)

    # Melt for seaborn plotting
    melted = gsva_scores.melt(id_vars=group_col, var_name='Pathway', value_name='GSVA Score')

    # Restrict to top N most variable pathways
    top_pathways = (
        melted.groupby('Pathway')['GSVA Score']
        .std()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    melted = melted[melted['Pathway'].isin(top_pathways)]

    # Custom color palette
    palette = {'recurrent': 'firebrick', 'non-recurrent': 'mediumseagreen'}

    plt.figure(figsize=(12, 0.5 + top_n * 0.6))
    sns.boxplot(data=melted, x='GSVA Score', y='Pathway', hue=group_col, palette=palette, fliersize=0)
    sns.stripplot(data=melted, x='GSVA Score', y='Pathway', hue=group_col, palette=palette,
                  dodge=True, alpha=0.5, linewidth=0.5, jitter=0.2, edgecolor='black', marker='o')

    plt.title(title)
    plt.tight_layout()
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], title=group_col)  # avoid duplicate legends

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


def run_gsva_and_test_by_group(
    data_matrix, pathway_dict, metadata,
    selected_pathways=None, group_col='group',
    group1='recurrent', group2='non-recurrent'
):
    """
    Robust GSVA wrapper:
      - Ensure unique genes (rows) and sample columns
      - Feed gseapy stable, short, unique IDs for pathways to avoid 'term' collisions
      - Map IDs back to original names after GSVA
      - Return (gsva_scores_df, results_df)
    """
    import numpy as np
    import pandas as pd
    from collections import defaultdict
    from scipy.stats import ttest_ind
    import gseapy as gp

    # ---------- subset pathways ----------
    if selected_pathways is None:
        selected_pathways = list(pathway_dict.keys())
    subset_pathway_dict = {k: pathway_dict[k] for k in selected_pathways if k in pathway_dict}

    # ---------- data hygiene: genes x samples expected ----------
    # data_matrix comes in as genes x samples in your current calls (you pass data_matrix.T upstream)
    X = data_matrix.copy()
    X.index = X.index.astype(str).str.strip()     # genes
    X.columns = X.columns.astype(str).str.strip() # samples

    # drop duplicate sample columns (keep first)
    if X.columns.duplicated().any():
        n = X.columns.duplicated().sum()
        print(f"[run_gsva] Dropping {n} duplicated sample columns before GSVA.")
        X = X.loc[:, ~X.columns.duplicated(keep='first')]

    # collapse duplicate gene rows by mean (gseapy expects unique gene names)
    if X.index.duplicated().any():
        n = X.index.duplicated().sum()
        print(f"[run_gsva] Collapsing {n} duplicated gene rows by mean before GSVA.")
        X = X.groupby(level=0).mean()

    # ensure samples align to metadata (but don't reorder genes)
    meta = metadata.copy()
    meta.index = meta.index.astype(str).str.strip()
    X = X.loc[:, X.columns.intersection(meta.index)]
    if X.shape[1] == 0:
        raise ValueError("[run_gsva] No overlapping samples between expression and metadata.")

    # ---------- supply collision-proof IDs to gseapy ----------
    # filter each geneset to available genes and require >=2 genes
    genes_available = set(X.index)
    filtered = {}
    for nm, genes in subset_pathway_dict.items():
        g = [g for g in genes if g in genes_available]
        if len(g) >= 2:
            filtered[nm] = list(dict.fromkeys(g))
    if not filtered:
        # return empty frames with expected shape
        return pd.DataFrame(index=[], columns=X.columns), pd.DataFrame(columns=["Pathway","mean_diff","pval"])

    # build ID -> genes dict and ID -> original-name map
    id2genes, id2name = {}, {}
    for i, (nm, genes) in enumerate(filtered.items(), start=1):
        term_id = f"PW_{i:06d}"   # short, unique, safe
        id2genes[term_id] = genes
        id2name[term_id]  = nm

    # ---------- run GSVA (genes x samples) ----------
    gv = gp.gsva(data=X, gene_sets=id2genes, outdir=None, processes=1, verbose=False)
    gv.run()

    # Extract scores robustly: prefer a matrix pathways x samples if available
    scores = getattr(gv, "scores", None)
    if isinstance(scores, pd.DataFrame):
        P = scores.copy()               # pathways x samples with term IDs as index
    else:
        # fallback to res2d (long form): columns ['Term','Name','ES']
        res2d = gv.res2d.copy()
        # make sure 'Term' is unique via our IDs
        # (it should be because we supplied unique IDs)
        P = res2d.pivot(index="Term", columns="Name", values="ES")

    # map IDs back to original pathway names
    P.index = P.index.astype(str)
    P.rename(index=id2name, inplace=True)

    # If, for any reason, duplicates appear after mapping back (very rare), uniquify them
    if P.index.duplicated().any():
        print(f"[run_gsva] WARNING: {P.index.duplicated().sum()} duplicated pathway names after mapping back; uniquifying.")
        counts = defaultdict(int)
        new_idx = []
        for nm in P.index:
            counts[nm] += 1
            new_idx.append(nm if counts[nm] == 1 else f"{nm}__{counts[nm]}")
        P.index = new_idx

    # final matrix: pathway x sample, numeric
    gsva_scores_df = P.apply(pd.to_numeric, errors="coerce")

    # ---------- group comparison ----------
    g1_ids = meta.index[meta[group_col] == group1]
    g2_ids = meta.index[meta[group_col] == group2]
    g1 = gsva_scores_df.columns.intersection(g1_ids)
    g2 = gsva_scores_df.columns.intersection(g2_ids)

    results = []
    for pathway in gsva_scores_df.index:
        x = gsva_scores_df.loc[pathway, g1].dropna()
        y = gsva_scores_df.loc[pathway, g2].dropna()
        if len(x) > 1 and len(y) > 1:
            stat, pval = ttest_ind(x, y, equal_var=False)
            results.append({"Pathway": pathway, "mean_diff": x.mean() - y.mean(), "pval": float(pval)})

    results_df = pd.DataFrame(results).sort_values("pval") if results else pd.DataFrame(columns=["Pathway","mean_diff","pval"])
    return gsva_scores_df, results_df


# def run_gsva_and_test_by_group(data_matrix, pathway_dict, metadata,
#                                 selected_pathways=None, group_col='group',
#                                 group1='recurrent', group2='non-recurrent'):
#     import pandas as pd
#     from gseapy import gsva
#     from scipy.stats import ttest_ind
#     from collections import defaultdict

#     # ✅ Subset and merge duplicate pathways
#     if selected_pathways is None:
#         selected_pathways = list(pathway_dict.keys())

#     subset_pathway_dict = {k: pathway_dict[k] for k in selected_pathways if k in pathway_dict}

#     # ✅ Ensure gene names are strings
#     data_matrix.index = data_matrix.index.astype(str)

#     # if data_matrix.columns.duplicated().any():
#     #     print("⚠️ Duplicate sample columns found. Deduplicating...")
#     #     data_matrix = data_matrix.loc[:, ~data_matrix.columns.duplicated()]


#     #data_matrix = data_matrix.T

#     # 🧬 Run GSVA
#     gsva_result = gsva(data_matrix, subset_pathway_dict, method='gsva', outdir=None)

#     # 📊 Handle duplicated Terms in GSVA result
#     res2d = gsva_result.res2d.copy()
#     print("🔎 GSVA result res2d shape:", res2d.shape)
#     print("🔎 Top duplicate Terms:", res2d['Term'].value_counts().head())
#     print("🔎 Unique Terms:", res2d['Term'].nunique())

#     # if res2d['Term'].duplicated().any():
#     #     print("⚠️ Duplicate pathway names in GSVA results. Deduplicating...")
#     #     res2d['Term'] = make_unique(res2d['Term'].tolist())

#     # Pivot to pathway x sample scores
#     gsva_scores_df = res2d.pivot(index='Term', columns='Name', values='ES')
#     gsva_scores_df = gsva_scores_df.apply(pd.to_numeric, errors='coerce')

#     # 🧪 Run t-tests between groups
#     group1_ids = metadata[metadata[group_col] == group1].index.astype(str)
#     group2_ids = metadata[metadata[group_col] == group2].index.astype(str)

#     results = []
#     for pathway in gsva_scores_df.index:
#         vals1 = gsva_scores_df.loc[pathway, gsva_scores_df.columns.intersection(group1_ids)].dropna()
#         vals2 = gsva_scores_df.loc[pathway, gsva_scores_df.columns.intersection(group2_ids)].dropna()

#         if len(vals1) > 1 and len(vals2) > 1:
#             mean_diff = vals1.mean() - vals2.mean()
#             stat, pval = ttest_ind(vals1, vals2, equal_var=False)
#             results.append({
#                 "Pathway": pathway,
#                 "mean_diff": mean_diff,
#                 "pval": pval
#             })

#     results_df = pd.DataFrame(results).sort_values(by='pval')

#     return gsva_scores_df, results_df


# def run_gsva_and_test_by_group(data_matrix, pathway_dict, metadata,
#                                 selected_pathways=None, group_col='group',
#                                 group1='recurrent', group2='non-recurrent'):
#     import pandas as pd
#     from gseapy import gsva
#     from scipy.stats import ttest_ind
#     from collections import defaultdict

#     # --- Deduplication helper ---
#     def make_unique(keys):
#         seen = defaultdict(int)
#         result = []
#         for k in keys:
#             new_k = k
#             while new_k in result:
#                 seen[k] += 1
#                 new_k = f"{k}_{seen[k]}"
#             result.append(new_k)
#         return result

#     # ✅ Select and deduplicate pathway names
#     if selected_pathways is None:
#         selected_pathways = list(pathway_dict.keys())

#     subset_pathway_dict = {k: pathway_dict[k] for k in selected_pathways if k in pathway_dict}

#     if len(set(subset_pathway_dict.keys())) != len(subset_pathway_dict):
#         print("⚠️ Warning: Duplicate pathway names detected — renaming...")
#         unique_keys = make_unique(list(subset_pathway_dict.keys()))
#         subset_pathway_dict = dict(zip(unique_keys, subset_pathway_dict.values()))

#     # ✅ Ensure gene names are strings
#     data_matrix.index = data_matrix.index.astype(str)

#         # Ensure sample names (columns) are unique
#     if data_matrix.columns.duplicated().any():
#         print("⚠️ Duplicate sample columns found in expression matrix. Deduplicating...")
#         data_matrix.columns = make_unique(data_matrix.columns.tolist())


#     # 🧬 Run GSVA
#     gsva_result = gsva(data_matrix, subset_pathway_dict, method='gsva', outdir=None)

#     # 📊 Pivot to pathway x sample scores
#   #  gsva_scores_df = gsva_result.res2d.pivot(index='Term', columns='Name', values='ES')
#     # Ensure 'Term' is unique before pivot
#     res2d = gsva_result.res2d.copy()
#     print("🔎 GSVA result res2d shape:", gsva_result.res2d.shape)
#     print("🔎 Top duplicate Terms:", gsva_result.res2d['Term'].value_counts().head())
#     print("🔎 Unique Terms:", gsva_result.res2d['Term'].nunique())

    
#     if res2d['Term'].duplicated().any():
#         print("⚠️ Duplicate pathway names in GSVA results. Deduplicating...")
#         res2d['Term'] = make_unique(res2d['Term'].tolist())

#     gsva_scores_df = res2d.pivot(index='Term', columns='Name', values='ES')

#     gsva_scores_df = gsva_scores_df.apply(pd.to_numeric, errors='coerce')

#     # 🧪 Run t-tests between groups
#     group1_ids = metadata[metadata[group_col] == group1].index.astype(str)
#     group2_ids = metadata[metadata[group_col] == group2].index.astype(str)

#     results = []
#     for pathway in gsva_scores_df.index:
#         vals1 = gsva_scores_df.loc[pathway, gsva_scores_df.columns.intersection(group1_ids)].dropna()
#         vals2 = gsva_scores_df.loc[pathway, gsva_scores_df.columns.intersection(group2_ids)].dropna()

#         if len(vals1) > 1 and len(vals2) > 1:
#             mean_diff = vals1.mean() - vals2.mean()
#             stat, pval = ttest_ind(vals1, vals2, equal_var=False)
#             results.append({
#                 "Pathway": pathway,
#                 "mean_diff": mean_diff,
#                 "pval": pval
#             })

#     results_df = pd.DataFrame(results).sort_values(by='pval')

#     return gsva_scores_df, results_df



# def run_gsva_and_test_by_group(data_matrix, pathway_dict, metadata,
#                                 selected_pathways=None, group_col='group',
#                                 group1='recurrent', group2='non-recurrent'):
#     import pandas as pd
#     from gseapy import gsva
#     from scipy.stats import ttest_ind

#     # Select pathways
#     if selected_pathways is None:
#         selected_pathways = list(pathway_dict.keys())

#     subset_pathway_dict = {k: pathway_dict[k] for k in selected_pathways if k in pathway_dict}

#     # ✅ Deduplicate pathway names here
#     unique_keys = make_unique(list(subset_pathway_dict.keys()))
#     subset_pathway_dict = dict(zip(unique_keys, subset_pathway_dict.values()))
# # ✅ Ensure gene names are strings
#     data_matrix.index = data_matrix.index.astype(str)

#     # Then run GSVA
#     gsva_result = gsva(data_matrix, subset_pathway_dict, method='gsva', outdir=None)
   
#     # Run GSVA
#     #gsva_result = gsva(data_matrix, subset_pathway_dict, method='gsva', outdir=None)
#     gsva_scores_df = gsva_result.res2d.pivot(index='Term', columns='Name', values='ES')
#     #gsva_scores_df.index = [f"{i}_{term}" for i, term in enumerate(gsva_scores_df.index)]

#     # ✅ Force numeric dtype
#     gsva_scores_df = gsva_scores_df.apply(pd.to_numeric, errors='coerce')

#     # Perform group-wise t-test
#     group1_ids = metadata[metadata[group_col] == group1].index.astype(str)
#     group2_ids = metadata[metadata[group_col] == group2].index.astype(str)

#     results = []
#     for pathway in gsva_scores_df.index:
#         vals1 = gsva_scores_df.loc[pathway, gsva_scores_df.columns.intersection(group1_ids)].dropna()
#         vals2 = gsva_scores_df.loc[pathway, gsva_scores_df.columns.intersection(group2_ids)].dropna()

#         if len(vals1) > 1 and len(vals2) > 1:
#             mean_diff = vals1.mean() - vals2.mean()
#             stat, pval = ttest_ind(vals1, vals2, equal_var=False)
#             results.append({
#                 "Pathway": pathway,
#                 "mean_diff": mean_diff,
#                 "pval": pval
#             })

#     results_df = pd.DataFrame(results).sort_values(by='pval')
#     #results_df = results_df.set_index('Pathway', drop=False)
#     return gsva_scores_df, results_df

import matplotlib.pyplot as plt
import seaborn as sns

def plot_top_gsva_differences(gsva_scores_df, metadata, test_results_df, top_n=10, group_col='group',
                               group_colors=None, output_dir='./', prefix='gsva'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import pandas as pd

    # Ensure 'Pathway' is index if not already
    if 'Pathway' in test_results_df.columns:
        test_results_df = test_results_df.set_index('Pathway')
        
    #top_pathways = test_results_df[test_results_df['pval'] < 0.06]
    top_pathways = test_results_df.head(top_n).index.tolist()

    # Melt GSVA scores for selected pathways into long format
    df_list = []
    for term in top_pathways:
        tmp = gsva_scores_df.loc[term].reset_index()
        tmp.columns = ['Sample', 'GSVA']
        tmp['Pathway'] = term
                # Filter to samples that exist in metadata
        valid_samples = tmp['Sample'].isin(metadata.index)
        tmp = tmp[valid_samples].copy()
        tmp[group_col] = tmp['Sample'].map(metadata[group_col])

        df_list.append(tmp)

    plot_df = pd.concat(df_list)

    # Create grouped boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=plot_df, x='GSVA', y='Pathway', hue=group_col, palette=group_colors)
    sns.stripplot(data=plot_df, x='GSVA', y='Pathway', hue=group_col,
                  dodge=True, color='black', alpha=0.5, jitter=True)

    plt.xticks(rotation=45, ha='right')
    plt.title(f"Top {top_n} Pathway GSVA Scores by Group")
    plt.tight_layout()

    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{prefix}_top{top_n}_gsva_boxplot.pdf'))
    plt.close()


