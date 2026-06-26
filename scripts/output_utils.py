import os
import pandas as pd
import numpy as np


def save_predictions(predictions, output_path):
    """
    Save classification predictions to CSV.
    """
    df = pd.DataFrame(predictions)
    out_file = os.path.join(output_path, "classification_results.csv")
    df.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")


def save_metrics(results_dict, output_path, block_name=None, fold=None):
    """
    Compute and save average evaluation metrics (accuracy, precision, etc.) for each model.

    Args:
        results_dict (dict): Dictionary of model evaluation results.
        output_path (str): Directory to save the output.
        block_name (str, optional): Used to label the file.
        fold (int, optional): Fold number for labeling.

    Returns:
        pd.DataFrame: DataFrame of averaged metrics with 'Classifier' column.
    """
    summary = {}
    for model_name, metrics in results_dict.items():
        summary[model_name] = {
            "accuracy": np.mean(metrics["accuracies"]),
            "precision": np.mean(metrics["precisions"]),
            "recall": np.mean(metrics["recalls"]),
            "f1": np.mean(metrics["f1s"]),
            "auc": np.mean(metrics["aucs"]) if metrics["aucs"] else None
        }

    df = pd.DataFrame.from_dict(summary, orient="index")
    df['Classifier'] = df.index

    # Construct output filename
    suffix = f"{block_name}_" if block_name else ""
    suffix += f"fold{fold}_" if fold is not None else ""
    filename = f"model_metrics_{suffix.rstrip('_')}.csv"
    out_file = os.path.join(output_path, filename)

    df.to_csv(out_file, index=False)
    print(f"💾 Saved: {out_file}")
    return df




def save_feature_importance(importances, output_path):
    """
    Save combined feature importance dataframe (SHAP, coefficients, etc.).
    """
    if importances:
        combined = pd.concat(importances, ignore_index=True)
        out_file = os.path.join(output_path, "feature_importance_summary.csv")
        combined.to_csv(out_file, index=False)
        print(f"Saved: {out_file}")


def save_top_genes_by_pc(loadings, top_pc_list, output_path, fold, top_n=20):
    """
    Save top contributing genes for each PC in the fold.
    """
    for pc in top_pc_list:
        col = f"{pc}_loading"
        if col in loadings.columns:
            top_genes = loadings[col].abs().sort_values(ascending=False).head(top_n)
            top_genes_df = top_genes.reset_index()
            top_genes_df.columns = ["Gene", f"{pc}_abs_loading"]
            out_file = os.path.join(output_path, f"top_genes_by_{pc}_fold{fold+1}.csv")
            top_genes_df.to_csv(out_file, index=False)
            print(f"Saved: {out_file}")


def mean_aucs(metrics_df, output_path):
    """
    Save model ranking summary by AUC and F1.
    """
    metrics_df["AUC_rank"] = metrics_df["auc"].rank(ascending=False)
    metrics_df["F1_rank"] = metrics_df["f1"].rank(ascending=False)
    ranked = metrics_df.sort_values(by=["AUC_rank", "F1_rank"])
    out_file = os.path.join(output_path, "model_ranking_summary.csv")
    ranked.to_csv(out_file)
    print(f"Saved: {out_file}")
