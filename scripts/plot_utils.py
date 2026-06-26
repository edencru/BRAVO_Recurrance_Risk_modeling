import os
import matplotlib.pyplot as plt
import numpy as np


def plot_roc(results, output_path, mean_fpr):
    """
    Plot mean ROC curves with shaded AUC confidence bands for each model.

    Args:
        results (dict): Dictionary of model results including 'tprs' and 'aucs'.
        output_path (str): Directory to save the ROC plot.
        mean_fpr (np.array): Common FPR values to interpolate TPRs against.
    """
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

    for model_name, metrics in results.items():
        if metrics['tprs']:
            mean_tpr = np.mean(metrics['tprs'], axis=0)
            mean_tpr[-1] = 1.0
            std_tpr = np.std(metrics['tprs'], axis=0)
            mean_auc = np.mean(metrics['aucs'])
            std_auc = np.std(metrics['aucs'])

            plt.plot(mean_fpr, mean_tpr, lw=2,
                     label=f"{model_name} (AUC = {mean_auc:.2f} ± {std_auc:.2f})")
            plt.fill_between(mean_fpr,
                             mean_tpr - std_tpr,
                             mean_tpr + std_tpr,
                             alpha=0.2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Mean ROC Curve (5-Fold CV)')
    plt.legend(loc="lower right", fontsize='small')
    plt.grid(True)
    out_file = os.path.join(output_path, "roc_curve.png")
    plt.savefig(out_file, dpi=300)
    print(f"Saved: {out_file}")
    plt.close()
