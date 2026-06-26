import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

def run_pca(
    X: pd.DataFrame,
    gene_features=None,
    output_path=None,
    fold=0,                         # can be int, str tag ("ASSESS"/"FINAL"/"PRED"), or None
    block_name="unknown",
    n_components=10,
    min_components=1
):
    """
    Run PCA on input DataFrame or return features directly if PCA cannot be performed.

    Args:
        X (pd.DataFrame): Full dataframe with all features.
        gene_features (list or Index): Subset of columns to use for PCA. If None, use all.
        output_path (str): Directory to save PCA artifacts (loadings/scores/explained variance).
        fold (int|str|None): Fold index or string tag for file naming ('ASSESS','FINAL','PRED', etc.).
        block_name (str): Name of the block (for naming output).
        n_components (int): Max number of principal components to retain.
        min_components (int): Minimum required to run PCA, otherwise skip.

    Returns:
        pca_df (pd.DataFrame): PCA-transformed data (scores) or original features if skipped.
        loadings_df (pd.DataFrame or None): PCA loadings per feature, or None if skipped.
    """
    import os
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Build a safe suffix for filenames
    def _fold_suffix(f):
        if f is None or (isinstance(f, str) and f.strip() == ""):
            return ""
        if isinstance(f, (int, np.integer)):
            return f"_fold{int(f) + 1}"
        return f"_{str(f)}"

    suffix = _fold_suffix(fold)

    # Select subset
    X_subset = X[gene_features].copy() if gene_features is not None else X.copy()

    # Determine feasible number of components
    max_components = min(n_components, X_subset.shape[0], X_subset.shape[1])
    if max_components < min_components:
        print(f"⚠️ Skipping PCA: only {max_components} components possible (min required: {min_components}). "
              f"Returning original features.")
        return X_subset.copy(), None

    # Optional: basic sanitization to avoid failures on NaNs/Infs
    X_subset = X_subset.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Standardize then PCA
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X_subset)

    pca = PCA(n_components=max_components)
    pca_result = pca.fit_transform(scaled)

    pc_columns = [f"PC{i+1}" for i in range(pca_result.shape[1])]
    pca_df = pd.DataFrame(pca_result, index=X_subset.index, columns=pc_columns)

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=X_subset.columns,
        columns=[f"{pc}_loading" for pc in pc_columns]
    )

    # Save artifacts
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        loadings_file = os.path.join(output_path, f"PCA_loadings_{block_name}{suffix}.csv")
        scores_file   = os.path.join(output_path, f"PCA_scores_{block_name}{suffix}.csv")
        explvar_file  = os.path.join(output_path, f"PCA_explained_variance_{block_name}{suffix}.csv")

        loadings_df.to_csv(loadings_file)
        pca_df.to_csv(scores_file)
        pd.DataFrame({
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_
        }, index=pc_columns).to_csv(explvar_file)

        print(f"💾 Saved: {loadings_file}")
        print(f"💾 Saved: {scores_file}")
        print(f"💾 Saved: {explvar_file}")

    return pca_df, loadings_df




def merge_pca_with_other_features(pca_df, other_df):
    """
    Merge PCA-transformed data with other features (e.g., clinical or non-RNA).

    Args:
        pca_df (pd.DataFrame): DataFrame with principal components.
        other_df (pd.DataFrame): Additional features to retain.

    Returns:
        merged_df (pd.DataFrame): Combined DataFrame.
    """
    merged = pd.concat([pca_df, other_df], axis=1)
    return merged.fillna(0)
