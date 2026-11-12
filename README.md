# BRAVO Recurrence-Risk Modeling Workflow

This repository contains the scripts used to assess, train, and apply subtype-specific recurrence-risk models from multi-omic breast cancer data. The workflow is organized around five main analysis stages:

1. Initial cross validations model-family selection
2. Iterated cross validation performance assessment
3. Final full-cohort model training with cross validation for hyperparamter selection
4. External validation feature processing
5. Prediction and validation plotting

A shared feature-selection module is used within the model-selection, iterated-CV, and final-training scripts. It can also be run as a standalone script for debugging, auditing, or generating feature-selection outputs directly, but its primary role in the modeling workflow is as an internal feature-construction and feature-selection function.

The commands below are templates. Replace the placeholder values with the exact parameters used for each subtype. Subtype-specific settings can be filled in for Basal, Luminal A, and Luminal B models separately.

---

## Shared Feature-Selection Module

Script:

```bash
feature_selection_only.py
```

Purpose:

The feature-selection script provides the shared feature-construction and feature-selection workflow used by the modeling scripts. It builds candidate RNA, CNV, MUT and CLINICAL features; applies label-free filtering; performs supervised feature selection; and writes selected block matrices.

In the main modeling workflow, this script is usually called internally through `run_feature_selection()`. This allows each modeling script to run feature selection in the correct context:

* During outer CV, feature selection is run inside each training fold to avoid leakage.
* During iterated CV, feature selection is repeated within each seeded fold split.
* During final training, feature selection is run on the full training cohort or reused from an existing final feature-selection directory.

The script can also be run directly when you want to inspect feature-selection outputs, troubleshoot feature construction, or generate a reusable feature-selection directory.

Optional standalone command template:

```bash
python feature_selection_only.py \
  --data_path <INTERNAL_DATA_DIR> \
  --output_dir <FEATURE_SELECTION_OUTPUT_DIR> \
  --subtype <SUBTYPE_NAME> \
  --status_col <DISEASE_STATUS_COLUMN> \
  --pathway_options <PATHWAY_OPTION_LIST> \
  --run_mwu \
  --run_cox_uni \
  --run_cox_multi \
  --cox_time_col <SURVIVAL_TIME_COLUMN> \
  --cox_event_col <SURVIVAL_EVENT_COLUMN> \
  --cox_covariates <COVARIATE_1,COVARIATE_2,...> \
  --fs_corr_threshold <REDUNDANCY_CORRELATION_THRESHOLD> \
  --mut_pathway_source <snv|import|none> \
  --ptmb_source <none|snv|import> \
  <SUBTYPE_SPECIFIC_FEATURE_SELECTION_PARAMETERS>
```

Outputs needed downstream:

```text
RNAblock_selected_post_redundancy.csv
CNVblock_selected_post_redundancy.csv
MUTblock_selected_post_redundancy.csv
CLINICALblock_selected_post_redundancy.csv
FINAL_features_only.csv
ALL_FEATURES_selection_table.csv
```

---

## 1. Initial Outer-CV Model-Family Selection

Script:

```bash
cross_validation_model_selection.py
```

Purpose:

This script runs a leak-free nested CV workflow to compare feature blocks and candidate model families. It was used to select model families to carry forward into final training.

Within each outer fold, `run_feature_selection()` is called using only the training samples for that fold. Hyperparameters are tuned using inner CV, and the tuned model is evaluated on the held-out outer fold. This provides model-family and feature-block comparison, while the broader repeated performance assessment is handled by the iterated CV script.

 The final locked model families were Gradient Boosting for Basal RNA, mutation, and ALL blocks, ExtraTrees for Basal copy number, and L2-penalized logistic regression for Basal clinical features; LinearSVC for Luminal A RNA and clinical features, Random Forest for Luminal A copy-number and mutation features, and elastic-net logistic regression for Luminal A ALL features; and Gradient Boosting for Luminal B RNA features, Random Forest for Luminal B copy-number features, LinearSVC for Luminal B mutation features, L1-penalized logistic regression for Luminal B clinical features, and L2-penalized logistic regression for Luminal B ALL features. L2-penalized logistic regression was used for all stacked ensemble models. 

Example command:

```bash
python cross_validation_model_selection.py \
  --data_path <INTERNAL_DATA_DIR> \
  --output_dir <MODEL_SELECTION_OUTPUT_DIR> \
  --subtype <SUBTYPE_NAME> \
  --status_col <DISEASE_STATUS_COLUMN> \
  --pathway_options <GSVA_PATHWAY_OPTION_LIST> \
  --outer_splits <N_OUTER_FOLDS> \
  --inner_splits <N_INNER_FOLDS> \
  --random_state <SEED> \
  --cv_blocks <RNA,CNV,MUT,CLINICAL,ALL> \
  --primary_test <mwu|cox> \
  --run_mwu <true_or_false> \
  --run_cox_uni <true_or_false> \
  --run_cox_multi <true_or_false> \
  --time_col <SURVIVAL_TIME_COLUMN> \
  --event_col <SURVIVAL_EVENT_COLUMN> \
  --cox_covariates <COVARIATE_1,COVARIATE_2,...> \
  --fs_p_thresh <FS_P_THRESHOLD> \
  --fs_fdr_thresh <FS_FDR_THRESHOLD> \
  --fs_corr_threshold <REDUNDANCY_CORRELATION_THRESHOLD> \
  --mut_pathway_source <snv|import|none> \
  --mut_binary_matrix_csv <OPTIONAL_PRECOMPUTED_BINARY_MUTATION_MATRIX> \
  --ptmb_source <none|snv|import> \
  --ptmb_matrix_csv <OPTIONAL_PRECOMPUTED_PTMB_MATRIX> \
  --stack_base_mode <best_per_block|fixed_family> \
  --stack_base_family <OPTIONAL_FIXED_MODEL_FAMILY> \
  --stack_meta_model <STACK_META_MODEL> \
  --stack_cv <STACK_INNER_CV>
```

Key outputs:

```text
<MODEL_SELECTION_OUTPUT_DIR>/<SUBTYPE>/cv_models/
<MODEL_SELECTION_OUTPUT_DIR>/<SUBTYPE>/predictions/
<MODEL_SELECTION_OUTPUT_DIR>/<SUBTYPE>/plots/
<MODEL_SELECTION_OUTPUT_DIR>/<SUBTYPE>/matrices/
<MODEL_SELECTION_OUTPUT_DIR>/<SUBTYPE>/preprocessing/
```

Use these outputs to decide which model family should be carried forward for each subtype.

---

## 2. Iterated Outer-CV Performance Assessment

Script:

```bash
cross_validation_iterations.py
```

Purpose:

This script runs one seeded CV assessment for each modeling strategy. It is intended to be repeated across multiple seeds to assess model stability and performance across different train/validation splits.

This script was used for the primary cross-validated performance assessment after the candidate model families had been selected (listed above). 

Example command for one iteration:

```bash
python cross_validation_iterations.py \
  --data_path <INTERNAL_DATA_DIR> \
  --output_dir <ITERATED_CV_OUTPUT_DIR> \
  --subtype <SUBTYPE_NAME> \
  --run_tag <SUBTYPE_RUN_TAG_OR_SEED_TAG> \
  --seed <SEED_MUST_BE_DIFFERENT_FOR_EVERY_ITERATION> \
  --status_col <DISEASE_STATUS_COLUMN> \
  --pathway_options <PATHWAY_OPTION_LIST> \
  --outer_splits <N_OUTER_FOLDS> \
  --inner_splits <N_INNER_FOLDS> \
  --cv_blocks <RNA,CNV,MUT,CLINICAL,ALL> \
  --primary_test <mwu|cox> \
  --run_mwu <true_or_false> \
  --run_cox_uni <true_or_false> \
  --run_cox_multi <true_or_false> \
  --time_col <SURVIVAL_TIME_COLUMN> \
  --event_col <SURVIVAL_EVENT_COLUMN> \
  --cox_covariates <COVARIATE_1,COVARIATE_2,...> \
  --fs_p_thresh <FS_P_THRESHOLD> \
  --fs_fdr_thresh <FS_FDR_THRESHOLD> \
  --fs_corr_threshold <REDUNDANCY_CORRELATION_THRESHOLD> \
  --model_id <SELECTED_MODEL_FAMILY_OR_ALL> \
  --block_family_map <OPTIONAL_BLOCK_TO_MODEL_MAP> \
  --stack_base_mode <best_per_block|fixed_family> \
  --stack_base_family <OPTIONAL_FIXED_MODEL_FAMILY> \
  --stack_meta_model <STACK_META_MODEL> \
  --mut_pathway_source <snv|import|none> \
  --mut_binary_matrix_csv <OPTIONAL_PRECOMPUTED_BINARY_MUTATION_MATRIX> \
  --ptmb_source <none|snv|import> \
  --ptmb_matrix_csv <OPTIONAL_PRECOMPUTED_PTMB_MATRIX>
```

This was run with 1-100 seeds and 1-100 run_tag to match. 

Key outputs per run:

```text
<ITERATED_CV_OUTPUT_DIR>/<SUBTYPE>/<RUN_TAG>/cv_models/
<ITERATED_CV_OUTPUT_DIR>/<SUBTYPE>/<RUN_TAG>/predictions/
<ITERATED_CV_OUTPUT_DIR>/<SUBTYPE>/<RUN_TAG>/plots/
<ITERATED_CV_OUTPUT_DIR>/<SUBTYPE>/<RUN_TAG>/matrices/
<ITERATED_CV_OUTPUT_DIR>/<SUBTYPE>/<RUN_TAG>/preprocessing/
```

---

## 3. Final Full-Cohort Model Training

Script:

```bash
final_training_w_hyperparam_cv.py
```

Purpose:

This script trains the final locked models on the full subtype-specific training cohort. It fits the selected final models on all available training samples and saves the artifacts needed for prediction and external validation.


Example command using an existing feature-selection directory:

```bash
python final_training_w_hyperparam_cv.py \
  --data_path <INTERNAL_DATA_DIR> \
  --output_dir <FINAL_MODEL_OUTPUT_DIR> \
  --subtype <SUBTYPE_NAME> \
  --status_col <DISEASE_STATUS_COLUMN> \
  --fs_dir_existing <FEATURE_SELECTION_DIR_FOR_FINAL_MODEL> \
  --model_id <SELECTED_MODEL_FAMILY> \
  --block_model_csv <OPTIONAL_PER_BLOCK_MODEL_CSV> \
  --use_cv_winners <true_or_false> \
  --blocks <RNA,CNV,MUT,CLINICAL,ALL> \
  --calibrate <true_or_false> \
  --calibrate_method <sigmoid|isotonic> \
  --mut_pathway_source <snv|import|none> \
  --mut_binary_matrix_csv <OPTIONAL_PRECOMPUTED_BINARY_MUTATION_MATRIX> \
  --ptmb_source <none|snv|import> \
  --ptmb_matrix_csv <OPTIONAL_PRECOMPUTED_PTMB_MATRIX> \
  --fit_stacked \
  --stack_blocks <RNA,CNV,MUT,CLINICAL> \
  --stack_meta_model <logreg|rf|gbdt> \
  --block_model_csv <OPTIONAL_PER_BLOCK_MODEL_CSV> 
  
  ```

Key outputs:

```text
<FINAL_MODEL_OUTPUT_DIR>/<SUBTYPE>/cv_models/
<FINAL_MODEL_OUTPUT_DIR>/<SUBTYPE>/preprocessing/
<FINAL_MODEL_OUTPUT_DIR>/<SUBTYPE>/predictions/
<FINAL_MODEL_OUTPUT_DIR>/<SUBTYPE>/plots/
<FINAL_MODEL_OUTPUT_DIR>/<SUBTYPE>/coefficients/
<FINAL_MODEL_OUTPUT_DIR>/<SUBTYPE>/shap/
```

Important saved files may include:

```text
ALL_FINAL.pkl
RNA_FINAL.pkl
CNV_FINAL.pkl
MUT_FINAL.pkl
CLINICAL_FINAL.pkl
STACKED_FINAL.pkl
global_cont_scaler.joblib
train_feature_order.csv
global_scaled_columns.csv
ALL_BLOCKS_FINAL_features.csv
ALL_BLOCKS_CONCAT_unscaled.csv
ALL_BLOCKS_CONCAT_scaled.csv
train_feature_stats.csv
```

---

## 4. External Validation Feature Processing

Script:

```bash
validation_processing.py
```

Purpose:

This script converts raw or semi-processed external validation cohort data into a model-ready feature matrix that matches the trained model’s expected feature schema.

It does not train models or run feature selection. Instead, it uses the training feature-selection outputs and final feature list to determine which validation features need to be rebuilt, then writes a final ordered validation matrix for prediction. Desgined for METABRIC or TCGA data. 

Example command:

```bash
python validation_processing.py \
  --outdir <VALIDATION_FEATURE_OUTPUT_DIR> \
  --prefix <VALIDATION_COHORT_PREFIX> \
  --training_fs_dir <FINAL_TRAINING_FS_DIR> \
  --training_final_feature_file <FINAL_FEATURE_LIST_FILE> \
  --expr_matrix <VALIDATION_EXPRESSION_MATRIX> \
  --expr_orientation <auto|samples_by_genes|genes_by_samples> \
  --id_map <GENE_ID_MAP_OR_GTF> \
  --id_map_format <auto|gtf|table> \
  --rna_gmt_paths <RNA_PATHWAY_GMT_1,RNA_PATHWAY_GMT_2,...> \
  --custom_gene_sets_file <CUSTOM_GENE_SET_FILE> \
  --precomputed_rna_geneset_csv <OPTIONAL_PRECOMPUTED_RNA_GENESET_MATRIX> \
  --cnv_matrix <VALIDATION_CNV_MATRIX> \
  --cnv_gmt_paths <CNV_PATHWAY_GMT_1,CNV_PATHWAY_GMT_2,...> \
  --mutation_matrix <VALIDATION_MUTATION_MATRIX> \
  --mut_gmt_paths <MUTATION_PATHWAY_GMT_1,MUTATION_PATHWAY_GMT_2,...> \
  --tmb_summary <OPTIONAL_TMB_SUMMARY_FILE> \
  --maf_path <OPTIONAL_VALIDATION_MAF_FILE> \
  --ptmb_gmt_path <OPTIONAL_PTMB_GMT_FILE> \
  --ptmb_gtf_path <OPTIONAL_GTF_FILE> \
  --cibersort_abs <OPTIONAL_CIBERSORT_ABS_FILE> \
  --tide_scores <OPTIONAL_TIDE_FILE> \
  --estimate_scores <OPTIONAL_ESTIMATE_FILE> \
  --ecotyper_base_dir <OPTIONAL_ECOTYPER_DIRECTORY> \
  --clinical_csv <OPTIONAL_CLINICAL_FILE> \
  --tcga_vial_strip_mode <none|per_block|final> \
  --mapping_log <OPTIONAL_FEATURE_MAPPING_LOG>
```

Key outputs:

```text
<VALIDATION_FEATURE_OUTPUT_DIR>/<VALIDATION_COHORT_PREFIX>__combined_pre.csv
<VALIDATION_FEATURE_OUTPUT_DIR>/<VALIDATION_COHORT_PREFIX>__combined_features.csv
<VALIDATION_FEATURE_OUTPUT_DIR>/<VALIDATION_COHORT_PREFIX>__selected_ordered_features.csv
```

---

## 5. Prediction and Validation Plotting

Script:

```bash
predict_script_with_hr_km_tables.py
```

Purpose:

This script applies trained final models to a prepared prediction matrix. It does not construct features or train models. It loads saved model artifacts, aligns the validation matrix, generates predicted recurrence-risk probabilities, and creates plots when labels or survival data are available.

Example prediction command:

```bash
python <PREDICTION_SCRIPT.py> \
  --model_output_dir <FINAL_MODEL_OUTPUT_DIR> \
  --prediction_output_dir <PREDICTION_OUTPUT_DIR> \
  --subtype <SUBTYPE_NAME> \
  --predict_matrix_csv <VALIDATION_ORDERED_FEATURE_MATRIX> \
  --predict_labels_csv <OPTIONAL_VALIDATION_LABELS_FILE> \
  --models <RNA,CNV,MUT,CLINICAL,ALL,STACKED> \
  --percentiles <median|quartiles|tertiles|deciles|CUSTOM_PERCENTILES> \
  --time_col <OPTIONAL_SURVIVAL_TIME_COLUMN> \
  --event_col <OPTIONAL_SURVIVAL_EVENT_COLUMN> \
  --status_col <OPTIONAL_RECURRENCE_STATUS_COLUMN> \
  --legend_mode <separate|embedded|both> \
  --km_show_ci <true_or_false> \
  --model_colors <OPTIONAL_MODEL_COLOR_MAP> \
  --group_colors <OPTIONAL_GROUP_COLOR_MAP> \
  --label_colors <OPTIONAL_LABEL_COLOR_MAP>
```


Key outputs:

```text
<PREDICTION_OUTPUT_DIR>/<LABEL_FILE_STEM>/predictions/
<PREDICTION_OUTPUT_DIR>/<LABEL_FILE_STEM>/plots/
```


## Basal Feature-Selection Thresholds

Feature selection for the Basal subtype was run on the full cohort subset (`n = 142`) using recurrence status as the primary label (`Disease_Status`; recurrent vs non-recurrent). 

### Label-Free Filtering

Variance filtering was performed before supervised testing using a top-fraction-by-group strategy. Within each feature group, the top 40% most variable features were retained.

```text
Variance method: top fraction by group
Top variance fraction retained: 0.40
Minimum non-missing samples: 10
```


### Correlation Prefiltering and Redundancy Pruning

A label-correlation prefilter was applied before supervised testing.

```text
Correlation prefilter: enabled
Minimum absolute feature-label correlation: 0.15
```

After supervised feature selection, redundancy pruning was performed using feature-feature correlation.

```text
Redundancy-pruning correlation threshold: |r| ≥ 0.90
Correlation-pruning mode: hierarchical
Priority order: multivariable Cox → univariate Cox → Mann-Whitney U or Fisher's significance test
```

### Supervised Feature-Selection Tests

Features were selected using Mann-Whitney U testing, univariate Cox regression, and multivariable Cox regression.

```text
MWU nominal or p-value threshold: 0.05
MWU FDR threshold: 0.25 for RNA/CNV, 1.0 for MUT

Univariate Cox nominal p-value threshold: 0.05
Univariate Cox FDR threshold: 0.25 for RNA/CNV, 1.0 for MUT

Multivariable Cox nominal p-value threshold: 0.05
Multivariable Cox FDR threshold: 0.25 for RNA/CNV, 1.0 for MUT
```

RNA limma-derived evidence used the following thresholds:

```text
limma p-value threshold: 0.05
limma adjusted p-value threshold: 1.0
limma absolute log2FC threshold: 0.25
```


## Luminal A Feature-Selection Thresholds

Feature selection for the Luminal A subtype was run on the full cohort subset (`n = 97`) using recurrence status as the primary label (`Disease_Status`; recurrent vs non-recurrent). 

### Label-Free Filtering

Variance filtering was performed before supervised testing using a top-fraction-by-group strategy. Within each feature group, the top 40% most variable features were retained.

```text
Variance method: top fraction by group
Top variance fraction retained: 0.40
Minimum non-missing samples: 5
```

### Correlation Prefiltering and Redundancy Pruning

A label-correlation prefilter was applied before supervised testing.

```text
Correlation prefilter: enabled
Minimum absolute feature-label correlation: 0.15
```

After supervised feature selection, redundancy pruning was performed using feature-feature correlation.

```text
Redundancy-pruning correlation threshold: |r| ≥ 0.90
Correlation-pruning mode: hierarchical
Priority order: multivariable Cox → univariate Cox → Mann-Whitney U or Fisher's significance test
```

### Supervised Feature-Selection Tests

Features were selected using Mann-Whitney U testing, univariate Cox regression, and multivariable Cox regression.

```text
MWU nominal p-value threshold: 0.05
MWU FDR threshold: 1.0

Univariate Cox nominal p-value threshold: 0.05
Univariate Cox FDR threshold: 1.0

Multivariable Cox nominal p-value threshold: 0.05
Multivariable Cox FDR threshold: 1.0
```

RNA limma-derived evidence used the following thresholds:

```text
limma p-value threshold: 0.05
limma adjusted p-value threshold: 1.0
limma absolute log2FC threshold: 0.25
```

## Luminal B Feature-Selection Thresholds

Feature selection for the Luminal B subtype was run on the full cohort subset (`n = 100`) using recurrence status as the primary label (`Disease_Status`; recurrent vs non-recurrent). 

### Label-Free Filtering

Variance filtering was performed before supervised testing using a top-fraction-by-group strategy. Within each feature group, the top 40% most variable features were retained.

```text
Variance method: top fraction by group
Top variance fraction retained: 0.40
Minimum non-missing samples: 10
```

### Correlation Prefiltering and Redundancy Pruning

A label-correlation prefilter was applied before supervised testing.

```text
Correlation prefilter: enabled
Minimum absolute feature-label correlation: 0.15
```

After supervised feature selection, redundancy pruning was performed using feature-feature correlation.

```text
Redundancy-pruning correlation threshold: |r| ≥ 0.90
Correlation-pruning mode: hierarchical
Priority order: multivariable Cox → univariate Cox → Mann-Whitney U or Fisher's significance test
```

### Supervised Feature-Selection Tests

Features were selected using Mann-Whitney U testing, univariate Cox regression, and multivariable Cox regression.

```text
MWU nominal p-value threshold: 0.05
MWU FDR threshold: 1.0

Univariate Cox nominal p-value threshold: 0.05
Univariate Cox FDR threshold: 1.0

Multivariable Cox nominal p-value threshold: 0.05
Multivariable Cox FDR threshold: 1.0
```

RNA limma-derived evidence used the following thresholds:

```text
limma p-value threshold: 0.05
limma adjusted p-value threshold: 1.0
limma absolute log2FC threshold: 0.25
```



---
### Versions of packages used 

```text
python: 3.11.3 
numpy: 1.26.4
pandas: 2.2.2
matplotlib: 3.8.4
sklearn: 1.6.1
xgboost: 2.0.3
lightgbm: 3.3.2
catboost: 1.2.5
```
