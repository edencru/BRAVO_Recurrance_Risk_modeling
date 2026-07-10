
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier

model_grid = {
    "Logistic Regression L1": (
        LogisticRegression(),
        {
            "penalty": ["l1"],
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "solver": ["liblinear"],
            "max_iter": [500, 1000]
        }
    ),
    "Logistic Regression L2": (
        LogisticRegression(),
        {
            "penalty": ["l2"],
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "solver": ["liblinear"],
            "max_iter": [500, 1000]
        }
    ),
    "Random Forest": (
        RandomForestClassifier(),
        {
            "n_estimators": [100, 200, 500],
            "max_depth": [None, 10, 20, 40],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2", None]
        }
    ),
    "SVM": (
        SVC(probability=True),
        {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
            "degree": [2, 3]
        }
    ),
    "Naive Bayes": (
        GaussianNB(),
        {}  # No hyperparameters to tune
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 1.0]
        }
    ),
    # "XGBoost": (
    #     XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    #     {
    #         "n_estimators": [100, 200],
    #         "learning_rate": [0.01, 0.1, 0.2],
    #         "max_depth": [3, 5, 7],
    #         "subsample": [0.8, 1.0],
    #         "colsample_bytree": [0.8, 1.0]
    #     }
    # ),
    # "LightGBM": (
    #     LGBMClassifier(),
    #     {
    #         "n_estimators": [100, 200],
    #         "learning_rate": [0.01, 0.1],
    #         "num_leaves": [31, 50],
    #         "subsample": [0.8, 1.0],
    #         "colsample_bytree": [0.8, 1.0]
    #     }
    # ),
    # "CatBoost": (
    #     CatBoostClassifier(verbose=0),
    #     {
    #         "iterations": [100, 200],
    #         "depth": [4, 6],
    #         "learning_rate": [0.01, 0.1]
    #     }
    # )
}



# model_grid = {
#     "Logistic Regression L1": (
#         LogisticRegression(),
#         {
#             "penalty": ["l1"],
#             "C": [0.001, 0.01, 0.1, 1, 10, 100],
#             "solver": ["liblinear"],
#             "max_iter": [500, 1000]
#         }
#     ),
#     "Logistic Regression L2": (
#         LogisticRegression(),
#         {
#             "penalty": ["l2"],
#             "C": [0.001, 0.01, 0.1, 1, 10, 100],
#             "solver": ["liblinear"],
#             "max_iter": [500, 1000]
#         }
#     ),
#     "Random Forest": (
#         RandomForestClassifier(),
#         {
#             "n_estimators": [100, 200, 500],
#             "max_depth": [None, 10, 20, 40],
#             "min_samples_split": [2, 5, 10],
#             "max_features": ["sqrt", "log2", None]
#         }
#     ),
#     "SVM": (
#         SVC(probability=True),
#         {
#             "C": [0.001, 0.01, 0.1, 1, 10, 100],
#             "kernel": ["linear", "rbf"],
#             "gamma": ["scale", "auto"],
#             "degree": [2, 3]
#         }
#     ),
#     "Gradient Boosting": (
#         GradientBoostingClassifier(),
#         {
#             "n_estimators": [100, 200],
#             "learning_rate": [0.01, 0.1, 0.2],
#             "max_depth": [3, 5, 7],
#             "subsample": [0.8, 1.0]
#         }
#     )
# }

