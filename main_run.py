"""
QSAR Biodegradation Classification

================================================================================
IMPLEMENTATION DETAILS

Description:
    Heterogeneous Stacking Ensemble for QSAR Biodegradation Classification.
    This script executes the complete pipeline and generates all results
    and figures reported in the accompanying paper.

Pipeline Architecture:
    1. Data Loading: Load 'QSAR_data.mat' (UCI QSAR Biodegradation dataset)
    2. Pre-processing: Remove duplicate samples and features
    3. Feature Selection:
       - Boruta: Non-linear feature relevance detection
       - LASSO: Sparse linear feature selection
    4. Base Learners:
       - Branch 1: Boruta features → HistGradientBoosting
       - Branch 2: LASSO features → Linear SVM
       - Branch 3: Full features → Distance-weighted KNN
    5. Meta-Learner: L2-regularised Logistic Regression
    6. Evaluation: Nested Cross-Validation (3 outer folds, 5 inner folds)

================================================================================
EXTERNAL LIBRARIES REQUIRED
    - py version python-3.13
    - numpy (>=1.21.0)
    - pandas (>=1.3.0)
    - matplotlib (>=3.4.0)
    - seaborn (>=0.11.0)
    - scikit-learn (>=1.0.0)
    - scipy (>=1.7.0)
    - boruta (>=0.3)

    Install via: pip install numpy pandas matplotlib seaborn scikit-learn scipy boruta

================================================================================
CUSTOM MODULES

    - data_loader.py: Functions for loading .mat files
    - pre_process.py: Data cleaning and duplicate removal
    - boruta_class.py: Sklearn-compatible Boruta transformer
    - ensemble_model.py: Stacking ensemble builder
    - evaluate_base_learners.py: Base learner evaluation utilities
    - plots.py: Visualisation functions

================================================================================
OUTPUT FILES GENERATED

    - fig1_confusion_matrix.pdf: Normalised confusion matrix
    - fig2_roc_curve.pdf: ROC curve with AUC
    - fig3_pr_curve.pdf: Precision-recall curve
    - fig4_learning_curve.pdf: Learning curve (F1 vs training size)
    - fig5_meta_weights.pdf: Meta-learner coefficients
    - fig6_branch_roc_comparison.pdf: Base learner ROC comparison
"""

# ================================================================
# Standard Library Imports
import os
import sys
import joblib

# ================================================================
# Core Scientific Python Stack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================================================================
# Scikit-Learn: Model Selection & Metrics
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    recall_score
)
# ================================================================
# Custom Local Modules

from data_loader import load_mat_file
from pre_process import (
    data_checks,
    remove_duplicate_samples,
    remove_duplicate_features
)
from boruta_class import BorutaTransformer
from ensemble_model import build_heterogeneous_stack
from evaluate_base_learners import evaluate_base_learners
from plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_learning_curve,
    plot_meta_weights,
    plot_branch_roc_curves,
    boruta_learning_curve
)

# ============================================================================
# Main Code

DATASET_FILENAME = 'QSAR_data.mat'

def main():
    #Generates all results and figures for the report 
    print(" QSAR Classification Pipeline ")

    # =========================================================================
    # 1. Data Loading
    print(f"1/5 Loading Dataset: '{DATASET_FILENAME}'")

    if not os.path.exists(DATASET_FILENAME):
        print(f"File '{DATASET_FILENAME}' not found.")
        sys.exit()

    df = load_mat_file(DATASET_FILENAME)
    print("-" * 72)
    print(" 2/5  Pre-processing... ")
    # Check for missing values and class distribution
    data_checks(df)

    # Remove exact duplicate rows to prevent data leakage
    df = remove_duplicate_samples(df)

    # Separate features (X) and target (y)
    # All columns except last are molecular descriptors
    # Last column is biodegradability label (0 = NRB, 1 = RB)
    X_raw = df.iloc[:, :-1]
    y = df.iloc[:, -1].astype(int)

    # Remove duplicate/constant features to reduce redundancy
    X = remove_duplicate_features(X_raw).copy()

    print(f"Final Dataset: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Class Distribution: {np.bincount(y)} (NRB: {np.sum(y==0)}, RB: {np.sum(y==1)})")
    # =========================================================================
    # 3.  Nested cross-validation on the outer loop
   
    # Outer loop estimates generalisation performance
    # Inner loop (inside StackingClassifier) trains base learners
    print("-" * 72)
    print(" 3/5 Nested Cross-Validation ")

    # Stratified split preserves class proportions in each fold
    outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    outer_f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), 1):
        print(f" Outer Fold {fold}")

        # Split data for this fold
        X_train_outer = X.iloc[train_idx]
        y_train_outer = y.iloc[train_idx]

        X_val_outer = X.iloc[val_idx]
        y_val_outer = y.iloc[val_idx]

        # --------------------------------------------------
        #  Fitting boruta on outer trainning data
       
        boruta = BorutaTransformer(
            n_estimators=100,
            max_iter=50,
            random_state=42
        )
        boruta.fit(X_train_outer, y_train_outer)

        # --------------------------------------------------
        #  Building stack with frozen Boruta Coefficients
        
        model = build_heterogeneous_stack(prefit_boruta=boruta)

        # --------------------------------------------------
        #  Fitting model 
    
        model.fit(X_train_outer, y_train_outer)

        # --------------------------------------------------
        #  Evaluation on outer validation data
        
        y_val_pred = model.predict(X_val_outer)
        fold_f1 = f1_score(y_val_outer, y_val_pred)

        print(f"Fold {fold} F1: {fold_f1:.4f}")
        outer_f1_scores.append(fold_f1)

    # Summary statistics across folds
    print(" Nested CV F1 scores:", outer_f1_scores)
    print(" Mean F1:", np.mean(outer_f1_scores))

    # =========================================================================
    # 4. Final train/test split for reporting
    print("-" * 72)
    print(" 4/5  Final Model Training (80/20 Split)")

    # Stratified split ensures class proportions match full dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ---------------------------------------------------------
    # Fit Boruta on final training data
   
   # Boruta on training set only - same procedure as nested CV
    boruta_final = BorutaTransformer(
        n_estimators=100,
        max_iter=50,
        random_state=42
    )
    boruta_final.fit(X_train, y_train)

    # ---------------------------------------------------------
    # Build ensemble with frozen Boruta selector
  
    model = build_heterogeneous_stack(prefit_boruta=boruta_final)
    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # Generate predictions on held-out test set
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    auc_score = roc_auc_score(y_test, y_proba)
    print(f"Final AUC: {auc_score:.4f}")

    # =========================================================================
    # 5. Plots
    print("-" * 72)
    print(" 5/5 Generating Plots")

    plot_confusion_matrix(
        y_test, y_pred,
        "fig1_confusion_matrix.pdf")

    plot_roc_curve(
        y_test, y_proba,
        "fig2_roc_curve.pdf")

    plot_pr_curve(
        y_test, y_proba,
        "fig3_pr_curve.pdf")

    # Learning curve calculation
    train_sizes, train_scores, cv_scores = boruta_learning_curve(
        X=X,
        y=y,
        build_model_fn=build_heterogeneous_stack,
        train_sizes=np.linspace(0.2, 1.0, 5),
        scoring="f1")

    print("Learning curve shapes:")
    print("train_sizes :", train_sizes.shape)
    print("train_scores:", train_scores.shape)
    print("cv_scores   :", cv_scores.shape)
    print("Train mean:", np.mean(train_scores, axis=1))
    print("CV mean   :", np.mean(cv_scores, axis=1))

    plot_learning_curve(
        train_sizes,
        train_scores,
        cv_scores,
        "fig4_learning_curve.pdf")

    # Meta-learner weights
    plot_meta_weights(
        model.final_estimator_,
        ["HGB", "SVM", "KNN"],
        "fig5_meta_weights.pdf")

    # Branch ROC comparison
    plot_branch_roc_curves(
        fitted_stack=model,
        X_test=X_test,
        y_test=y_test,
        save_path="fig6_branch_roc_comparison.pdf")
        
    # =========================================================================
    # 6. Final Performance Report
    
     # Compute ensemble metrics
    
    ensemble_sens = recall_score(y_test, y_pred, pos_label=1)
    ensemble_spec = recall_score(y_test, y_pred, pos_label=0)
    final_f1 = f1_score(y_test, y_pred)

    ensemble_metrics = {
        "AUC": auc_score,
        "F1": final_f1,
        "Sensitivity": ensemble_sens,
        "Specificity": ensemble_spec
    }

    print(f"    Test Set: {len(y_test)} samples")
    print(f"    Positive Class Rate: {np.mean(y_test):.3f}")
    print("-" * 72)
    print(f"    ROC-AUC:     {auc_score:.4f}")
    print(f"    F1 Score:    {final_f1:.4f}")
    print(f"    Sensitivity: {ensemble_sens:.4f}")
    print(f"    Specificity: {ensemble_spec:.4f}")

    # Base learner comparison
    print("-" * 72)
    print("    Base Learner Comparison")
    print("-" * 72)
    base_results = evaluate_base_learners(model, X_test, y_test)

    # Generate latex table
    from plots import generate_latex_table
    latex_table = generate_latex_table(
        ensemble_metrics,
        base_results,
        save_path="table_comparison.tex"
    )
    print("    LaTeX Table:")
    print(latex_table)

    # Classification report
    
    print("    Classification Report")
    print("-" * 72)
    print(classification_report(y_test, y_pred, digits=4))  

    
    # Summary of outputs
    print("Plots Generated")
    print("-" * 72)

    print(" fig1_confusion_matrix.svg")
    print(" fig2_roc_curve.svg")
    print(" fig3_pr_curve.svg")
    print(" fig4_learning_curve.svg")
    print(" fig5_meta_weights.svg")

    print(" Pipeline Complete")
    

if __name__ == "__main__":
    main()