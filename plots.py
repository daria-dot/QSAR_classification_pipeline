# ============================================================
# plots.py  visualisation utilities
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from boruta_class import BorutaTransformer

# ------------------------------------------------------------
# Global styling 

sns.set_style("whitegrid")
plt.rcParams.update({
    "axes.edgecolor": "0.3",
    "axes.linewidth": 1.0,
    "font.size": 12,              # base font
    "axes.labelsize": 13,         # x/y labels
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 120
})

# ============================================================
# 1. CONFUSION MATRIX

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        square=True,
        cbar=True,
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

# ============================================================
# 2. ROC CURVE

def plot_roc_curve(y_true, y_proba, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    ax.plot(
        fpr, tpr,
        lw=2.5,
        color="darkorange",
        label=f"AUC = {roc_auc:.3f}"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(color="lightgray", linestyle="--", linewidth=0.6)

    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

# ============================================================
# 3. PRECISION–RECALL CURVE


def plot_pr_curve(y_true, y_proba, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = auc(recall, precision)   # IEEE metrics

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.plot(
        recall, precision,
        lw=2.5,
        color="navy",
        label=f"AP = {ap:.3f}"
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right")
    ax.grid(color="lightgray", linestyle="--", linewidth=0.6)

    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

# ============================================================
# 4. META-LEARNER WEIGHTS


def plot_meta_weights(meta_model, branch_names, save_path):
    coef = meta_model.coef_.ravel()
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(6, 3))

    ax.barh(
        branch_names,
        coef,
        color=colors[:len(coef)],
        edgecolor="black"
    )
    """""
    for i, val in enumerate(coef):
        ax.text(
            val + np.sign(val) * 0.02,
            i,
            f"{val:.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=11
        )
    """""
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Coefficient Value")
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 6. BORUTA-SAFE LEARNING CURVE (COMPUTATION)


def boruta_learning_curve(
    X,
    y,
    build_model_fn,
    train_sizes=np.linspace(0.2, 1.0, 5),
    outer_cv=None,
    scoring="f1",
    random_state=42,
    min_train_samples=30
):
    #Compute learning curve with Boruta feature selection applied
    #independently at each training size 

     # Default to 3-fold stratified CV
    if outer_cv is None:
        outer_cv = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=random_state
        )

    # Select scoring function
    scorer = f1_score if scoring == "f1" else accuracy_score

    # Initialise score arrays
    n_sizes = len(train_sizes)
    n_folds = outer_cv.get_n_splits()
    # Iterate over CV folds
    train_scores = np.zeros((n_sizes, n_folds))
    cv_scores = np.zeros_like(train_scores)

    for j, (train_idx, val_idx) in enumerate(outer_cv.split(X, y)):
        X_train_full = X.iloc[train_idx]
        y_train_full = y.iloc[train_idx]

        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        n_train_full = len(X_train_full)
        # Convert fractions to actual sample counts
        fold_train_sizes = np.clip(
            (train_sizes * n_train_full).astype(int),
            a_min=min_train_samples,
            a_max=n_train_full
        )
        # Convert fractions to actual sample counts
        for i, size in enumerate(fold_train_sizes):

            if size >= n_train_full:
                X_train = X_train_full
                y_train = y_train_full
            else:
                sss = StratifiedShuffleSplit(
                    n_splits=1,
                    train_size=size,
                    random_state=random_state + 100 * j + i # Unique seed per fold
                )
                sub_idx, _ = next(sss.split(X_train_full, y_train_full))
                X_train = X_train_full.iloc[sub_idx]
                y_train = y_train_full.iloc[sub_idx]

            # Fit Boruta on this training subset only
            boruta = BorutaTransformer(
                n_estimators=100,
                max_iter=50,
                random_state=random_state
            )
            boruta.fit(X_train, y_train)
            # Build and train model with frozen Boruta
            model = build_model_fn(prefit_boruta=boruta)
            model.fit(X_train, y_train)
            # Evaluate on training and validation sets
            train_scores[i, j] = scorer(y_train, model.predict(X_train))
            cv_scores[i, j] = scorer(y_val, model.predict(X_val))

    mean_train_sizes = np.mean(
        [(train_sizes * len(X.iloc[train_idx])).astype(int)
         for train_idx, _ in outer_cv.split(X, y)],
        axis=0
    ).astype(int)

    return mean_train_sizes, train_scores, cv_scores

# ============================================================
# 7. LEARNING CURVE (PLOT)

def plot_learning_curve(train_sizes, train_scores, cv_scores, save_path):
    #Plot learning curve showing training and cross-validation scores
    #as a function of training set size.

    # Compute mean and standard deviation across folds
    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores, axis=1)

    cv_mean = np.mean(cv_scores, axis=1)
    cv_std  = np.std(cv_scores, axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Training score curve with shaded confidence band
    ax.plot(
        train_sizes,
        train_mean,
        color="red",
        marker="o",
        lw=2.5,
        label="Training Score"
    )
    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        color="red",
        alpha=0.20
    )

    # Cross-validation score curve with shaded confidence band
    ax.plot(
        train_sizes,
        cv_mean,
        color="green",
        marker="o",
        lw=2.5,
        label="Cross-Validation Score"
    )
    ax.fill_between(
        train_sizes,
        cv_mean - cv_std,
        cv_mean + cv_std,
        color="green",
        alpha=0.20
    )
    # y-axis limits with padding
    ymin = min(train_mean.min(), cv_mean.min()) - 0.05
    ymax = max(train_mean.max(), cv_mean.max()) + 0.02
    ax.set_ylim(ymin, min(1.05, ymax))

    ax.set_xlabel("Training Examples")
    ax.set_ylabel("F1 Score")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

# ============================================================
# 8. PER-BRANCH ROC COMPARISON

def plot_branch_roc_curves(fitted_stack, X_test, y_test, save_path):
    #Plots ROC curves comparing stacked ensemble against individual base learners.
    fig, ax = plt.subplots(figsize=(7, 5))

    y_proba_stack = fitted_stack.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba_stack)
    auc_stack = auc(fpr, tpr)

    ax.plot(
        fpr, tpr,
        lw=2.8,
        color="black",
        label=f"Stacked Ensemble (AUC = {auc_stack:.3f})"
    )
    # Colour scheme for each branch
    base_colors = {
        "nonlinear_branch": "#4C72B0",
        "svm_branch": "#55A868",
        "knn_branch": "#C44E52"
    }
    
    for name, estimator in fitted_stack.named_estimators_.items():
        if estimator == "drop":
            continue
        # Get probability scores
        # HGB and KNN have predict_proba, SVM uses decision_function
        if hasattr(estimator, "predict_proba"):
            y_proba = estimator.predict_proba(X_test)[:, 1]
        else:
            # Normalise decision function to [0, 1] for ROC calculation
            scores = estimator.decision_function(X_test)
            y_proba = (scores - scores.min()) / (scores.max() - scores.min())

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = auc(fpr, tpr)

        ax.plot(
            fpr, tpr,
            lw=2,
            linestyle="--",
            color=base_colors.get(name, "gray"),
            label=f"{name.replace('_', ' ').title()} (AUC = {auc_score:.3f})"
        )

    # Diagonal reference line (random classifier)
    ax.plot([0, 1], [0, 1], linestyle=":", color="gray", lw=1)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def generate_latex_table(ensemble_metrics, base_results, save_path=None):
    """
    Generate LaTeX table comparing ensemble and base learner performance.

    """
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Performance comparison on the independent test set.}",
        r"\label{tab:comparison}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & AUC & F1 & Sens. & Spec. \\",
        r"\midrule",
        f"Stacked Ensemble & \\textbf{{{ensemble_metrics['AUC']:.3f}}} & "
        f"\\textbf{{{ensemble_metrics['F1']:.3f}}} & "
        f"{ensemble_metrics['Sensitivity']:.3f} & "
        f"{ensemble_metrics['Specificity']:.3f} \\\\",
    ]
    
    for name, metrics in base_results.items():
        lines.append(
            f"{name} & {metrics['AUC']:.3f} & {metrics['F1']:.3f} & "
            f"{metrics['Sensitivity']:.3f} & {metrics['Specificity']:.3f} \\\\"
        )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    table_str = "\n".join(lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(table_str)
        print(f"    Saved: {save_path}")
    
    return table_str