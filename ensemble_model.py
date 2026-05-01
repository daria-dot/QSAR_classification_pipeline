# ============================================================
# ensemble_model.py  builds the pipeline architecture
# ============================================================

import numpy as np
# ================================================================
# Scikit-Learn: Base Classes
from sklearn.base import BaseEstimator, TransformerMixin, clone
# ================================================================
# Scikit-Learn: Preprocessing & Pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
# ================================================================
# Scikit-Learn: Linear Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
# ================================================================
# Scikit-Learn: Non-Linear Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
# ================================================================
# Scikit-Learn: Ensemble & Model Selection
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
# ================================================================

# SafeLassoSelector embedded directly due to errors

class SafeLassoSelector(BaseEstimator, TransformerMixin):
    #LASSO feature selector with fallback for single-class folds.
    #Includes safety fallbacks for edge cases in cross-validation:
    #- Single-class folds: selects all features
    #- Too few features selected: falls back to top-k by coefficient magnitude
    def __init__(self, C=0.05, threshold="mean", min_features=5, random_state=42):
        self.C = C
        self.threshold = threshold
        self.min_features = min_features
        self.random_state = random_state
        self.selector_ = None
        self.fallback_used_ = False
        self.n_features_in_ = None
        self.selected_mask_ = None
    #Fit LASSO selector on training data.
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.n_features_in_ = X.shape[1]
        
        n_classes = len(np.unique(y))
        # Fallback: single-class fold (can occur in stratified CV with small data)
        if n_classes < 2:
            self.fallback_used_ = True
            self.selected_mask_ = np.ones(self.n_features_in_, dtype=bool)
        else:
            # L1-regularised logistic regression for sparse feature selection
            try:
                lasso = LogisticRegression(
                    penalty='l1',
                    solver='liblinear', #  for L1 penalty
                    C=self.C,
                    class_weight='balanced',
                    max_iter=5000,
                    random_state=self.random_state
                )
                # Select features with coefficients above threshold
                self.selector_ = SelectFromModel(lasso, threshold=self.threshold)
                self.selector_.fit(X, y)
                self.selected_mask_ = self.selector_.get_support()
                # Fallback: too few features selected
                if self.selected_mask_.sum() < self.min_features:
                    coefs = np.abs(self.selector_.estimator_.coef_).ravel()
                    top_idx = np.argsort(coefs)[-self.min_features:]
                    self.selected_mask_ = np.zeros(self.n_features_in_, dtype=bool)
                    self.selected_mask_[top_idx] = True
                print(f"    [LASSO] Selected {self.selected_mask_.sum()} features")
                self.fallback_used_ = False
            except Exception:
                 # Catch-all fallback: select all features
                self.fallback_used_ = True
                self.selected_mask_ = np.ones(self.n_features_in_, dtype=bool)
        
        return self

    def transform(self, X):
        #Transform X to selected features only.
        X = np.asarray(X)
        return X[:, self.selected_mask_]

    def get_support(self, indices=False):
        #Get mask or indices of selected features
        if indices:
            return np.where(self.selected_mask_)[0]
        return self.selected_mask_

# ============================================================
# Building Heterogeneous Stacked Ensemble

def build_heterogeneous_stack(prefit_boruta, random_state=42):
    #Builds the ensemble model
    #params -> prefit_boruta selector
    #       -> random_state : int, for reproducibility
    RAND = random_state

    print("\n" + "=" * 55)
    print("[EnsembleBuilder] Boruta -> HGB | Lasso -> SVM |  KNN")
    print("=" * 55)

    # ---------------------------------------------------------
    # BRANCH 1: Boruta -> HistGradientBoostingClassifier
   
    nonlinear_pipe = Pipeline([
        ('selector', clone(prefit_boruta)),  # frozen selector# Clone to avoid refitting
        ('model', HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=8,
            max_iter=600,
            l2_regularization=1.0,
            random_state=RAND
        ))
    ])
    print("  - Branch 1: Boruta -> HistGradientBoostingClassifier")

    # ---------------------------------------------------------
    # BRANCH 2: Lasso -> Linear SVM
 
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),# SVM requires scaled features
        ('selector', SafeLassoSelector(
            C=0.05,
            random_state=RAND
        )),
        ('model', SVC(
            kernel='linear',
            C=1.0,
            class_weight='balanced',
            probability=False,   #  for stack_method='auto'
            random_state=RAND
        ))
    ])
    print("  - Branch 2: Lasso -> Linear SVM")

    # ---------------------------------------------------------
    # BRANCH 3: Lasso -> KNN

    knn_pipe = Pipeline([
        ('scaler', StandardScaler()),  # KNN requires scaled features
        
        ('model', KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
        ))
    ])
    print("  - Branch 3:  KNN")

    # ---------------------------------------------------------
    # META-LEARNER
  
    meta_learner = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        class_weight='balanced',
        max_iter=3000,
        random_state=RAND
    )
    print("  - Meta-Learner: LogisticRegression (L2)")

    # ---------------------------------------------------------
    # STACKING CLASSIFIER
    
    clf_stack = StackingClassifier(
        estimators=[
            ('nonlinear_branch', nonlinear_pipe),
            ('svm_branch', svm_pipe),
            ('knn_branch', knn_pipe)
        ],
        final_estimator=meta_learner,
        cv=StratifiedKFold(
            n_splits=5,
            shuffle =True,
            random_state=RAND
        ),
        passthrough=False,
        stack_method="auto",  #  auto selects decision_function where possible
        n_jobs=1              #  avoids nested-CV multiprocessing crashes
    )

    print("\n Stack successfully configured.\n")
    return clf_stack