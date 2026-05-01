
# =========================================================================
# Bourtua_class.py  Boruta feature selection transformer
# =========================================================================
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy


class BorutaTransformer(BaseEstimator, TransformerMixin):
    
    #Sklearn-compatible Boruta feature selector with single-class fallback.
    

    def __init__(self, n_estimators=500, max_iter=100, perc=90, random_state=42):
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.random_state = random_state
        self.perc = perc
        self.boruta_ = None
        self.selected_indices_ = None
        self.n_features_in_ = None
        self.fallback_used_ = False

    def fit(self, X, y):
        #Fit Boruta selector with single-class protection
        X = np.asarray(X)
        y = np.asarray(y)

        self.n_features_in_ = X.shape[1]

        if X.shape[0] != len(y):
            raise ValueError(f"X has {X.shape[0]} samples but y has {len(y)}")

        n_classes = len(np.unique(y))
        #------------------------------------------------
        # Single class fallback
        if n_classes < 2:
            print(f"    [Boruta]: Only {n_classes} class -> selecting all features")
            self.selected_indices_ = np.arange(self.n_features_in_)
            self.fallback_used_ = True
            return self
        #------------------------------------------------
        # Normal Boruta fitting
        try:

            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=5,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )

            self.boruta_ = BorutaPy(
                estimator=rf,
                n_estimators=self.n_estimators,
                max_iter=self.max_iter,
                perc=self.perc,
                random_state=self.random_state,
                verbose=0
            )

            self.boruta_.fit(X, y)

            # Select confirmed + tentative features
            self.selected_indices_ = np.where(
                self.boruta_.support_ | self.boruta_.support_weak_
            )[0]

            n_selected = len(self.selected_indices_)

            # Fallback if too few selected
            if n_selected == 0:
                fallback_k = max(5, int(0.2 * self.n_features_in_))
                self.selected_indices_ = np.argsort(self.boruta_.ranking_)[:fallback_k]
                print(f"    [Boruta] : 0 selected -> using top {fallback_k}")
            elif n_selected < 3:
                fallback_k = 10
                self.selected_indices_ = np.argsort(self.boruta_.ranking_)[:fallback_k]
                print(f"    [Boruta] : Only {n_selected} -> augmented to top {fallback_k}")
            else:
                print(f"    [Boruta] Selected {n_selected} features")

            self.fallback_used_ = False

        except Exception as e:
            # Catch-all fallback
            print(f"    [Boruta] : {e} -> selecting all features")
            self.selected_indices_ = np.arange(self.n_features_in_)
            self.fallback_used_ = True

        return self

    def transform(self, X):
        #Transform X to selected features
        if self.selected_indices_ is None:
            raise RuntimeError("Must fit before transform")

        X = np.asarray(X)
        return X[:, self.selected_indices_]

    def get_support(self, indices=False):
        #Get mask or indices of selected features
        if indices:
            return self.selected_indices_
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_indices_] = True
        return mask