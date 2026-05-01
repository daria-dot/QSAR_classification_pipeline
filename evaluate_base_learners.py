# =========================================================================
# evaluate_base_learners.py evaluates each base learner in the ensemble
# =========================================================================
from sklearn.metrics import roc_auc_score, f1_score, recall_score

import numpy as np  
def evaluate_base_learners(fitted_stack, X_test, y_test):
    
   # Evaluate each base learner using named_estimators_ for fully refit pipelines.
    
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    results = {}
    pipelines = {
        "HGB (Boruta)": fitted_stack.named_estimators_["nonlinear_branch"],
        "SVM (LASSO)": fitted_stack.named_estimators_["svm_branch"],
        "KNN (Full)": fitted_stack.named_estimators_["knn_branch"],
    }

    for name, pipeline in pipelines.items():

        y_pred = pipeline.predict(X_test)

        if hasattr(pipeline, "predict_proba"):
            y_score = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            scores = pipeline.decision_function(X_test)
            y_score = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        else:
            y_score = y_pred.astype(float)
        #metrics calculation
        auc = roc_auc_score(y_test, y_score)
        f1 = f1_score(y_test, y_pred)
        sens = recall_score(y_test, y_pred, pos_label=1)
        spec = recall_score(y_test, y_pred, pos_label=0)

        results[name] = {
            "AUC": auc,
            "F1": f1,
            "Sensitivity": sens,
            "Specificity": spec,
        }

        print(f"{name}")
        print(f"  AUC : {auc:.4f}")
        print(f"  F1  : {f1:.4f}")
        print(f"  Sens: {sens:.4f}")
        print(f"  Spec: {spec:.4f}")
        print()

    return results
    
    

