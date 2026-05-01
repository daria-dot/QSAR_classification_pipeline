# ============================================================
# pre_process.py  checking and cleaning functions
# ============================================================
import pandas as pd
import numpy as np


# ================================================================
# 1. Basic Data check

def data_checks(df):
    #Prints basic dataset information 
    print("Data Check")
    print(f"Missing Values: {df.isna().sum().sum()}")
    print(f"Class Balance: {df.iloc[:, -1].value_counts().to_dict()}")
    print("-"*72)
    

# ================================================================
# 2. Duplicate Sample Removal

def remove_duplicate_samples(df):
    #Removes duplicate rows (samples).
    print("   [Pre-Process] Checking for duplicate samples...")
    before = df.shape[0]
    df_clean = df.drop_duplicates()
    
    removed = before - df_clean.shape[0]
    if removed > 0:
        print(f" Removed {removed} duplicate samples.")
    return df_clean


# ================================================================
# 3. Duplicate Feature Removal

def remove_duplicate_features(X):
    #Removes duplicated columns (identical descriptors)
    print(" Pre-Process Checking for duplicate features...")
    
    before = X.shape[1]
    X_clean = X.T.drop_duplicates().T.infer_objects()
    removed = before - X_clean.shape[1]
    
    if removed > 0:
        print(f"Removed {removed} duplicate features.")
    return X_clean


