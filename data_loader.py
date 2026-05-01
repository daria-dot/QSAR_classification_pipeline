# =========================================================================
# data_loader.py custom .mat file loader
# =========================================================================
import scipy.io
import pandas as pd
import numpy as np


def load_mat_file(filepath):
    
    #Custom function to load a .mat file and convert it to a Pandas DataFrame.
    try:
        mat = scipy.io.loadmat(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {filepath}")

    # 1. Filter out the internal MATLAB metadata -keys starting with '__'
    variables = {k: v for k, v in mat.items() if not k.startswith('__')}
    
    if not variables:
        raise ValueError("empty .MAT file")

    print(f"      DEBUG: Found MATLAB variables: {list(variables.keys())}")
    
    # 2.  Find the variable with the largest size (elements)
  
    target_key = max(variables, key=lambda k: variables[k].size)
    
    data_array = variables[target_key]
    print(f"      Selected variable '{target_key}' based on size ({data_array.size} elements).")
    
    # 3. Convert to DataFrame
    df = pd.DataFrame(data_array)
    
    print(f"      Successfully converted variable '{target_key}' to DataFrame.")
    return df
