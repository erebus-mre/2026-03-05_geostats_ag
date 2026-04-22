import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple

def generate_table_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a summary table for geological data with universal unique value counts.
    
    This version ensures 'unique_values' is populated for all data types, 
    while mean and variance remain reserved for continuous float data.

    Args:
        df (pd.DataFrame): The input drillhole or spatial dataset.

    Returns:
        pd.DataFrame: Summary table with columns: [variable, type, n_rows, 
                      n_informed, n_blank, n_na, mean, variance, unique_values]
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    summary_list = []

    for col in df.columns:
        series = df[col]
        
        # 1. Basic Counts
        n_rows = len(series)
        n_na = series.isna().sum()
        
        # Identify blanks (only relevant for object/string types)
        if pd.api.types.is_object_dtype(series):
            n_blank = (series.astype(str).str.strip() == "").sum()
            # Clean series for unique count: drop NA and empty strings
            valid_for_unique = series.dropna()
            valid_for_unique = valid_for_unique[valid_for_unique.astype(str).str.strip() != ""]
        else:
            n_blank = 0
            # Clean series for unique count: just drop NA
            valid_for_unique = series.dropna()
            
        n_informed = n_rows - n_na - n_blank

        # 2. Universal Unique Count (Calculated for all types)
        unique_count = valid_for_unique.nunique()

        # 3. Type-Specific Stats
        mean_val = np.nan
        var_val = np.nan
        
        if pd.api.types.is_float_dtype(series):
            mean_val = series.mean()
            var_val = series.var(ddof=0)

        summary_list.append({
            "variable": col,
            "type": str(series.dtype),
            "n_rows": n_rows,
            "n_informed": n_informed,
            "n_blank": n_blank,
            "n_na": n_na,
            "unique_values": unique_count,
            "mean": mean_val,
            "variance": var_val,
            
        })

    return pd.DataFrame(summary_list)



def find_proximal_points(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    z_col: str, 
    threshold: float
) -> pd.DataFrame:
    """
    Identifies pairs of points within a 3D distance threshold using a KD-Tree.

    Theory:
        Uses a K-Dimensional Tree to efficiently query neighbors in 3D space. 
        This avoids the 'all-to-all' distance matrix calculation, which is 
        memory-intensive for large geological datasets.

    Args:
        df: Input pandas DataFrame.
        x_col, y_col, z_col: Column names for the 3D coordinates.
        threshold: The distance limit (Euclidean) for identifying proximal points.

    Returns:
        pd.DataFrame: Columns ['index_a', 'index_b', 'distance'] representing 
                      the original indices of the proximal pairs.
    """
    # 1. Validation
    for col in [x_col, y_col, z_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
    
    if threshold <= 0:
        raise ValueError("Threshold must be a positive non-zero value.")

    # 2. Extract coordinates and build KD-Tree
    coords = df[[x_col, y_col, z_col]].values
    tree = cKDTree(coords)

    # 3. Query the tree for pairs within the threshold
    # output is a set of tuples (i, j) where distance < threshold
    pairs = tree.query_pairs(r=threshold)

    if not pairs:
        return pd.DataFrame(columns=['index_a', 'index_b', 'distance'])

    # 4. Extract actual distances and map back to original indices
    results = []
    original_indices = df.index.values

    for i, j in pairs:
        # Calculate precise distance for the identified pairs
        dist = np.linalg.norm(coords[i] - coords[j])
        results.append({
            'index_a': original_indices[i],
            'index_b': original_indices[j],
            'distance': round(dist, 4)
        })

    return pd.DataFrame(results)    