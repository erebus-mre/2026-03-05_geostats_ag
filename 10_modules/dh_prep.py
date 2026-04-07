import pandas as pd
import numpy as np

import pyvista as pv
from typing import List, Dict, Tuple

def audit_drillhole_consistency(dataframes: Dict[str, pd.DataFrame], bhid_col: str = 'BHID') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs a cross-consistency audit across multiple drillhole datasets.
    
    Identifies missing Borehole IDs by comparing each individual DataFrame 
    against the union of all unique IDs found across the entire project.
    
    Args:
        dataframes (Dict[str, pd.DataFrame]): A dictionary where keys are filenames/table names 
            and values are the respective DataFrames.
        bhid_col (str): The column name representing the Drillhole ID. Defaults to 'BHID'.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - summary_df: A table showing the count of missing holes per file.
            - flag_df: A boolean matrix (True if present, False if absent) for every hole.
            
    Raises:
        ValueError: If a provided DataFrame does not contain the specified bhid_col.
    """
    # 1. Extract sets of unique BHIDs from each file
    id_sets = {}
    for name, df in dataframes.items():
        if bhid_col not in df.columns:
            raise ValueError(f"Column '{bhid_col}' not found in dataset: {name}")
        # Drop NaNs and ensure string type for consistent comparison
        id_sets[name] = set(df[bhid_col].dropna().astype(str).unique())

    # 2. Define the Universal Set (Global list of all drillholes)
    all_unique_ids = sorted(list(set().union(*id_sets.values())))
    
    # 3. Create the Flag DataFrame (Presence/Absence Matrix)
    # Using a dictionary comprehension for vectorized-style construction
    presence_map = {
        name: [hole_id in id_sets[name] for hole_id in all_unique_ids]
        for name in dataframes.keys()
    }
    
    flag_df = pd.DataFrame(presence_map, index=all_unique_ids)
    flag_df.index.name = bhid_col

    # 4. Generate Summary Table (Missing counts)
    summary_data = []
    total_holes = len(all_unique_ids)
    
    for name, ids in id_sets.items():
        missing_count = total_holes - len(ids)
        summary_data.append({
            'Dataset': name,
            'Total_Records': len(dataframes[name]),
            'Unique_Holes': len(ids),
            'Missing_Holes': missing_count
        })
        
    summary_df = pd.DataFrame(summary_data)

    # Simple Validation Assertion
    assert not flag_df.isnull().values.any(), "Audit failed: Null values detected in flag matrix."
    
    return summary_df, flag_df


def merge_intervals(df1, df2, holeid_col='HOLEID', from_col='FROM', to_col='TO'):
    """
    Performs a true topological outer join of two drillhole interval tables.
    Retains all gaps, unlogged sections, and unsampled intervals by creating 
    a master set of boundaries and mapping attributes via interval midpoints.
    
    Parameters:
    df1 (pd.DataFrame): First drillhole dataframe (e.g., Assays).
    df2 (pd.DataFrame): Second drillhole dataframe (e.g., Lithology).
    holeid_col (str): Column name for Drillhole ID.
    from_col (str): Column name for the 'From' downhole distance.
    to_col (str): Column name for the 'To' downhole distance.
    
    Returns:
    pd.DataFrame: A fully merged DataFrame containing all intervals and gaps.
    """
    
    # 1. Extract every unique depth boundary to create a continuous downhole framework
    b1 = df1[[holeid_col, from_col]].rename(columns={from_col: 'DEPTH'})
    b2 = df1[[holeid_col, to_col]].rename(columns={to_col: 'DEPTH'})
    b3 = df2[[holeid_col, from_col]].rename(columns={from_col: 'DEPTH'})
    b4 = df2[[holeid_col, to_col]].rename(columns={to_col: 'DEPTH'})
    
    # Combine, drop duplicates, and sort all boundaries down the hole
    all_boundaries = pd.concat([b1, b2, b3, b4]).drop_duplicates()
    all_boundaries = all_boundaries.sort_values([holeid_col, 'DEPTH']).reset_index(drop=True)
    
    # 2. Create master intervals from these consecutive boundaries
    all_boundaries['NEXT_DEPTH'] = all_boundaries.groupby(holeid_col)['DEPTH'].shift(-1)
    master = all_boundaries.dropna().rename(columns={'DEPTH': from_col, 'NEXT_DEPTH': to_col}).copy()
    
    # Calculate the midpoint of each master interval for "point-in-polygon" style mapping
    master['MIDPOINT'] = (master[from_col] + master[to_col]) / 2.0
    
    # 3. Map attributes from df1
    # Merge on HOLEID, then filter where the master midpoint falls inside df1's intervals
    m1 = pd.merge(master, df1, on=holeid_col, how='left', suffixes=('', '_drop'))
    mask1 = (
        ((m1['MIDPOINT'] >= m1[from_col + '_drop']) & (m1['MIDPOINT'] < m1[to_col + '_drop'])) 
        | m1[from_col + '_drop'].isna()
    )
    m1 = m1[mask1].drop(columns=[from_col + '_drop', to_col + '_drop']).drop_duplicates(subset=[holeid_col, from_col, to_col])
    
    # 4. Map attributes from df2 using the exact same midpoint logic
    m2 = pd.merge(m1, df2, on=holeid_col, how='left', suffixes=('', '_drop'))
    mask2 = (
        ((m2['MIDPOINT'] >= m2[from_col + '_drop']) & (m2['MIDPOINT'] < m2[to_col + '_drop'])) 
        | m2[from_col + '_drop'].isna()
    )
    m2 = m2[mask2].drop(columns=[from_col + '_drop', to_col + '_drop', 'MIDPOINT']).drop_duplicates(subset=[holeid_col, from_col, to_col])
    
    # 5. Clean up, calculate final lengths, and reorder columns
    m2['LENGTH'] = m2[to_col] - m2[from_col]
    
    front_cols = [holeid_col, from_col, to_col, 'LENGTH']
    other_cols = [c for c in m2.columns if c not in front_cols]
    final_df = m2[front_cols + other_cols].reset_index(drop=True)
    
    return final_df

def fill_drillhole_gaps(df, holeid_col='HOLEID', from_col='FROM', to_col='TO'):
    """
    Finds gaps between drillhole intervals and fills them with explicit dummy rows.
    Missing attribute columns will automatically be filled with NaN.
    """
    # 1. Sort the dataframe to ensure downhole sequence
    df_sorted = df.sort_values(by=[holeid_col, from_col]).reset_index(drop=True)
    
    # 2. Shift the 'TO' column down to compare with the next row's 'FROM'
    df_sorted['PREV_TO'] = df_sorted.groupby(holeid_col)[to_col].shift(1)
    
    # 3. Identify internal gaps (where current FROM is greater than previous TO)
    gaps = df_sorted[df_sorted[from_col] > df_sorted['PREV_TO']].copy()
    
    # Create a list to hold our new dummy dataframes
    dummy_list = []
    
    # 4. Process internal gaps
    if not gaps.empty:
        internal_dummies = pd.DataFrame({
            holeid_col: gaps[holeid_col],
            from_col: gaps['PREV_TO'],
            to_col: gaps[from_col]
        })
        dummy_list.append(internal_dummies)
        
    # 5. Check for missing collar intervals (gaps between 0.0 and the first logged interval)
    first_intervals = df_sorted.groupby(holeid_col).first().reset_index()
    collar_gaps = first_intervals[first_intervals[from_col] > 0.0].copy()
    
    if not collar_gaps.empty:
        collar_dummies = pd.DataFrame({
            holeid_col: collar_gaps[holeid_col],
            from_col: 0.0,
            to_col: collar_gaps[from_col]
        })
        dummy_list.append(collar_dummies)
        
    # 6. If no gaps exist, return the original dataframe cleanly
    if not dummy_list:
        return df_sorted.drop(columns=['PREV_TO'])
        
    # 7. Concatenate the original data with the new dummy intervals
    all_dummies = pd.concat(dummy_list, ignore_index=True)
    
    # Because we only provided HOLEID, FROM, and TO for the dummies, 
    # pandas will automatically fill Grade/Lithology columns with NaN!
    df_continuous = pd.concat([df, all_dummies], ignore_index=True)
    
    # 8. Sort downhole one last time and clean up
    df_continuous = df_continuous.sort_values(by=[holeid_col, from_col]).reset_index(drop=True)
    
    return df_continuous    

def align_end_of_hole(df1, df2, holeid_col='HOLEID', from_col='FROM', to_col='TO'):
    """
    Finds the maximum depth (EOH) for each drillhole across two tables.
    Appends a dummy interval to the shorter table so both match perfectly.
    """
    # 1. Find the max TO depth (EOH) for each hole in both dataframes
    eoh1 = df1.groupby(holeid_col)[to_col].max().rename('EOH1')
    eoh2 = df2.groupby(holeid_col)[to_col].max().rename('EOH2')
    
    # 2. Combine to find the absolute maximum EOH per hole
    # fillna(0) handles the edge case where a hole exists in one table but not the other
    eoh_combined = pd.concat([eoh1, eoh2], axis=1).fillna(0)
    eoh_combined['MAX_EOH'] = eoh_combined[['EOH1', 'EOH2']].max(axis=1)
    
    # 3. Identify holes where df1 is shorter than the true max EOH
    short_df1 = eoh_combined[eoh_combined['EOH1'] < eoh_combined['MAX_EOH']].copy()
    
    # 4. Identify holes where df2 is shorter than the true max EOH
    short_df2 = eoh_combined[eoh_combined['EOH2'] < eoh_combined['MAX_EOH']].copy()
    
    # 5. Create dummy extensions for df1 if needed
    if not short_df1.empty:
        ext1 = pd.DataFrame({
            holeid_col: short_df1.index,
            from_col: short_df1['EOH1'],
            to_col: short_df1['MAX_EOH']
        })
        df1_extended = pd.concat([df1, ext1], ignore_index=True)
    else:
        df1_extended = df1.copy()
        
    # 6. Create dummy extensions for df2 if needed
    if not short_df2.empty:
        ext2 = pd.DataFrame({
            holeid_col: short_df2.index,
            from_col: short_df2['EOH2'],
            to_col: short_df2['MAX_EOH']
        })
        df2_extended = pd.concat([df2, ext2], ignore_index=True)
    else:
        df2_extended = df2.copy()
        
    # 7. Sort downhole to keep things tidy
    df1_extended = df1_extended.sort_values(by=[holeid_col, from_col]).reset_index(drop=True)
    df2_extended = df2_extended.sort_values(by=[holeid_col, from_col]).reset_index(drop=True)
    
    return df1_extended, df2_extended  


def drillhole_merge_pipeline(df1, df2, holeid_col='HOLEID', from_col='FROM', to_col='TO'):
    """
    Master pipeline to safely merge two drillhole interval tables.
    Executes gap filling, End of Hole (EOH) alignment, and geometric outer union.
    
    Parameters:
    df1, df2 (pd.DataFrame): The raw drillhole tables (e.g., Assays, Lithology).
    holeid_col, from_col, to_col (str): Column names for spatial tracking.
    
    Returns:
    pd.DataFrame: A fully merged, contiguous DataFrame with all intervals and gaps.
    """
    #print("Step 1: Filling implicit gaps in Table 1...")
    df1_filled = fill_drillhole_gaps(df1, holeid_col, from_col, to_col)
    
    #print("Step 2: Filling implicit gaps in Table 2...")
    df2_filled = fill_drillhole_gaps(df2, holeid_col, from_col, to_col)
    
    #print("Step 3: Aligning End of Hole (EOH) depths across both tables...")
    df1_aligned, df2_aligned = align_end_of_hole(df1_filled, df2_filled, holeid_col, from_col, to_col)
    
    #print("Step 4: Executing geometric outer merge...")
    final_merged_db = merge_intervals(df1_aligned, df2_aligned, holeid_col, from_col, to_col)
    
    #print("✅ Pipeline complete! Database is ready for compositing.")
    return final_merged_db   


def check_internal_overlaps(df, table_name="Database", holeid_col='HOLEID', from_col='FROM', to_col='TO'):
    """
    Checks a drillhole table for overlapping intervals within the same hole.
    Returns a DataFrame of the offending rows if overlaps exist, otherwise returns None.
    """
    # 1. Sort the data strictly by Hole ID and From depth
    # First copy index
    df_local = df.copy()
    df_local['orig_index'] = df_local.index 
    df_sorted = df_local.sort_values(by=[holeid_col, from_col]).reset_index(drop=True)
    
    # 2. Shift the 'TO' column down by one row, grouped by hole
    # This aligns the previous interval's TO depth with the current interval's FROM depth
    df_sorted['PREV_TO'] = df_sorted.groupby(holeid_col)[to_col].shift(1)
    
    # 3. An overlap exists if the current FROM is strictly less than the PREV_TO
    # (FROM == PREV_TO is perfectly fine, that just means continuous sampling)
    mask = df_sorted[from_col] < df_sorted['PREV_TO']
    overlaps = df_sorted[mask].copy()
    err =  overlaps[[holeid_col,'orig_index', from_col, to_col, 'PREV_TO']]
    
    # 4. Report the results
    if not overlaps.empty:
        print(f"⚠️ ERROR in {table_name}: Found {len(overlaps)} overlapping intervals!")
        # Return just the columns needed to track down the error
        return err
    else:
        print(f"✅ {table_name} validation passed: No internal overlaps found.")
        return None

import pandas as pd
import numpy as np

def composite_drillholes(df, holeid_col='HOLEID', from_col='FROM', to_col='TO', 
                         domain_col='LITH', num_cols=None, cat_cols=None, 
                         comp_len=2.0, min_len=1.0):
    """
    Composites drillhole data to a fixed length, respecting hard boundaries (domains).
    Numeric variables are length-weighted. Categorical variables use length-weighted mode.
    Short final composites are merged with the previous composite or discarded if isolated.
    """
    # 1. Protect the original dataframe (the pass-by-reference fix!)
    df_raw = df.copy()
    num_cols = num_cols or []
    cat_cols = cat_cols or []
    
    # 2. Identify contiguous blocks of the same domain (Lithology)
    # A block changes if the HOLEID changes OR the domain changes
    df_raw['block_change'] = (
        (df_raw[holeid_col] != df_raw[holeid_col].shift(1)) | 
        (df_raw[domain_col] != df_raw[domain_col].shift(1))
    )
    df_raw['block_id'] = df_raw['block_change'].cumsum()
    
    # 3. Get the absolute start and end depths for each continuous domain block
    blocks = df_raw.groupby(['block_id', holeid_col, domain_col]).agg(
        BLOCK_FROM=(from_col, 'min'),
        BLOCK_TO=(to_col, 'max')
    ).reset_index()
    
    # 4. Generate the target composite intervals (The Framework)
    comp_records = []
    comp_id = 0
    
    for _, row in blocks.iterrows():
        b_from, b_to = row['BLOCK_FROM'], row['BLOCK_TO']
        
        # Generate mathematical boundaries for the composites
        bins = list(np.arange(b_from, b_to, comp_len))
        if not np.isclose(bins[-1], b_to): 
            bins.append(b_to) # Ensure the final boundary caps at the exact block end
            
        # Handle the short final composite (End of block logic)
        if len(bins) > 2:
            last_len = bins[-1] - bins[-2]
            if last_len < min_len:
                bins.pop(-2) # Merge short tail with the previous composite
        elif len(bins) == 2:
            if (bins[1] - bins[0]) < min_len:
                continue # Discard if it's the only composite in the block and it's too short
                
        # Create the empty composite intervals
        for i in range(len(bins)-1):
            comp_records.append({
                'COMP_ID': comp_id,
                holeid_col: row[holeid_col],
                'block_id': row['block_id'],
                domain_col: row[domain_col],
                'COMP_FROM': bins[i],
                'COMP_TO': bins[i+1],
                'COMP_LENGTH': bins[i+1] - bins[i]
            })
            comp_id += 1
            
    comp_df = pd.DataFrame(comp_records)
    if comp_df.empty:
        print("Warning: No composites generated. Check your minimum length settings.")
        return comp_df
        
    # 5. Intersect raw data with the new composite framework
    merged = pd.merge(df_raw, comp_df, on=[holeid_col, 'block_id', domain_col])
    overlaps = merged[(merged[from_col] < merged['COMP_TO']) & 
                      (merged[to_col] > merged['COMP_FROM'])].copy()
                      
    # Calculate exact intersecting lengths (Weights)
    overlaps['YIELD_FROM'] = np.maximum(overlaps[from_col], overlaps['COMP_FROM'])
    overlaps['YIELD_TO'] = np.minimum(overlaps[to_col], overlaps['COMP_TO'])
    overlaps['WEIGHT'] = overlaps['YIELD_TO'] - overlaps['YIELD_FROM']
    
    # 6. Initialize final output dataframe with our framework
    out_cols = ['COMP_ID', holeid_col, 'COMP_FROM', 'COMP_TO', 'COMP_LENGTH', domain_col]
    final_df = comp_df[out_cols].copy()
    
    # 7. Aggregate Numerical Columns (Length-Weighted Average)
    for col in num_cols:
        overlaps[f'{col}_x_W'] = overlaps[col] * overlaps['WEIGHT']
        # Only use weight where the grade is actually present (ignore NaNs)
        overlaps[f'{col}_valid_W'] = overlaps['WEIGHT'].where(overlaps[col].notnull())
        
        sum_wx = overlaps.groupby('COMP_ID')[f'{col}_x_W'].sum()
        sum_w = overlaps.groupby('COMP_ID')[f'{col}_valid_W'].sum()
        
        # Calculate grade and map back to final dataframe
        weighted_avg = (sum_wx / sum_w).rename(col)
        final_df = final_df.merge(weighted_avg, on='COMP_ID', how='left')
        
    # 8. Aggregate Categorical Columns (Length-Weighted Mode)
    for col in cat_cols:
        # Sum the lengths (weights) of each category within each composite
        cat_weights = overlaps.groupby(['COMP_ID', col])['WEIGHT'].sum().reset_index()
        # Sort by weight descending, then take the top category
        top_cats = cat_weights.sort_values('WEIGHT', ascending=False).drop_duplicates('COMP_ID')
        final_df = final_df.merge(top_cats[['COMP_ID', col]], on='COMP_ID', how='left')
        
    # Clean up and rename to standard FROM/TO
    final_df = final_df.drop(columns=['COMP_ID']).rename(columns={'COMP_FROM': from_col, 'COMP_TO': to_col})
    return final_df.sort_values(by=[holeid_col, from_col]).reset_index(drop=True)

# ==========================================
# Example Execution
# ==========================================
if __name__ == "__main__":
    # Mock Database: Note the internal lithology variations
    raw_db = pd.DataFrame({
        'HOLEID': ['DH001', 'DH001', 'DH001', 'DH001'],
        'FROM': [0.0, 1.5, 3.0, 4.2],
        'TO': [1.5, 3.0, 4.2, 5.0],
        'LITH': ['Oxide', 'Oxide', 'Fresh', 'Fresh'],
        'Au_ppm': [1.0, 2.0, 5.0, 1.0],
        'Alteration': ['Silica', 'Clay', 'Silica', 'Silica']
    })

    print("--- RAW DATA ---")
    print(raw_db.to_string())

    # Create 2m composites, minimum length 1m
    composites = composite_drillholes(
        df=raw_db, 
        domain_col='LITH',
        num_cols=['Au_ppm'], 
        cat_cols=['Alteration'],
        comp_len=2.0, 
        min_len=1.0
    )

    print("\n--- 2m COMPOSITES ---")
    print(composites.to_string())

import pandas as pd
import numpy as np

def desurvey_composites(df_comps, df_collar, df_survey, holeid_col='HOLEID'):
    """
    Desurveys drillhole composites to calculate exact X, Y, Z coordinates for 
    the start (FROM), middle (MID), and end (TO) of each composite interval.
    """
    # 1. Calculate the MID depth for each composite
    comps = df_comps.copy()
    comps['MID'] = (comps['FROM'] + comps['TO']) / 2.0
    
    # 2. Extract all the unique depths we need coordinates for
    depths_from = comps[[holeid_col, 'FROM']].rename(columns={'FROM': 'DEPTH'})
    depths_mid  = comps[[holeid_col, 'MID']].rename(columns={'MID': 'DEPTH'})
    depths_to   = comps[[holeid_col, 'TO']].rename(columns={'TO': 'DEPTH'})
    survey_deps = df_survey[[holeid_col, 'DEPTH', 'DIP', 'AZIMUTH']]
    
    # Ensure every hole has a starting depth at 0.0 (the collar)
    collar_deps = df_collar[[holeid_col]].copy()
    collar_deps['DEPTH'] = 0.0
    
    # 3. Combine all depths into a single master downhole track and sort
    track = pd.concat([depths_from, depths_mid, depths_to, survey_deps, collar_deps])
    track = track.sort_values([holeid_col, 'DEPTH']).drop_duplicates(subset=[holeid_col, 'DEPTH']).reset_index(drop=True)
    
    # 4. Interpolate missing Dip and Azimuth for the composite depths
    # We use forward fill (.ffill) which applies the Tangent Method (the orientation 
    # remains constant until the next survey point is reached).
    track['DIP'] = track.groupby(holeid_col)['DIP'].ffill().bfill()
    track['AZIMUTH'] = track.groupby(holeid_col)['AZIMUTH'].ffill().bfill()
    
    # 5. Calculate the length (L) of every segment down the hole
    track['L'] = track.groupby(holeid_col)['DEPTH'].diff().fillna(0.0)
    
    # 6. Trigonometry: Calculate delta X, Y, and Z for each segment
    dip_rad = np.radians(track['DIP'])
    az_rad = np.radians(track['AZIMUTH'])
    
    track['dX'] = track['L'] * np.cos(dip_rad) * np.sin(az_rad)
    track['dY'] = track['L'] * np.cos(dip_rad) * np.cos(az_rad)
    track['dZ'] = track['L'] * np.sin(dip_rad)
    
    # 7. Cumulative sum to get the local coordinates relative to the collar
    track['local_X'] = track.groupby(holeid_col)['dX'].cumsum()
    track['local_Y'] = track.groupby(holeid_col)['dY'].cumsum()
    track['local_Z'] = track.groupby(holeid_col)['dZ'].cumsum()
    
    # 8. Add the absolute Collar coordinates to convert local to real-world XYZ
    track = pd.merge(track, df_collar[[holeid_col, 'X', 'Y', 'Z']], on=holeid_col, how='left')
    track['X_COORD'] = track['X'] + track['local_X']
    track['Y_COORD'] = track['Y'] + track['local_Y']
    track['Z_COORD'] = track['Z'] + track['local_Z']
    
    # 9. Lookup dictionary to easily map coordinates back to the composites
    track_lookup = track.set_index([holeid_col, 'DEPTH'])[['X_COORD', 'Y_COORD', 'Z_COORD']]
    
    def get_coords(df, depth_col, prefix):
        # Helper to join coordinates and rename them nicely
        joined = df.join(track_lookup, on=[holeid_col, depth_col])
        return joined.rename(columns={'X_COORD': f'{prefix}_X', 
                                      'Y_COORD': f'{prefix}_Y', 
                                      'Z_COORD': f'{prefix}_Z'})
    
    # 10. Map FROM, MID, and TO coordinates back to the composite dataframe
    comps = get_coords(comps, 'FROM', 'FROM')
    comps = get_coords(comps, 'MID', 'MID')
    comps = get_coords(comps, 'TO', 'TO')
    
    return comps

# ==========================================
# Example Execution
# ==========================================
if __name__ == "__main__":
    # 1. Mock Collar File (X, Y, Z)
    collars = pd.DataFrame({
        'HOLEID': ['DH001'],
        'X': [50000.0],
        'Y': [60000.0],
        'Z': [300.0]
    })

    # 2. Mock Survey File (Notice the hole starts plunging south-east)
    surveys = pd.DataFrame({
        'HOLEID': ['DH001', 'DH001', 'DH001'],
        'DEPTH': [0.0, 50.0, 100.0],
        'DIP': [-60.0, -58.0, -55.0],  # Negative is down
        'AZIMUTH': [135.0, 138.0, 140.0]
    })

    # 3. Mock Composites (Output from our previous subroutine)
    composites = pd.DataFrame({
        'HOLEID': ['DH001', 'DH001'],
        'FROM': [10.0, 12.0],
        'TO': [12.0, 14.0],
        'Au_ppm': [1.5, 4.2],
        'LITH': ['Oxide', 'Oxide']
    })

    # Run the desurvey!
    desurveyed_comps = desurvey_composites(composites, collars, surveys)
    
    # To keep the printout clean, let's just show the MID coordinates
    print("--- COMPOSITE MIDPOINT COORDINATES ---")
    cols_to_print = ['HOLEID', 'FROM', 'TO', 'MID_X', 'MID_Y', 'MID_Z', 'Au_ppm']
    print(desurveyed_comps[cols_to_print].to_string())




def visualize_composites_3d(df_comps, x_col='MID_X', y_col='MID_Y', z_col='MID_Z', grade_col='Au_ppm'):
    """
    Takes a dataframe of desurveyed composites and plots them as 3D spheres 
    in PyVista, colored by their assay grade.
    """
    # 1. Extract the X, Y, Z coordinates into a 2D numpy array (N x 3)
    # PyVista requires points to be in this specific mathematical format
    points = df_comps[[x_col, y_col, z_col]].values
    
    # 2. Initialize a PyVista PolyData object (a point cloud)
    point_cloud = pv.PolyData(points)
    
    # 3. Attach the grade array to the point cloud's data
    # This tells PyVista what values to use for the color scale
    point_cloud[grade_col] = df_comps[grade_col].values
    
    # 4. Set up the 3D Plotter environment
    plotter = pv.Plotter()
    
    # Add a bounding box with coordinates to help you orient yourself
    plotter.show_bounds(grid='front', location='outer', all_edges=True)
    plotter.show_axes()
    
    # 5. Add the points to the scene
    plotter.add_points(
        point_cloud, 
        render_points_as_spheres=True,  # Makes them look like 3D balls instead of flat pixels
        point_size=15.0,                # Adjust this depending on your zoom level
        scalars=grade_col,              # The column we attached in step 3
        cmap="turbo",                   # 'turbo' or 'jet' are classic mining colormaps
        show_scalar_bar=True,
        scalar_bar_args={'title': f'{grade_col} (g/t)'}
    )
    
    # 6. Launch the interactive render window
    print("Launching PyVista 3D Viewer. Close the window to continue your script.")
    plotter.show()

# ==========================================
# Example Execution
# ==========================================
if __name__ == "__main__":
    # Mocking up a few drillholes curving down into an orebody
    np.random.seed(42)
    
    # Generate a spiral/curve of points to simulate multiple holes
    z_coords = np.linspace(300, 0, 100)
    x_coords = 50000 + np.sin(z_coords / 50) * 100 + np.random.normal(0, 5, 100)
    y_coords = 60000 + np.cos(z_coords / 50) * 100 + np.random.normal(0, 5, 100)
    
    # Generate some mock grades (lognormal distribution is great for Au)
    grades = np.random.lognormal(mean=0.5, sigma=0.8, size=100)
    
    mock_desurveyed_df = pd.DataFrame({
        'MID_X': x_coords,
        'MID_Y': y_coords,
        'MID_Z': z_coords,
        'Au_ppm': grades
    })

    # Run the visualizer!
    visualize_composites_3d(mock_desurveyed_df)        