import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from IPython.display import display, HTML

def analysis(
    df: pd.DataFrame, 
    variables: list, 
    weight_col: str = None, 
    log_vars: list = None,
    colors: list = None,
    output_format: str = 'markdown'
):
    """
    Performs univariate analysis strictly on numerical geological data.
    
    Theory:
        Focuses on continuous variables. Calculates the weighted Coefficient of 
        Variation (CV) to assess the relative dispersion of the data.
    
    Args:
        df (pd.DataFrame): The input drillhole dataset.
        variables (list): List of column names to analyze.
        weight_col (str, optional): Column name for declustering weights.
        log_vars (list, optional): Subset of 'variables' to display with log-scaling.
        colors (list, optional): List of colors for histograms.
        output_format (str): 'markdown' or 'html'.
    """
    # 1. & 5. Validation and Deduplication
    seen = set()
    duplicates = [v for v in variables if v in seen or (seen.add(v) or False)]
    if duplicates:
        print(f"**Warning:** Duplicate variables detected in input: {duplicates}")
    
    unique_vars = []
    [unique_vars.append(v) for v in variables if v not in unique_vars]
    
    # 6. Filter for Numeric and handle missing
    numeric_vars = [v for v in unique_vars if v in df.columns and np.issubdtype(df[v].dtype, np.number)]
    non_numeric = [v for v in unique_vars if v in df.columns and not np.issubdtype(df[v].dtype, np.number)]
    
    if non_numeric:
        print(f"**Note:** Skipping non-numeric variables: {non_numeric}")
    
    if not numeric_vars:
        print("No valid numeric variables found for analysis.")
        return None

    # Sync log_vars and colors to the filtered numeric_vars list
    log_vars = log_vars if log_vars else []
    if not colors:
        colors = plt.cm.get_cmap('tab10').colors # Default to a professional palette
    
    summary_records = []
    
    for i, var in enumerate(numeric_vars):
        # Handle NaNs and Weights
        cols_to_use = [var] + ([weight_col] if weight_col else [])
        data_clean = df[cols_to_use].dropna()
        
        vals = data_clean[var].values
        weights = data_clean[weight_col].values if weight_col else np.ones(len(vals))
        
        # Geostatistical Summary Math
        w_mean = np.average(vals, weights=weights)
        w_var = np.average((vals - w_mean)**2, weights=weights)
        w_std = np.sqrt(w_var)
        cv = w_std / w_mean if w_mean != 0 else np.nan
        
        summary_records.append({
            'Variable': var,
            'Count': len(vals),
            'Min': np.min(vals),
            'Max': np.max(vals),
            'W_Mean': round(w_mean, 4),
            'W_Std': round(w_std, 4),
            'CV': round(cv, 4)
        })
        
        # Visualization
        is_log = var in log_vars
        current_color = colors[i % len(colors)]
        _plot_dist_v2(vals, weights, var, is_log, current_color)

    # 3. & 4. Table Construction
    summary_df = pd.DataFrame(summary_records)
    
    if output_format == 'html':
        display(HTML("<h3>Numerical Summary Table</h3>" + summary_df.to_html(index=False)))
    else:
        print("\n### Numerical Statistical Summary")
        print(summary_df.to_markdown(index=False))
        
    return summary_df

def _plot_dist_v2(vals, weights, var_name, log_scale, color):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    
    # Histogram logic (Ensure values > 0 for log scaling)
    if log_scale and np.any(vals > 0):
        pos_vals = vals[vals > 0]
        bins = np.geomspace(pos_vals.min(), pos_vals.max(), 30)
        ax[0].set_xscale('log')
    else:
        bins = 30

    ax[0].hist(vals, weights=weights, bins=bins, color=color, edgecolor='black', alpha=0.6)
    ax[0].set_title(f'{var_name} Histogram (Log={log_scale})')
    
    # Cumulative Distribution Function (CDF)
    idx = np.argsort(vals)
    cdf = np.cumsum(weights[idx]) / np.sum(weights)
    ax[1].plot(vals[idx], cdf, color='black', lw=2)
    if log_scale and np.any(vals > 0):
        ax[1].set_xscale('log')
    ax[1].set_title(f'{var_name} CDF')
    ax[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()







def compare_distributions(df, col_x, col_y, use_log=False, 
                          color_x='C0', color_y='C1', 
                          color_qq='C2', color_scatter='C0'):
    """
    Compares two variable distributions using Histograms, ECDFs, QQ plots, and Scatter plots.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data.
    col_x : str
        The column name for the first variable (x-axis / baseline).
    col_y : str
        The column name for the second variable (y-axis / comparison).
    use_log : bool
        If True, applies log scaling to the axes while displaying real values.
    color_x, color_y, color_qq, color_scatter : str
        Optional color overrides for the plots.
    """
    
    # 1. Data Preparation
    data = df[[col_x, col_y]].dropna().copy()
    
    if use_log:
        # Guard against zero or negative values for log scaling
        initial_len = len(data)
        data = data[(data[col_x] > 0) & (data[col_y] > 0)]
        if len(data) < initial_len:
            print(f"Warning: Dropped {initial_len - len(data)} rows containing zeros or negative values for log scaling.")
            
    if data.empty:
        raise ValueError(f"No valid rows remaining in '{col_x}' and '{col_y}'.")

    x = data[col_x].values
    y = data[col_y].values

    # 2. Generate Well-Structured Summary Table
    print(f"\n--- Statistical Summary: {col_x} vs {col_y} ---")
    stats_df = data.describe().T
    stats_df['Pearson_r'] = [stats.pearsonr(x, y)[0], None] # Add correlation to the first row
    
    # Use pandas to display a clean table (works well in Jupyter Notebooks and console)
    try:
        from IPython.display import display
        display(stats_df.round(3))
    except ImportError:
        print(stats_df.round(3))

    # 3. Plot Setup
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_hist, ax_cdf, ax_qq, ax_scatter = axes.ravel()
    
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())

    # Formatter to ensure real numbers are shown instead of scientific notation (e.g. 10^2)
    formatter = mticker.ScalarFormatter()
    formatter.set_scientific(False)

    # --- Plot 1: Overlapping Histograms ---
    if use_log:
        bins = np.geomspace(min_val, max_val, 40)
    else:
        bins = np.linspace(min_val, max_val, 40)
        
    ax_hist.hist(x, bins=bins, density=True, alpha=0.6, label=col_x, color=color_x, edgecolor='none')
    ax_hist.hist(y, bins=bins, density=True, alpha=0.5, label=col_y, color=color_y, edgecolor='none')
    ax_hist.set_title('Overlapping Histograms (Density)')
    ax_hist.set_xlabel('Value')
    ax_hist.legend()

    # --- Plot 2: Empirical CDFs ---
    def compute_ecdf(a):
        return np.sort(a), np.arange(1, len(a) + 1) / len(a)

    x_sort, x_ecdf = compute_ecdf(x)
    y_sort, y_ecdf = compute_ecdf(y)

    ax_cdf.plot(x_sort, x_ecdf, label=col_x, color=color_x, linewidth=2)
    ax_cdf.plot(y_sort, y_ecdf, label=col_y, color=color_y, linewidth=2)
    ax_cdf.set_title('Empirical CDFs')
    ax_cdf.set_xlabel('Value')
    ax_cdf.set_ylabel('Cumulative Probability')
    ax_cdf.legend()
    ax_cdf.grid(alpha=0.2)

    # --- Plot 3: QQ Plot ---
    nq = min(len(x), len(y))
    q = np.linspace(0, 1, nq, endpoint=False) + 0.5 / nq
    qx = np.quantile(x, q)
    qy = np.quantile(y, q)

    ax_qq.scatter(qx, qy, s=10, color=color_qq)
    
    # 1:1 line
    lims = [min_val, max_val]
    ax_qq.plot(lims, lims, linestyle='--', color='gray', zorder=0, label='1:1 Line')
    
    # Linear fit of quantiles (calculated on raw data)
    slope, intercept = np.polyfit(qx, qy, 1)
    ax_qq.plot(lims, intercept + slope * np.array(lims), color='red', linewidth=1, label='Linear Fit')
    r_q = np.corrcoef(qx, qy)[0, 1]
    
    ax_qq.set_title('QQ Plot (Sample Quantiles)')
    ax_qq.set_xlabel(f'{col_x} Quantiles')
    ax_qq.set_ylabel(f'{col_y} Quantiles')
    ax_qq.text(0.02, 0.95, f'slope = {slope:.3f}\n$r$ = {r_q:.3f}', transform=ax_qq.transAxes,
               va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    ax_qq.legend(loc='lower right')
    ax_qq.grid(alpha=0.2)

    # --- Plot 4: Scatter Plot ---
    ax_scatter.scatter(x, y, s=20, alpha=0.6, color=color_scatter)
    
    # Manual linear regression line (calculated on raw data)
    slope_s, intercept_s, r_val, p_val, std_err = stats.linregress(x, y)
    x_fit = np.linspace(min_val, max_val, 100)
    y_fit = intercept_s + slope_s * x_fit
    ax_scatter.plot(x_fit, y_fit, color='red', linewidth=1.5)
    
    ax_scatter.set_title('Scatter Plot with Linear Regression')
    ax_scatter.set_xlabel(col_x)
    ax_scatter.set_ylabel(col_y)
    ax_scatter.text(0.02, 0.95, f'Pearson $r$ = {r_val:.3f}\n$p$-val = {p_val:.2g}', 
                    transform=ax_scatter.transAxes, va='top', 
                    bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    ax_scatter.grid(alpha=0.2)

    # --- Apply Log Scaling ---
    if use_log:
        for ax in [ax_hist, ax_cdf]:
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(formatter)
            
        for ax in [ax_qq, ax_scatter]:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.show()

# ==========================================
# Example Usage:
# compare_distributions(gc_comps, 'U3O8_ppm', 'eU3O8_CALC', use_log=True)
#
# Custom color example:
# compare_distributions(gc_comps, 'U3O8_ppm', 'eU3O8_CALC', use_log=False, 
#                       color_x='purple', color_y='orange', color_qq='teal', color_scatter='navy')
# ==========================================