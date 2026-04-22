import pyvista as pv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import binned_statistic_2d
from matplotlib.colors import LogNorm, Normalize
from matplotlib.backends.backend_pdf import PdfPages
from typing import Optional, Union



def plot_spatial_data(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: Optional[str] = None,
    val_col: Optional[str] = None,
    is_discrete: bool = False,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Axes:
    """
    Generates a geostatistically-sound spatial scatter plot with forced equal aspect ratio.

    Theory:
        Ensures spatial integrity by maintaining an isometric scale (1 unit X = 1 unit Y).
        Supports both continuous (e.g., Grade) and discrete (e.g., Lithology) variables
        using appropriate color mapping.

    Args:
        df: The input dataframe containing spatial coordinates and variables.
        x_col: Name of the column representing the Easting/X coordinate.
        y_col: Name of the column representing the Northing/Y coordinate.
        val_col: Optional; Column name for color mapping.
        is_discrete: If True, treats val_col as categorical data.
        ax: Optional; Existing matplotlib axis to plot onto.
        **kwargs: Additional arguments passed to plt.scatter (e.g., s=size, cmap, alpha).

    Returns:
        plt.Axes: The axis object with the spatial plot.

    Raises:
        ValueError: If coordinates contain NaNs or if specified columns are missing.
    """
    # 1. Validation
    required_cols = [x_col, y_col]
    if val_col:
        required_cols.append(val_col)
        
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
    
    if df[[x_col, y_col]].isnull().values.any():
        raise ValueError("Coordinates contain NaN values. Please clean the data first.")

    # 2. Setup Plotting Surface
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # 3. Handle Default Styling
    # We set defaults but allow kwargs to override them
    params = {
        's': 20,
        'edgecolor': 'black',
        'linewidth': 0.5,
        'cmap': 'viridis' if not is_discrete else 'Set1'
    }
    params.update(kwargs)

    # 4. Logic for Continuous vs Discrete
    if val_col:
        if is_discrete:
            # Map categories to integers for plotting
            categories = df[val_col].astype('category').cat
            scatter = ax.scatter(df[x_col], df[y_col], c=categories.codes, **params)
            
            # Add legend for discrete classes
            handles, _ = scatter.legend_elements()
            ax.legend(handles, categories.categories, title=val_col, loc='best')
        else:
            # Continuous plotting with colorbar
            scatter = ax.scatter(df[x_col], df[y_col], c=df[val_col], **params)
            plt.colorbar(scatter, ax=ax, label=val_col, shrink=0.8)
    else:
        # Simple coordinate plot
        ax.scatter(df[x_col], df[y_col], **params)

    # 5. Geostatistical Requirement: Equal Aspect Ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Labeling
    ax.set_xlabel(f"Easting ({x_col})")
    ax.set_ylabel(f"Northing ({y_col})")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.6)

    # 6. Coordinate Formatting (The Update)
    # Disable scientific notation
    formatter = ticker.ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # Set tick label fontsize to 8
    ax.tick_params(axis='both', which='major', labelsize=8)

    return ax

# Simple internal assertion test
if __name__ == "__main__":
    test_df = pd.DataFrame({'E': [0, 1], 'N': [0, 1]})
    try:
        plot_spatial_data(test_df, 'E', 'N')
        print("Validation Check: Passed")
    except Exception as e:
        print(f"Validation Check: Failed - {e}")


def visualize_geostat_point_cloud(
    df: pd.DataFrame, 
    coords: list = ['X', 'Y', 'Z'], 
    data_col: str = 'Au_gpt', 
    mip_threshold: float = None,
    point_size: float = 5.0,
    cmap: str = 'inferno'
):
    """
    Visualizes geological drillhole data in 3D using PyVista.
    
    Theory:
        Uses a VTK-backed point cloud to render spatial data. The MIP 
        functionality is simulated by high-intensity thresholding to 
        reveal the core of the mineralized body.

    Args:
        df (pd.DataFrame): Dataframe containing coordinates and values.
        coords (list): Column names for [Easting, Northing, Elevation].
        data_col (str): The numeric attribute to visualize (e.g., Grade).
        mip_threshold (float, optional): If provided, only points above 
                                         this 'intensity' are shown.
        point_size (float): Diameter of the rendered points.
        cmap (str): Matplotlib-style colormap.
    """
    # 1. Validation and NaN handling
    required_cols = coords + [data_col]
    data_clean = df[required_cols].dropna().copy()
    
    if mip_threshold is not None:
        data_clean = data_clean[data_clean[data_col] >= mip_threshold]
        if data_clean.empty:
            print(f"Warning: No points found above threshold {mip_threshold}")
            return

    # 2. Convert to PyVista PolyData
    points = data_clean[coords].values
    point_cloud = pv.PolyData(points)
    point_cloud[data_col] = data_clean[data_col].values

    # 3. Setup Plotter
    plotter = pv.Plotter(title="Geostatistical 3D Visualization")
    plotter.set_background("black") # Standard for geological software
    
    # 4. Add Point Cloud
    # We use 'render_points_as_spheres' for high-quality visual representation
    actor = plotter.add_mesh(
        point_cloud, 
        scalars=data_col, 
        cmap=cmap, 
        point_size=point_size, 
        render_points_as_spheres=True,
        opacity=0.8,
        scalar_bar_args={'title': f"{data_col} Intensity"}
    )

    # 5. Add spatial context
    plotter.add_axes()
    plotter.show_grid()
    
    print(f"Rendering {len(data_clean)} points. Close window to continue...")
    plotter.show()

# Example Usage
if __name__ == "__main__":
    # Generate a synthetic mineralized vein (plunging cylinder)
    n_samples = 2000
    z = np.linspace(0, 100, n_samples)
    x = 5 * np.sin(z/10) + np.random.normal(0, 1, n_samples)
    y = z * 0.5 + np.random.normal(0, 1, n_samples)
    grade = np.exp(np.random.normal(0, 0.5, n_samples)) * (z/10) # Grade increases with depth

    df_synthetic = pd.DataFrame({'X': x, 'Y': y, 'Z': z, 'Au': grade})

    # Visualize with a "Maximum Intensity" threshold
    # Show only the top 20% of grades to simulate a high-intensity core
    top_20_percentile = df_synthetic['Au'].quantile(0.8)
    
    visualize_geostat_point_cloud(
        df_synthetic, 
        coords=['X', 'Y', 'Z'], 
        data_col='Au', 
        mip_threshold=top_20_percentile,
        cmap='hot'
    )




def geostat_2d_mip_projection(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    value_col: str, 
    bin_size: float = 5.0,
    cmap: str = 'magma',
    log_scale: bool = True
):
    """
    Creates a 2D Maximum Intensity Projection (MIP) from 3D point data.

    Theory:
        Discretizes the 2D plane into pixels of size 'bin_size'. For every 
        pixel, it searches the entire depth of the 3D volume and assigns 
        the maximum value found to that pixel.

    Args:
        df (pd.DataFrame): Input dataset.
        x_col (str): The horizontal axis for the projection (e.g., 'X').
        y_col (str): The vertical axis for the projection (e.g., 'Z' for a section).
        value_col (str): The variable to project (e.g., 'Au_gpt').
        bin_size (float): The resolution of the projection in coordinate units.
        cmap (str): Colormap for intensity.
        log_scale (bool): Use log-normalization for the color scale.
    """
    # 1. Setup Bins based on data extents
    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()
    
    x_bins = np.arange(x_min, x_max + bin_size, bin_size)
    y_bins = np.arange(y_min, y_max + bin_size, bin_size)

    # 2. Perform the Projection (The "MIP" Core)
    # binned_statistic_2d is vectorized and highly efficient
    mip_map, x_edges, y_edges, _ = binned_statistic_2d(
        df[x_col], 
        df[y_col], 
        df[value_col], 
        statistic='max', 
        bins=[x_bins, y_bins]
    )

    # 3. Visualization
    plt.figure(figsize=(12, 8))
    
    # Handle Log Scaling for visualization
    from matplotlib.colors import LogNorm, Normalize
    norm = LogNorm() if log_scale else Normalize()

    # Transpose mip_map because binned_statistic_2d returns (nx, ny)
    im = plt.imshow(
        mip_map.T, 
        origin='lower', 
        extent=[x_min, x_max, y_min, y_max],
        aspect='equal', 
        cmap=cmap, 
        norm=norm
    )

    plt.colorbar(im, label=f'Max Intensity ({value_col})')
    plt.xlabel(f'Coordinate: {x_col}')
    plt.ylabel(f'Coordinate: {y_col}')
    plt.title(f'2D Maximum Intensity Projection (MIP)\nResolution: {bin_size} units')
    plt.grid(alpha=0.2)
    plt.show()

    return mip_map    




def geostat_sectional_mip(
    df: pd.DataFrame,
    section_axis: str,
    section_coord: float,
    half_width: float,
    x_col: str,
    z_col: str,
    value_col: str,
    bin_size: float = 2.0,
    cmap: str = 'viridis',
    log_scale: bool = True
):
    """
    Generates a 2D MIP for a specific spatial corridor (slab).
    
    Theory:
        Filters the 3D point cloud into a 'slab' defined by 
        section_coord +/- half_width along the section_axis. It then 
        performs a 2D Maximum Intensity Projection on the remaining points.

    Args:
        df (pd.DataFrame): Drillhole dataset.
        section_axis (str): The axis to slice along (e.g., 'Y' or 'Northing').
        section_coord (float): The center coordinate of the slice.
        half_width (float): Distance from center to slice edge.
        x_col, z_col (str): Coordinates for the 2D projection plane.
        value_col (str): Variable to project (e.g., 'Au_gpt').
        bin_size (float): Pixel resolution.
    """
    
    # 1. Slab Filtering
    mask = (df[section_axis] >= (section_coord - half_width)) & \
           (df[section_axis] <= (section_coord + half_width))
    
    df_slab = df[mask].copy()
    
    if df_slab.empty:
        print(f"No data found in section {section_axis}: {section_coord} +/- {half_width}")
        return None

    # 2. Define Bins for the 2D plane
    # We use global limits or local limits? Usually, local for a section is better.
    x_bins = np.arange(df_slab[x_col].min(), df_slab[x_col].max() + bin_size, bin_size)
    z_bins = np.arange(df_slab[z_col].min(), df_slab[z_col].max() + bin_size, bin_size)

    # 3. MIP Calculation
    mip_map, x_e, z_e, _ = binned_statistic_2d(
        df_slab[x_col],
        df_slab[z_col],
        df_slab[value_col],
        statistic='max',
        bins=[x_bins, z_bins]
    )

    # 4. Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    norm = LogNorm() if log_scale else Normalize()
    
    im = ax.imshow(
        mip_map.T,
        origin='lower',
        extent=[x_e[0], x_e[-1], z_e[0], z_e[-1]],
        cmap=cmap,
        norm=norm,
        aspect='auto'
    )
    
    plt.colorbar(im, label=f'Max {value_col}')
    ax.set_title(f"Section {section_axis}: {section_coord}\nWidth: {half_width*2} units")
    ax.set_xlabel(x_col)
    ax.set_ylabel(z_col)
    
    plt.tight_layout()
    plt.show()
    
    return mip_map    


def export_mip_atlas_to_pdf(
    df: pd.DataFrame,
    filename: str,
    section_axis: str,
    start_coord: float,
    end_coord: float,
    step: float,
    half_width: float,
    x_col: str,
    z_col: str,
    value_col: str,
    bin_size: float = 2.0,
    log_scale: bool = True,
    cmap: str = 'magma'
):
    """
    Generates a multi-page PDF Section Atlas of Maximum Intensity Projections.

    Theory:
        Iterates through a spatial range, creating 2D slices (slabs). 
        Each slice is rendered as a MIP to highlight the highest grade 
        within that corridor, then saved to a serial PDF.

    Args:
        df (pd.DataFrame): The drillhole dataset.
        filename (str): Name of the output PDF (e.g., 'Section_Atlas_Au.pdf').
        section_axis (str): The coordinate to step along (e.g., 'Northing').
        start_coord, end_coord: The spatial range for the atlas.
        step: The distance between section centers.
        half_width: The thickness of the 'look-through' on either side of center.
    """
    
    # Define the centers of the sections
    section_centers = np.arange(start_coord, end_coord + step, step)
    
    with PdfPages(filename) as pdf:
        for center in section_centers:
            # 1. Slice the Slab
            mask = (df[section_axis] >= (center - half_width)) & \
                   (df[section_axis] <= (center + half_width))
            df_slab = df[mask].copy()
            
            if df_slab.empty:
                continue

            # 2. Setup Figure
            fig, ax = plt.subplots(figsize=(11, 8.5)) # Standard US Letter / A4 Landscape
            
            # 3. Calculate MIP
            x_bins = np.arange(df_slab[x_col].min(), df_slab[x_col].max() + bin_size, bin_size)
            z_bins = np.arange(df_slab[z_col].min(), df_slab[z_col].max() + bin_size, bin_size)
            
            mip_map, x_e, z_e, _ = binned_statistic_2d(
                df_slab[x_col], df_slab[z_col], df_slab[value_col],
                statistic='max', bins=[x_bins, z_bins]
            )

            # 4. Render
            norm = LogNorm() if log_scale else Normalize()
            im = ax.imshow(
                mip_map.T, origin='lower',
                extent=[x_e[0], x_e[-1], z_e[0], z_e[-1]],
                cmap=cmap, norm=norm, aspect='auto'
            )
            
            # Formatting
            plt.colorbar(im, label=f'Max {value_col}')
            ax.set_title(
                f"SECTION ATLAS: {section_axis} {center:.1f}\n"
                f"Slab Corridor: [{center-half_width:.1f} to {center+half_width:.1f}] | "
                f"Variable: {value_col}", 
                fontsize=14, fontweight='bold'
            )
            ax.set_xlabel(f"Coordinate {x_col}")
            ax.set_ylabel(f"Coordinate {z_col}")
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # Save current page
            pdf.savefig(fig)
            plt.close(fig) # Memory management is crucial for large atlases
            
    print(f"Successfully generated Section Atlas: {filename}")    