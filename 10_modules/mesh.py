import numpy as np
import pandas as pd
import trimesh
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pyvista as pv




def import_stl_to_trimesh(file_path: str) -> trimesh.Trimesh:
    """
    Imports an STL file and returns a trimesh object.
    
    Parameters:
    -----------
    file_path : str
        The absolute or relative path to the .stl file.
        
    Returns:
    --------
    trimesh.Trimesh
        The loaded 3D mesh object. Returns None if loading fails.
    """
    # 1. Check if the file actually exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file was not found at: {file_path}")
        
    # 2. Check the file extension
    if not file_path.lower().endswith('.stl'):
        raise ValueError("The provided file does not have an .stl extension.")

    try:
        # 3. Load the mesh
        mesh = trimesh.load(file_path,merge_norm=True, merge_tex=True)
        #mesh.merge_vertices()
        #mesh.remove_duplicate_faces()
        
        # Trimesh sometimes loads complex files as a Scene. 
        # For STL, it is almost always a single Trimesh object, but we can enforce it:
        if isinstance(mesh, trimesh.Scene):
            # If it's a scene, grab the first geometry (or combine them)
            if len(mesh.geometry) > 0:
                mesh = list(mesh.geometry.values())[0]
            else:
                raise ValueError("The loaded STL scene contains no geometry.")

        #mesh.remove_unreferenced_vertices()
        #mesh.merge_vertices()
        #mesh.remove_duplicate_faces()
        #mesh.process()

        print(f"Successfully loaded mesh: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices.")
        return mesh
        
    except Exception as e:
        print(f"An error occurred while loading the STL file: {e}")
        return None

# ==========================================
# Example Usage:
# my_mesh = import_stl_to_trimesh('./models/my_part.stl')
# if my_mesh:
#     my_mesh.show() # Opens a 3D viewer window
# ==========================================



def import_surface_stl(file_path: str) -> trimesh.Trimesh:
    """
    Imports an STL file representing an open surface (e.g., topography, 
    geological contact) into a trimesh object.
    
    Parameters:
    -----------
    file_path : str
        The absolute or relative path to the .stl file.
        
    Returns:
    --------
    trimesh.Trimesh
        The loaded 3D mesh object. Returns None if loading fails.
    """
    
    # 1. Validate that the file exists to prevent vague errors
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file was not found at: {file_path}")
        
    # 2. Ensure the file has the correct extension
    if not file_path.lower().endswith('.stl'):
        raise ValueError("The provided file does not have an .stl extension.")

    try:
        # 3. Load the mesh from the file path
        # Trimesh handles both ASCII and Binary STL formats automatically
        mesh = trimesh.load(file_path)
        
        # 4. Handle edge cases where trimesh wraps the import in a Scene object
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) > 0:
                # Extract the first piece of geometry from the scene dictionary
                mesh = list(mesh.geometry.values())[0]
            else:
                raise ValueError("The loaded STL scene contains no geometry.")
                
        # 5. Clean the mesh topology
        # This is critical for surfaces: it merges coincident vertices and 
        # removes duplicate or degenerate (zero-area) faces.
        mesh.process()
        
        # 6. Validation: Check the topological state of the mesh
        # A true "surface" should have boundary edges (edges shared by only 1 face).
        # If it is 'watertight', it is actually a closed volume/wireframe.
        if mesh.is_watertight:
            print("Notice: The imported mesh is completely closed (watertight). "
                  "It is a volume, not an open surface.")
        else:
            print("Notice: Mesh is an open surface (contains boundary edges).")
            
        print(f"Successfully loaded surface: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices.")
        
        return mesh
        
    except Exception as e:
        print(f"An error occurred while loading the STL file: {e}")
        return None

# ==========================================
# Example Usage:
# topo_surface = import_surface_stl('./surfaces/topography.stl')
# ==========================================

import numpy as np
import pandas as pd
import trimesh

def create_surface_masks(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, 
                         surface_mesh: trimesh.Trimesh) -> tuple:
    """
    Creates two 0 or 1 masks for points in a dataframe based on whether they are 
    above or below a trimesh surface. Points outside the mesh footprint default to 0.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the coordinates.
    x_col, y_col, z_col : str
        The column names for the X, Y, and Z coordinates.
    surface_mesh : trimesh.Trimesh
        The open 3D surface mesh to compare against.
        
    Returns:
    --------
    tuple of numpy.ndarray
        (mask_above, mask_below) representing arrays of 1s and 0s.
    """
    
    # 1. Extract coordinates from the dataframe
    x_coords = df[x_col].values
    y_coords = df[y_col].values
    z_coords = df[z_col].values

    # 2. Setup Ray Origins and Directions
    # Find a safe "ceiling" height above the highest point of the mesh
    ceiling_z = surface_mesh.bounds[1][2] + 1000.0 
    
    # Create ray origins at (X, Y, ceiling_z) for every point
    ray_origins = np.column_stack((x_coords, y_coords, np.full(len(df), ceiling_z)))
    
    # Create ray directions pointing straight down [0, 0, -1]
    ray_directions = np.tile([0, 0, -1], (len(df), 1))

    # 3. Perform Ray Tracing
    # intersects_location returns the [X, Y, Z] hit locations and the ray index
    locations, index_ray, index_tri = surface_mesh.ray.intersects_location(
        ray_origins=ray_origins, 
        ray_directions=ray_directions,
        multiple_hits=False # Grabs the first (highest) hit
    )

    # 4. Map the intersection Z-elevations back to the original points
    # Initialize an array with NaNs (for points outside the mesh footprint)
    surface_elevations = np.full(len(df), np.nan)
    
    # Assign the Z-coordinate of the intersection to the corresponding ray index
    surface_elevations[index_ray] = locations[:, 2]

    # 5. Create both masks simultaneously
    # Comparisons with NaN automatically safely evaluate to False (0)
    mask_above = (z_coords > surface_elevations).astype(int)
    mask_below = (z_coords < surface_elevations).astype(int)

    return mask_above, mask_below



def create_volume_mask(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, 
                           volume_mesh: trimesh.Trimesh) -> np.ndarray:
    
    # 1. Convert trimesh to PyVista PolyData (using our previous subroutine)
    pv_mesh = trimesh_to_pyvista(volume_mesh)
    
    # 2. Convert dataframe points to a PyVista Point Cloud
    valid_coords = df[[x_col, y_col, z_col]].dropna().values
    cloud = pv.PolyData(valid_coords)
    
    # 3. Run the VTK C++ Enclosure Filter (Lightning Fast)
    print("Running VTK enclosure filter...")
    selected = cloud.select_interior_points(pv_mesh, check_surface=False)
    
    # 'SelectedPoints' is an array of 0s and 1s provided by PyVista
    inside_array = selected['selected_points']
    
    # 4. Map back to dataframe (handling NaNs as we did before)
    final_mask = np.zeros(len(df), dtype=int)
    valid_mask = df[[x_col, y_col, z_col]].notna().all(axis=1)
    final_mask[valid_mask] = inside_array
    
    return final_mask

# ==========================================
# Example Usage:
#
# # Load your closed solid (e.g., an ore domain)
# ore_solid = trimesh.load('ore_domain.stl')
# ore_solid.process() # Clean the topology
#
# # Create the mask
# comps['in_ore'] = create_volume_mask(comps, 'MID_X', 'MID_Y', 'MID_Z', ore_solid)
#
# # See how many points fell inside
# print(f"Points inside wireframe: {comps['in_ore'].sum()}")
# ==========================================    


def diagnose_surface_selection(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, 
                               mesh: trimesh.Trimesh, visualize: bool = True):
    """
    Diagnoses common issues when comparing a DataFrame of points to a 3D surface mesh.
    """
    print("--- DIAGNOSTIC REPORT ---")
    
    # 1. Bounding Box Check (Coordinate Mismatch)
    df_min = [df[x_col].min(), df[y_col].min(), df[z_col].min()]
    df_max = [df[x_col].max(), df[y_col].max(), df[z_col].max()]
    mesh_min = mesh.bounds[0]
    mesh_max = mesh.bounds[1]
    
    print("\n1. BOUNDING BOXES (Check for Coordinate/Scale Mismatch):")
    print(f"   DataFrame X: {df_min[0]:.2f} to {df_max[0]:.2f} | Mesh X: {mesh_min[0]:.2f} to {mesh_max[0]:.2f}")
    print(f"   DataFrame Y: {df_min[1]:.2f} to {df_max[1]:.2f} | Mesh Y: {mesh_min[1]:.2f} to {mesh_max[1]:.2f}")
    print(f"   DataFrame Z: {df_min[2]:.2f} to {df_max[2]:.2f} | Mesh Z: {mesh_min[2]:.2f} to {mesh_max[2]:.2f}")
    
    # Check if X/Y footprints actually overlap
    if (df_max[0] < mesh_min[0] or df_min[0] > mesh_max[0] or 
        df_max[1] < mesh_min[1] or df_min[1] > mesh_max[1]):
        print("\n   [!] CRITICAL ERROR: The X/Y footprints do not overlap at all. "
              "Your points and your mesh are in different locations or scales.")
    else:
        print("\n   [OK] X/Y footprints overlap.")

    # 2. Mesh Health Check
    print("\n2. MESH HEALTH:")
    print(f"   Watertight (Closed Volume): {mesh.is_watertight}")
    print(f"   Consistent Normals: {mesh.is_winding_consistent}")

    # 3. Ray Tracing Test (Check for Overhangs / Multiple Hits)
    print("\n3. OVERHANG TEST:")
    # Shoot 100 random rays from the dataframe bounds to see if they hit multiple times
    sample_x = np.random.uniform(max(df_min[0], mesh_min[0]), min(df_max[0], mesh_max[0]), 100)
    sample_y = np.random.uniform(max(df_min[1], mesh_min[1]), min(df_max[1], mesh_max[1]), 100)
    ray_origins = np.column_stack((sample_x, sample_y, np.full(100, mesh_max[2] + 100)))
    ray_directions = np.tile([0, 0, -1], (100, 1))
    
    # Force multiple_hits=True to see if rays pass through the surface more than once
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=True
    )
    
    # If a single ray index appears more than once, it hit multiple Z elevations
    unique_rays, counts = np.unique(index_ray, return_counts=True)
    multiple_hits = np.sum(counts > 1)
    
    if multiple_hits > 0:
        print(f"   [!] WARNING: {multiple_hits}% of test rays hit the surface multiple times. "
              "Your surface has vertical overlaps, folds, or is a closed volume.")
    else:
        print("   [OK] No overhangs detected in test sample. Surface is clean 2.5D.")

    # 4. Interactive Visualization
    if visualize:
        print("\nLaunching 3D Viewer... Close the window to continue your script.")
        # Convert DataFrame points to a Trimesh PointCloud
        points = df[[x_col, y_col, z_col]].dropna().values
        point_cloud = trimesh.points.PointCloud(points, colors=[255, 0, 0, 255]) # Red points
        
        # Make the mesh semi-transparent blue
        mesh.visual.face_colors = [0, 0, 255, 100] 
        
        # Create a scene and show it
        scene = trimesh.Scene([mesh, point_cloud])
        scene.show()

# ==========================================
# Example Usage:
# diagnose_surface_selection(comps, 'MID_X', 'MID_Y', 'MID_Z', pit)
# ==========================================


def plot_mesh_section(mesh: trimesh.Trimesh, axis: str, coord: float, 
                      ax: plt.Axes = None, color: str = 'blue', 
                      linewidth: float = 1.5, **kwargs) -> plt.Axes:
    """
    Slices a 3D trimesh object and plots the raw 2D boundary segments.
    Bypasses graph chaining to physically prevent artificial lines between polygons.
    """
    
    axis = axis.lower()
    if axis not in ['x', 'y', 'z']:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    # 1. Define the slicing plane normal, origin, and 2D mapping indices
    if axis == 'x':
        normal, origin = [1, 0, 0], [coord, 0, 0]
        idx_u, idx_v = 1, 2      # Plot Y vs Z
        xlabel, ylabel = 'Y', 'Z'
    elif axis == 'y':
        normal, origin = [0, 1, 0], [0, coord, 0]
        idx_u, idx_v = 0, 2      # Plot X vs Z
        xlabel, ylabel = 'X', 'Z'
    else: # 'z'
        normal, origin = [0, 0, 1], [0, 0, coord]
        idx_u, idx_v = 0, 1      # Plot X vs Y
        xlabel, ylabel = 'X', 'Y'

    if ax is None:
        fig, ax = plt.subplots()

    # 2. Slice the ENTIRE mesh at once
    slice_3d = mesh.section(plane_origin=origin, plane_normal=normal)

    if slice_3d is None:
        print(f"Warning: No geometry found at {axis.upper()} = {coord}")
        return ax

    # 3. Extract RAW line segments (Bypasses networkx loop-chaining completely)
    lines_2d = []
    for entity in slice_3d.entities:
        # Get the 3D vertices for this exact triangle intersection
        curve_3d = slice_3d.vertices[entity.points]
        
        # Map to the 2D plot plane
        curve_2d = curve_3d[:, [idx_u, idx_v]]
        lines_2d.append(curve_2d)

    # 4. Plot all segments instantly using LineCollection
    lc = LineCollection(lines_2d, colors=color, linewidths=linewidth, **kwargs)
    ax.add_collection(lc)

    # 5. Update axis limits (LineCollection doesn't auto-scale automatically)
    all_pts = np.vstack(lines_2d)
    ax.update_datalim(all_pts)
    ax.autoscale_view()

    # 6. Format the plot
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'Mesh Section at {axis.upper()} = {coord}')
    ax.grid(alpha=0.3)

    return ax





def trimesh_to_pyvista(mesh: trimesh.Trimesh) -> pv.PolyData:
    """
    Converts a trimesh.Trimesh object into a pyvista.PolyData object.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        The loaded 3D mesh object from trimesh.
        
    Returns:
    --------
    pyvista.PolyData
        A PyVista mesh object ready to be passed into plotter.add_mesh()
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Input must be a trimesh.Trimesh object.")

    # 1. Extract vertices
    vertices = mesh.vertices
    
    # 2. Format faces for VTK/PyVista
    # Create a column of 3s (since trimesh only uses triangles)
    padding = np.full((len(mesh.faces), 1), 3, dtype=np.int_)
    
    # Horizontally stack the 3s next to the face indices, then flatten it into a 1D array
    pv_faces = np.hstack((padding, mesh.faces)).flatten()
    
    # 3. Create and return the PyVista PolyData object
    pv_mesh = pv.PolyData(vertices, pv_faces)
    
    return pv_mesh

# ==========================================
# Example Usage:
#
# # 1. Load your trimesh
# my_trimesh = trimesh.load('my_pit.stl')
# 
# # 2. Convert it using the subroutine
# pv_surface = trimesh_to_pyvista(my_trimesh)
#
# # 3. Plot it using PyVista
# plotter = pv.Plotter()
# plotter.add_mesh(pv_surface, color='lightblue', show_edges=True)
# plotter.show()
# ==========================================    

import pandas as pd
import pyvista as pv
import numpy as np

def plot_notebook_point_cloud(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, 
                              scalar_col: str, point_size: float = 5.0, 
                              cmap: str = 'viridis') -> None:
    """
    Plots an interactive 3D point cloud directly inside a Jupyter Notebook.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the coordinates and scalar data.
    x_col, y_col, z_col : str
        The column names for the X, Y, and Z coordinates.
    scalar_col : str
        The column name of the variable to color the points by (e.g., grade, density).
    point_size : float, optional
        The visual size of the points in the 3D viewer.
    cmap : str, optional
        The matplotlib colormap to use (e.g., 'viridis', 'plasma', 'jet').
    """
    
    # 1. Drop any rows with blank coordinates or blank scalar values to prevent crashes
    clean_df = df[[x_col, y_col, z_col, scalar_col]].dropna()
    
    if len(clean_df) == 0:
        print("Error: No valid data points left after removing NaNs.")
        return

    # 2. Extract arrays
    points = clean_df[[x_col, y_col, z_col]].values
    scalars = clean_df[scalar_col].values

    # 3. Create the PyVista Point Cloud object
    cloud = pv.PolyData(points)
    
    # Attach the scalar data to the point cloud
    cloud[scalar_col] = scalars

    # 4. Setup the Plotter
    plotter = pv.Plotter(notebook=True)
    
    # Add the point cloud to the scene
    plotter.add_mesh(
        cloud, 
        scalars=scalar_col,
        cmap=cmap,
        point_size=point_size,
        render_points_as_spheres=True, # Makes points look like nice 3D spheres instead of flat squares
        show_scalar_bar=True
    )
    
    # Optional: Add an orientation axis in the corner
    plotter.add_axes()

    # 5. Show the plot using the 'trame' backend for Jupyter interactivity
    plotter.show(jupyter_backend='trame')

# ==========================================
# Example Usage:
# plot_notebook_point_cloud(comps, 'MID_X', 'MID_Y', 'MID_Z', 'Fe_grade', point_size=8.0)
# ==========================================



def plot_gaussian_trend_shell(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, 
                              scalar_col: str, radius: float = 0.05, 
                              grid_res: int = 50, cutoff_percent: float = 0.25) -> None:
    """
    Converts a point cloud into a continuous 3D volume using Gaussian splatting,
    then extracts a solid isosurface (grade shell) to visualize the spatial trend.
    
    Parameters:
    -----------
    radius : float
        The "smear" size of each point as a percentage of the bounding box. (0.05 = 5%).
    grid_res : int
        The resolution of the 3D voxel grid (Higher = smoother, but slower).
    cutoff_percent : float
        The density threshold to draw the shell. 0.25 draws a shell around the 
        top 25% highest density areas.
    """
    
    # 1. Clean the data
    clean_df = df[[x_col, y_col, z_col, scalar_col]].dropna()
    points = clean_df[[x_col, y_col, z_col]].values
    scalars = clean_df[scalar_col].values

    # 2. Build the PyVista Point Cloud
    cloud = pv.PolyData(points)
    cloud[scalar_col] = scalars
    cloud.set_active_scalars(scalar_col)

    print("Calculating Gaussian Splat Volume (this may take a few seconds)...")
    
    # 3. Splat points into a 3D voxel grid
    try:
        volume = cloud.gaussian_splatting(radius=radius, dimensions=(grid_res, grid_res, grid_res))
    except AttributeError:
        print("Error: gaussian_splatting requires PyVista version 0.46 or newer. "
              "Please run: %pip install -U pyvista")
        return

    # 4. Extract the Trend Shell (Isosurface)
    # The splatting math alters the absolute values into a density field.
    # We find the max density and extract a shell at a percentage of that max.
    splat_max = volume.active_scalars.max()
    iso_val = splat_max * cutoff_percent
    
    print(f"Extracting trend isosurface...")
    trend_shell = volume.contour([iso_val])

    # 5. Plot the result
    plotter = pv.Plotter(notebook=True)
    
    # Plot the original points faintly in the background for context
    plotter.add_mesh(cloud, color='lightgrey', point_size=3.0, opacity=0.1, render_points_as_spheres=True)
    
    # Plot the solid trend shell
    plotter.add_mesh(trend_shell, color='red', opacity=0.7, show_edges=False)
    
    plotter.add_axes()
    plotter.show(jupyter_backend='trame')

# ==========================================
# Example Usage:
# mh.plot_gaussian_trend_shell(comps, 'MID_X', 'MID_Y', 'MID_Z', 'AD', radius=0.08)
# ==========================================



def plot_ew_sections(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, 
                     scalar_col: str, n_sections: int = 5, radius: float = 0.05, 
                     grid_res: int = 50) -> None:
    """
    Creates East-West cross-sections through a Gaussian splat volume.
    
    Parameters:
    -----------
    n_sections : int
        The number of slices to cut along the Northing (Y) axis.
    radius : float
        The Gaussian "smear" size (0.05 = 5% of bounding box).
    grid_res : int
        The voxel resolution of the background volume.
    """
    
    # 1. Clean the data
    clean_df = df[[x_col, y_col, z_col, scalar_col]].dropna()
    points = clean_df[[x_col, y_col, z_col]].values
    scalars = clean_df[scalar_col].values

    # 2. Build the PyVista Point Cloud
    cloud = pv.PolyData(points)
    cloud[scalar_col] = scalars
    cloud.set_active_scalars(scalar_col)

    print("Calculating continuous 3D volume...")
    try:
        volume = cloud.gaussian_splatting(radius=radius, dimensions=(grid_res, grid_res, grid_res))
    except AttributeError:
        print("Error: Please update PyVista to use gaussian_splatting.")
        return

    # 3. Slice the Volume
    print(f"Cutting {n_sections} East-West sections...")
    # Slicing along the 'y' axis cuts planes spanning X (Easting) and Z (Elevation)
    slices = volume.slice_along_axis(n=n_sections, axis='y')

    # 4. Plot the result
    plotter = pv.Plotter(notebook=True)
    
    # Plot the 2D heat map sections
    plotter.add_mesh(slices, cmap='viridis', opacity=0.9, show_scalar_bar=True)
    
    # Plot the original points faintly in black so you can see where the data actually is
    plotter.add_mesh(cloud, color='black', point_size=2.0, opacity=0.15, render_points_as_spheres=True)
    
    # Add an outline box for spatial context
    plotter.add_bounding_box(color='grey')
    plotter.add_axes()
    
    # Force the camera to look directly North (down the Y-axis) so sections face you flat-on
    plotter.view_xz()
    
    plotter.show(jupyter_backend='trame')

# ==========================================
# Example Usage:
# mh.plot_ew_sections(comps, 'MID_X', 'MID_Y', 'MID_Z', 'AD', n_sections=8)
# ==========================================



def plot_interactive_ew_section(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, 
                                scalar_col: str, radius: float = 0.05, 
                                grid_res: int = 50) -> None:
    """
    Creates an interactive East-West cross-section through a Gaussian splat volume.
    Adds a slider widget to move the plane seamlessly along the Y (Northing) axis.
    
    Parameters:
    -----------
    radius : float
        The Gaussian "smear" size (0.05 = 5% of bounding box).
    grid_res : int
        The voxel resolution of the background volume. Higher resolution makes
        smoother slices but takes longer to compute.
    """
    
    # 1. Clean the data
    clean_df = df[[x_col, y_col, z_col, scalar_col]].dropna()
    points = clean_df[[x_col, y_col, z_col]].values
    scalars = clean_df[scalar_col].values

    # 2. Build the PyVista Point Cloud
    cloud = pv.PolyData(points)
    cloud[scalar_col] = scalars
    cloud.set_active_scalars(scalar_col)

    print("Calculating continuous 3D volume (this takes a moment)...")
    try:
        volume = cloud.gaussian_splatting(radius=radius, dimensions=(grid_res, grid_res, grid_res))
    except AttributeError:
        print("Error: Please update PyVista to use gaussian_splatting.")
        return

    # 3. Setup the Plotter
    plotter = pv.Plotter(notebook=True)
    
    # Plot the original points faintly in black for spatial context
    plotter.add_mesh(cloud, color='black', point_size=2.0, opacity=0.1, render_points_as_spheres=True)
    
    # 4. Add the Interactive Slice Widget
    # Setting normal='y' forces the plane to face North/South (an East-West cut)
    print("Generating interactive slice widget...")
    plotter.add_mesh_slice(
        volume, 
        normal='y', 
        cmap='viridis', 
        show_scalar_bar=True,
        
    )
    
    # Add an outline box and axes
    plotter.add_bounding_box(color='grey')
    plotter.add_axes()
    
    # Force the camera to look directly North so the section faces you flat-on
    plotter.view_xz()
    
    print("Launching viewer! Click and drag the border of the plane to move it through the volume.")
    plotter.show(jupyter_backend='trame')

# ==========================================
# Example Usage:
# mh.plot_interactive_ew_section(comps, 'MID_X', 'MID_Y', 'MID_Z', 'AD', radius=0.08)
# ==========================================