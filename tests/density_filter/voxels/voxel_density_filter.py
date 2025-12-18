import laspy
import numpy as np
from scipy.ndimage import convolve, label


def detect_dense_vertical_structures(
    input_las_path: str,
    output_las_path: str,
    colorize_clusters: bool = False,
    voxel_size: float = 0.05,
    window_size: tuple[int, int, int] = (3, 3, 3),
    min_points_in_voxel: int = 10,
    min_neighbors_3d: int = 5,
    min_height_voxels: int = 20,
    height_ratio: float = 2.0
    ):
    '''
    Detects dense vertical structures (e.g. tree stems, poles) in a LiDAR point cloud
    using 3D voxel density analysis and connected component filtering.

    The algorithm works as follows:
      1. Voxelizes the point cloud with a fixed voxel size.
      2. Keeps only voxels containing at least `min_points_in_voxel` points.
      3. Applies a 3D neighborhood density filter using convolution
         to retain only voxels surrounded by enough dense neighbors.
      4. Extracts connected components in the 3D voxel grid.
      5. Filters connected components by geometric shape:
         components must be sufficiently tall in Z and vertically elongated
         compared to their horizontal extent.
      6. Outputs only points belonging to valid vertical components.
      7. Optionally assigns a unique RGB color to each detected component.

    Parameters
    ----------
    input_las_path : str
        Path to the input LAS/LAZ file containing the point cloud.
    output_las_path : str
        Path where the filtered LAS file will be written.
    colorize_clusters : bool, optional
        If True, assigns a unique color to each detected vertical component.
        Default is False.
    voxel_size : float, optional
        Edge length of a voxel in meters. Default is 0.05.
    window_size : tuple[int, int, int], optional
        Size of the 3D neighborhood window used for voxel connectivity
        and density filtering. Default is (3, 3, 3).
    min_points_in_voxel : int, optional
        Minimum number of points required for a voxel to be considered dense.
        Default is 10.
    min_neighbors_3d : int, optional
        Minimum number of dense neighboring voxels within the 3D window
        required to keep a voxel. Default is 5.
    min_height_voxels : int, optional
        Minimum vertical size (in voxels) of a connected component.
        Default is 20.
    height_ratio : float, optional
        Minimum ratio between vertical extent (Z) and horizontal extent (X/Y)
        required for a component to be considered vertical. Default is 2.0.

    Raises
    ------
    RuntimeError
        If no dense voxels, connected components, or valid vertical structures
        are found during processing.
    '''

    # Load point cloud
    print("Loading LAS...")
    las = laspy.read(input_las_path)
    points = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    N = len(points)
    print(f"Points loaded: {N}")

    # Voxelization
    mins = points.min(axis=0)
    idx = np.floor((points - mins) / voxel_size).astype(np.int32)

    print("Grouping points into voxels...")
    unique_voxels, counts = np.unique(idx,
                                      axis=0,
                                      return_counts=True)

    # Mask for voxels by density
    dense_voxels = counts >= min_points_in_voxel
    print(f"Dense voxels: {dense_voxels.sum()} / {len(counts)}")

    if dense_voxels.sum() == 0:
        raise RuntimeError("No dense voxels found")

    # Minimal grid that covers all voxels
    max_idx = unique_voxels.max(axis=0) + 1
    grid = np.zeros(max_idx, dtype=bool)

    # Select only dense voxels
    coords_of_dense_voxels = unique_voxels[dense_voxels]

    # Set True in the grid for each dense voxel
    grid[coords_of_dense_voxels[:, 0],
        coords_of_dense_voxels[:, 1],
        coords_of_dense_voxels[:, 2]] = True

    # Convolution of the binary 3D grid with a kernel of size window_size.
    # This computes the number of dense voxels in the neighborhood of each voxel.
    kernel = np.ones(window_size, dtype=np.int32)
    neighbor_count = convolve(grid.astype(np.int32), kernel, mode="constant")

    # Select dense voxels that have a sufficient number of dense neighbors
    grid_connected = grid & (neighbor_count >= min_neighbors_3d)
    print(f"Voxels after neighbor filter: {grid_connected.sum()}")

    if grid_connected.sum() == 0:
        raise RuntimeError("No connected dense voxels found")

    # Merge neighboring voxels into clusters, labeled 1..n_labels
    # labels has the same shape as grid_connected, 
    # but stores the cluster label number for each voxel.
    labels, n_labels = label(grid_connected, structure=np.ones((3, 3, 3)))
    print(f"Connected components: {n_labels}")

    # Coordinates of all voxels that belong to clusters
    coords = np.argwhere(labels > 0)
    # Cluster label for each voxel
    lbls = labels[coords[:, 0], coords[:, 1], coords[:, 2]]
    max_label = lbls.max()

    # Find the minimum and maximum voxel coordinates for each cluster
    mins = np.full((max_label + 1, 3),  np.inf, dtype=np.float32)
    np.minimum.at(mins, lbls, coords)

    maxs = np.full((max_label + 1, 3), -np.inf, dtype=np.float32)
    np.maximum.at(maxs, lbls, coords)

    # Compute cluster sizes along X, Y, Z axes (in voxels)
    extent = maxs - mins
    height = extent[:, 2]
    width  = np.maximum(extent[:, 0], extent[:, 1])

    # Mask for valid clusters by shape
    valid_mask = ((height >= min_height_voxels) &
                  (height >= height_ratio * width))

    # Select valid clusters
    valid_labels = np.nonzero(valid_mask)[0]
    valid_labels = valid_labels[valid_labels > 0]

    print(f"Valid vertical components: {len(valid_labels)}")

    if len(valid_labels) == 0:
        raise RuntimeError("No vertical structures detected")

    # Final voxel label for each point
    voxel_labels = labels[idx[:, 0],
                          idx[:, 1],
                          idx[:, 2]]

    # Mask for points from valid voxels
    mask = np.isin(voxel_labels, valid_labels)
    print(f"Remaining points: {mask.sum()}")

    # Colorize clusters
    if colorize_clusters:
        print("Colorizing clusters...")

        # Generate stable colors for each cluster
        rng = np.random.default_rng(42)
        colors = {
            lbl: rng.integers(0, 65535, size=3, dtype=np.uint16)
            for lbl in valid_labels
        }

        # Initialize colors
        las.red[:]   = 0
        las.green[:] = 0
        las.blue[:]  = 0

        # Assign a color to each point according to its cluster
        for lbl, (r, g, b) in colors.items():
            pts = voxel_labels == lbl
            las.red[pts]   = r
            las.green[pts] = g
            las.blue[pts]  = b

    print(f"Remaining points: {mask.sum()}")
    print("Writing LAS...")
    las[mask].write(output_las_path)
    print("Done.")


if __name__ == "__main__":

    input_las_path = "/home/chazelam/Code/LiDAR/PMF_out/non_ground.las"
    output_las_dir = "/home/chazelam/Code/LiDAR/data/voxel_density_filter_out"
    prefix = "large"

    voxel_size=0.05
    min_points_in_voxel=50
    
    window_size = (5, 5, 5)
    min_neighbors_3d=25
    min_height_voxels=20
    height_ratio=2.0

    output_las_path = f"{prefix}{output_las_dir}/vs{voxel_size}x{min_points_in_voxel} - ws{window_size}x{min_neighbors_3d}.las"
    
    detect_dense_vertical_structures(
        input_las_path=input_las_path,
        output_las_path=output_las_path,
        voxel_size=voxel_size,
        min_points_in_voxel=min_points_in_voxel,
        colorize_clusters=True,

        window_size=window_size,
        min_neighbors_3d=min_neighbors_3d,
        min_height_voxels=min_height_voxels,
        height_ratio=height_ratio)
