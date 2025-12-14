import laspy
import numpy as np
import time


def las_voxel_downsample(input_las_path: str,
                         output_las_path: str,
                         n_points: int = 1,
                         voxel_size: float = 0.05,
                         min_points_in_voxel: int = 10):
    print("Loading LAS...")
    las = laspy.read(input_las_path)
    points = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)

    N = las.header.point_count
    print(f"{N} points loaded.")

    print("Grouping points into voxels...")
    # Assign each point an index in the voxel grid
    mins = points.min(axis=0)
    idx = np.floor((points - mins) / voxel_size).astype(np.int32)

    # Group points by voxel:
    #  - inverse: index of the unique voxel for each point
    #  - counts:  number of points in each voxel
    unique_voxels, inverse, counts = np.unique(
        idx,
        axis=0,
        return_inverse=True,
        return_counts=True)

    n_voxels = len(unique_voxels)

    print(f"Voxels: {n_voxels}")

    # Filter voxels by minimum number of points
    valid_voxels = counts >= min_points_in_voxel
    valid_count = np.sum(valid_voxels)

    if valid_count == 0:
        raise ValueError("No voxels left after applying min_points_in_voxel filter")

    print(f"Voxels after min_points filter: {valid_count} (min = {min_points_in_voxel})")

    # Mask for points
    mask = np.zeros(N, dtype=bool)

    if n_points == 1:
        # Mask points that belong to voxels that passed the filter
        valid_voxel_mask = valid_voxels[inverse]

        # Set inverse index to -1 for points in invalid voxels
        inverse[~valid_voxel_mask] = -1

        # Extract the first point index from each valid voxel
        _, first_inverse = np.unique(inverse, return_index=True)
        first = first_inverse[inverse[first_inverse] != -1]

        # Build the mask
        mask[first] = True
    elif n_points == -1:
        mask = valid_voxels[inverse]

    elif n_points > 1:
        valid_mask = valid_voxels[inverse]
        idx = np.where(valid_mask)[0]
        vox = inverse[valid_mask]

        order = np.lexsort((idx, vox))
        idx = idx[order]
        vox = vox[order]

        starts = np.r_[0, np.flatnonzero(vox[1:] != vox[:-1]) + 1]
        pos = np.arange(len(vox)) - np.repeat(starts, np.diff(np.r_[starts, len(vox)]))

        keep = pos < n_points
        keep_global = idx[keep]
        mask[keep_global] = True

    else:
        print("Invalid parameter")

    print(f"Remaining points: {mask.sum()}")
    print("Creating new LAS...")
    new_las = las[mask]
    print("Writing file...")
    new_las.write(output_las_path)
    print("Done.")


if __name__ == "__main__":

    input_las_path = "data/test_data/non_ground.las"

    start_time = time.time()
    las_voxel_downsample(
        input_las_path,
        "data/test_data/alg3.las",
        voxel_size=0.05,
        min_points_in_voxel=50,
        n_points=50)
    print(f'10 points time: {time.time() - start_time:.2f}\n')

    