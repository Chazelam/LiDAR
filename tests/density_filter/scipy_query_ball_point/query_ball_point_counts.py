import numpy as np
import laspy
from scipy.spatial import cKDTree
import sys
import os
import gc
import time


def neighbor_density(points, radius, output_file_name):
    start_time = time.time()
    print(f"[INFO] r = {radius}: запуск подсчёта соседей...")

    tree = cKDTree(points)
    counts = tree.query_ball_point(points, r=radius, return_length=True, workers=-1)
    np.save(output_file_name, counts)

    del tree, counts
    gc.collect()

    elapsed = time.time() - start_time
    print(f"[INFO] r = {radius}: завершено за {elapsed:.2f} сек.\n")


if __name__ == "__main__":
    r_list = [0.01, 0.02, 0.03, 0.04, 0.05,
              0.06, 0.07, 0.08, 0.09, 0.1,
              0.2, 0.3, 0.4, 0.5]

    file_name = sys.argv[1]

    output_dir = f"{file_name.replace(' ', '_').replace('.las', '')}_density_data"
    os.makedirs(output_dir, exist_ok=True)

    las = laspy.read(file_name)

    points = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)

    del las
    gc.collect()

    print(f"Loaded {len(points):,} points\n")

    for r in r_list:
        output_file_name = f"{output_dir}/density_r{r}m.npy"
        neighbor_density(points, r, output_file_name)
        print(" -------- ")
