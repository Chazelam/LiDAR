import laspy
import numpy as np


def detect_trunk_points_2d(
        input_las_path: str,
        output_las_path: str,
        cell_size: float = 0.2,
        min_points_in_cell: int = 100
):
    print("Reading LAS...")
    las = laspy.read(input_las_path)

    N = las.header.point_count

    x = las.x
    y = las.y

    print("Computing bounds...")
    min_x = x.min()
    min_y = y.min()

    max_x = x.max()
    max_y = y.max()

    cols = int(np.ceil((max_x - min_x) / cell_size))
    rows = int(np.ceil((max_y - min_y) / cell_size))

    print(f"Grid: {rows} x {cols} = {rows * cols} cells")

    # ===== ИНДЕКС ЯЧЕЙКИ ДЛЯ КАЖДОЙ ТОЧКИ =====
    col_idx = np.floor((x - min_x) / cell_size).astype(np.int32)
    row_idx = np.floor((y - min_y) / cell_size).astype(np.int32)

    # защита
    valid = (
        (row_idx >= 0) & (row_idx < rows) &
        (col_idx >= 0) & (col_idx < cols)
    )

    row_idx = row_idx[valid]
    col_idx = col_idx[valid]

    print("Creating flat indices...")
    flat_idx = row_idx * cols + col_idx

    print("Counting points per cell...")
    counts = np.bincount(flat_idx, minlength=rows * cols)

    print("Applying threshold...")
    valid_cells = counts >= min_points_in_cell

    # маска точек
    mask = np.zeros(N, dtype=bool)

    # проверка для каждой точки: входит ли её ячейка в valid_cells
    point_is_valid = valid_cells[flat_idx]

    # применяем только к валидным
    valid_indices = np.where(valid)[0]
    mask[valid_indices[point_is_valid]] = True

    print(f"Total points: {N}")
    print(f"Selected (trunk) points: {mask.sum()}")

    if mask.sum() == 0:
        raise ValueError("No trunk points found with given parameters")

    print("Saving result...")
    las[mask].write(output_las_path)
    print("Done.")


detect_trunk_points_2d(
    input_las_path="data/test_data/non_ground.las",
    output_las_path="data/test_data/trunks.las",
    cell_size=0.2,           # 20 см на ячейку
    min_points_in_cell=20*10**4     # порог "ствола"
)
