import laspy
import numpy as np
from scipy.ndimage import convolve, label


def detect_dense_vertical_structures(
    input_las_path: str,
    output_las_path: str,
    voxel_size: float = 0.05,          # размер вокселя (м)
    window_size: tuple[int, int, int] = (3, 3, 3),
    min_points_in_voxel: int = 10,     # плотность вокселя
    min_neighbors_3d: int = 5,         # минимум плотных соседей (3x3x3)
    min_height_voxels: int = 20,       # минимальная высота компоненты
    height_ratio: float = 2.0          # Z должен быть в X раз больше XY
):

    # Загрузка обака точек
    print("Loading LAS...")
    las = laspy.read(input_las_path)
    points = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    N = len(points)
    print(f"Points loaded: {N}")

    # Вокселизация
    mins = points.min(axis=0)
    idx = np.floor((points - mins) / voxel_size).astype(np.int32)

    print("Grouping points into voxels...")
    unique_voxels, counts = np.unique(idx,
                                      axis=0,
                                      return_counts=True)

    # Маска для вокселей по плотности
    dense_voxels = counts >= min_points_in_voxel
    print(f"Dense voxels: {dense_voxels.sum()} / {len(counts)}")

    if dense_voxels.sum() == 0:
        raise RuntimeError("No dense voxels found")

    # Минимальная сетка покрывающая все воксели
    max_idx = unique_voxels.max(axis=0) + 1
    grid = np.zeros(max_idx, dtype=bool)

    # Отбираем только плотные воксели
    coords_of_dense_voxels = unique_voxels[dense_voxels]

    # Для каждого плотного вокселя проставляем true в сетке
    grid[coords_of_dense_voxels[:, 0],
         coords_of_dense_voxels[:, 1],
         coords_of_dense_voxels[:, 2]] = True

    # Свертка grid с еденичной матрицей 3x3x3, 
    # получаем количество соседей для каждого вокселя в window_size.
    kernel = np.ones(window_size, dtype=np.int32)
    neighbor_count = convolve(grid.astype(np.int32), kernel, mode="constant")

    # Отбираем только достаточно плотные воклели у которых достаточно соседей
    grid_connected = grid & (neighbor_count >= min_neighbors_3d)
    print(f"Voxels after neighbor filter: {grid_connected.sum()}")

    if grid_connected.sum() == 0:
        raise RuntimeError("No connected dense voxels found")

    # Объединяем соседние воксели в кластеры, с номерами 1-n_labels
    # labels - той же формы что grid_connected, но у каждого вокскля его номер кластера
    labels, n_labels = label(grid_connected, structure=np.ones((3, 3, 3)))
    print(f"Connected components: {n_labels}")
    valid_labels = []

    # Идем по всем кластерам, 
    # индексы кластеров начинаются с 1 тк:
    # маркеровка 0 означет - не входит не в один кластер
    for lbl in range(1, n_labels + 1):
        # Координаты всех вокселей в этом кластере
        coords = np.argwhere(labels == lbl)
        if len(coords) == 0:
            continue

        # Размер кластера по каждому измерению (в вокселях)
        extent = coords.max(axis=0) - coords.min(axis=0)

        # вертикальная и горизонтальная протяжённость
        height = extent[2]
        width = max(extent[0], extent[1])

        # Фильтр по форме
        if height >= min_height_voxels and height >= height_ratio * width:
            valid_labels.append(lbl)

    print(f"Valid vertical components: {len(valid_labels)}")

    if not valid_labels:
        raise RuntimeError("No vertical structures detected")

    # Итоговая маска
    voxel_labels = labels[idx[:, 0],
                          idx[:, 1],
                          idx[:, 2]]

    mask = np.isin(voxel_labels, valid_labels)
    print(f"Remaining points: {mask.sum()}")

    # Сохранение файла
    print("Writing LAS...")
    las[mask].write(output_las_path)
    print("Done.")


if __name__ == "__main__":

    input_las_path = "/home/chazelam/Code/LiDAR/data/test_data/non_ground.las"
    # input_las_path = "/home/chazelam/Code/LiDAR/data/test_data/raw.las"
    output_las_dir = "/home/chazelam/Code/LiDAR/data/voxel_density_filter_out"

    voxel_size=0.05
    min_points_in_voxel=50
    
    window_size = (5, 5, 5)
    min_neighbors_3d=25
    min_height_voxels=20
    height_ratio=2.0

    output_las_path = f"{output_las_dir}/vs{voxel_size}x{min_points_in_voxel} - ws{window_size}x{min_neighbors_3d}.las"
    
    detect_dense_vertical_structures(
        input_las_path=input_las_path,
        output_las_path=output_las_path,
        voxel_size=voxel_size,
        min_points_in_voxel=min_points_in_voxel,

        window_size=window_size,
        min_neighbors_3d=min_neighbors_3d,
        min_height_voxels=min_height_voxels,
        height_ratio=height_ratio)
