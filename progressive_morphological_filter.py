import numpy as np
import laspy
from scipy import ndimage
import os


def update_ground_mask(points, 
                       grid_opened, 
                       ground_mask, 
                       min_x, min_y, 
                       cell_size, 
                       current_threshold):

    # Размеры сетки
    rows, cols = grid_opened.shape
    # Загружаем точки
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Для кажой точки находим ее индекс в сетке
    col_idx = np.floor((x - min_x) / cell_size).astype(np.int32)
    row_idx = np.floor((y - min_y) / cell_size).astype(np.int32)

    # Маска точек, попадающих внутрь сетки
    valid = (row_idx >= 0) & (row_idx < rows) & (col_idx >= 0) & (col_idx < cols)
    row_idx, col_idx, z_valid = row_idx[valid], col_idx[valid], z[valid]

    # Извлекаем значения из открытой поверхности
    opened_vals = grid_opened[row_idx, col_idx]

    # Проверяем условие
    mask_valid = (z_valid - opened_vals) <= current_threshold

    # Обновляем исходную маску (дизъюнкция с перведущей)
    ground_mask[valid] |= mask_valid

    return ground_mask


def progressive_morphological_filter(points,
                                     cell_size=1.0,
                                     max_window_size=20, 
                                     initial_threshold=0.5, 
                                     slope=0.3):
    

    min_x, min_y = np.min(points[:, :2], axis=0)
    max_x, max_y = np.max(points[:, :2], axis=0)
    
    

    # find lovest points --------

    # Построение пространственной сетки
    cols = np.ceil((max_x - min_x) / cell_size).astype(int)
    rows = np.ceil((max_y - min_y) / cell_size).astype(int)
    grid = np.full((rows, cols), np.inf, dtype=np.float32)
    
    # Для кажой точки находим ее индекс в сетке
    col_idx = np.floor((points[:,0] - min_x) / cell_size).astype(np.int64)
    row_idx = np.floor((points[:,1] - min_y) / cell_size).astype(np.int64)

    # Проверка чтобы точки не вышли за сетку
    valid = (row_idx >= 0) & (row_idx < rows) & (col_idx >= 0) & (col_idx < cols)
    row_idx = row_idx[valid]
    col_idx = col_idx[valid]

    # отделяем z координаты в одномерный массив
    z_vals  = points[valid, 2].astype(np.float32)

    # Сводим друмерную сетку к одномерной
    flat_idx = row_idx * cols + col_idx
    flat_grid = grid.ravel()

    # Находим минимум среди всех z_vals с одним индексом
    # и помещаем в flat_grid
    np.minimum.at(flat_grid, flat_idx, z_vals)

    # возвращаем сетку в двумерный вид
    grid = flat_grid.reshape((rows, cols))

    # ------------

    # Пуская маска земли
    ground_mask = np.zeros(len(points), dtype=bool)

    # Возможные значаения окон
    window_sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
        
    for window_size in window_sizes:
        if window_size > max_window_size:
            break
            
        footprint = np.ones((window_size, window_size))
        eroded = ndimage.grey_erosion(grid, footprint=footprint)
        opened = ndimage.grey_dilation(eroded, footprint=footprint)
        current_threshold = initial_threshold + slope * (window_size / 2) * cell_size

        ground_mask = update_ground_mask(points, 
                                         opened, 
                                         ground_mask, 
                                         min_x, min_y,
                                         cell_size,
                                         current_threshold)
    
    return ground_mask


if __name__ == "__main__":
    input_dir = "/home/chazelam/Code/LiDAR/5+1_split_output"

    output_dir = "PMF_out"
    os.makedirs(output_dir, exist_ok=True)

    file_name = os.path.join(input_dir, "tile_4_4.las")

    las = laspy.read(file_name)
    points = np.vstack((las.x, las.y, las.z)).T
    print(f"file contain {len(points):,} points")


    cell_size=0.5
    max_window_size=15
    initial_threshold=0.3
    slope=0.2

    ground_mask = progressive_morphological_filter(points = points,
                                                   cell_size = cell_size,
                                                   max_window_size = max_window_size, 
                                                   initial_threshold = initial_threshold, 
                                                   slope = slope)


    non_ground_mask = ~ground_mask

    # non_ground_points = points[non_ground_mask]

    non_ground_path = os.path.join(output_dir, "non_ground.las")
    non_ground = las[non_ground_mask]
    non_ground.write(non_ground_path)

    print(f"Saved {non_ground.header.point_count:,} points to {non_ground_path}")


    ground_path = os.path.join(output_dir, "ground.las")
    ground = las[ground_mask]
    ground.write(ground_path)

    print(f"Saved {ground.header.point_count:,} points to {ground_path}")
