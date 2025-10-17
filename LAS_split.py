import laspy
import numpy as np
from tqdm import tqdm
from laspy.lasappender import LasAppender
import os


def split_las_streamed(input_las_file,
                       output_dir = "tiles",
                       tile_size = 20,
                       tile_overlap = 0,
                       las_read_chunk_size = 5_000_000):
    """
    Stream splits a file without loading it into memory
    TODO:
     - Добавить зазоры
     - Добавить многопоточность
     - Прибраться в комментариях
    """

    # По необходимости создаем директорию
    os.makedirs(output_dir, exist_ok=True)

    with laspy.open(input_las_file) as f:
        header = f.header
        total_points = header.point_count
        min_x, max_x = header.mins[0], header.maxs[0]
        min_y, max_y = header.mins[1], header.maxs[1]

        num_x_tiles = int(np.ceil((max_x - min_x) / tile_size))
        num_y_tiles = int(np.ceil((max_y - min_y) / tile_size))

        print(f"В файле {total_points} точек")
        print(f"Разбиваем на {num_x_tiles} × {num_y_tiles} тайлов")


        # Заголовок для будущих файлов
        tile_header = laspy.LasHeader(point_format = header.point_format, 
                                      version = header.version)
        tile_header.offsets = header.offsets
        tile_header.scales = header.scales
        
        # Словарь с путями к файлам
        tiles = {}
        for ix in range(num_x_tiles):
            for iy in range(num_y_tiles):
                tile_name = f"tile_{ix}_{iy}.las"
                tile_path = os.path.join(output_dir, tile_name)

                tiles[(ix, iy)] = tile_path

        # Потоковое чтение
        total_chunks = total_points // las_read_chunk_size 
        total_chunks += (1 if total_points % las_read_chunk_size else 0)

        for points in tqdm(f.chunk_iterator(las_read_chunk_size), 
                           total=total_chunks, 
                           desc="  Разбиение файла"):

            # для каждой точки определяем, в какой тайл она попадает
            x, y = points.x, points.y
            ix = np.floor((x - min_x) / tile_size).astype(int)
            iy = np.floor((y - min_y) / tile_size).astype(int)

            # # Собираем пары индексов
            # tile_indices = np.vstack((ix, iy)).T

            # # Уникальные отсортированые тайлы и индексы точек в них
            # unique_tiles, inverse = np.unique(tile_indices, axis=0, return_inverse=True)

            # # Раскидываем точки по файлам
            # for t_idx, tile_key in enumerate(unique_tiles):
            #     # Для кажой уникальной пары индексов(сектора) делаем маску на все точки из нее
            #     mask = inverse == t_idx
                
            base_indices = np.vstack((ix, iy)).T
            unique_tiles = np.unique(base_indices, axis=0)

            # Для каждого тайла проверяем попадание с учетом tile_overlap
            for tx, ty in unique_tiles:
                # Границы тайла
                x_min_tile = min_x + tx * tile_size - tile_overlap
                x_max_tile = min_x + (tx + 1) * tile_size + tile_overlap
                y_min_tile = min_y + ty * tile_size - tile_overlap
                y_max_tile = min_y + (ty + 1) * tile_size + tile_overlap

                # Маска попадания точек
                mask = (x >= x_min_tile) & (x < x_max_tile) & (y >= y_min_tile) & (y < y_max_tile)
                if not np.any(mask):
                    continue
                
                # Находим соответствующий путь к файлу
                # tile_path = tiles.get(tuple(tile_key))
                tile_path = tiles.get((tx, ty))
                if tile_path is None:
                    continue

                # Если файл еще не существует, то создаем
                if not os.path.isfile(tile_path):
                    laspy.LasData(tile_header).write(tile_path)

                # Фильтруем точки
                pts_chunk = points[mask]

                # Копируем в новый las объект всю онформацию о выбраных точках
                new_las = laspy.LasData(tile_header)
                new_las.points = pts_chunk

                # Дополняем соответствующий файл новыми точками
                with open(tile_path, "rb+") as dest:
                    with LasAppender(dest) as appender:
                        appender.append_points(new_las.points)

    print("Разбиение завершено.")


if __name__ == "__main__":
    input_las_file = "data/pine_forest.las"
    output_dir = "LAS_split output"
    tile_size = 40
    tile_overlap = 0
    las_read_chunk_size = 5_000_000

    split_las_streamed(input_las_file = input_las_file,
                       output_dir     = output_dir, 
                       las_read_chunk_size = las_read_chunk_size,
                       tile_size  = tile_size,
                       tile_overlap= tile_overlap)
