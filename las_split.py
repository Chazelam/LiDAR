import laspy
import numpy as np
from tqdm import tqdm
from laspy.lasappender import LasAppender
import os
import sys


def split_las_streamed(input_las_file,
                       output_dir = "tiles",
                       tile_size = 20,
                       tile_overlap = 0,
                       las_read_chunk_size = 5_000_000):
    """
    Stream splits a file without loading it into memory
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


def merge_tiles(input_dir, 
                output_las_file, 
                tile_overlap, 
                las_read_chunk_size = 5_000_000):
    """
    Merges all tiles in the input directory into a single LAS file.
    """
    # список фалов las в директории
    tile_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.las')]
    
    if not tile_files:
        print("No tile files found in the specified directory.")
        return

    for tile_file in tqdm(tile_files, total=len(tile_files)):

        with laspy.open(tile_file) as f:
            header = f.header
            total_points = header.point_count
            min_x, max_x = header.mins[0]+tile_overlap, header.maxs[0]-tile_overlap
            min_y, max_y = header.mins[1]+tile_overlap, header.maxs[1]-tile_overlap

            if not os.path.isfile(output_las_file):
                laspy.LasData(header).write(output_las_file)

            total_chunks = total_points // las_read_chunk_size 
            total_chunks += (1 if total_points % las_read_chunk_size else 0)

            # for points in tqdm(f.chunk_iterator(las_read_chunk_size), 
            #                total=total_chunks, 
            #                desc="  Слияние файлов"):

            for points in f.chunk_iterator(las_read_chunk_size):


                # # Загружаем точки из чанка
                # x, y = points.x, points.y

                # Отбрачаваем точки из запаса
                # mask = (x > min_x) & (x < max_x) & (y > min_y) & (y < max_y)
                # # x = x[x > min_x]
                # # x = x[x < max_x]
                # # y = y[y > min_y]
                # # y = y[y < max_y]

                # # Фильтруем точки
                pts_chunk = points#[mask]

                # Копируем в новый las объект всю онформацию о выбраных точках
                new_las = laspy.LasData(header)
                new_las.points = pts_chunk

                # Дополняем соответствующий файл новыми точками
                with open(output_las_file, "rb+") as dest:
                    with LasAppender(dest) as appender:
                        appender.append_points(new_las.points)


if __name__ == "__main__":
    # input_las_file = "data/pine_input_40x40.las"
    output_dir = "LAS_split_output"
    las_read_chunk_size = 5_000_000

    input_las_file = sys.argv[1]
    tile_size      = int(sys.argv[2])
    tile_overlap   = int(sys.argv[3])

    split_las_streamed(input_las_file = input_las_file,
                       output_dir     = output_dir, 
                       las_read_chunk_size = las_read_chunk_size,
                       tile_size  = tile_size,
                       tile_overlap= tile_overlap)

    # merge_tiles(input_dir=output_dir,
    #             output_las_file="out2.las",
    #             tile_overlap=tile_overlap,
    #             las_read_chunk_size=las_read_chunk_size)