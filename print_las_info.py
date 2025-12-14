import laspy 
import numpy as np
import os, sys

def print_las_info(las_path):
    """
    Выводит основную информацию о LAS/LAZ файле:
    количество точек, границы (X, Y, Z), размер области и примерную плотность.
    x, y - 
    z - высота
    """
    with laspy.open(las_path) as f:
        header = f.header
        num_points = header.point_count
        min_x, max_x = header.mins[0], header.maxs[0]
        min_y, max_y = header.mins[1], header.maxs[1]
        min_z, max_z = header.mins[2], header.maxs[2]

        width = max_x - min_x
        lenght = max_y - min_y
        area = width * lenght if width > 0 and lenght > 0 else np.nan
        density = num_points / area if area and not np.isnan(area) else np.nan

        print(f"\nФайл: {os.path.basename(las_path)}")
        print(f"Формат версии: {header.version}")
        print(f"Количество точек: {num_points:,}")
        print(f"Диапазон координат (X, Y, Z):")
        print(f"  X: {min_x:.2f} - {max_x:.2f}  (ширина {width:.2f} м)")
        print(f"  Y: {min_y:.2f} - {max_y:.2f}  (размах {lenght:.2f} м)")
        print(f"  Z: {min_z:.2f} - {max_z:.2f}  (высота {max_z - min_z:.2f} м)")
        print(f"Площадь покрытия: {area:.2f} м²")

        print(f"Средняя плотность: {density:.2f} точек/м²\n")

        # Первые несколько точек
        print("Пример точек:")
        for points in f.chunk_iterator(3):
            for i in range(3):
                print(f"x: {points.x[i]:.2f} y: {points.y[i]:.2f} z: {points.z[i]:.2f}")
            break

if __name__ == "__main__":
    # las_file = "data/pine_forest.las"
    input_las_file = sys.argv[1]


    print_las_info(input_las_file)
