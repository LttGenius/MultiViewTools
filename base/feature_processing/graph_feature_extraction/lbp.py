import numpy as np
from cv2 import cvtColor
from cv2 import COLOR_BGR2GRAY
from math import floor, ceil


def local_binary_pattern(image, n_points, radius):
    """

    :param image: 图像
    :param n_points: 总点数
    :param radius: 圆角
    :return:
    """
    # 灰度图片转换
    gray_image = cvtColor(image, COLOR_BGR2GRAY)
    gray_image_matrix = np.array(gray_image, dtype=np.float32)
    dst = gray_image_matrix.copy()
    # 计算LBP值
    height = gray_image_matrix.shape[0]
    width = gray_image_matrix.shape[1]
    neighbours = np.zeros((1, n_points), dtype=np.uint8)
    lbp_value = np.zeros((1, n_points), dtype=np.uint8)
    for x in range(radius, width - radius - 1):
        for y in range(radius, height - radius - 1):
            lbp = 0.
            # 先计算共n_points个点对应的像素值，使用双线性插值法
            for n in range(n_points):
                theta = float(2 * np.pi * n) / n_points
                x_n = x + radius * np.cos(theta)
                y_n = y - radius * np.sin(theta)
                # 向下取整
                x1 = int(floor(x_n))
                y1 = int(floor(y_n))
                # 向上取整
                x2 = int(ceil(x_n))
                y2 = int(ceil(y_n))
                # 将坐标映射到0-1之间
                tx = np.abs(x - x1)
                ty = np.abs(y - y1)
                # 根据0-1之间的x，y的权重计算公式计算权重
                w1 = (1 - tx) * (1 - ty)
                w2 = tx * (1 - ty)
                w3 = (1 - tx) * ty
                w4 = tx * ty
                # 根据双线性插值公式计算第k个采样点的灰度值
                neighbour = gray_image_matrix[y1, x1] * w1 + gray_image_matrix[y2, x1] * w2\
                            + gray_image_matrix[y1, x2] * w3 + gray_image_matrix[y2, x2] * w4
                neighbours[0, n] = neighbour
            center = gray_image_matrix[y, x]
            for n in range(n_points):
                if neighbours[0, n] > center:
                    lbp_value[0, n] = 1
                else:
                    lbp_value[0, n] = 0
            for n in range(n_points):
                lbp += lbp_value[0, n] * 2 ** n
            # 转换到0-255的灰度空间，比如n_points=16位时结果会超出这个范围，对该结果归一化
            dst[y, x] = int(lbp / (2 ** n_points - 1) * 255)
    return dst
