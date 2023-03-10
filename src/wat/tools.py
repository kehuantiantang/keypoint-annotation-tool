# coding=utf-8
# @Project  ：keypoint-annotation-tool 
# @FileName ：tools.py
# @Author   ：SoberReflection
# @Revision : sober 
# @Date     ：2023/1/27 11:32
import numpy as np

def gen_gaussian2d(shape, sigma=1):
    h, w = [_ // 2 for _ in shape]
    y, x = np.ogrid[-h : h + 1, -w : w + 1]
    gaussian = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0
    return gaussian


def draw_gaussian(density, center, radius, k=1, delte=6, overlap="add"):
    diameter = 2 * radius + 1
    gaussian = gen_gaussian2d((diameter, diameter), sigma=diameter / delte)
    gaussian = gaussian / gaussian.sum()
    height, width = density.shape[0:2]
    x, y = center.astype(np.int32)
    x = min(x, width - 1)
    y = min(y, height - 1)
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    left, right, top, bottom, radius = map(int, [left, right, top, bottom, radius])
    if overlap == "max":
        masked_density = density[y - top : y + bottom, x - left : x + right]
        masked_gaussian = gaussian[
            radius - top : radius + bottom, radius - left : radius + right
        ]
        np.maximum(masked_density, masked_gaussian * k, out=masked_density)
    elif overlap == "add":
        density[y - top : y + bottom, x - left : x + right] += gaussian[
            radius - top : radius + bottom, radius - left : radius + right
        ]
    else:
        raise NotImplementedError


def _min_dis_global(points):
    """
    points: m x 2, m x [x, y]
    """
    dis_min = float("inf")
    for point in points:
        point = point[None, :]  # 2 -> 1 x 2
        dis = np.sqrt(np.sum((points - point) ** 2, axis=1))  # m x 2 -> m
        dis = sorted(dis)[1]
        if dis_min > dis:
            dis_min = dis
    return dis_min


def _min_dis_local(point, points):
    """
    point: [x, y]
    points: m x 2, m x [x, y]
    """
    point = point[None, :]  # 2 -> 1 x 2
    dis = np.sqrt(np.sum((points - point) ** 2, axis=1))  # m x 2 -> m
    dis = sorted(dis)[1]
    return dis


def points2density(points, max_scale, max_radius, image_size):
    """
    points: m x 2, m x [x, y]
    """
    num_points = points.shape[0]
    density = np.zeros(image_size, dtype=np.float32)  # [h, w]
    if num_points == 0:
        return density
    elif num_points == 1:
        radius = max_radius
        for point in points:
            draw_gaussian(density, point, radius, overlap="add")
    else:
        dis_min = _min_dis_global(points)
        for point in points:
            # calculate the minimum distance of a point between its neighbors
            dis = _min_dis_local(point, points)
            dis = min(dis, max_scale * dis_min, max_radius)
            radius = max(3, int(dis))
            draw_gaussian(density, point, radius, overlap="add")
    return density