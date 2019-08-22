import numpy as np
import open3d as o3d

def load_ply(ply_path, img2d_size=640):
    ''' read PLY file, generation 3D point cloud
    Args:
        ply_path: the PLY file path
        img2d_size: rendered 2D image size
    Returns:
            n: number of vertices of PLY
        point_cloud_o3d:  o3d.PointCloud   class of PointCloud in open3d
        point_cloud_regular: [n,3]  According to the size of 2D image, the regularized point cloud
        point_cloud_colors: [n,3]  The color of the point cloud
        point_cloud_in_img:[n,3]  Point clouds in picture coordinates
    '''
    point_cloud_o3d = o3d.io.read_point_cloud(ply_path)
    # o3d.visualization.draw_geometries([point_cloud])
    point_cloud = np.asarray(point_cloud_o3d.points)
    point_cloud_xyz = point_cloud.copy()
    point_cloud_colors = np.asarray(point_cloud_o3d.colors)

    min_x = min(point_cloud_xyz[:, 0])
    min_y = min(point_cloud_xyz[:, 1])

    point_cloud_xyz[:, 0] = point_cloud_xyz[:, 0] - min_x
    point_cloud_xyz[:, 1] = point_cloud_xyz[:, 1] - min_y

    max_x = max(point_cloud_xyz[:, 0])
    max_y = max(point_cloud_xyz[:, 1])
    max_xy = max(max_x, max_y)

    point_cloud_xyz[:, 0] = np.rint(point_cloud_xyz[:, 0] * (img2d_size - 1) / max_xy)
    point_cloud_xyz[:, 1] = np.rint(point_cloud_xyz[:, 1] * (img2d_size - 1) / max_xy)

    point_cloud_regular = point_cloud_xyz.astype(int)

    point_cloud_in_img = point_cloud_regular.copy()
    point_cloud_in_img[:, 1] = img2d_size - 1 - point_cloud_in_img[:, 1]

    return point_cloud_o3d, point_cloud_regular, point_cloud_colors, \
           point_cloud_in_img