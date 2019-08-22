import numpy as np
import open3d as o3d
import cv2

from . import load
from face_point_cloud_landmark.dlib import dlib_landmark

class PointCloudLandmark():

    def __init__(self,ply_path,img2d_size = 640, GaussianBlur_size=5):
        '''
        n: number of vertices of PLY
            point_cloud_o3d:  o3d.PointCloud   class of PointCloud in open3d
            point_cloud_regular: [n,3]  According to the size of 2D image, the regularized point cloud
            point_cloud_colors: [n,3]  The color of the point cloud
            point_cloud_in_img:[n,3]  Point clouds in picture coordinates
            img2d_size：the size of rendered 2D image
            GaussianBlur_size：the gaussian kernel is used to blur the image
        '''
        self.point_cloud_o3d, self.point_cloud_regular, \
        self.point_cloud_colors, self.point_cloud_in_img = load.load_ply(ply_path, img2d_size)

        self.img2d_size = img2d_size
        self.GaussianBlur_size = GaussianBlur_size


    def project(self, is_show = "false" ,is_save= "false",save_folder = "projection_img"):
        ''' Orthogonal projection of a 3D point cloud with texture onto a 2D image
        Args:
            is_show:  "false" or "true"   show for the 2D image
            is_save:  "false" or "true"
            save_folder: position for save 2D image
        Returns:
            img_projected: ndarray [ self.img2d_size, self.img2d_size, 3 ]
        '''
        image = np.zeros((self.img2d_size, self.img2d_size, 3),dtype=np.uint8)

        for i in range(self.point_cloud_regular.shape[0]):
            image[-self.point_cloud_regular[i, 1], self.point_cloud_regular[i, 0]] = 255 * self.point_cloud_colors[i]

        img_projected = cv2.merge([image[:, :, 2], image[:, :, 1], image[:, :, 0]])

        if is_show == "true":
            cv2.imshow("img_projected", img_projected)
            cv2.waitKey(0)
        if is_save == "true":
            cv2.imwrite(save_folder, img_projected)
        return img_projected

    def get_3d_point(self, landmarks_2d, is_show = "false"):
        ''' Calculate 3D landmarks
        Args:
            is_show:  "false" or "true"   show for the 3D landmarks
            landmarks_2d: [68,2]  68 landmarks
        Returns:
            index: [ 68 ]  landmarks indexes of point cloud
            landmarks_3d_numpy:[ 68 , 3 ] landmarks coordinates
        '''
        index = np.zeros((68), dtype=np.int)
        for i in range(68):
            distance = (self.point_cloud_in_img[:, 0] - landmarks_2d[i, 0]) ** 2 + \
                       (self.point_cloud_in_img[:, 1] - landmarks_2d[i, 1]) ** 2
            a = np.where(distance == np.min(distance))
            index[i] = np.max(a[0])

        if is_show == "true":
            print("the 3d landmark index is :" + str(index))

        point_cloud = np.asarray(self.point_cloud_o3d.points)
        landmarks_3d_numpy = point_cloud[index]

        if is_show == "true":
            print("the 3d landmark is :" + str(landmarks_3d_numpy))

        if is_show == "true":
            landmarks_3d = o3d.geometry.PointCloud()
            landmarks_3d.points = o3d.utility.Vector3dVector(landmarks_3d_numpy)
            o3d.visualization.draw_geometries([landmarks_3d])

            lines = np.zeros((16, 2), dtype=np.float)
            for i in range(16):
                lines[i] = [i, i + 1]

            line2 = np.array([[17, 18], [18, 19], [19, 20], [20, 21]])
            line3 = np.array([[22, 23], [23, 24], [24, 25], [25, 26]])
            line4 = np.array(
                [[37, 38], [38, 39], [39, 40], [40, 41], [36, 37], [36, 41]])
            line5 = np.array(
                [[42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [42, 47]])
            line6 = np.array(
                [[27, 28], [28, 29], [29, 30], [31, 32], [32, 33], [33, 34], [34, 35]])
            line7 = np.array(
                [[48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57],
                 [57, 58], [58, 59], [48, 59], [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67],
                 [67, 60]])

            lines = np.vstack((lines, line2, line3, line4, line5, line6, line7))

            colors = [[1, 0, 0] for i in range(len(lines))]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(landmarks_3d_numpy)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            # o3d.visualization.draw_geometries([line_set])

            o3d.visualization.draw_geometries([landmarks_3d])
            o3d.visualization.draw_geometries([landmarks_3d, self.point_cloud_o3d, line_set])

        return index,landmarks_3d_numpy

    def get_landmark_byPLY(self):
        '''  Calculate 3D landmarks directly
        Returns:
            index: [ 68 ]  landmarks indexes of point cloud
            landmarks_3d_numpy:[ 68 , 3 ] landmarks coordinates
        '''
        project_img = self.project()
        landmarks_2d = dlib_landmark(project_img)
        index, landmarks_3d_numpy = self.get_3d_point(landmarks_2d)

        return index, landmarks_3d_numpy
