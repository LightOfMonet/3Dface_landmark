from face_point_cloud_landmark import PointCloudLandmark
from face_point_cloud_landmark.dlib import dlib_landmark

if __name__ == "__main__":
    scan = PointCloudLandmark("Data/someone2.ply")
    project_img = scan.project()
    landmarks_2d = dlib_landmark(project_img, is_show ='true')
    index, landmarks_3d_numpy = scan.get_3d_point(landmarks_2d,is_show = "true")

