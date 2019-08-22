from face_point_cloud_landmark import PointCloudLandmark

if __name__ == "__main__":
    scan = PointCloudLandmark("Data/someone.ply")
    index, landmarks_3d_numpy = scan.get_landmark_byPLY()

    print(landmarks_3d_numpy)