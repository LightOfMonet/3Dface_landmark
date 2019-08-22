import dlib
import cv2
import numpy as np

def dlib_landmark(img_cv, is_show = 'false', is_save = 'false',
                  save_folder= "2D_landmark_img.png" ):
    '''
    Args:
        img_cv:  [  ,  ， 3] input image for landmark detecte
        is_show:  "false" or "true"   show for the 2D landmarks image
        is_save:  "false" or "true"
        save_folder: positive for save 2D image
    Returns:
        landmarks_2d: [68,2]  68 landmarks by dlib
    '''

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')

    img_rd = cv2.GaussianBlur(img_cv, ksize=(5, 5), sigmaX=0, sigmaY=0)
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)

    faces = detector(img_gray, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if len(faces) != 0:
        for i in range(len(faces)):
            landmarks_2d = np.matrix([[p.x, p.y] for p in predictor(img_rd, faces[i]).parts()])
            for idx, point in enumerate(landmarks_2d):
                pos = (point[0, 0], point[0, 1])
                cv2.circle(img_rd, pos, 4, color=(0, 0, 255), thickness=-1)
                cv2.putText(img_rd, str(idx + 1), pos, font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "faces: " + str(len(faces)), (20, 50), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        landmarks_2d = np.array(landmarks_2d)
    else:
        print("no face detected in this picture")
        exit(0)
    if is_show == "true":
        cv2.imshow("landmark_2d", img_rd)
        cv2.waitKey(0)
    if is_save == "true":
        cv2.imwrite(save_folder, img_rd)
    return landmarks_2d