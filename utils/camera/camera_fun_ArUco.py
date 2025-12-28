import numpy as np
import time
import cv2
import cv2.aruco as aruco
import math
import scipy
from rs_func import get_camera


############################ Aruco ###########################

def init_aruco_detector(dict_type=aruco.DICT_4X4_50):
    """
    初始化 Aruco 偵測器
    dict_type 可以換成 DICT_6X6_50, DICT_7X7_1000 等
    """
    aruco_dict = aruco.getPredefinedDictionary(dict_type)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    return detector


def find_aruco(img, aruco_detector, camera_obj, draw=False):
    """
    偵測 Aruco 並回傳 Transformation Matrix
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco_detector.detectMarkers(gray)

    transformation_matrix = []
    if draw and ids is not None:
        for i, corner in enumerate(corners):
            pts = corner.reshape((4, 2)).astype(int)
            # # 畫四個角
            # for j, pt in enumerate(pts):
            #     cv2.circle(img, tuple(pt), 4, (255, 0, 0), 2)
            #     cv2.putText(img, str(j), tuple(pt),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            # # 畫中心
            # center = np.mean(pts, axis=0).astype(int)
            # cv2.circle(img, tuple(center), 4, (0, 0, 255), 2)

            # 從相機深度資料轉換到 3D 座標
            center3d = camera_obj.get_XYZ_point(int(center[1]), int(center[0]), camera_obj.get_Depth_image())
            corner0 = camera_obj.get_XYZ_point(int(pts[0][1]), int(pts[0][0]), camera_obj.get_Depth_image())
            corner1 = camera_obj.get_XYZ_point(int(pts[1][1]), int(pts[1][0]), camera_obj.get_Depth_image())
            corner2 = camera_obj.get_XYZ_point(int(pts[2][1]), int(pts[2][0]), camera_obj.get_Depth_image())

            x_axis = ((corner0 - center3d) + (corner1 - center3d))
            x_axis /= np.linalg.norm(x_axis)
            y_axis = ((corner1 - center3d) + (corner2 - center3d))
            y_axis /= np.linalg.norm(y_axis)
            z_axis = np.cross(x_axis, y_axis)
            z_axis /= np.linalg.norm(z_axis)

            transformation_matrix = np.eye(4)
            rotation_matrix = np.concatenate((x_axis.reshape(-1, 1),
                                              y_axis.reshape(-1, 1),
                                              z_axis.reshape(-1, 1)), axis=1)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = center3d.reshape(-1)

    flag = ids is not None and len(ids) > 0
    return flag, corners, img, transformation_matrix

def camera_xy_to_uv(cam_x, cam_y, camera_obj):
    """
    將相機正規化座標轉換為影像像素座標
    cam_x, cam_y: 相機座標系統的 x, y
    返回: u, v (影像像素座標)
    """
    u = cam_x * camera_obj.rs_para.RGB_fu + camera_obj.rs_para.RGB_Cu
    v = cam_y * camera_obj.rs_para.RGB_fv + camera_obj.rs_para.RGB_Cv
    return int(u), int(v)


def uv_to_camera_xy_aruco(img, aruco_detector, camera_obj, draw=False):
    """
    回傳 Aruco 四角點的 camera xy
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco_detector.detectMarkers(gray)
    transformation_matrix = []

    desired_pose_camera = [
        (-0.1543, -0.173),
        ( 0.1799, -0.1736),
        ( 0.174,  0.179),
        (-0.166,  0.167)
    ]

    # if draw and ids is not None:
    #     for i, corner in enumerate(corners):
    #         pts = corner.reshape((4, 2)).astype(int)
    #         # 四個角
    #         for j, pt in enumerate(pts):
    #             cv2.circle(img, tuple(pt), 4, (255, 0, 0), 2)
    #             cv2.putText(img, str(j), tuple(pt),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    #         # 正規化座標
    #         for k, pt in enumerate(pts):
    #             cam_x = (pt[0] - camera_obj.rs_para.RGB_Cu) / camera_obj.rs_para.RGB_fu
    #             cam_y = (pt[1] - camera_obj.rs_para.RGB_Cv) / camera_obj.rs_para.RGB_fv
    #             print(f"corner{k+1} → camera_x: {cam_x:.4f}, camera_y: {cam_y:.4f}")

    # if draw:
    #     for idx, (cam_x, cam_y) in enumerate(desired_pose_camera):
    #         u, v = camera_xy_to_uv(cam_x, cam_y, camera_obj)
    #         cv2.circle(img, (u, v), 6, (0, 0, 255), 2)  # 紅色實心圓
    #         cv2.putText(img, str(idx), (u+10, v+10),
    #                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    flag = ids is not None and len(ids) > 0
    return flag, corners, img, transformation_matrix


############################ Main ###########################

if __name__ == "__main__":
    camera_obj = get_camera()
    aruco_detector = init_aruco_detector()

    cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)

    while True:
        ts = time.time()
        camera_obj.set_image_frame()
        Depth_image = camera_obj.get_Depth_image()
        RGB_image = camera_obj.get_RGB_image()

        # detect_flag, corners, Drawed_img, pose = find_aruco(RGB_image, aruco_detector, camera_obj, draw=True)
        detect_flag, corners, Drawed_img, pose = uv_to_camera_xy_aruco(RGB_image, aruco_detector, camera_obj, draw=True)

        if detect_flag:
            center = np.mean(corners[0].reshape((4, 2)), axis=0).astype(int)
            camera_save_point = camera_obj.get_XYZ_point(int(center[1]), int(center[0]), Depth_image)
            print("camera_save_point : ", camera_save_point)
        else:
            print("No aruco detect")

        cv2.imshow("Image", Drawed_img)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):  # 按 's' 存圖片
            filename = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(filename, Drawed_img)
            print(f"已儲存圖片：{filename}")
