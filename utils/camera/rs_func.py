import pyrealsense2 as rs
import open3d as o3d
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


class rs_params:
    fps = 30
    res_u = 640
    res_v = 480
    background_removed_color = 153 # Grey
    depth_scale = 1.0

    RGB_Cu = None
    RGB_Cv = None
    RGB_fu = None
    RGB_fv = None

    Depth_Cu = None
    Depth_Cv = None
    Depth_fu = None
    Depth_fv = None

class Realsense_Camera():
    def __init__(self, serial_num, rs_para):
        self.device = serial_num
        self.rs_para = rs_para
        self.init_camera()

    def init_camera(self):
        self.pipeline = rs.pipeline()
        self.config   = rs.config()
        self.config.enable_device(self.device)
        self.stream_res_u = self.rs_para.res_u
        self.stream_res_v = self.rs_para.res_v
        self.stream_fps   = self.rs_para.fps

        self.config.enable_stream(rs.stream.depth, self.stream_res_u, self.stream_res_v, rs.format.z16, self.stream_fps)
        self.config.enable_stream(rs.stream.color, self.stream_res_u, self.stream_res_v, rs.format.bgr8, self.stream_fps)
        
        self.profile = self.pipeline.start(self.config)
        self.init_Depth_camera()
        self.init_RGB_camera()
        self.set_align()

    def init_Depth_camera(self):
        self.depth_profile = self.profile.get_stream(rs.stream.depth)
        self.depth_intr = self.depth_profile.as_video_stream_profile().get_intrinsics()
        self.rs_para.Depth_Cu = self.depth_intr.ppx
        self.rs_para.Depth_Cv = self.depth_intr.ppy
        self.rs_para.Depth_fu = self.depth_intr.fx
        self.rs_para.Depth_fv = self.depth_intr.fy

    def init_RGB_camera(self):
        self.color_profile = self.profile.get_stream(rs.stream.color)
        self.color_intr = self.color_profile.as_video_stream_profile().get_intrinsics()
        self.rs_para.RGB_Cu = self.color_intr.ppx
        self.rs_para.RGB_Cv = self.color_intr.ppy
        self.rs_para.RGB_fu = self.color_intr.fx
        self.rs_para.RGB_fv = self.color_intr.fy
    
    def set_align(self):
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def set_image_frame(self):
        camera_frames = self.pipeline.wait_for_frames()
        camera_aligned_frames = self.align.process(camera_frames)

        self.color_frame = camera_aligned_frames.get_color_frame()
        self.RGB_image = np.asanyarray(self.color_frame.get_data())

        self.aligned_depth_frame = camera_aligned_frames.get_depth_frame()
        self.Depth_image = np.asanyarray(self.aligned_depth_frame.get_data())
        # print("self.Depth_image : ",self.Depth_image.shape)
        
    def get_RGB_image(self):
        return self.RGB_image

    def get_Depth_image(self):
        return self.Depth_image
    
    def get_XYZ_point(self, cX, cY, Depth_image):
        depth_intrin = self.aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        _x, _y, _z = rs.rs2_deproject_pixel_to_point(depth_intrin, [cX, cY], Depth_image[cX, cY])
        return np.array([_x, -_y, _z])
    
    def colorized_Depth_image(self):
        normalized_depth = cv2.normalize(self.Depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = normalized_depth.astype(np.uint8)
        heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

        return heatmap
    
    def get_aligned_RGB_image(self):
        depth_image_3d = np.dstack((self.Depth_image, self.Depth_image, self.Depth_image))
        background_removed = np.where((depth_image_3d > 2000) | (depth_image_3d <= 0), self.rs_para.background_removed_color, self.RGB_image)

        return background_removed
    
    def get_point_cloud(self):
        pass

    def save_point_cloud(self):
        pcd = np.zeros((self.rs_para.res_v, self.rs_para.res_u, 6))
        for u in range(self.rs_para.res_u):
            for v in range(self.rs_para.res_v):
                pcd[v, u, 3:6] = self.get_XYZ_point(v, u)
        pcd[:, :, 0:3] = np.array(self.RGB_image)
        cv2.imwrite('rgb_pcd.jpg', self.RGB_image)
        np.savetxt('./pcd.txt', pcd.reshape(-1, 6))
        
def get_camera(device_num=0):
    realsense_ctx = rs.context()  #獲取相機資訊
    connected_devices = []
    for i in range(len(realsense_ctx.devices)):
        detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)  #相機編號
        print(f"{detected_camera}")
        connected_devices.append(detected_camera)

    camera_obj = Realsense_Camera(connected_devices[device_num], rs_params())
    return camera_obj

if __name__ == "__main__":

    camera_obj = get_camera()
    cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
    window_rect = cv2.getWindowImageRect("Image")
    print(f"視窗位置與大小: {window_rect}")
    print(f"深度Cx: {camera_obj.rs_para.Depth_Cu}")
    print(f"深度Cy: {camera_obj.rs_para.Depth_Cv}")
    print(f"深度fx: {camera_obj.rs_para.Depth_fu}")
    print(f"深度fy: {camera_obj.rs_para.Depth_fv}")
    print(f"RGB Cx: {camera_obj.rs_para.RGB_Cu}")
    print(f"RGB Cy: {camera_obj.rs_para.RGB_Cv}")
    print(f"RGB fx: {camera_obj.rs_para.RGB_fu}")
    print(f"RGB fy: {camera_obj.rs_para.RGB_fv}")
    while True:
        ts = time.time()
        camera_obj.set_image_frame()
        Depth_image = camera_obj.get_Depth_image()
        RGB_image = camera_obj.get_RGB_image()
        # print(np.max(Depth_image))
        # cv2.imshow("Image", RGB_image)
        # key = cv2.waitKey(1)

        Aligned_RGB_image = camera_obj.get_aligned_RGB_image()
        cv2.imshow("Image", RGB_image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        # print(f"FPS : {1/(time.time()-ts)}")