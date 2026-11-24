"""Camera RealSense D435 giả lập trong MuJoCo.

Ngoài đời, RealSense trả về 2 ảnh cùng kích thước, đã căn khớp (aligned) với nhau:
    color : ảnh màu  H x W x 3 (uint8)
    depth : ảnh độ sâu H x W (mét), mỗi điểm ảnh = khoảng cách dọc trục quang học
và bộ thông số nội (intrinsics) fx, fy, cx, cy để đổi điểm ảnh -> toạ độ 3D.

Class này cho ra đúng những thứ đó, kèm nhiễu giống cảm biến thật:
    - nhiễu Gauss tỉ lệ với bình phương khoảng cách (đặc trưng của camera stereo)
    - lượng tử hoá 1 mm (D435 xuất depth dạng uint16, đơn vị 1 mm)
    - một ít điểm ảnh "thủng" (depth = 0, không đo được), như ở mép vật ngoài đời
Nhờ vậy code xử lý ảnh viết ở đây có thể chuyển sang camera thật (pyrealsense2) gần như nguyên vẹn.
"""

import mujoco
import numpy as np


class CameraRealSense:
    # Thông số gần với D435: 640x480, góc nhìn dọc ~58 độ (MuJoCo dùng điểm ảnh vuông nên fx = fy)
    def __init__(self, model, ten_camera="realsense", rong=640, cao=480,
                 nhieu=0.002, ti_le_thung=0.01, seed=0):
        self.m = model
        self.cam = model.camera(ten_camera).id
        self.W, self.H = rong, cao
        self.nhieu = nhieu              # độ lệch chuẩn (m) của nhiễu depth ở khoảng cách 1 m
        self.ti_le_thung = ti_le_thung  # tỉ lệ điểm ảnh mất depth
        self.rng = np.random.default_rng(seed)
        self.r_mau = mujoco.Renderer(model, cao, rong)
        self.r_sau = mujoco.Renderer(model, cao, rong)
        self.r_sau.enable_depth_rendering()
        # Camera thật không nhìn thấy "site" (chấm đánh dấu TCP, vòng tròn đích...) -> tắt hiển thị site
        self.tuy_chon = mujoco.MjvOption()
        self.tuy_chon.sitegroup[:] = 0

        # ---- Thông số nội (intrinsics) ----
        fovy = np.radians(model.cam_fovy[self.cam])
        self.fy = (cao / 2) / np.tan(fovy / 2)
        self.fx = self.fy
        self.cx, self.cy = (rong - 1) / 2, (cao - 1) / 2

        # Sai số hiệu chỉnh tay-mắt: robot TƯỞNG camera đặt ở (R_lech·R_that, t_that + t_lech).
        # Mặc định = 0 (hiệu chỉnh hoàn hảo). Bài bai_do_ben dùng để thử độ bền.
        self.R_lech = np.eye(3)
        self.t_lech = np.zeros(3)
        # Hiệu chỉnh do robot TỰ ƯỚC LƯỢNG (tu_hieu_chinh.py): điểm đo y được sửa thành R_sua·y + t_sua
        self.R_sua = np.eye(3)
        self.t_sua = np.zeros(3)

    def chup(self, d):
        """Chụp 1 khung hình: trả về (color, depth)."""
        self.r_mau.update_scene(d, self.cam, self.tuy_chon)
        color = self.r_mau.render().copy()
        self.r_sau.update_scene(d, self.cam, self.tuy_chon)
        depth = self.r_sau.render().copy()

        # Nhiễu: sigma tăng theo z^2 (stereo), rồi lượng tử 1 mm, rồi đục lỗ ngẫu nhiên
        depth += self.rng.normal(0, 1, depth.shape) * self.nhieu * depth**2
        depth = np.round(depth * 1000) / 1000
        depth[self.rng.random(depth.shape) < self.ti_le_thung] = 0.0
        depth[depth > 3.0] = 0.0   # ngoài tầm đo
        return color, depth

    def chieu_nguoc(self, u, v, z):
        """Điểm ảnh (u = cột, v = hàng) + depth z -> điểm 3D trong hệ toạ độ THẾ GIỚI.

        Giống rs2_deproject_pixel_to_point của RealSense, cộng thêm bước đổi sang hệ thế giới.
        Hệ camera của MuJoCo: x sang phải, y hướng LÊN, camera nhìn theo -z
        (khác OpenCV/RealSense: y hướng xuống, nhìn theo +z) -> phải đổi dấu y và z.
        """
        u, v, z = np.asarray(u, float), np.asarray(v, float), np.asarray(z, float)
        p_cam = np.stack([(u - self.cx) * z / self.fx,
                          -(v - self.cy) * z / self.fy,
                          -z], axis=-1)
        return self._the_gioi(p_cam)

    def chieu_xuoi(self, p_world):
        """Điểm 3D thế giới -> điểm ảnh (u, v). Dùng để vẽ kết quả lên ảnh."""
        d = self._d
        R = d.cam_xmat[self.cam].reshape(3, 3)
        p = R.T @ (np.asarray(p_world) - d.cam_xpos[self.cam])
        z = -p[2]
        return self.cx + self.fx * p[0] / z, self.cy - self.fy * p[1] / z

    # Thông số ngoại (extrinsics): vị trí + hướng camera trong thế giới.
    # Ngoài đời phải đo bằng hiệu chỉnh tay-mắt (hand-eye calibration); trong mô phỏng thì đọc thẳng.
    def cap_nhat_ngoai(self, d):
        self._d = d

    def _the_gioi(self, p_cam):
        d = self._d
        R = self.R_lech @ d.cam_xmat[self.cam].reshape(3, 3)
        y = p_cam @ R.T + d.cam_xpos[self.cam] + self.t_lech
        return y @ self.R_sua.T + self.t_sua
