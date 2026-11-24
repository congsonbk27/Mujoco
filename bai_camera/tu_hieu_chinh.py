"""Tự hiệu chỉnh động học (kinematic self-calibration) bằng camera.

Vấn đề: encoder mỗi khớp lệch điểm 0 một góc δ_i (chưa biết). Robot ra lệnh góc q nhưng khớp
thật đứng ở q + δ. Hậu quả: bàn tay vừa DỊCH vừa NGHIÊNG so với chỗ robot tưởng.

Cách làm:
  1. Cho tay đi qua K tư thế khác nhau (lệnh q_k).
  2. Ở mỗi tư thế, camera đo vị trí 3D của các marker trên mu bàn tay: y_k.
  3. Mô hình: marker ở FK(q_k + δ) (động học thuận của MuJoCo).
  4. Tìm δ làm nhỏ nhất  Σ_k || W (FK(q_k + δ) − y_k) ||²   bằng Gauss-Newton
     (Jacobian tính bằng sai phân hữu hạn trên một MjData riêng).
  5. Từ đó về sau: muốn khớp đứng ở q thì ra lệnh q − δ̂.

Vì cả mục tiêu (cốc) lẫn bàn tay đều được đo bằng CÙNG camera, δ̂ còn "hấp thụ" một phần sai số
hiệu chỉnh tay-mắt: tay và camera trở nên nhất quán với nhau, đó mới là điều cần cho việc xếp cốc.
"""

import mujoco
import numpy as np

from nhan_dang import hsv

TEN_MARKER = ["marker_trai", "marker_phai"]


def do_marker_3d(cam, color, depth, so_diem_toi_thieu=12):
    """Trả về list các tâm 3D (mặt trên) của từng cụm điểm hồng trong ảnh."""
    h, s, v = hsv(color)
    mask = (s > 0.35) & (v > 0.12) & (h > 280) & (h < 325) & (depth > 0)
    vv, uu = np.nonzero(mask)
    if len(uu) < so_diem_toi_thieu:
        return []
    p = cam.chieu_nguoc(uu, vv, depth[vv, uu])
    # gom cụm đơn giản: 2 marker cách nhau ~10 cm; điểm cách nhau > 3 cm (theo xy) là khác cụm
    cum = []
    for diem in p[np.argsort(p[:, 0])]:
        for c in cum:
            if np.linalg.norm(c[-1][:2] - diem[:2]) < 0.03:
                c.append(diem)
                break
        else:
            cum.append([diem])
    return [np.median(np.array(c), axis=0) for c in cum if len(c) >= so_diem_toi_thieu]


class TuHieuChinh:
    def __init__(self, robot, cam):
        self.robot, self.cam = robot, cam
        m = robot.m
        self.m = m
        self.d = mujoco.MjData(m)                       # MjData riêng để tính FK
        self.qadr = robot.ik.qadr
        self.gid = [m.geom(t).id for t in TEN_MARKER]
        self.mau = []                                   # (q_lenh, chỉ số marker, vị trí đo)

    def fk_marker(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        # tâm MẶT TRÊN của đĩa marker (camera nhìn thấy mặt trên): tâm geom lùi nửa bề dày theo
        # trục z của tay (z tay chúc xuống -> mặt trên nằm phía -z)
        ra = []
        for g in self.gid:
            R = self.d.geom_xmat[g].reshape(3, 3)
            ra.append(self.d.geom_xpos[g] - R[:, 2] * self.m.geom_size[g, 1])
        return ra

    def thu_mau(self, cac_dich):
        """Đưa tay tới từng điểm (TCP, tay chúc xuống), chụp ảnh, ghi lại marker đo được."""
        rb = self.robot
        for p in cac_dich:
            q = rb.giai_ik_moi(p)
            rb.toi_khop(q, T=1.0)
            rb.buoc(75)
            self.cam.cap_nhat_ngoai(rb.d)
            color, depth = self.cam.chup(rb.d)
            du_doan = self.fk_marker(rb.q)              # dự đoán với δ = 0
            for y in do_marker_3d(self.cam, color, depth):
                # gán cụm đo được cho marker dự đoán gần nhất (lệch vài cm << 10 cm giữa 2 marker)
                k = int(np.argmin([np.linalg.norm(y[:2] - dd[:2]) for dd in du_doan]))
                if np.linalg.norm(y[:2] - du_doan[k][:2]) < 0.04:
                    self.mau.append((rb.q.copy(), k, y))

    def giai(self, so_vong=15, w_z=0.3):
        """Gauss-Newton tìm δ. Trọng số trục z nhỏ hơn vì depth nhiễu hơn x, y."""
        W = np.array([1.0, 1.0, w_z])
        delta = np.zeros(len(self.qadr))
        eps = 1e-5

        def phan_du(dl):
            r = []
            for q, k, y in self.mau:
                r.append(W * (self.fk_marker(q + dl)[k] - y))
            return np.concatenate(r)

        for _ in range(so_vong):
            r = phan_du(delta)
            J = np.zeros((len(r), len(delta)))
            for j in range(len(delta)):
                dd = delta.copy()
                dd[j] += eps
                J[:, j] = (phan_du(dd) - r) / eps
            buoc = np.linalg.lstsq(J, -r, rcond=None)[0]
            delta += buoc
            if np.linalg.norm(buoc) < 1e-6:
                break
        rms = np.sqrt(np.mean(phan_du(delta).reshape(-1, 3) ** 2, axis=0)) / W
        return delta, rms


# Các điểm hiệu chỉnh: rải khắp vùng làm việc, 2 độ cao, để các khớp gập ở nhiều góc khác nhau
# (cùng một tư thế thì không tách được δ của vai / khuỷu / cổ tay).
DIEM_HIEU_CHINH = [np.array([r * np.cos(a), r * np.sin(a), z])
                   for z in (0.18, 0.34) for r in (0.3, 0.45) for a in (-1.0, 0.0, 1.3)]


def tu_hieu_chinh(robot, cam):
    """Chạy toàn bộ quy trình, gắn kết quả vào robot (robot.bu = δ̂). Trả về (δ̂, rms, số mẫu)."""
    hc = TuHieuChinh(robot, cam)
    hc.thu_mau(DIEM_HIEU_CHINH)
    if len(hc.mau) < 8:
        return None, None, len(hc.mau)
    delta, rms = hc.giai()
    robot.bu = delta
    return delta, rms, len(hc.mau)
