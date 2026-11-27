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

import nhan_dang

LAM_CANG_TAY = 0.3
# Tay 6DOF: 8 điểm (không kém 12 điểm) với tốc độ cũ -> ~9.7 s thay vì ~14.5 s. Đi nhanh hơn (0.5 s, chờ 50%)
# thì 6DOF kém rõ ở bài độ bền (lệch encoder 4°: 77 -> 70%, 3°: 97 -> 87%) dù sai số ước lượng trông vẫn nhỏ.
# PiPER đặt riêng (TayPiper.T_DI_HIEU_CHINH...).
T_DI = 1.0          # giây di chuyển giữa 2 tư thế hiệu chỉnh
HE_SO_CHO = 1.0     # nhân với thời gian chờ tay đứng yên trước khi chụp
LAM_GAP_CO_TAY = 0.02   # gập cổ tay: đo được nhờ tư thế xoay tay 90° -> không cần ép mạnh
TEN_MARKER = ["marker_trai", "marker_phai"]


def do_marker_3d(cam, color, depth, so_diem_toi_thieu=12):
    """Trả về list các tâm 3D (mặt trên) của từng cụm điểm hồng trong ảnh."""
    mask = nhan_dang.mat_na(color)["marker"] & (depth > 0)
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
        self.do_phan_tan = []                           # độ lệch chuẩn z giữa các khung -> ước lượng nhiễu camera

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
        for i, p in enumerate(cac_dich):
            # Xen kẽ 2 hướng bàn tay. Hướng thường: 2 marker nằm dọc trục khớp gập cổ tay -> khớp này
            # lệch thì marker gần như không đổi (không đo được). Hướng xoay 90°: 2 marker nằm dọc cánh
            # tay, gập cổ tay lệch 4° làm 2 marker chênh cao ~7 mm -> đo được.
            y = (0.0, 1.0, 0.0) if i % 2 == 0 else (np.cos(np.arctan2(p[1], p[0])), np.sin(np.arctan2(p[1], p[0])), 0.0)
            q = rb.giai_ik_moi(p, y)
            rb.toi_khop(q, T=getattr(rb, "T_DI_HIEU_CHINH", T_DI))
            rb.buoc(int((75 + rb.CHO_THEM) * getattr(rb, "HE_SO_CHO_HIEU_CHINH", HE_SO_CHO)))
            self.cam.cap_nhat_ngoai(rb.d)
            du_doan = self.fk_marker(rb.q)              # dự đoán với δ = 0
            # 3 khung hình mỗi tư thế, gom lại rồi lấy trung vị theo từng marker (lọc nhiễu)
            theo_marker, doc = {}, []
            for _ in range(3):
                # Ghép số đo camera với góc ENCODER đọc lúc chụp (không phải góc lệnh): tay chưa đứng yên
                # hẳn vẫn đúng -> chờ ngắn được (PiPER: servo mềm, đứng yên hẳn mất ~1 s mỗi tư thế).
                doc.append(rb.doc_encoder())
                color, depth = self.cam.chup(rb.d)
                for y in do_marker_3d(self.cam, color, depth):
                    # gán cụm đo được cho marker dự đoán gần nhất (lệch vài cm << 10 cm giữa 2 marker)
                    k = int(np.argmin([np.linalg.norm(y[:2] - dd[:2]) for dd in du_doan]))
                    if np.linalg.norm(y[:2] - du_doan[k][:2]) < 0.04:
                        theo_marker.setdefault(k, []).append(y)
                rb.buoc(10)
            for k, ys in theo_marker.items():
                self.mau.append((np.mean(doc, axis=0), k, np.median(ys, axis=0)))
                if len(ys) >= 2:
                    self.do_phan_tan.append(np.std(np.array(ys)[:, 2]))

    def giai(self, so_vong=20, w_z=0.3, lam=0.02, lam_co_tay=0.3, lam_cang_tay=LAM_CANG_TAY):
        """Gauss-Newton tìm ĐỒNG THỜI:
             δ (5)  : lệch điểm 0 encoder các khớp
             r (2)  : sai số góc camera quanh x, y  (xoay quanh z trùng với δ của khớp đế -> bỏ)
             b (3)  : sai số vị trí camera
           Mô hình:  FK(q + δ) = Rot(r)·y + b     (y = vị trí marker camera đo được)
        Nếu chỉ ước lượng δ, mọi sai số của camera bị "đổ" hết cho encoder -> δ sai -> tay nghiêng.
        lam: regularization (Tikhonov) kéo tham số về 0 khi dữ liệu không đủ để xác định
        (đơn vị chuẩn hoá: 1 rad ~ 1 m ~ 1).
        lam_co_tay: regularization MẠNH hơn cho các khớp cổ tay (xoay_cang_tay, co_tay, xoay_tay). Marker gắn sát cổ
        tay (~4 cm từ trục khớp) nên cổ tay quay một chút marker gần như không đổi -> khó đo; nhưng
        TCP cách trục ~18 cm -> ước lượng sai một chút là tay nghiêng. Đo được: không có sai số nào
        mà vẫn ước lượng cổ tay lệch -1..-1.6 độ -> tay nghiêng, cốc đổ khi camera nhiễu."""
        W = np.array([1.0, 1.0, w_z])
        n_d = len(self.qadr)
        th = np.zeros(n_d + 5)
        eps = 1e-6
        lam_vec = np.full(len(th), lam)
        # các khớp cổ tay (xoay cẳng tay, gập cổ tay, xoay tay): 3 khớp cuối trong KHOP_TAY
        lam_vec[3:n_d] = lam_co_tay
        lam_vec[3] = lam_cang_tay
        # Gập cổ tay chỉ đo được qua chênh cao ~mm giữa 2 marker -> nhạy với nhiễu depth. Robot tự đo
        # nhiễu (độ phân tán z giữa 3 khung hình cùng tư thế): camera sạch (~0 mm) -> tin số đo, ép nhẹ;
        # camera nhiễu (>= 1 mm) -> ép mạnh như cũ, tránh "thấy" lệch giả làm tay nghiêng.
        nhieu_mm = 1000 * np.median(self.do_phan_tan) if self.do_phan_tan else 0.0
        lam_vec[4] = LAM_GAP_CO_TAY + (lam_co_tay - LAM_GAP_CO_TAY) * min(1.0, nhieu_mm / 1.0)

        def tach(t):
            return t[:n_d], ma_tran_xoay_nho(t[n_d], t[n_d + 1]), t[n_d + 2:]

        def phan_du(t):
            dl, R, b = tach(t)
            r = [W * (self.fk_marker(q + dl)[k] - (R @ y + b)) for q, k, y in self.mau]
            return np.concatenate(r + [lam_vec * t])

        for _ in range(so_vong):
            r = phan_du(th)
            J = np.zeros((len(r), len(th)))
            for j in range(len(th)):
                tt = th.copy()
                tt[j] += eps
                J[:, j] = (phan_du(tt) - r) / eps
            buoc = np.linalg.lstsq(J, -r, rcond=None)[0]
            th += buoc
            if np.linalg.norm(buoc) < 1e-7:
                break
        dl, R, b = tach(th)
        r = phan_du(th)[:-len(th)].reshape(-1, 3) / W   # bỏ các số hạng regularization
        return dl, R, b, np.sqrt(np.mean(r ** 2, axis=0))


def ma_tran_xoay_nho(rx, ry):
    """Ma trận xoay góc nhỏ quanh x rồi y."""
    cx, sx, cy, sy = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Ry @ Rx


# Các điểm hiệu chỉnh: rải khắp vùng làm việc, 2 độ cao, để các khớp gập ở nhiều góc khác nhau
# (cùng một tư thế thì không tách được δ của vai / khuỷu / cổ tay).
DIEM_HIEU_CHINH = [np.array([r * np.cos(a), r * np.sin(a), z])
                   for z in (0.18, 0.34) for r in (0.3, 0.45) for a in (-1.0, 0.0, 1.3)]
DIEM_HIEU_CHINH = DIEM_HIEU_CHINH[:4] + DIEM_HIEU_CHINH[8:]


def tu_hieu_chinh(robot, cam):
    """Chạy toàn bộ quy trình: gắn δ̂ vào robot (robot.bu) và phép sửa vào camera.
    Trả về (δ̂, (R, b), rms, số mẫu)."""
    hc = TuHieuChinh(robot, cam)
    hc.thu_mau(getattr(robot, "DIEM_HIEU_CHINH", DIEM_HIEU_CHINH))   # PiPER: tầm với khác -> điểm riêng
    if len(hc.mau) < 8:
        return None, None, None, len(hc.mau)
    delta, R, b, rms = hc.giai()
    # Lọc mẫu ngoại lai: marker bị che một phần (vd. cụm motor cổ tay PiPER) cho tâm đo lệch 5-6 mm, kéo
    # lời giải ra lệch giả (PiPER: vai +1° khi không có lỗi nào). Bỏ mẫu có phần dư xy > 3x trung vị rồi giải lại.
    for _ in range(2):
        r = np.array([np.linalg.norm((hc.fk_marker(q + delta)[k] - (R @ y + b))[:2]) for q, k, y in hc.mau])
        giu = r < max(3 * np.median(r), 0.002)
        if giu.all() or giu.sum() < 8:
            break
        hc.mau = [x for x, g in zip(hc.mau, giu) if g]
        delta, R, b, rms = hc.giai()
    robot.bu = delta
    # sửa camera: từ nay mọi điểm camera đo được (cốc, marker) đều qua phép biến đổi R·y + b
    cam.R_sua, cam.t_sua = R, b
    return delta, (R, b), rms, len(hc.mau)
