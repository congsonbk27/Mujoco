"""Bài: Cánh tay robot gắp và đặt vật trong MuJoCo.

Chạy:
    python gap_do.py              # chạy headless, in kết quả kiểm tra
    python gap_do.py --xem        # mở cửa sổ viewer để xem trực tiếp
    python gap_do.py --gif        # xuất gap_do.gif (cần MUJOCO_GL=egl khi không có màn hình)
    python gap_do.py --ngau-nhien 7   # vật ở vị trí + góc xoay ngẫu nhiên (seed 7)

Ý tưởng chính:
  1. IK (inverse kinematics - động học ngược) bằng phương pháp Damped Least Squares:
     cho trước vị trí + hướng mong muốn của TCP -> tìm góc khớp.
  2. Lập kế hoạch quỹ đạo: chuỗi các điểm đích (waypoint), nội suy mượt giữa chúng.
  3. Máy trạng thái: tới trên vật -> hạ xuống -> kẹp -> nhấc -> di chuyển -> hạ -> thả -> rút lên.
  4. Servo vị trí trong MuJoCo bám theo góc khớp do IK tính ra. Vật được giữ bằng MA SÁT thật.
"""

import argparse
import pathlib
import time

import mujoco
import numpy as np

THU_MUC = pathlib.Path(__file__).parent
MODEL_XML = THU_MUC / "canh_tay.xml"

# 6 khớp của cánh tay tham gia IK (không tính 2 ngón kẹp).
# Bản 6DOF thêm "xoay_cang_tay": cùng co_tay, xoay_tay tạo thành cổ tay cầu.
KHOP_TAY = ["xoay_de", "vai", "khuyu", "xoay_cang_tay", "co_tay", "xoay_tay"]
NGON_MO = 0.04    # độ mở mỗi ngón (m)
NGON_DONG = 0.0   # lệnh đóng: ngón bị vật chặn lại, lực kẹp bị giới hạn bởi forcerange


# ---------------------------------------------------------------------------
# 1. Động học ngược (IK)
# ---------------------------------------------------------------------------
class GiaiIK:
    """IK Damped Least Squares cho site 'tcp'.

    Mục tiêu (6 ràng buộc):
      - vị trí TCP = p_dich                       (3)
      - trục z của bàn tay chỉ thẳng xuống        (2 hiệu dụng)
      - trục y của bàn tay (hướng ngón mở/đóng) = y_dich  (1 hiệu dụng)
    Dùng một MjData riêng để tính toán, không ảnh hưởng mô phỏng thật.
    """

    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        self.site = model.site("tcp").id
        self.dof = np.array([model.joint(j).dofadr[0] for j in KHOP_TAY])
        self.qadr = np.array([model.joint(j).qposadr[0] for j in KHOP_TAY])
        self.lo = np.array([model.joint(j).range[0] for j in KHOP_TAY])
        self.hi = np.array([model.joint(j).range[1] for j in KHOP_TAY])
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

    def giai(self, q0, p_dich, y_dich, so_vong=100, lam=0.05, tol=1e-4, z_dich=None):
        """Trả về (q, sai_so_vi_tri). q0 là nghiệm khởi đầu (warm start).
        z_dich: hướng trục z bàn tay; mặc định chúc thẳng xuống (gắp từ trên)."""
        q = q0.copy()
        z_dich = np.array([0.0, 0.0, -1.0]) if z_dich is None else np.asarray(z_dich, float)
        for _ in range(so_vong):
            self.d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)  # cần cho mj_jacSite

            p = self.d.site_xpos[self.site]
            R = self.d.site_xmat[self.site].reshape(3, 3)  # cột = trục x,y,z của TCP
            e_p = p_dich - p
            # Sai số hướng: a x b là trục quay cần để đưa vector a về b
            e_r = np.cross(R[:, 2], z_dich) + np.cross(R[:, 1], y_dich)
            e = np.concatenate([e_p, 0.5 * e_r])  # trọng số hướng nhỏ hơn vị trí
            if np.linalg.norm(e_p) < tol and np.linalg.norm(e_r) < 1e-3:
                break

            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.site)
            J = np.vstack([self.jacp[:, self.dof], 0.5 * self.jacr[:, self.dof]])
            # dq = J^T (J J^T + lam^2 I)^-1 e   -> ổn định cả gần điểm kỳ dị
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), e)
            q = np.clip(q + dq, self.lo, self.hi)
        return q, np.linalg.norm(e_p)


# ---------------------------------------------------------------------------
# 2. Kế hoạch quỹ đạo
# ---------------------------------------------------------------------------
def huong_kep(yaw_vat, y_hien_tai):
    """Chọn hướng trục y của bàn tay vuông góc với một mặt của khối vật.

    Khối lập phương đối xứng 90 độ nên có 4 lựa chọn; lấy hướng gần y hiện tại nhất
    để bàn tay xoay ít nhất.
    """
    ung_vien = [yaw_vat + k * np.pi / 2 for k in range(4)]
    vecs = [np.array([np.cos(a), np.sin(a), 0.0]) for a in ung_vien]
    return max(vecs, key=lambda v: v @ y_hien_tai)


def yaw_tu_quat(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def lap_ke_hoach(p_vat, yaw_vat, p_dat, y_hien_tai):
    """Danh sách các bước: (tên, vị trí TCP, hướng y, lệnh ngón, thời gian [s])."""
    cao = 0.15             # độ cao khi di chuyển
    # TCP (cách đầu ngón 2 cm) đặt cao hơn tâm vật 5 mm -> đầu ngón cách sàn ~5 mm, không cạ sàn
    z_kep = p_vat[2] + 0.005
    y_gap = huong_kep(yaw_vat, y_hien_tai)
    # khi đặt: giữ nguyên hướng bàn tay (vật xoay theo), chỉ cần vị trí đúng
    tren_vat = np.array([p_vat[0], p_vat[1], cao])
    tai_vat = np.array([p_vat[0], p_vat[1], z_kep])
    tren_dich = np.array([p_dat[0], p_dat[1], cao])
    tai_dich = np.array([p_dat[0], p_dat[1], z_kep + 0.003])
    return [
        ("toi_tren_vat", tren_vat,  y_gap, NGON_MO,   1.5),
        ("ha_xuong",     tai_vat,   y_gap, NGON_MO,   1.0),
        ("kep",          tai_vat,   y_gap, NGON_DONG, 0.6),
        ("nhac_len",     tren_vat,  y_gap, NGON_DONG, 1.0),
        ("di_chuyen",    tren_dich, y_gap, NGON_DONG, 2.0),
        ("ha_dat",       tai_dich,  y_gap, NGON_DONG, 1.0),
        ("tha",          tai_dich,  y_gap, NGON_MO,   0.6),
        ("rut_len",      tren_dich, y_gap, NGON_MO,   1.0),
    ]


def muot(s):
    """Nội suy mượt (smoothstep bậc 5): vận tốc và gia tốc = 0 ở 2 đầu."""
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def nlerp(a, b, s):
    v = (1 - s) * a + s * b
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# 3. Vòng lặp mô phỏng
# ---------------------------------------------------------------------------
def chay(xem=False, gif=False, seed=None):
    m = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("san_sang").id)
    if seed is not None:
        # Đặt vật ở vị trí + góc xoay ngẫu nhiên trong vùng với tới của tay (bán kính 0.3-0.5 m)
        rng = np.random.default_rng(seed)
        r, goc, yaw = rng.uniform(0.3, 0.5), rng.uniform(-1.2, 0.2), rng.uniform(-np.pi, np.pi)
        qa = m.joint("vat").qposadr[0]
        d.qpos[qa:qa + 3] = [r * np.cos(goc), r * np.sin(goc), 0.02]
        d.qpos[qa + 3:qa + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(m, d)

    ik = GiaiIK(m)
    act_tay = np.array([m.actuator(j).id for j in KHOP_TAY])
    act_ngon = m.actuator("kep").id
    vat = m.body("vat").id
    tcp = m.site("tcp").id

    # "Cảm biến": đọc vị trí + góc xoay của vật và vị trí đích
    p_vat = d.xpos[vat].copy()
    yaw_vat = yaw_tu_quat(d.xquat[vat])
    p_dat = d.site_xpos[m.site("dich").id].copy()
    print(f"Vật ở {np.round(p_vat, 3)}, xoay {np.degrees(yaw_vat):.1f} độ; đích {np.round(p_dat[:2], 3)}")

    y_ht = d.site_xmat[tcp].reshape(3, 3)[:, 1].copy()
    ke_hoach = lap_ke_hoach(p_vat, yaw_vat, p_dat, y_ht)

    q = d.qpos[ik.qadr].copy()
    p_truoc = d.site_xpos[tcp].copy()
    y_truoc = y_ht
    ngon_truoc = d.ctrl[act_ngon]
    buoc_dk = 10                 # cập nhật IK mỗi 10 bước mô phỏng (50 Hz)
    khung_hinh, ghi = [], {}

    viewer = None
    if xem:
        import mujoco.viewer as mjv
        viewer = mjv.launch_passive(m, d)
    renderer = mujoco.Renderer(m, 300, 400) if gif else None
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.2, 0.15, 0.12]
    cam.distance, cam.azimuth, cam.elevation = 1.2, -135, -25

    for ten, p_dich, y_dich, ngon, T in ke_hoach:
        n = int(T / m.opt.timestep)
        sai_max = 0.0
        for i in range(n):
            if i % buoc_dk == 0:
                s = muot((i + buoc_dk) / n)
                p_ref = (1 - s) * p_truoc + s * p_dich
                y_ref = nlerp(y_truoc, y_dich, s)
                q, sai = ik.giai(q, p_ref, y_ref)
                sai_max = max(sai_max, sai)
                d.ctrl[act_tay] = q
                d.ctrl[act_ngon] = (1 - s) * ngon_truoc + s * ngon
            mujoco.mj_step(m, d)
            if viewer is not None:
                if not viewer.is_running():
                    return
                viewer.sync()
                time.sleep(m.opt.timestep)
            if renderer is not None and i % 20 == 0:
                renderer.update_scene(d, cam)
                khung_hinh.append(renderer.render().copy())
        p_truoc, y_truoc, ngon_truoc = p_dich, y_dich, ngon
        ghi[ten] = d.xpos[vat].copy()
        tcp_sai = np.linalg.norm(d.site_xpos[tcp] - p_dich)
        print(f"  {ten:<13} xong | vật z={d.xpos[vat][2]:.3f} m | "
              f"TCP lệch đích {tcp_sai*1000:5.1f} mm | IK sai số max {sai_max*1000:.2f} mm")

    # Để vật ổn định 0.5 s rồi kiểm tra kết quả
    for _ in range(int(0.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    p_cuoi = d.xpos[vat]
    lech = np.linalg.norm(p_cuoi[:2] - p_dat[:2])
    nhac_ok = ghi["di_chuyen"][2] > 0.1
    dat_ok = lech < 0.02 and abs(p_cuoi[2] - 0.02) < 0.005
    print(f"\nKết quả: vật được nhấc lên {ghi['di_chuyen'][2]:.3f} m khi di chuyển -> "
          f"{'ĐẠT' if nhac_ok else 'KHÔNG ĐẠT'}")
    print(f"         vật cuối ở {np.round(p_cuoi, 3)}, lệch đích {lech*1000:.1f} mm -> "
          f"{'ĐẠT' if dat_ok else 'KHÔNG ĐẠT'}")

    if renderer is not None:
        from PIL import Image
        anh = [Image.fromarray(k) for k in khung_hinh]
        duong_dan = THU_MUC / "gap_do.gif"
        anh[0].save(duong_dan, save_all=True, append_images=anh[1:], duration=40, loop=0)
        print(f"Đã lưu {duong_dan}")
    if viewer is not None:
        viewer.close()
    return nhac_ok and dat_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xem", action="store_true", help="mở viewer tương tác")
    ap.add_argument("--gif", action="store_true", help="xuất gap_do.gif")
    ap.add_argument("--ngau-nhien", type=int, metavar="SEED", help="đặt vật ở vị trí/góc ngẫu nhiên")
    a = ap.parse_args()
    raise SystemExit(0 if chay(a.xem, a.gif, a.ngau_nhien) else 1)
