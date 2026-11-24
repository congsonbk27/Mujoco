"""Bài: Xếp kim tự tháp 3 cốc, tìm cốc bằng camera RealSense (giả lập) + điều khiển vòng kín.

Khác bài xep_coc (đọc thẳng vị trí cốc từ mô phỏng, chỉ đọc 1 lần):
  - Vị trí cốc lấy từ ẢNH màu + độ sâu có nhiễu, giống camera thật.
  - NHÌN LẠI trước mỗi lần gắp -> cốc bị dời đi vẫn gắp trúng.
  - KIỂM TRA sau khi kẹp: ngón đóng sát = kẹp hụt -> nhìn lại và gắp lại.
  - Trước khi đặt cốc trên cùng, đo vị trí THỰC của 2 cốc dưới và đặt vào giữa chúng.

Chạy:
    python xep_coc_camera.py                 # chạy bình thường
    python xep_coc_camera.py --pha           # có người "phá": dời cốc khi tay đang làm
    python xep_coc_camera.py --ngau-nhien 3  # rải cốc ngẫu nhiên
    python xep_coc_camera.py --gif           # xuất GIF + ảnh camera (MUJOCO_GL=egl nếu không có màn hình)
    python xep_coc_camera.py --xem           # mở viewer
"""

import argparse
import pathlib
import sys
import time

import mujoco
import numpy as np

THU_MUC = pathlib.Path(__file__).parent
sys.path.insert(0, str(THU_MUC.parent / "bai_gap_do"))
sys.path.insert(0, str(THU_MUC.parent / "bai_xep_coc"))
from gap_do import KHOP_TAY, GiaiIK, muot, nlerp  # noqa: E402
from xep_coc import (CAO, NGON_DONG, NGON_MO, TEN_COC, THA_CAO, Z_DI, Z_KEP,  # noqa: E402
                     vi_tri_kim_tu_thap)

from nhan_dang import tim_coc, tim_marker  # noqa: E402
from realsense import CameraRealSense  # noqa: E402
from tu_hieu_chinh import tu_hieu_chinh  # noqa: E402

XML_GOC = THU_MUC.parent / "bai_xep_coc" / "xep_coc.xml"
# Tư thế "đứng chờ" (góc khớp): tay dựng gần thẳng đứng -> nhìn từ camera trên cao, tay chỉ chiếm
# một vùng nhỏ quanh đế, không che cốc. Cho bằng GÓC KHỚP chứ không bằng vị trí TCP.
Q_CHO = np.array([0.0, -0.2, 0.3, 1.0, 0.0])
NGUONG_KEP_HUT = 0.01                    # ngón đóng < 1 cm -> giữa 2 ngón không có cốc
SO_LAN_THU = 3


def tao_model(nang_cap=False):
    """Lấy model của bài xep_coc và GẮN THÊM camera bằng MjSpec (không phải chép lại file XML).
    nang_cap=True: gắn thêm 2 marker màu hồng lên mu bàn tay để camera đo vị trí thật của tay."""
    spec = mujoco.MjSpec.from_file(str(XML_GOC))
    if nang_cap:
        tay = spec.body("ban_tay")
        for dau in (-1, 1):
            # "Tai" nhô ra 2 bên mu bàn tay (theo trục x của tay), ra ngoài bóng của khớp cổ tay
            # để camera trên cao luôn nhìn thấy. Chỉ để nhìn (không va chạm).
            tay.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.012, 0.004],
                         pos=[dau * 0.05, 0, 0.004], rgba=[0.4, 0.4, 0.45, 1], contype=0, conaffinity=0)
            tay.add_geom(name=f"marker{'_trai' if dau < 0 else '_phai'}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                         size=[0.011, 0.0015], pos=[dau * 0.052, 0, -0.0015],
                         rgba=[0.95, 0.1, 0.85, 1], contype=0, conaffinity=0)
    # Camera treo trên cao 1.25 m, nhìn thẳng xuống (camera MuJoCo mặc định nhìn theo -z).
    # Đặt lệch về phía tháp (y = 0.05) và đủ cao để khi tay cầm cốc ở trên tháp (cao ~0.43 m)
    # bàn tay vẫn nằm trong khung hình (cần cho bước visual servoing).
    # fovy 58 độ ~ góc nhìn dọc của RealSense D435.
    than = spec.worldbody.add_body(name="gia_camera", pos=[0.25, 0.05, 1.25])
    than.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.045, 0.0125, 0.0125],
                  rgba=[0.15, 0.15, 0.15, 1], contype=0, conaffinity=0)
    than.add_camera(name="realsense", pos=[0, 0, -0.015], fovy=58)
    return spec.compile()


class RoBot:
    """Gói phần điều khiển chuyển động (như bài trước) thành các hàm dùng lại được."""

    def __init__(self, m, d, viewer=None, renderer=None):
        self.m, self.d = m, d
        self.ik = GiaiIK(m)
        self.act_tay = np.array([m.actuator(j).id for j in KHOP_TAY])
        self.act_ngon = m.actuator("kep").id
        self.tcp = m.site("tcp").id
        self.q = d.qpos[self.ik.qadr].copy()
        self.p = d.site_xpos[self.tcp].copy()
        self.y = np.array([0.0, 1.0, 0.0])
        self.ngon = d.ctrl[self.act_ngon]
        self.viewer, self.renderer = viewer, renderer
        self.khung_hinh = []
        self.cam_xem = mujoco.MjvCamera()
        self.cam_xem.lookat[:] = [0.2, 0.05, 0.15]
        self.cam_xem.distance, self.cam_xem.azimuth, self.cam_xem.elevation = 1.45, -105, -28
        self.su_kien = []   # (thời điểm, hàm) - dùng để "phá" giữa chừng
        # Độ lệch điểm 0 encoder ước lượng được (tự hiệu chỉnh). Muốn khớp đứng ở q -> ra lệnh q - bu.
        self.bu = np.zeros(len(KHOP_TAY))

    def buoc(self, n):
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)
            for ev in list(self.su_kien):
                if self.d.time >= ev[0]:
                    ev[1]()
                    self.su_kien.remove(ev)
            if self.viewer is not None:
                if not self.viewer.is_running():
                    raise SystemExit(0)
                self.viewer.sync()
                time.sleep(self.m.opt.timestep)
            if self.renderer is not None and int(round(self.d.time / self.m.opt.timestep)) % 40 == 0:
                self.renderer.update_scene(self.d, self.cam_xem)
                self.khung_hinh.append(self.renderer.render().copy())

    def toi(self, p_dich, ngon=None, T=1.2, y=None):
        """Đưa TCP tới p_dich trong T giây (quỹ đạo smoothstep), đồng thời đổi độ mở ngón."""
        p_dich = np.asarray(p_dich, float)
        ngon = self.ngon if ngon is None else ngon
        y = self.y if y is None else y
        n, bdk = int(T / self.m.opt.timestep), 10
        for i in range(0, n, bdk):
            s = muot((i + bdk) / n)
            self.q, _ = self.ik.giai(self.q, (1 - s) * self.p + s * p_dich, nlerp(self.y, y, s))
            self.d.ctrl[self.act_tay] = self.q - self.bu
            self.d.ctrl[self.act_ngon] = (1 - s) * self.ngon + s * ngon
            self.buoc(bdk)
        self.p, self.y, self.ngon = np.array(p_dich, float), y, ngon

    def toi_khop(self, q_dich, ngon=None, T=1.5):
        """Di chuyển trong KHÔNG GIAN KHỚP: nội suy thẳng từ góc khớp hiện tại tới q_dich.
        Dùng cho các quãng đường dài (đổi hướng đế nhiều): tránh IK bám theo nghiệm cũ rồi
        rơi vào nhánh nghiệm xấu (tay vươn lật qua đầu, chạm giới hạn khớp)."""
        ngon = self.ngon if ngon is None else ngon
        q0 = self.q.copy()
        n, bdk = int(T / self.m.opt.timestep), 10
        for i in range(0, n, bdk):
            s = muot((i + bdk) / n)
            self.d.ctrl[self.act_tay] = (1 - s) * q0 + s * q_dich - self.bu
            self.d.ctrl[self.act_ngon] = (1 - s) * self.ngon + s * ngon
            self.buoc(bdk)
        self.q, self.ngon = np.array(q_dich, float), ngon
        # cập nhật vị trí/hướng TCP tương ứng q_dich (động học thuận trên MjData của IK)
        self.ik.d.qpos[self.ik.qadr] = self.q
        mujoco.mj_kinematics(self.m, self.ik.d)
        self.p = self.ik.d.site_xpos[self.tcp].copy()
        self.y = self.ik.d.site_xmat[self.tcp].reshape(3, 3)[:, 1].copy()

    def giai_ik_moi(self, p_dich):
        """Giải IK từ một điểm khởi đầu 'hợp lý': đế quay về phía đích, tay gập vừa phải."""
        q0 = np.array([np.arctan2(p_dich[1], p_dich[0]), 0.3, 1.0, 1.84, 0.0])
        q, sai = self.ik.giai(q0, np.asarray(p_dich, float), np.array([0.0, 1.0, 0.0]), so_vong=500)
        assert sai < 1e-3, f"IK không hội tụ tới {p_dich} (sai {sai:.4f} m)"
        return q

    def do_mo_ngon(self):
        return self.d.qpos[self.m.joint("ngon_trai").qposadr[0]]


def nhin(robot, cam, ghi_anh=None):
    """Lùi tay về chỗ chờ (không che camera), chụp ảnh và tìm cốc."""
    robot.toi_khop(Q_CHO, T=1.5)
    robot.buoc(50)                      # chờ 0.1 s cho tay hết rung
    cam.cap_nhat_ngoai(robot.d)
    color, depth = cam.chup(robot.d)
    thay = tim_coc(cam, color, depth)
    if ghi_anh is not None:
        ghi_anh.append((color, depth, thay))
    return thay


def gap(robot, p_coc, nang_cap=False, cam=None):
    """Gắp cốc có gốc ở p_coc. Trả về True nếu kẹp được (kiểm tra bằng độ mở ngón).
    nang_cap=True: kiểm tra thêm SAU KHI NHẤC (cốc có thể tuột khi nhấc, ví dụ do ma sát thấp)."""
    x, y, z = p_coc
    robot.toi_khop(robot.giai_ik_moi([x, y, Z_DI]), NGON_MO, T=1.5)
    if nang_cap:
        # Chỉnh tay đúng tâm cốc TRƯỚC khi hạ xuống: nếu tay lệch theo trục x (dọc mặt ngón),
        # 2 ngón không tự kéo cốc vào giữa -> cốc nằm lệch trong kẹp -> đặt cũng lệch theo.
        x, y = chinh_tay(robot, cam, [x, y], NGON_MO)
    robot.toi([x, y, z + Z_KEP], NGON_MO, T=1.2)
    robot.toi([x, y, z + Z_KEP], NGON_DONG, T=0.6)
    mo = robot.do_mo_ngon()
    if mo < NGUONG_KEP_HUT:
        print(f"    ! kẹp hụt (ngón chỉ mở {mo*1000:.1f} mm) -> mở kẹp, nhìn lại")
        robot.toi([x, y, z + Z_KEP], NGON_MO, T=0.5)
        robot.toi([x, y, Z_DI], NGON_MO, T=1.0)
        return False
    robot.toi([x, y, Z_DI], NGON_DONG, T=1.2)
    if nang_cap and con_coc(robot) is False:
        print("    ! cốc tuột khi nhấc lên -> mở kẹp, nhìn lại")
        robot.toi([x, y, Z_DI], NGON_MO, T=0.5)
        return False
    return True


def con_coc(robot):
    """Giữa 2 ngón còn cốc không? (ngón đang ra lệnh đóng mà mở > ngưỡng -> có vật chặn)"""
    return robot.do_mo_ngon() >= NGUONG_KEP_HUT


SAI_SO_CHO_PHEP = 0.0015   # chỉnh tay tới khi lệch < 1.5 mm
LECH_TOI_DA = 0.02         # đo thấy tay lệch > 2 cm sau tự hiệu chỉnh -> coi là đo sai, không chỉnh
SO_KHUNG = 3
SO_LAN_CHINH = 3


def do_vi_tri_tay(robot, cam):
    """Chụp ảnh, tìm 2 marker trên mu bàn tay -> vị trí xy THẬT của trục bàn tay (= xy của TCP
    khi tay chúc thẳng xuống). Trả về None nếu không thấy đủ 2 marker."""
    cam.cap_nhat_ngoai(robot.d)
    # Lọc theo thời gian: 3 khung hình (cách nhau 20 ms), lấy trung vị -> bớt nhiễu depth
    kq = []
    for _ in range(SO_KHUNG):
        color, depth = cam.chup(robot.d)
        xy = tim_marker(cam, color, depth, robot.p[:2])
        if xy is not None:
            kq.append(xy)
        robot.buoc(10)
    return np.median(kq, axis=0) if kq else None


def chinh_tay(robot, cam, xy_muon, ngon):
    """VISUAL SERVOING: đang lơ lửng ở độ cao Z_DI, dùng camera đo vị trí THẬT của bàn tay
    (2 marker) rồi dịch tay tới khi trục tay trùng xy_muon (lệch < 1.5 mm).
    Trả về xy cần RA LỆNH (khác xy_muon đúng bằng sai số động học/encoder tại chỗ đó)."""
    lenh = np.array(xy_muon[:2], float)
    for k in range(SO_LAN_CHINH):
        robot.buoc(50)                        # chờ tay hết rung rồi mới chụp
        xy_that = do_vi_tri_tay(robot, cam)
        if xy_that is None:
            print("    (không thấy marker, bỏ qua bước chỉnh)")
            break
        lech = np.array(xy_muon[:2]) - xy_that
        print(f"    chỉnh lần {k+1}: tay lệch {np.linalg.norm(lech)*1000:.1f} mm")
        if np.linalg.norm(lech) > LECH_TOI_DA:
            print("    (lệch quá lớn -> nghi đo sai, không chỉnh)")
            break
        if np.linalg.norm(lech) < SAI_SO_CHO_PHEP:
            break
        lenh = lenh + lech
        robot.toi([lenh[0], lenh[1], Z_DI], ngon, T=0.6)
    return lenh


def dat(robot, p_dat, cam=None, nang_cap=False):
    """Đặt cốc đang cầm xuống p_dat. Trả về False nếu phát hiện cốc đã rơi trên đường đi.

    nang_cap=True:
      - kiểm tra còn cốc trong kẹp khi tới trên đích
      - VISUAL SERVOING: camera đo vị trí thật của bàn tay (qua 2 marker), so với đích, dịch tay
        bù độ lệch; lặp tới khi lệch < 1.5 mm. Sai số encoder, sai số động học... đều bị bù.
        Vì đích (đo C1, C2) và tay đều đo bằng CÙNG một camera, sai số hiệu chỉnh tay-mắt
        cũng gần như triệt tiêu khi đặt C3 lên C1, C2.
    """
    x, y, z = p_dat
    robot.toi([x, y, Z_DI], NGON_DONG, T=2.0)
    lenh = np.array([x, y])                   # vị trí RA LỆNH (có thể khác vị trí muốn tới)
    if nang_cap:
        if not con_coc(robot):
            print("    ! cốc đã rơi trên đường chở -> quay lại tìm")
            robot.toi([x, y, Z_DI], NGON_MO, T=0.5)
            return False
        lenh = chinh_tay(robot, cam, [x, y], NGON_DONG)
    robot.toi([lenh[0], lenh[1], z + Z_KEP + THA_CAO], NGON_DONG, T=1.8)
    robot.toi([lenh[0], lenh[1], z + Z_KEP + THA_CAO], NGON_MO, T=0.6)
    robot.toi([lenh[0], lenh[1], Z_DI], NGON_MO, T=1.2)
    return True


def chay(xem=False, gif=False, seed=None, pha=False, tuy_chinh=None, nang_cap=False):
    """tuy_chinh(m, d, cam): hàm tuỳ chọn, gọi trước khi chạy để bơm sai số (xem bai_do_ben).
    nang_cap: bật phát hiện rơi cốc + visual servoing khi đặt (xem README, mục Nâng cấp)."""
    m = tao_model(nang_cap)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("san_sang").id)
    qadr = {c: m.joint(m.body(c).jntadr[0]).qposadr[0] for c in TEN_COC}
    if seed is not None:
        rng = np.random.default_rng(seed)
        da_dat = []
        while len(da_dat) < 3:
            r, goc = rng.uniform(0.28, 0.5), rng.uniform(-1.3, 0.1)
            p = np.array([r * np.cos(goc), r * np.sin(goc)])
            if all(np.linalg.norm(p - q) > 0.16 for q in da_dat):
                da_dat.append(p)
        for c, p in zip(TEN_COC, da_dat):
            d.qpos[qadr[c]:qadr[c] + 2] = p
    mujoco.mj_forward(m, d)

    viewer = None
    if xem:
        import mujoco.viewer as mjv
        viewer = mjv.launch_passive(m, d)
    robot = RoBot(m, d, viewer, mujoco.Renderer(m, 270, 360) if gif else None)
    cam = CameraRealSense(m, seed=0 if seed is None else seed)
    if tuy_chinh is not None:
        tuy_chinh(m, d, cam)
    anh = []
    if nang_cap:
        delta, (R_sua, b_sua), rms, n = (lambda k: k if k[0] is not None else (None, (None, None), None, k[3]))(
            tu_hieu_chinh(robot, cam))
        if delta is None:
            print(f"Tự hiệu chỉnh: chỉ thu được {n} mẫu marker, bỏ qua")
        else:
            goc = np.degrees(np.arccos(np.clip((np.trace(R_sua) - 1) / 2, -1, 1)))
            print(f"Tự hiệu chỉnh ({n} mẫu): lệch encoder = {np.round(np.degrees(delta), 2)} độ; "
                  f"sửa camera: dịch {np.round(b_sua * 1000, 1)} mm, xoay {goc:.2f} độ; "
                  f"sai số còn lại (rms x,y,z) = {np.round(rms * 1000, 1)} mm")

    def cho_trong(ten, goi_y):
        """Chọn chỗ đầu tiên trong goi_y cách mọi cốc khác >= 14 cm (không đặt cốc chồng lấn cốc)."""
        khac = [d.body(c).xpos[:2] for c in TEN_COC if c != ten]
        for xy in goi_y:
            if all(np.linalg.norm(np.asarray(xy) - k) > 0.14 for k in khac):
                return np.asarray(xy, float)
        return None

    def doi_coc(ten, xy, ly_do):
        if xy is None:
            print("    >>> NGƯỜI PHÁ: không còn chỗ trống, bỏ qua")
            return
        a = qadr[ten]
        d.qpos[a:a + 3] = [xy[0], xy[1], 0.0]
        d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        d.qvel[m.joint(m.body(ten).jntadr[0]).dofadr[0]:][:6] = 0
        print(f"    >>> NGƯỜI PHÁ: {ly_do}, dời {ten} tới {np.round(xy, 3)}")

    tam = d.site_xpos[m.site("dich").id][:2].copy()
    dich = vi_tri_kim_tu_thap(tam, np.array([1.0, 0.0, 0.0]))

    for i, ten in enumerate(TEN_COC):
        print(f"\n== Cốc {ten} ==")
        for lan in range(1, SO_LAN_THU + 1):
            thay = nhin(robot, cam, anh)
            if ten not in thay:
                print(f"    camera không thấy {ten}!")
                return False
            p_coc = thay[ten]["goc"]
            p_that = d.body(ten).xpos
            print(f"  lần {lan}: camera thấy {ten} ở {np.round(p_coc, 3)} "
                  f"(sai số so với thật {np.linalg.norm(p_coc - p_that)*1000:.1f} mm)")

            if pha and ten == "coc3" and lan == 1:
                # Phá kiểu 2: đúng lúc tay đang hạ xuống, kéo cốc đi 12 cm -> tay sẽ kẹp hụt
                goi_y = [p_that[:2] + 0.12 * np.array([np.cos(a), np.sin(a)])
                         for a in np.linspace(-np.pi, np.pi, 12, endpoint=False)
                         if 0.27 < np.linalg.norm(p_that[:2] + 0.12 * np.array([np.cos(a), np.sin(a)])) < 0.52]
                robot.su_kien.append((d.time + 2.0, lambda g=goi_y: doi_coc(
                    "coc3", cho_trong("coc3", g), "tay đang hạ xuống")))
            if not gap(robot, p_coc, nang_cap, cam):
                continue
            if pha and ten == "coc1" and lan == 1:
                # Phá kiểu 1: khi tay đang chở C1, có người dời C3 sang chỗ khác
                # mọi chỗ gợi ý đều cách đế 0.3-0.46 m: trong tầm với của tay ở độ cao di chuyển
                goi_y = [[0.22, -0.30], [0.25, -0.38], [0.42, -0.18], [0.35, 0.0], [0.12, -0.36]]
                robot.su_kien.append((d.time + 1.0, lambda: doi_coc(
                    "coc3", cho_trong("coc3", goi_y), "tay đang chở C1")))

            p_dat = dich[i]
            if ten == "coc3" and "coc1" in thay and "coc2" in thay:
                # Vòng kín: đặt C3 vào giữa vị trí THỰC của C1, C2 (đo trong ảnh vừa chụp trước khi
                # gắp C3) chứ không phải vị trí dự kiến.
                p_dat = (thay["coc1"]["goc"] + thay["coc2"]["goc"]) / 2 + np.array([0, 0, CAO])
                print(f"  camera đo C1, C2 -> đặt C3 tại {np.round(p_dat, 3)} (dự kiến {np.round(dich[2], 3)})")
            if dat(robot, p_dat, cam, nang_cap):
                break
        else:
            print("  Gắp thất bại quá số lần thử")
            return False

    robot.toi_khop(Q_CHO, T=1.5)
    print("\nChờ 2 s kiểm tra độ vững...")
    robot.buoc(int(2.0 / m.opt.timestep))

    # ---- Đánh giá: so với vị trí đúng tính từ vị trí THỰC của 2 cốc dưới ----
    p1, p2, p3 = (d.body(c).xpos.copy() for c in TEN_COC)
    ok = True
    print("\nKết quả:")
    for ten, p, p_mong in [("C1", p1, dich[0]), ("C2", p2, dich[1]),
                           ("C3", p3, (p1 + p2) / 2 + [0, 0, CAO])]:
        truc_z = d.body("coc" + ten[1]).xmat.reshape(3, 3)[:, 2]
        lech = np.linalg.norm(p - p_mong)
        nghieng = np.degrees(np.arccos(np.clip(truc_z[2], -1, 1)))
        dung = lech < 0.008 and nghieng < 3
        ok &= dung
        print(f"  {ten}: ở {np.round(p, 3)}, lệch {lech*1000:4.1f} mm, nghiêng {nghieng:4.1f} độ"
              f" -> {'ĐẠT' if dung else 'KHÔNG ĐẠT'}")
    print(f"Kim tự tháp: {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")

    if gif:
        from PIL import Image
        hinh = [Image.fromarray(k) for k in robot.khung_hinh]
        hinh[0].save(THU_MUC / "xep_coc_camera.gif", save_all=True, append_images=hinh[1:],
                     duration=50, loop=0)
        ve_anh_camera(cam, anh[0], THU_MUC / "camera_realsense.png")
        print(f"Đã lưu xep_coc_camera.gif, camera_realsense.png")
    if viewer is not None:
        viewer.close()
    return ok


def ve_anh_camera(cam, khung, duong_dan, phong=2):
    """Ghép ảnh color (đánh dấu cốc tìm được) và ảnh depth (tô màu) cạnh nhau,
    cắt quanh vùng có cốc và phóng to để dễ nhìn."""
    from PIL import Image, ImageDraw
    color, depth, thay = khung
    uv = np.array([t["uv"] for t in thay.values()])
    u0, v0 = np.maximum(uv.min(0) - 70, 0).astype(int)
    u1, v1 = np.minimum(uv.max(0) + 110, [cam.W, cam.H]).astype(int)

    # Depth: dải cố định 0.95-1.12 m (đỉnh cốc ~1.01 m, sàn ~1.1 m). Gần = sáng; không đo được = đen
    chuan = np.clip((1.12 - depth) / 0.17, 0, 1)
    sau = np.stack([255 * chuan, 255 * chuan**2, 160 * (1 - chuan)], -1).astype(np.uint8)
    sau[depth <= 0] = 0

    def cat(a):
        im = Image.fromarray(a[v0:v1, u0:u1])
        return im.resize((im.width * phong, im.height * phong), Image.NEAREST)

    mau, sau = cat(color), cat(sau)

    def chu(ve, xy, txt):
        ve.text(xy, txt, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))

    ve = ImageDraw.Draw(mau)
    for ten, t in thay.items():
        u, v = (t["uv"][0] - u0) * phong, (t["uv"][1] - v0) * phong
        ve.ellipse([u - 12, v - 12, u + 12, v + 12], outline=(0, 255, 0), width=2)
        ve.line([u - 18, v, u + 18, v], fill=(0, 255, 0), width=1)
        ve.line([u, v - 18, u, v + 18], fill=(0, 255, 0), width=1)
        g = t["goc"]
        chu(ve, (u + 16, v + 8), f"{ten}: x={g[0]:.3f} y={g[1]:.3f}")
    W, H = mau.size
    ghep = Image.new("RGB", (W * 2 + 6, H), (0, 0, 0))
    ghep.paste(mau, (0, 0))
    ghep.paste(sau, (W + 6, 0))
    ve = ImageDraw.Draw(ghep)
    chu(ve, (8, 6), "COLOR + nhan dang (tam mat day coc)")
    chu(ve, (W + 14, 6), "DEPTH (gan = sang, cham den = mat diem do)")
    ghep.save(duong_dan)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xem", action="store_true")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--pha", action="store_true", help="dời cốc khi tay đang làm")
    ap.add_argument("--ngau-nhien", type=int, metavar="SEED")
    ap.add_argument("--nang-cap", action="store_true", help="phát hiện rơi cốc + visual servoing khi đặt")
    a = ap.parse_args()
    raise SystemExit(0 if chay(a.xem, a.gif, a.ngau_nhien, a.pha, nang_cap=a.nang_cap) else 1)
