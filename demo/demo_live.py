"""Demo live cho khách: robot 6DOF xếp kim tự tháp 3 cốc bằng camera, khách được "phá".

Chạy:
    python demo/demo_live.py              # mở viewer, chạy liên tục
    python demo/demo_live.py --thu 50     # không cửa sổ: chạy 50 lần, tự bấm phím ngẫu nhiên, đếm lỗi
    python demo/demo_live.py --piper      # dùng cánh tay AgileX PiPER (cần model Menagerie, xem bai_piper)

Phím (bấm trong cửa sổ viewer):
    Space   tạm dừng / chạy tiếp
    R       rải lại cốc ngẫu nhiên, chạy lại từ đầu
    1 2 3   dời cốc C1 / C2 / C3 tới chỗ ngẫu nhiên (cốc chưa xếp, tay chưa cầm)
    T       làm đổ một cốc chưa xếp -> robot thấy cốc nằm, tự dựng lại (cần 6DOF)
    N       bật/tắt nhiễu camera mạnh (10 mm)
    E       bật/tắt lệch encoder 2° mọi khớp -> bản nâng cấp tự hiệu chỉnh lại
    C       chuyển bản nâng cấp <-> bản cũ (so sánh ngay trước mặt khách)

Các phím này trùng phím tắt hiển thị của viewer MuJoCo (T = trong suốt, C = điểm tiếp xúc, 1-3 = nhóm
geom...). Demo tự trả các cờ hiển thị về như cũ, nên bấm phím không làm cảnh bị đổi.
"""

import argparse
import pathlib
import sys
import time
import unicodedata

import mujoco
import numpy as np

GOC = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "bai_camera"))
sys.path.insert(0, str(GOC / "bai_coc_nam"))
import xep_coc_camera as X  # noqa: E402
from coc_nam import dat_coc_nam, ke_hoach_dung_coc  # noqa: E402
from gap_do import KHOP_TAY  # noqa: E402
from realsense import CameraRealSense  # noqa: E402
from tu_hieu_chinh import tu_hieu_chinh  # noqa: E402
from xep_coc import CAO, NGON_MO, TEN_COC, vi_tri_kim_tu_thap  # noqa: E402

NHIEU_THUONG, NHIEU_MANH = 0.002, 0.010
LECH_ENCODER = 2.0      # độ
SO_LAN_THU = 5          # mỗi cốc: số lần nhìn-gắp-đặt tối đa
PHIM = {32: "space", ord("R"): "R", ord("1"): "1", ord("2"): "2", ord("3"): "3",
        ord("T"): "T", ord("N"): "N", ord("E"): "E", ord("C"): "C"}


class ChayLai(Exception):
    pass


def khong_dau(s):
    """Font của viewer MuJoCo chỉ có ASCII -> bỏ dấu tiếng Việt."""
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


class BuocDemo:
    """Trộn vào lớp robot: mỗi bước mô phỏng xử lý phím / tạm dừng / vẽ lớp phủ."""

    def __init__(self, demo, *a, **k):
        super().__init__(*a, **k)
        self.demo = demo

    def buoc(self, n):
        for _ in range(n):
            self.demo.xu_ly()
            super().buoc(1)
            if self.demo.ghi is not None and int(round(self.d.time / self.m.opt.timestep)) % 20 == 0:
                self.demo.quay_khung()                 # 20 bước x 2 ms = 25 khung/giây thời gian mô phỏng


class RoBotDemo(BuocDemo, X.RoBot):
    # Tư thế chờ khi chụp: quay đế ra phía sau (không có cốc). Tư thế của bài camera (đế = 0) làm cánh
    # tay che dải y ~ 0 -> chỉ thấy nửa cốc nằm, nhận nhầm là cốc đứng (đo: bỏ sót 5/40 -> 1/40).
    Q_CHO = np.r_[2.6, X.Q_CHO[1:]]


class Demo:
    def __init__(self, xem=True, seed=0, quay=None, piper=False):
        self.piper = piper
        if piper:
            # PiPER (bai_piper): cần model Menagerie; camera trên cao điều khiển, như xep_coc_piper_camera.py
            sys.path.insert(0, str(GOC / "bai_piper"))
            import xep_coc_piper_camera as C
            self.C = C
            self.m = C.tao_model()
        else:
            self.m = X.tao_model(nang_cap=True)   # luôn gắn marker; bản cũ chỉ không dùng tới
        self.lech_enc_piper = np.zeros(6)
        self.ghi, self.anh_cuoi, self.kich_ban, self.tu_bam = None, None, [], False
        if quay:
            import imageio
            from PIL import ImageFont
            self.m.vis.global_.offwidth, self.m.vis.global_.offheight = 1280, 720
            self.rd = mujoco.Renderer(self.m, 720, 1280)
            self.cam_quay = mujoco.MjvCamera()
            self.cam_quay.lookat[:] = [0.22, 0.0, 0.25]      # nhìn chéo từ trước-trái: thấy cốc, tay, chỗ xếp tháp
            self.cam_quay.distance, self.cam_quay.azimuth, self.cam_quay.elevation = 1.55, -150, -30
            self.ghi = imageio.get_writer(quay, fps=25, quality=7, macro_block_size=8)
            self.font = self.font_nho = None
            for ten in ("arial.ttf", "DejaVuSans.ttf"):   # font có dấu tiếng Việt (Windows / Ubuntu)
                try:
                    self.font, self.font_nho = ImageFont.truetype(ten, 26), ImageFont.truetype(ten, 20)
                    break
                except OSError:
                    pass
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.cam = CameraRealSense(self.m, seed=seed)
        self.lenh = []                            # phím chờ xử lý (viewer gọi từ luồng khác)
        self.tam_dung = False
        self.nang_cap, self.nhieu_manh, self.lech_enc = True, False, False
        self.can_hieu_chinh = True
        self.bu = np.zeros(len(KHOP_TAY))
        self.so_lan = self.so_dat = 0
        self.trang_thai = ""
        self.da_xep, self.dang_cam = set(), None
        self.dem = 0
        self.viewer = None
        if xem:
            import mujoco.viewer as mjv
            self.viewer = mjv.launch_passive(self.m, self.d, key_callback=self.phim)
            with self.viewer.lock():
                self.co0 = self.viewer.opt.flags.copy()
                self.nhom0 = self.viewer.opt.geomgroup.copy()
            self.viewer.cam.lookat[:] = [0.2, 0.05, 0.12]
            self.viewer.cam.distance, self.viewer.cam.azimuth, self.viewer.cam.elevation = 1.5, -105, -30

    # ---------------- phím ----------------
    def phim(self, ma):
        if ma in PHIM:
            self.lenh.append(PHIM[ma])

    def xu_ly(self):
        while self.lenh:
            self.lam(self.lenh.pop(0))
        while self.tam_dung and self.viewer is not None and self.viewer.is_running():
            while self.lenh:
                self.lam(self.lenh.pop(0))
            self.ve()
            self.viewer.sync()
            time.sleep(0.02)
        self.dem += 1
        if self.dem % 25 == 0:
            self.ve()

    def lam(self, k):
        m, d = self.m, self.d
        if k == "space":
            self.tam_dung = not self.tam_dung
            print("TẠM DỪNG" if self.tam_dung else "Chạy tiếp")
        elif k == "R":
            self.tam_dung = False
            raise ChayLai()
        elif k in "123":
            ten = TEN_COC[int(k) - 1]
            if ten in self.da_xep or ten == self.dang_cam:
                print(f">>> Khách dời {ten}: cốc đã xếp / đang cầm -> bỏ qua")
                return
            xy = self.cho_trong(ten)
            if xy is None:
                return
            a = m.joint(m.body(ten).jntadr[0]).qposadr[0]
            d.qpos[a:a + 7] = [xy[0], xy[1], 0.0, 1, 0, 0, 0]
            d.qvel[m.joint(m.body(ten).jntadr[0]).dofadr[0]:][:6] = 0
            print(f">>> Khách dời {ten} tới {np.round(xy, 2)}")
        elif k == "T":
            ung_vien = [c for c in TEN_COC if c not in self.da_xep and c != self.dang_cam]
            if not ung_vien:
                print(">>> Không còn cốc nào để làm đổ")
                return
            ten = ung_vien[int(self.rng.integers(len(ung_vien)))]
            dat_coc_nam(m, d, ten, d.body(ten).xpos[:2].copy(), self.rng.uniform(-np.pi, np.pi))
            d.qvel[m.joint(m.body(ten).jntadr[0]).dofadr[0]:][:6] = 0
            print(f">>> Khách làm đổ {ten}")
        elif k == "N":
            self.nhieu_manh = not self.nhieu_manh
            self.cam.nhieu = NHIEU_MANH if self.nhieu_manh else NHIEU_THUONG
            print(f">>> Nhiễu camera: {self.cam.nhieu * 1000:.0f} mm")
        elif k == "E":
            self.lech_enc = not self.lech_enc
            dau = self.rng.choice([-1, 1], size=len(KHOP_TAY))
            if self.piper:
                # PiPER có khâu tích phân bám ENCODER -> lệch encoder mô phỏng bằng sai số đọc encoder
                # (biasprm như tay tự dựng sẽ bị khâu tích phân bù mất)
                self.lech_enc_piper = np.radians(LECH_ENCODER) * dau * self.lech_enc
                self.robot.lech_enc = self.lech_enc_piper.copy()
            for i, j in enumerate(KHOP_TAY if not self.piper else []):
                a = m.actuator(j).id
                m.actuator_biasprm[a, 0] = (m.actuator_gainprm[a, 0] * np.radians(LECH_ENCODER) * dau[i]
                                            if self.lech_enc else 0.0)
            self.can_hieu_chinh = True
            print(f">>> Encoder: {'LỆCH 2 độ' if self.lech_enc else 'chuẩn'}")
        elif k == "C":
            self.nang_cap = not self.nang_cap
            if self.nang_cap:
                self.can_hieu_chinh = True
            else:
                self.bu[:] = 0
                self.robot.bu[:] = 0
                self.cam.R_sua, self.cam.t_sua = np.eye(3), np.zeros(3)
            print(f">>> Chuyển sang bản {'NÂNG CẤP' if self.nang_cap else 'CŨ'}")

    def cho_trong(self, ten):
        """Chỗ ngẫu nhiên trong tầm với, cách cốc khác và khu xếp tháp >= 16 cm."""
        khac = [self.d.body(c).xpos[:2] for c in TEN_COC if c != ten] + [self.tam]
        r0, r1, kc = (0.22, 0.34, 0.15) if self.piper else (0.3, 0.46, 0.16)   # PiPER tầm với ngắn hơn
        for _ in range(200):
            r, g = self.rng.uniform(r0, r1), self.rng.uniform(-1.2, 0.0)
            xy = np.array([r * np.cos(g), r * np.sin(g)])
            if all(np.linalg.norm(xy - q) > kc for q in khac):
                return xy
        return None

    # ---------------- lớp phủ ----------------
    def ve(self, anh=None):
        v = self.viewer
        if v is None:
            return
        with v.lock():
            v.opt.flags[:] = self.co0                  # chặn phím tắt hiển thị của viewer
            v.opt.geomgroup[:] = self.nhom0
        trai = "Trang thai\nBan\nCamera\nEncoder\nKet qua"
        phai = "\n".join([
            khong_dau(self.trang_thai)[:70] + ("   [TAM DUNG]" if self.tam_dung else ""),
            "NANG CAP (tu hieu chinh + chinh tay bang camera)" if self.nang_cap else "CU (khong hieu chinh)",
            "nhieu manh 10 mm" if self.nhieu_manh else "binh thuong",
            "LECH 2 do" if self.lech_enc else "chuan",
            f"{self.so_dat}/{self.so_lan} lan dat"])
        phim = "Space dung | R chay lai | 1 2 3 doi coc | T lam do coc | N nhieu | E encoder | C cu/moi"
        v.set_texts([(None, mujoco.mjtGridPos.mjGRID_TOPLEFT, trai, phai),
                     (None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, phim, None)])
        if anh is not None and v.viewport is not None:
            h, w = anh.shape[:2]
            v.set_images((mujoco.MjrRect(v.viewport.width - w - 10, 40, w, h), anh))

    def anh_camera(self, color, thay):
        """Ảnh camera thu nhỏ 1/2, đánh dấu cốc tìm được (xanh = đứng, đỏ = nằm)."""
        from PIL import Image, ImageDraw
        im = Image.fromarray(color[::2, ::2].copy())
        ve = ImageDraw.Draw(im)
        for ten, t in thay.items():
            u, v = t["uv"][0] / 2, t["uv"][1] / 2
            mau = (255, 40, 40) if t.get("nam") else (0, 255, 0)
            ve.ellipse([u - 9, v - 9, u + 9, v + 9], outline=mau, width=2)
            ve.text((u + 11, v - 6), ten.upper() + (" NAM" if t.get("nam") else ""), fill=mau,
                    stroke_width=2, stroke_fill=(0, 0, 0))
        ve.text((6, 4), "REALSENSE", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        return np.asarray(im)

    # ---------------- một lần xếp tháp ----------------
    def nhin(self):
        ghi = []
        thay = X.nhin(self.robot, self.cam, ghi)
        if self.viewer is not None or self.ghi is not None:
            self.anh_cuoi = self.anh_camera(ghi[-1][0], thay)
            self.ve(self.anh_cuoi)
        return thay

    def quay_khung(self):
        """Một khung video: cảnh 3D + dòng trạng thái (có dấu) + ô camera."""
        from PIL import Image, ImageDraw
        self.rd.update_scene(self.d, self.cam_quay)
        im = Image.fromarray(self.rd.render())
        if self.anh_cuoi is not None:
            o = Image.fromarray(self.anh_cuoi)
            im.paste(o, (1280 - o.width - 16, 720 - o.height - 16))
        ve = ImageDraw.Draw(im)
        f, fn = self.font, self.font_nho
        chu = (khong_dau if f is None else str)
        dong = [chu(self.trang_thai)[:80],
                "Bản: " + ("NÂNG CẤP (tự hiệu chỉnh + chỉnh tay bằng camera)" if self.nang_cap else "CŨ"),
                "Camera: " + ("nhiễu mạnh 10 mm" if self.nhieu_manh else "bình thường")
                + "    Encoder: " + ("LỆCH 2°" if self.lech_enc else "chuẩn"),
                f"Kết quả: {self.so_dat}/{self.so_lan} lần đạt"]
        ve.rectangle([0, 0, 1280, 132], fill=(0, 0, 0))
        for i, s in enumerate(dong):
            ve.text((16, 8 + i * 30), chu(s), font=f if i == 0 else fn, fill=(255, 255, 255) if i else (255, 220, 80))
        self.ghi.append_data(np.asarray(im))

    def dung_coc(self, ten, t):
        print(f"Thấy {ten} NẰM ĐỔ -> dựng lại (xoay bàn tay 90 độ)")
        rb = self.robot
        if self.piper:                                 # PiPER: nghiêng ±45° (cổ tay chỉ ±70°), xem bai_piper
            self.C.dung_coc(rb, t["mieng"], t["truc"])
            return
        buoc = ke_hoach_dung_coc(t["mieng"], t["truc"])
        _, p, y, _, n, T = buoc[0]
        rb.toi_khop(rb.giai_ik_moi(p, y), n, T=1.5)
        for _, p, y, z, n, T in buoc[1:]:
            rb.toi(p, n, T, y=y, z=z)

    def mot_lan(self, seed):
        m, d = self.m, self.d
        rng = np.random.default_rng(seed)
        if self.piper:                                 # model PiPER không có keyframe: đặt tay ở tư thế chờ
            C = self.C
            mujoco.mj_resetData(m, d)
            d.qpos[[m.joint(j).qposadr[0] for j in C.P.KHOP]] = C.TayPiper.Q_CHO
            d.ctrl[[m.actuator(j).id for j in C.P.KHOP]] = C.TayPiper.Q_CHO
            vung = (0.22, 0.34, -1.2, 0.2, 0.15)
        else:
            mujoco.mj_resetDataKeyframe(m, d, m.key("san_sang").id)
            vung = (0.28, 0.5, -1.3, 0.1, 0.16)
        r0, r1, g0, g1, kc = vung
        da_dat, lan = [], 0
        while len(da_dat) < 3:
            lan += 1
            if lan % 200 == 0:                         # vùng nhỏ (PiPER): 2 cốc đầu có thể chiếm hết chỗ
                da_dat = []
            r, g = rng.uniform(r0, r1), rng.uniform(g0, g1)
            p = np.array([r * np.cos(g), r * np.sin(g)])
            if all(np.linalg.norm(p - q) > kc for q in da_dat) and \
                    (not self.piper or np.linalg.norm(p - self.C.P.TAM_THAP) > 0.15):
                da_dat.append(p)
        for c, p in zip(TEN_COC, da_dat):
            a = m.joint(m.body(c).jntadr[0]).qposadr[0]
            d.qpos[a:a + 2] = p
        mujoco.mj_forward(m, d)
        lop = type("PiperDemo", (BuocDemo, self.C.TayPiper), {}) if self.piper else RoBotDemo
        self.robot = rb = lop(self, m, d, self.viewer)
        rb.bu = self.bu.copy()
        if self.piper:
            rb.lech_enc = self.lech_enc_piper.copy()
        if self.tu_bam:
            self.hen_phim(rb)
        for t, p in self.kich_ban:                     # chế độ quay video: phím hẹn sẵn theo kịch bản
            rb.su_kien.append((d.time + t, lambda p=p: self.lenh.append(p)))
        self.tam = d.site_xpos[m.site("dich").id][:2].copy()
        dich = vi_tri_kim_tu_thap(self.tam, np.array([1.0, 0.0, 0.0]))
        self.da_xep, self.dang_cam = set(), None

        for i, ten in enumerate(TEN_COC):
            print(f"Xếp {ten}")
            for _ in range(SO_LAN_THU):
                if self.nang_cap and self.can_hieu_chinh:
                    print("Tự hiệu chỉnh encoder + camera...")
                    self.can_hieu_chinh = False   # xoá TRƯỚC: bấm E giữa chừng thì lần sau hiệu chỉnh lại
                    # Đo từ trạng thái "chưa bù": nếu còn bộ bù cũ, kết quả là phần lệch CÒN LẠI chứ không
                    # phải lệch thật -> ghi đè lên bộ bù làm sai số cộng dồn mỗi lần bấm E (2 -> 4 -> 7 độ...).
                    rb.bu[:] = 0
                    self.cam.R_sua, self.cam.t_sua = np.eye(3), np.zeros(3)
                    delta = tu_hieu_chinh(rb, self.cam)[0]
                    if delta is not None and np.degrees(np.abs(delta)).max() < 6:
                        self.bu = rb.bu.copy()
                        print(f"Tự hiệu chỉnh xong: lệch encoder {np.round(np.degrees(delta), 1)} độ")
                    else:                         # kết quả vô lý (bị phá giữa chừng...) -> bỏ, lần sau làm lại
                        rb.bu[:] = 0
                        self.bu[:] = 0
                        self.cam.R_sua, self.cam.t_sua = np.eye(3), np.zeros(3)
                        self.can_hieu_chinh = True
                        print("Tự hiệu chỉnh không tin được -> bỏ, sẽ làm lại")
                thay = self.nhin()
                try:
                    nam = [c for c, t in thay.items() if t.get("nam") and c not in self.da_xep]
                    if nam:
                        self.dung_coc(nam[0], thay[nam[0]])
                        continue
                    if ten not in thay:
                        print(f"Camera không thấy {ten}")
                        continue
                    self.dang_cam = ten
                    if not X.gap(rb, thay[ten]["goc"], self.nang_cap, self.cam):
                        continue
                    p_dat = dich[i]
                    if ten == "coc3" and "coc1" in thay and "coc2" in thay:
                        p_dat = (thay["coc1"]["goc"] + thay["coc2"]["goc"]) / 2 + np.array([0, 0, CAO])
                    if X.dat(rb, p_dat, self.cam, self.nang_cap):
                        self.da_xep.add(ten)
                        break
                except AssertionError:                # IK không tới được (cốc lăn ra ngoài tầm với)
                    print("Cốc ngoài tầm với -> nhìn lại")
                    rb.toi(rb.p + [0, 0, 0.05], NGON_MO, T=0.5)
                finally:
                    self.dang_cam = None
            else:
                print(f"Không xếp được {ten} sau {SO_LAN_THU} lần")
                return False

        rb.toi_khop(rb.Q_CHO, NGON_MO, T=1.5)
        rb.buoc(int(2.0 / m.opt.timestep))
        # Tiêu chí "tháp đứng vững" (như bai_do_ben): không cốc nào nghiêng > 3°, C1 C2 trên sàn,
        # C3 nằm trên chúng. Không chấm vị trí tuyệt đối: nhiễu camera 10 mm làm cả tháp xê dịch vài mm.
        p1, p2, p3 = (d.body(c).xpos for c in TEN_COC)
        nghieng = [np.degrees(np.arccos(np.clip(d.body(c).xmat[8], -1, 1))) for c in TEN_COC]
        return (max(nghieng) < 3 and p1[2] < 0.005 and p2[2] < 0.005 and abs(p3[2] - CAO) < 0.005
                and np.linalg.norm(p3[:2] - (p1[:2] + p2[:2]) / 2) < 0.02)

    def chay(self, so_lan=None, tu_bam=False):
        k = 0
        while so_lan is None or k < so_lan:
            if self.viewer is not None and not self.viewer.is_running():
                break
            k += 1
            self.tu_bam = tu_bam
            try:
                ok = self.mot_lan(self.seed * 1000 + k)
            except ChayLai:
                print("Chạy lại từ đầu")
                continue
            self.so_lan += 1
            self.so_dat += ok
            print(f"KIM TỰ THÁP: {'ĐẠT' if ok else 'KHÔNG ĐẠT'}  ({self.so_dat}/{self.so_lan})")
            if self.viewer is not None:
                try:                                   # đứng yên chờ khách bấm R
                    while self.viewer.is_running():
                        self.robot.buoc(10)
                except ChayLai:
                    pass

    def hen_phim(self, rb):
        """Chế độ thử: hẹn 0-2 lần "khách bấm phím" ngẫu nhiên trong lần chạy này."""
        cac = ["1", "2", "3", "T", "N", "E"]
        for _ in range(int(self.rng.integers(0, 3))):
            p = cac[int(self.rng.integers(len(cac)))]
            rb.su_kien.append((self.d.time + self.rng.uniform(3, 40), lambda p=p: self.lenh.append(p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thu", type=int, metavar="N", help="không cửa sổ: chạy N lần, tự bấm phím ngẫu nhiên")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--piper", action="store_true", help="dùng cánh tay AgileX PiPER (cần model Menagerie)")
    ap.add_argument("--quay", metavar="FILE.mp4", help="không cửa sổ: quay video dự phòng theo kịch bản cố định")
    a = ap.parse_args()

    demo = Demo(xem=a.thu is None and a.quay is None, seed=a.seed, quay=a.quay, piper=a.piper)
    thuc = sys.stdout

    class Ghi:
        """In ra terminal như thường, đồng thời lấy dòng cuối làm dòng trạng thái trên màn hình."""
        def write(self, s):
            thuc.write(s)
            if s.strip():
                demo.trang_thai = s.strip().splitlines()[-1]

        def flush(self):
            thuc.flush()

    sys.stdout = Ghi()
    if a.quay:
        # Kịch bản video (thời điểm tính bằng giây mô phỏng kể từ đầu mỗi lần chạy):
        #   lần 1: tự hiệu chỉnh (~23 s) -> khách làm đổ 1 cốc -> robot dựng lại -> khách dời C3, xếp xong
        #   lần 2: bật lệch encoder 2° + nhiễu camera -> robot tự hiệu chỉnh lại, vẫn xếp được
        for lan, (truoc, hen) in enumerate([([], [(25, "T"), (48, "3")]), (["E", "N"], [])], start=1):
            for p in truoc:
                demo.lam(p)
            demo.kich_ban = hen
            ok = demo.mot_lan(a.seed * 1000 + lan)
            demo.so_lan += 1
            demo.so_dat += ok
            print(f"KIM TỰ THÁP: {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
            demo.robot.buoc(int(3 / demo.m.opt.timestep))   # dừng 3 s trên kết quả
        demo.ghi.close()
        sys.stdout = thuc
        print(f"Đã lưu {a.quay}: đạt {demo.so_dat}/{demo.so_lan}")
        return 0
    if a.thu is None:
        demo.chay()
        return 0
    demo.chay(a.thu, tu_bam=True)
    sys.stdout = thuc
    print(f"\nTHỬ {a.thu} LẦN: đạt {demo.so_dat}/{demo.so_lan}")
    return 0 if demo.so_lan == a.thu else 1


if __name__ == "__main__":
    raise SystemExit(main())
