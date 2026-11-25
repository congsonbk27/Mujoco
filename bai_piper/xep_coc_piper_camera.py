"""Bài: PiPER + camera RealSense: tìm cốc bằng camera, tự hiệu chỉnh, chỉnh tay bằng camera,
phát hiện rơi cốc, và DỰNG CỐC ĐỔ — các tính năng của cánh tay tự dựng, chuyển sang AgileX PiPER.

Chạy (cần model PiPER, xem xep_coc_piper.py):
    python bai_piper/xep_coc_piper_camera.py --ngau-nhien 3               # camera, vòng kín
    python bai_piper/xep_coc_piper_camera.py --ngau-nhien 3 --nang-cap    # + tự hiệu chỉnh, chỉnh tay, phát hiện rơi
    python bai_piper/xep_coc_piper_camera.py --do-coc --nang-cap          # 1 cốc bị đổ từ đầu -> robot dựng lại
    python bai_piper/xep_coc_piper_camera.py --lech-encoder 2 --nang-cap  # encoder lệch 2° mọi khớp
    thêm --xem để mở viewer

Dùng lại nguyên các hàm của bài camera (nhin, gap, dat, chinh_tay, tu_hieu_chinh, tim_coc): chúng đọc
thông số cánh tay từ lớp RoBot, ở đây lớp TayPiper chỉ ghi đè thông số và vài phương thức.

Điểm khác của PiPER khi DỰNG CỐC: khớp gập cổ tay chỉ ±70° -> không xoay bàn tay nằm ngang được
(cách của bai_coc_nam). Thay vào đó xoay 90° chia đôi: gắp với bàn tay nghiêng -45°, xoay sang +45°.
Nghiêng SANG NGANG (theo hướng tiếp tuyến quanh đế) thì PiPER với tới, nghiêng về phía đế thì không
-> nếu cốc nằm theo hướng xuyên tâm, trước tiên gắp từ trên xuống, xoay cốc nằm theo hướng tiếp tuyến.
"""

import argparse
import pathlib
import sys

import mujoco
import numpy as np

THU_MUC = pathlib.Path(__file__).parent
sys.path.insert(0, str(THU_MUC))
sys.path.insert(0, str(THU_MUC.parent / "bai_camera"))
sys.path.insert(0, str(THU_MUC.parent / "bai_coc_nam"))
import xep_coc_camera as X  # noqa: E402
import xep_coc_piper as P  # noqa: E402
from coc_nam import dat_coc_nam  # noqa: E402
from realsense import CameraRealSense  # noqa: E402
from tu_hieu_chinh import tu_hieu_chinh  # noqa: E402
from xep_coc import CAO, TEN_COC, vi_tri_kim_tu_thap  # noqa: E402

EZ = np.array([0.0, 0.0, 1.0])


def tao_model():
    spec = P.tao_spec()
    # camera trên cao + 2 marker hồng ở mu bàn tay PiPER, xa trục 8.5 cm: cụm motor cổ tay PiPER to,
    # gắn 5.2 cm như tay tự dựng thì camera chỉ thấy 1 marker (bị che một phần) -> đo sai ~9 mm
    X.gan_camera(spec, "link6", z_tai=0.0, kc=TayPiper.KC_MARKER)
    return spec.compile()


class TayPiper(X.RoBot):
    KHOP = P.KHOP
    ACT_NGON, KHOP_NGON = "gripper", "joint7"
    # Tư thế chờ khi chụp: cánh tay ngả ra SAU đế (không có cốc ở đó) -> không che cốc
    Q_CHO = np.array([2.6, 1.2, -1.0, 0.0, 0.8, 0.0])
    Z_KEP, Z_DI, THA_CAO = P.Z_KEP, P.Z_DI, P.THA_CAO
    NGON_MO, NGON_DONG = P.NGON_MO, P.NGON_DONG
    KC_MARKER = 0.085
    CHO_THEM = 400        # servo PiPER (mềm + tích phân) cần ~0.8 s mới đứng yên trong ~1 mm
    # Điểm tự hiệu chỉnh: trong vùng PiPER chúc tay thẳng xuống được (TCP thấp hơn mặt đế + 11 cm),
    # đầu ngón (TCP - 3 cm) cao hơn đỉnh cốc (9 cm) -> không đụng cốc
    DIEM_HIEU_CHINH = [np.array([r * np.cos(a), r * np.sin(a), z])
                       for z in (0.14, 0.22) for r in (0.22, 0.32) for a in (-1.0, -0.3, 0.5)]

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.tich = np.zeros(6)
        self.lech_enc = np.zeros(6)   # sai số encoder THẬT (chỉ mô phỏng biết; robot phải tự ước lượng)
        self.rng = np.random.default_rng(0)

    def lenh_khop(self, q):
        """Ghi nhớ góc encoder muốn tới; lệnh thật (có tích phân) do buoc() cập nhật mỗi chu kỳ."""
        self.muc = q - self.bu
        return self.muc + self.tich

    def buoc(self, n):
        """Khâu tích phân như servo thật (xem xep_coc_piper.KI), chạy MỌI chu kỳ 20 ms kể cả khi đứng yên
        (chỉ cập nhật lúc tay đang đi thì khi dừng lệnh giữ phần tích phân dư -> khớp trôi tới ~5°).
        Servo bám theo ENCODER: encoder lệch δ thì khớp thật đứng ở (lệnh + δ) -> như bài độ bền."""
        for _ in range(n):
            if int(round(self.d.time / self.m.opt.timestep)) % 10 == 0 and hasattr(self, "muc"):
                doc = self.d.qpos[self.ik.qadr] - self.lech_enc
                self.tich = np.clip(self.tich + P.KI * (self.muc - doc), -0.15, 0.15)
                self.d.ctrl[self.act_tay] = self.muc + self.tich
            super().buoc(1)

    def giai_ik(self, p, y, z=(0, 0, -1.0)):
        """IK nhiều điểm khởi đầu ngẫu nhiên (PiPER có giới hạn khớp hẹp, 1 điểm khởi đầu dễ kẹt)."""
        lo, hi = self.ik.lo, self.ik.hi
        tot = min((self.ik.giai(lo + (hi - lo) * self.rng.random(6), np.asarray(p, float), np.asarray(y, float),
                                so_vong=200, z_dich=np.asarray(z, float)) for _ in range(30)), key=lambda k: k[1])
        assert tot[1] < 2e-3, f"PiPER không với tới {np.round(p, 3)} (sai {tot[1] * 1000:.0f} mm)"
        return tot[0]

    def giai_ik_moi(self, p_dich, y_dich=(0.0, 1.0, 0.0)):
        return self.giai_ik(p_dich, y_dich)

    def cho_toi(self, xy, ngon):
        """Chở cốc vòng quanh đế, chậm (~12 cm/s) — lý do xem xep_coc_piper.lap_ke_hoach."""
        a, b = self.p, np.array([xy[0], xy[1], self.Z_DI])
        ga, gb = np.arctan2(a[1], a[0]), np.arctan2(b[1], b[0])
        r = max(np.linalg.norm(a[:2]), np.linalg.norm(b[:2]))
        giua = np.array([r * np.cos((ga + gb) / 2), r * np.sin((ga + gb) / 2), self.Z_DI])
        for dich in (giua, b):
            self.toi(dich, ngon, T=max(1.0, np.linalg.norm(dich[:2] - self.p[:2]) / 0.12))


S_KEP_NAM = 0.05     # gắp cốc nằm ở điểm trên trục cách vành miệng 5 cm (gần giữa thân)
LUI = 0.025          # lùi TCP khỏi trục cốc theo hướng tiếp cận: lòng bàn tay (TCP - 1.6 cm) không đè thân cốc


def xoay_coc_tiep_tuyen(rb, mieng, truc):
    """Cốc nằm theo hướng xuyên tâm: gắp từ trên xuống, xoay cho trục cốc theo hướng tiếp tuyến, đặt lại.
    Trả về (mieng, truc) mới."""
    a = np.array([truc[0], truc[1], 0.0])
    a /= np.linalg.norm(a)
    tam = mieng + 0.045 * truc
    r_hat = np.array([tam[0], tam[1], 0.0]) / np.linalg.norm(tam[:2])
    t_hat = np.cross(EZ, r_hat)
    a_moi = t_hat if a @ t_hat >= 0 else -t_hat
    phi = np.arctan2(np.cross(a, a_moi)[2], a @ a_moi)        # góc xoay (|phi| <= 90°)
    c, s = np.cos(phi), np.sin(phi)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    y0 = np.cross(EZ, a)
    z_kep = 0.055                                            # trục cốc ~3 cm + lòng bàn tay phải trên đỉnh thân
    print(f"  cốc nằm xuyên tâm -> xoay {np.degrees(phi):.0f}° cho nằm theo hướng tiếp tuyến")
    rb.toi_khop(rb.giai_ik([tam[0], tam[1], rb.Z_DI], y0), rb.NGON_MO, T=1.5)
    rb.toi([tam[0], tam[1], z_kep], rb.NGON_MO, T=1.2, y=y0)
    rb.toi([tam[0], tam[1], z_kep], rb.NGON_DONG, T=0.6, y=y0)
    rb.toi([tam[0], tam[1], 0.12], rb.NGON_DONG, T=1.0, y=y0)
    rb.toi([tam[0], tam[1], 0.12], rb.NGON_DONG, T=1.2, y=Rz @ y0)
    rb.toi([tam[0], tam[1], z_kep + 0.003], rb.NGON_DONG, T=1.0, y=Rz @ y0)
    rb.toi([tam[0], tam[1], z_kep + 0.003], rb.NGON_MO, T=0.5, y=Rz @ y0)
    rb.toi([tam[0], tam[1], rb.Z_DI], rb.NGON_MO, T=1.0, y=Rz @ y0)
    t_moi = Rz @ truc
    return tam - 0.045 * t_moi, t_moi


def dung_coc(rb, mieng, truc):
    """Dựng cốc nằm: gắp với bàn tay nghiêng 45° dọc trục cốc, nhấc, xoay 90° quanh trục ngón, đặt xuống."""
    tam = mieng + 0.045 * truc
    r_hat = np.array([tam[0], tam[1], 0.0]) / np.linalg.norm(tam[:2])
    a = np.array([truc[0], truc[1], 0.0])
    a /= np.linalg.norm(a)
    if abs(a @ r_hat) > 0.5:
        mieng, truc = xoay_coc_tiep_tuyen(rb, mieng, truc)
        a = np.array([truc[0], truc[1], 0.0])
        a /= np.linalg.norm(a)
    k = np.sqrt(0.5)
    # Hai hướng tiếp cận nghiêng ±45° dọc trục cốc; xoay từ hướng này sang hướng kia (90° quanh trục ngón y)
    # phải đưa trục cốc (miệng -> đáy) lên THẲNG ĐỨNG (cốc úp, như các cốc khác) -> chọn chiều cho đúng.
    for dau in (1, -1):
        z0, z1 = k * (-EZ) + dau * k * a, k * (-EZ) - dau * k * a
        y = np.cross(z0, z1)
        y /= np.linalg.norm(y)
        R = quay(y, np.pi / 2)
        if np.allclose(R @ z0, z1, atol=1e-6) and (R @ truc)[2] > 0.9:
            break
    G = mieng + S_KEP_NAM * truc
    tcp0 = G - LUI * z0
    p_xoay = tcp0 + [0, 0, 0.10]
    m_sau = R @ (mieng - tcp0)                       # vành miệng so với TCP, sau khi xoay
    tcp_dat = np.array([p_xoay[0], p_xoay[1], 0.004 - m_sau[2]])
    print("  dựng cốc: gắp nghiêng 45°, xoay 90° quanh trục ngón")
    # Tới trên cao (tay chúc xuống) bằng không gian khớp rồi mới hạ + nghiêng theo đường thẳng: đi thẳng
    # trong không gian khớp tới điểm gắp thấp thì ngón quét ngang qua cốc bên cạnh (đã gặp: hất đổ C1).
    tren = tcp0 - 0.06 * z0
    rb.toi_khop(rb.giai_ik([tren[0], tren[1], rb.Z_DI], y), rb.NGON_MO, T=1.5)
    rb.toi(tren, rb.NGON_MO, T=1.5, y=y, z=z0)
    rb.toi(tcp0, rb.NGON_MO, T=1.0, y=y, z=z0)
    rb.toi(tcp0, rb.NGON_DONG, T=0.6, y=y, z=z0)
    rb.toi(p_xoay, rb.NGON_DONG, T=1.2, y=y, z=z0)
    rb.toi(p_xoay, rb.NGON_DONG, T=2.5, y=y, z=z1)
    rb.toi(tcp_dat, rb.NGON_DONG, T=1.2, y=y, z=z1)
    rb.toi(tcp_dat, rb.NGON_MO, T=0.5, y=y, z=z1)
    rb.toi(tcp_dat - 0.06 * z1, rb.NGON_MO, T=0.8, y=y, z=z1)
    rb.toi(tcp_dat - 0.06 * z1 + [0, 0, 0.08], rb.NGON_MO, T=0.8, y=y, z=z1)


def quay(truc, goc):
    """Ma trận quay quanh trục đơn vị (Rodrigues)."""
    K = np.array([[0, -truc[2], truc[1]], [truc[2], 0, -truc[0]], [-truc[1], truc[0], 0]])
    return np.eye(3) + np.sin(goc) * K + (1 - np.cos(goc)) * K @ K


def chay(xem=False, seed=None, nang_cap=False, do_coc=False, lech_encoder=0.0):
    m = tao_model()
    d = mujoco.MjData(m)
    qadr = {c: m.joint(m.body(c).jntadr[0]).qposadr[0] for c in TEN_COC}
    rng = np.random.default_rng(seed)
    if seed is not None:
        da_dat, lan = [], 0
        while len(da_dat) < 3:
            lan += 1
            if lan % 200 == 0:
                da_dat = []
            r, g = rng.uniform(0.22, 0.34), rng.uniform(-1.2, 0.2)
            p = np.array([r * np.cos(g), r * np.sin(g)])
            if all(np.linalg.norm(p - q) > 0.15 for q in da_dat) and np.linalg.norm(p - P.TAM_THAP) > 0.15:
                da_dat.append(p)
        for c, p in zip(TEN_COC, da_dat):
            d.qpos[qadr[c]:qadr[c] + 2] = p
    if do_coc:
        ten = TEN_COC[int(rng.integers(3))]
        dat_coc_nam(m, d, ten, d.qpos[qadr[ten]:qadr[ten] + 2].copy(), rng.uniform(-np.pi, np.pi))
        print(f"{ten} bị ĐỔ")
    d.qpos[[m.joint(j).qposadr[0] for j in P.KHOP]] = TayPiper.Q_CHO
    d.ctrl[[m.actuator(j).id for j in P.KHOP]] = TayPiper.Q_CHO
    mujoco.mj_forward(m, d)

    viewer = None
    if xem:
        import mujoco.viewer as mjv
        viewer = mjv.launch_passive(m, d)
        viewer.cam.lookat[:] = [0.15, 0.0, 0.1]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.1, -150, -30
    rb = TayPiper(m, d, viewer)
    rb.buoc(300)                                     # cốc đổ nằm yên
    if lech_encoder:
        rb.lech_enc = np.radians(lech_encoder) * rng.choice([-1, 1], size=6)
        print(f"Encoder lệch {np.round(np.degrees(rb.lech_enc), 1)} độ")
    cam = CameraRealSense(m, seed=0 if seed is None else seed)
    if nang_cap:
        delta = tu_hieu_chinh(rb, cam)[0]
        print("Tự hiệu chỉnh: " + ("không đủ mẫu marker" if delta is None
                                   else f"lệch encoder ước lượng {np.round(np.degrees(delta), 1)} độ"))

    dich = vi_tri_kim_tu_thap(np.array(P.TAM_THAP), np.array([1.0, 0.0, 0.0]))
    da_xep = set()
    for i, ten in enumerate(TEN_COC):
        print(f"== {ten}")
        for _ in range(5):
            thay = X.nhin(rb, cam)
            try:
                if lam_mot_lan(rb, cam, thay, ten, i, dich, da_xep, nang_cap, d):
                    break
            except AssertionError as e:                 # cốc lăn ra ngoài tầm với
                print(f"  {e}")
                rb.toi(rb.p + [0, 0, 0.05], rb.NGON_MO, T=0.5)
        else:
            print(f"Không xếp được {ten}")
            return False

    rb.toi_khop(rb.Q_CHO, rb.NGON_MO, T=1.5)
    rb.buoc(int(2.0 / m.opt.timestep))
    # Tiêu chí "tháp đứng vững" như bai_do_ben / demo
    p1, p2, p3 = (d.body(c).xpos for c in TEN_COC)
    nghieng = [np.degrees(np.arccos(np.clip(d.body(c).xmat[8], -1, 1))) for c in TEN_COC]
    ok = (max(nghieng) < 3 and p1[2] < 0.005 and p2[2] < 0.005 and abs(p3[2] - CAO) < 0.005
          and np.linalg.norm(p3[:2] - (p1[:2] + p2[:2]) / 2) < 0.02)
    print(f"Kim tự tháp: {'ĐẠT' if ok else 'KHÔNG ĐẠT'} (nghiêng tối đa {max(nghieng):.1f}°)")
    if viewer is not None:
        viewer.close()
    return ok


def lam_mot_lan(rb, cam, thay, ten, i, dich, da_xep, nang_cap, d):
    """Một lần thử với cốc `ten` từ ảnh vừa chụp. Trả về True khi đã đặt xong cốc."""
    nam = [c for c, t in thay.items() if t.get("nam") and c not in da_xep]
    if nam:
        print(f"  camera thấy {nam[0]} NẰM ĐỔ")
        dung_coc(rb, thay[nam[0]]["mieng"], thay[nam[0]]["truc"])
        return False
    if ten not in thay:
        print(f"  camera không thấy {ten}")
        return False
    print(f"  camera thấy {ten} ở {np.round(thay[ten]['goc'][:2], 3)} "
          f"(sai {np.linalg.norm(thay[ten]['goc'][:2] - d.body(ten).xpos[:2]) * 1000:.1f} mm)")
    if not X.gap(rb, thay[ten]["goc"], nang_cap, cam):
        return False
    p_dat = dich[i]
    if ten == "coc3" and "coc1" in thay and "coc2" in thay:
        p_dat = (thay["coc1"]["goc"] + thay["coc2"]["goc"]) / 2 + np.array([0, 0, CAO])
    if X.dat(rb, p_dat, cam, nang_cap):
        da_xep.add(ten)
        return True
    return False

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xem", action="store_true")
    ap.add_argument("--ngau-nhien", type=int, metavar="SEED")
    ap.add_argument("--nang-cap", action="store_true", help="tự hiệu chỉnh + chỉnh tay bằng camera + phát hiện rơi")
    ap.add_argument("--do-coc", action="store_true", help="làm đổ 1 cốc từ đầu")
    ap.add_argument("--lech-encoder", type=float, default=0.0, metavar="DO")
    a = ap.parse_args()
    raise SystemExit(0 if chay(a.xem, a.ngau_nhien, a.nang_cap, a.do_coc, a.lech_encoder) else 1)
