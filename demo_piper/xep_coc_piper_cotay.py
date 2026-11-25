"""Bài: PiPER với camera GẮN TRÊN CỔ TAY (eye-in-hand), như RealSense D405 lắp trên PiPER thật.

Khác bản camera trên cao (xep_coc_piper_camera.py, eye-to-hand):
  - Camera đi theo bàn tay -> phải QUÉT: đưa tay tới vài điểm nhìn trên cao, chụp, gộp kết quả.
  - Trước khi gắp, đưa tay tới ngay trên cốc, NHÌN GẦN lần nữa rồi mới hạ xuống (chính xác hơn vì gần).
  - Robot biết vị trí camera qua ĐỘNG HỌC THUẬN từ encoder (+ hiệu chỉnh tay-mắt cố định). Encoder lệch thì
    robot "thấy" cốc lệch GIỐNG như tay lệch -> nhìn gần từ ngay trên cốc thì 2 sai số phần lớn triệt tiêu.
    Vì thế không cần marker hay tự hiệu chỉnh như bản camera trên cao.

Chạy (cần model PiPER, xem xep_coc_piper.py):
    python bai_piper/xep_coc_piper_cotay.py --ngau-nhien 3
    python bai_piper/xep_coc_piper_cotay.py --ngau-nhien 3 --lech-encoder 2   # encoder lệch, không tự hiệu chỉnh
    python bai_piper/xep_coc_piper_cotay.py --ngau-nhien 3 --do-coc           # 1 cốc đổ từ đầu
    python bai_piper/xep_coc_piper_cotay.py --ngau-nhien 3 --pha-do           # cốc đổ giữa chừng
    thêm --xem để mở viewer
"""

import argparse
import pathlib
import sys

import mujoco
import numpy as np

THU_MUC = pathlib.Path(__file__).parent
sys.path.insert(0, str(THU_MUC))
import xep_coc_piper_camera as C  # noqa: E402  (kéo theo đường dẫn tới bai_camera, bai_coc_nam...)
import xep_coc_piper as P  # noqa: E402
import xep_coc_camera as X  # noqa: E402
from nhan_dang import tim_coc  # noqa: E402
from realsense import CameraRealSense  # noqa: E402
from xep_coc import CAO, TEN_COC, vi_tri_kim_tu_thap  # noqa: E402

NGHIENG, CAM_X = C.NGHIENG_COTAY, C.CAM_X_COTAY   # vị trí camera cổ tay (xem C.gan_cam_cotay)
Z_NHIN = 0.23                     # TCP khi quét: cao nhất PiPER còn chúc tay thẳng xuống được -> nhìn rộng nhất
# Điểm quét (xy của TCP): 2 điểm phủ vùng cốc, 1 điểm trên chỗ xếp tháp
DIEM_QUET = [np.array([0.28 * np.cos(a), 0.28 * np.sin(a)]) for a in (-0.85, -0.15)]
# Tư thế nhìn tháp = tư thế đặt cốc lên tháp (trên tâm tháp, độ cao di chuyển): kiểm tra tháp và đo chỗ đặt C3
# đều chụp từ đây -> nhất quán với lúc đặt, kể cả khi encoder lệch
TU_THE_THAP = (np.array(P.TAM_THAP) + [CAM_X * np.sin(NGHIENG), 0.0], P.Z_DI)


def tao_model():
    spec = P.tao_spec()
    C.gan_cam_cotay(spec)             # camera robot dùng
    X.gan_camera(spec)                # camera trên cao: chỉ để XEM (ô thứ 2 trên viewer), không marker
    return spec.compile()


class CameraCoTay(CameraRealSense):
    """Ảnh chụp từ vị trí THẬT của camera; nhưng đổi điểm ảnh -> 3D dùng vị trí camera robot TƯỞNG
    (động học thuận từ góc encoder + phần bù encoder robot đang dùng), như ngoài đời."""

    def cap_nhat_ngoai(self, d, rb=None):
        if rb is None:
            return super().cap_nhat_ngoai(d)
        if not hasattr(self, "d_tin"):
            self.d_tin = mujoco.MjData(self.m)
        self.d_tin.qpos[:] = d.qpos
        self.d_tin.qpos[rb.ik.qadr] = d.qpos[rb.ik.qadr] - rb.lech_enc + rb.bu
        mujoco.mj_kinematics(self.m, self.d_tin)
        mujoco.mj_camlight(self.m, self.d_tin)        # mj_kinematics KHÔNG tính vị trí camera
        self._d = self.d_tin


def chup(rb, cam):
    rb.buoc(50 + rb.CHO_THEM)                        # tay đứng yên rồi mới chụp
    cam.cap_nhat_ngoai(rb.d, rb)
    color, depth = cam.chup(rb.d)
    thay = {}
    for c, t in tim_coc(cam, color, depth).items():
        u, v = t["uv"]
        # Cốc sát mép ảnh bị cắt mất 1 phần: thấy 1 mẩu dài -> dễ nhận nhầm là cốc nằm. Bỏ, nhìn từ chỗ khác.
        if min(u, v, cam.W - u, cam.H - v) < 60:
            continue
        t["lech_tam"] = np.hypot(u - cam.cx, v - cam.cy)   # độ "giữa ảnh": dùng để gộp nhiều lần chụp
        thay[c] = t
    rb.bao_anh(color, thay)
    return thay


def nhin_tai(rb, cam, xy, z=Z_NHIN):
    """Đưa tay (chúc xuống) tới trên xy rồi chụp."""
    rb.toi_khop(rb.giai_ik([xy[0], xy[1], z], (0.0, 1.0, 0.0)), T=1.5)
    return chup(rb, cam)


# Điểm quét THÊM khi không thấy cốc cần tìm (cốc bị xô / lăn ra chỗ khác): phủ phần còn lại của tầm với
# (lưới 2 bán kính x 8 hướng; quét tới khi thấy thì dừng). Lưới thưa hơn từng để lọt cốc vào khe giữa 2 điểm:
# ở cả 2 điểm cốc đều sát mép ảnh -> bị bỏ.
DIEM_THEM = [np.array([r * np.cos(a), r * np.sin(a)])
             for a in np.linspace(-1.4, 2.4, 8) for r in (0.22, 0.33)]


def tim_them(rb, cam, gop, can_tim):
    """Quét thêm từng điểm cho tới khi thấy đủ các cốc cần tìm."""
    for xy in DIEM_THEM:
        if all(c in gop for c in can_tim):
            break
        for c, t in nhin_tai(rb, cam, xy).items():
            if c not in gop or t["lech_tam"] < gop[c]["lech_tam"]:
                gop[c] = t


def quet(rb, cam):
    """Quét các điểm nhìn, gộp: mỗi cốc lấy lần thấy GẦN GIỮA ẢNH nhất (ít méo, ít bị che).
    Trả về (gop, thap): thap = ảnh chụp từ trên chỗ xếp tháp, chỉ dùng ảnh này để kiểm tra tháp.
    Encoder lệch thì sai số đo phụ thuộc TƯ THẾ tay: cùng 1 cốc, 2 điểm nhìn khác nhau cho kết quả
    lệch nhau tới ~3 cm -> chỉ so sánh những gì đo từ cùng một tư thế."""
    gop = {}
    for xy in DIEM_QUET:
        for c, t in nhin_tai(rb, cam, xy).items():
            if c not in gop or t["lech_tam"] < gop[c]["lech_tam"]:
                gop[c] = t
    thap = nhin_tai(rb, cam, *TU_THE_THAP)
    for c, t in thap.items():
        if c not in gop or t["lech_tam"] < gop[c]["lech_tam"]:
            gop[c] = t
    return gop, thap


def do_thap(rb, cam):
    """Đo C1, C2 từ ngay trên tháp, ở ĐÚNG độ cao sẽ đặt C3 -> chỗ đặt C3 nhất quán với tư thế lúc đặt."""
    thay = nhin_tai(rb, cam, *TU_THE_THAP)
    if "coc1" in thay and "coc2" in thay and not thay["coc1"].get("nam") and not thay["coc2"].get("nam"):
        return (thay["coc1"]["goc"] + thay["coc2"]["goc"]) / 2 + np.array([0, 0, CAO])
    return None


def nhin_gan(rb, cam, ten, goc):
    """Tới ngay trên cốc ở độ cao di chuyển, chụp lại -> vị trí chính xác hơn (gần hơn, giữa ảnh)."""
    thay = nhin_tai(rb, cam, goc[:2] + [CAM_X * np.sin(NGHIENG), 0.0], rb.Z_DI)
    return thay.get(ten)


def gap(rb, cam, ten, goc):
    t = nhin_gan(rb, cam, ten, goc)
    if t is not None and t.get("nam"):
        # Nhìn gần đáng tin hơn quét: cốc nằm sát cốc khác, nhìn xiên từ xa có lúc tưởng là cốc đứng.
        # Quét lại thì lại nhầm -> vòng lặp. Dựng luôn theo số đo nhìn gần.
        print(f"  nhìn gần: {ten} đang NẰM -> dựng lại")
        C.dung_coc(rb, t["mieng"], t["truc"])
        return False
    if t is None:
        print(f"  nhìn gần: không thấy {ten} -> quét lại")
        return False
    x, y = t["goc"][:2]
    print(f"  nhìn gần: {ten} ở {np.round(t['goc'][:2], 3)} (thật {np.round(rb.d.body(ten).xpos[:2], 3)})")
    rb.toi([x, y, rb.Z_DI], rb.NGON_MO, T=0.8, y=np.array([0.0, 1.0, 0.0]))
    rb.toi([x, y, t["goc"][2] + rb.Z_KEP], rb.NGON_MO, T=1.2)
    rb.toi([x, y, t["goc"][2] + rb.Z_KEP], rb.NGON_DONG, T=0.6)
    if not X.con_coc(rb):
        print("    ! kẹp hụt -> quét lại")
        rb.toi([x, y, rb.Z_DI], rb.NGON_MO, T=1.0)
        return False
    rb.toi([x, y, rb.Z_DI], rb.NGON_DONG, T=1.2)
    return True


def dat(rb, p_dat):
    rb.cho_toi(p_dat[:2], rb.NGON_DONG)
    if not X.con_coc(rb):
        print("    ! cốc rơi trên đường chở -> quét lại")
        rb.toi(rb.p, rb.NGON_MO, T=0.3)
        return False
    x, y, z = p_dat
    rb.toi([x, y, z + rb.Z_KEP + rb.THA_CAO], rb.NGON_DONG, T=1.8)
    rb.toi([x, y, z + rb.Z_KEP + rb.THA_CAO], rb.NGON_MO, T=0.6)
    rb.toi([x, y, rb.Z_DI], rb.NGON_MO, T=1.2)
    return True


def lam_mot_lan(rb, cam, thay, ten, dich, da_xep):
    """Một lần thử với cốc `ten` sau khi quét: dựng cốc nằm (nếu có), hoặc gắp + đặt `ten`."""
    nam = [c for c, t in thay.items() if t.get("nam") and c not in da_xep]
    if nam:
        print(f"  camera thấy {nam[0]} NẰM ĐỔ")
        C.dung_coc(rb, thay[nam[0]]["mieng"], thay[nam[0]]["truc"])
        return
    if ten not in thay:
        print(f"  quét không thấy {ten}")
        return
    p_dat = dich[TEN_COC.index(ten)]
    if ten == "coc3":
        p_dat = do_thap(rb, cam)
        if p_dat is None:
            print("  nhìn gần tháp: không thấy đủ C1, C2 -> quét lại")
            return
    rb.dang_cam = ten
    if not gap(rb, cam, ten, thay[ten]["goc"]):
        return
    if dat(rb, p_dat):
        da_xep.add(ten)


def vong_xep(rb, cam, rng, dich, pha_do):
    """Vòng quét - xử lý tối đa 18 lần cho tới khi đủ 3 cốc trong tháp. Trả về True khi xong."""
    m, d = rb.m, rb.d
    if pha_do:
        rb.khi_cho = {"coc1": lambda: C.xo_do(m, d, rng, "coc3", "khách xô đổ C3 (chưa xếp) khi tay đang chở C1"),
                      "coc3": lambda: C.tuot(rb, m, d)}
    da_xep, da_xo = set(), False
    for _ in range(18):
        if len(da_xep) == 3:
            return True
        if pha_do and not da_xo and {"coc1", "coc2"} <= da_xep:
            da_xo = True
            C.xo_do(m, d, rng, "coc1", "khách xô đổ C1 của tầng dưới", dxy=(-0.09, 0.0), goc=np.pi)
            rb.buoc(300)
        thay, thap = quet(rb, cam)
        C.kiem_tra_thap(thap, da_xep, dich)
        ten = next(c for c in TEN_COC if c not in da_xep)
        print(f"== {ten}")
        can_tim = [c for c in TEN_COC if c not in da_xep and c not in thay]
        if can_tim:
            print(f"  chưa thấy {can_tim} -> quét rộng thêm")
            tim_them(rb, cam, thay, can_tim)
        try:
            lam_mot_lan(rb, cam, thay, ten, dich, da_xep)
        except AssertionError as e:                      # cốc lăn ra ngoài tầm với
            print(f"  {e}")
            rb.toi(rb.p + [0, 0, 0.05], rb.NGON_MO, T=0.5)
        finally:
            rb.dang_cam = None
    print("Không xếp xong sau 18 lần thử")
    return False


def chay(xem=False, seed=None, do_coc=False, lech_encoder=0.0, pha_do=False):
    m, d, rng = C.tao_canh(seed, do_coc)
    viewer = C.mo_viewer(m, d) if xem else None
    rb = C.TayPiper(m, d, viewer)
    rb.buoc(300)
    if lech_encoder:
        rb.lech_enc = np.radians(lech_encoder) * rng.choice([-1, 1], size=6)
        print(f"Encoder lệch {np.round(np.degrees(rb.lech_enc), 1)} độ (KHÔNG tự hiệu chỉnh)")
    cam = CameraCoTay(m, "realsense_cotay", seed=0 if seed is None else seed)
    rb.gan_cam_hien(("CAMERA CO TAY (dieu khien)", cam), ("REALSENSE TREN CAO (chi xem)", CameraRealSense(m)))

    dich = vi_tri_kim_tu_thap(np.array(P.TAM_THAP), np.array([1.0, 0.0, 0.0]))
    if not vong_xep(rb, cam, rng, dich, pha_do):
        return False
    rb.toi_khop(rb.Q_CHO, rb.NGON_MO, T=1.5)
    rb.buoc(int(2.0 / m.opt.timestep))
    ok, nghieng = X.thap_dung_vung(d)
    print(f"Kim tự tháp: {'ĐẠT' if ok else 'KHÔNG ĐẠT'} (nghiêng tối đa {nghieng:.1f}°)")
    if viewer is not None:
        viewer.close()
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xem", action="store_true")
    ap.add_argument("--ngau-nhien", type=int, metavar="SEED")
    ap.add_argument("--do-coc", action="store_true", help="1 cốc đổ từ đầu")
    ap.add_argument("--lech-encoder", type=float, default=0.0, metavar="DO")
    ap.add_argument("--pha-do", action="store_true", help="cốc đổ giữa chừng")
    a = ap.parse_args()
    raise SystemExit(0 if chay(a.xem, a.ngau_nhien, a.do_coc, a.lech_encoder, a.pha_do) else 1)
