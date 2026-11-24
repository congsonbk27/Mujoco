"""Bài: Thử độ bền hệ thống xếp cốc bằng camera.

Câu hỏi: robot thật luôn có sai số (camera nhiễu, hiệu chỉnh lệch, servo lệch, cốc trơn/nặng...).
Mỗi loại sai số được phép lớn TỚI ĐÂU thì hệ thống vẫn xếp được tháp?

Cách làm: với mỗi NGUỒN sai số, tăng dần MỨC độ; ở mỗi mức chạy N lần với cốc rải ngẫu nhiên
(cùng một bộ seed cho mọi mức để so sánh công bằng) và đếm tỉ lệ thành công.

Chạy:
    MUJOCO_GL=egl python thu_do_ben.py             # đầy đủ (30 lần/mức, ~10 phút với 8 nhân)
    MUJOCO_GL=egl python thu_do_ben.py --nhanh     # 6 lần/mức để thử nhanh
Kết quả: ket_qua.csv (+ ve_bieu_do.py để vẽ biểu đồ)
"""

import argparse
import contextlib
import csv
import io
import multiprocessing as mp
import pathlib
import re
import sys
import time

import mujoco
import numpy as np

THU_MUC = pathlib.Path(__file__).parent
sys.path.insert(0, str(THU_MUC.parent / "bai_camera"))


# ---------------------------------------------------------------------------
# Các nguồn sai số. Mỗi hàm nhận (mức, rng) và trả về tuy_chinh(m, d, cam)
# để xep_coc_camera.chay() gọi trước khi chạy.
# ---------------------------------------------------------------------------
def vector_ngau_nhien(rng, ngang=False):
    v = rng.normal(size=3)
    if ngang:
        v[2] = 0
    return v / np.linalg.norm(v)


def ma_tran_xoay(truc, goc):
    """Công thức Rodrigues: ma trận xoay góc `goc` quanh trục đơn vị `truc`."""
    K = np.array([[0, -truc[2], truc[1]], [truc[2], 0, -truc[0]], [-truc[1], truc[0], 0]])
    return np.eye(3) + np.sin(goc) * K + (1 - np.cos(goc)) * K @ K


def nhieu_camera(muc, rng):
    """Độ lệch chuẩn nhiễu depth ở 1 m (mm). RealSense D435 ~ 2 mm @1 m khi điều kiện tốt."""
    def f(m, d, cam):
        cam.nhieu = muc / 1000
    return f


def lech_tay_mat_tinh_tien(muc, rng):
    """Hiệu chỉnh tay-mắt sai vị trí camera `muc` mm theo một hướng ngẫu nhiên (3D)."""
    v = vector_ngau_nhien(rng)
    def f(m, d, cam):
        cam.t_lech = v * muc / 1000
    return f


def lech_tay_mat_xoay(muc, rng):
    """Hiệu chỉnh tay-mắt sai hướng camera `muc` độ quanh một trục ngẫu nhiên."""
    R = ma_tran_xoay(vector_ngau_nhien(rng), np.radians(muc))
    def f(m, d, cam):
        cam.R_lech = R
    return f


def lech_encoder(muc, rng):
    """Mỗi khớp tay lệch 0 điểm (offset encoder) ±`muc` độ, dấu ngẫu nhiên.

    Actuator position: lực = kp·ctrl + biasprm[0] − kp·q. Đặt biasprm[0] = kp·δ thì khớp dừng ở
    q = ctrl + δ: robot ra lệnh góc ctrl nhưng thực tế đứng ở ctrl + δ, đúng như encoder lệch 0.
    """
    from gap_do import KHOP_TAY
    dau = rng.choice([-1, 1], size=len(KHOP_TAY))
    def f(m, d, cam):
        for i, ten in enumerate(KHOP_TAY):
            a = m.actuator(ten).id
            m.actuator_biasprm[a, 0] = m.actuator_gainprm[a, 0] * np.radians(muc) * dau[i]
    return f


def ma_sat_kep(muc, rng):
    """Hệ số ma sát giữa ngón và cốc (mặc định 1.5). Cốc nhựa trơn/ướt: 0.2-0.4.
    MuJoCo lấy MAX ma sát của 2 geom -> phải đặt cả ngón và cốc."""
    def f(m, d, cam):
        for g in ["coc1", "coc2", "coc3"]:
            m.geom_friction[m.geom(g).id, 0] = muc
        for b in ["ngon_trai", "ngon_phai"]:
            gid = m.body_geomadr[m.body(b).id]
            m.geom_friction[gid, 0] = muc
    return f


def khoi_luong_coc(muc, rng):
    """Khối lượng mỗi cốc (gam). Mặc định 30 g (cốc nhựa rỗng); cốc có nước ~ 200-300 g."""
    def f(m, d, cam):
        for c in ["coc1", "coc2", "coc3"]:
            b = m.body(c).id
            ti_le = (muc / 1000) / m.body_mass[b]
            m.body_mass[b] *= ti_le
            m.body_inertia[b] *= ti_le
        # BẮT BUỘC sau khi đổi khối lượng: tính lại các hằng số mà bộ giải ràng buộc dùng
        # (body_invweight0...). Thiếu bước này mô phỏng nổ (NaN) với cốc nặng.
        mujoco.mj_setConst(m, d)
    return f


def ket_hop(muc, rng):
    """Kết hợp mọi sai số cùng lúc. muc = 1 (tốt), 2 (bình thường), 3 (xấu)."""
    bang = {  # nhiễu mm, tịnh tiến mm, xoay độ, encoder độ, ma sát, khối lượng g
        1: (2, 2, 0.2, 0.2, 1.0, 30),
        2: (4, 5, 0.5, 0.5, 0.7, 60),
        3: (8, 10, 1.0, 1.0, 0.5, 100),
    }
    n, t, x, e, ms, kl = bang[muc]
    ham = [nhieu_camera(n, rng), lech_tay_mat_tinh_tien(t, rng), lech_tay_mat_xoay(x, rng),
           lech_encoder(e, rng), ma_sat_kep(ms, rng), khoi_luong_coc(kl, rng)]
    def f(m, d, cam):
        for h in ham:
            h(m, d, cam)
    return f


# (tên, hàm, đơn vị, các mức, mức mặc định của bài camera)
NGUON = [
    ("nhieu_camera",  nhieu_camera,           "mm @1m", [2, 5, 10, 20, 40, 80], 2),
    ("tay_mat_dich",  lech_tay_mat_tinh_tien, "mm",     [0, 5, 10, 15, 20, 30, 40], 0),
    ("tay_mat_xoay",  lech_tay_mat_xoay,      "độ",     [0, 0.5, 1, 1.5, 2, 3, 4], 0),
    ("lech_encoder",  lech_encoder,           "độ",     [0, 0.5, 1, 1.5, 2, 3, 4], 0),
    ("ma_sat_kep",    ma_sat_kep,             "μ",      [1.5, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05], 1.5),
    ("khoi_luong",    khoi_luong_coc,         "g",      [30, 100, 300, 600, 1000, 1500], 30),
    ("ket_hop",       ket_hop,                "cấp",    [1, 2, 3], None),
]


# ---------------------------------------------------------------------------
def mot_lan(viec):
    """Chạy 1 lần; trả về dict kết quả. Chạy trong process con."""
    ten, muc, seed, nang_cap = viec
    import xep_coc_camera as X
    ham = dict((n[0], n[1]) for n in NGUON)[ten]
    rng = np.random.default_rng(10_000 + seed)        # sai số ngẫu nhiên theo seed, tái lập được
    out = io.StringIO()
    t0 = time.time()
    giu = {}
    sai_so = ham(muc, rng)

    def tuy_chinh(m, d, cam):
        giu["d"] = d          # giữ lại để sau khi chạy kiểm tra mô phỏng có bị nổ không
        sai_so(m, d, cam)

    try:
        with contextlib.redirect_stdout(out):
            X.chay(seed=seed, tuy_chinh=tuy_chinh, nang_cap=nang_cap)
        log = out.getvalue()
        ok, ly_do, chi_so = danh_gia_thap(log)
        # Mô phỏng nổ (NaN) -> MuJoCo tự reset: kết quả vô nghĩa, tách riêng khỏi lỗi của robot
        if giu["d"].warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0:
            ok, ly_do = False, "mo_phong_no"
            chi_so["dep"] = 0
    except Exception as e:  # IK không hội tụ, mô phỏng nổ...
        log = out.getvalue()
        ok, ly_do, chi_so = False, f"loi_{type(e).__name__}", {}
    return {"nguon": ten, "muc": muc, "seed": seed, "dat": int(ok), "ly_do": ly_do,
            **chi_so, "so_lan_kep_hut": log.count("kẹp hụt"), "giay": round(time.time() - t0, 1)}


def danh_gia_thap(log):
    """Hai chỉ số (không đòi tháp nằm đúng toạ độ tuyệt đối - cả tháp dịch 1 cm cũng không sao):
      ĐỨNG VỮNG (chỉ số chính, thuần vật lý): sau 2 s chờ
         - không cốc nào nghiêng quá 3 độ
         - C1, C2 nằm trên sàn (z ~ 0), không chồng lấn (tâm cách nhau >= 6.5 cm)
         - C3 nằm ở tầng 2 (z = 9 cm ± 5 mm)  -> tức là đang được C1, C2 đỡ
      ĐẸP: đứng vững VÀ C3 lệch khỏi điểm giữa C1-C2 không quá 1 cm
    Trả về (đứng vững?, lý do hỏng, các chỉ số)."""
    if "camera không thấy" in log:
        return False, "camera_khong_thay", {}
    if "Gắp thất bại" in log:
        return False, "kep_hut_3_lan", {}
    p = {int(k): np.array([float(v) for v in xyz.split()])
         for k, xyz in re.findall(r"C([123]): ở \[ *([-\d.]+ +[-\d.]+ +[-\d.]+) *\]", log)}
    nghieng = [float(x) for x in re.findall(r"nghiêng +([\d.]+)", log)]
    lech = [float(x) for x in re.findall(r"lệch +([\d.]+) mm", log)]
    if len(p) < 3:
        return False, "khong_xong", {}
    kc12 = np.linalg.norm(p[1][:2] - p[2][:2])
    lech3 = np.linalg.norm(p[3][:2] - (p[1][:2] + p[2][:2]) / 2)
    chi_so = {"lech_tuyet_doi_max_mm": max(lech), "kc_C1C2_mm": round(kc12 * 1000, 1),
              "C3_lech_giua_mm": round(lech3 * 1000, 1), "nghieng_max_do": max(nghieng)}
    if max(nghieng) >= 3:
        ly_do = "coc_do"
    elif kc12 < 0.065 or abs(p[1][2]) > 0.005 or abs(p[2][2]) > 0.005:
        ly_do = "tang1_sai"
    elif abs(p[3][2] - 0.09) > 0.005:
        ly_do = "C3_roi"
    else:
        ly_do = ""
    chi_so["dep"] = int(ly_do == "" and lech3 <= 0.01)
    return ly_do == "", ly_do, chi_so


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nhanh", action="store_true")
    ap.add_argument("--so-lan", type=int, default=30)
    ap.add_argument("--nhan", type=int, default=mp.cpu_count())
    ap.add_argument("--nang-cap", action="store_true",
                    help="bản nâng cấp: tự hiệu chỉnh + visual servoing + phát hiện rơi cốc")
    a = ap.parse_args()
    n = 6 if a.nhanh else a.so_lan

    viec = [(ten, muc, seed, a.nang_cap) for ten, _, _, cac_muc, _ in NGUON for muc in cac_muc
            for seed in range(1, n + 1)]
    print(f"{len(viec)} lần chạy trên {a.nhan} nhân...")
    t0 = time.time()
    ket_qua = []
    with mp.get_context("spawn").Pool(a.nhan) as pool:
        for i, kq in enumerate(pool.imap_unordered(mot_lan, viec, chunksize=2), 1):
            ket_qua.append(kq)
            if i % 50 == 0:
                print(f"  {i}/{len(viec)}  ({time.time() - t0:.0f} s)", flush=True)

    ket_qua.sort(key=lambda r: ([x[0] for x in NGUON].index(r["nguon"]), r["muc"], r["seed"]))
    ten_file = "ket_qua_nang_cap.csv" if a.nang_cap else "ket_qua.csv"
    with open(THU_MUC / ten_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ket_qua[0]))
        w.writeheader()
        w.writerows(ket_qua)

    print(f"\nXong trong {time.time() - t0:.0f} s. Tỉ lệ tháp đứng vững | tháp đẹp:")
    for ten, _, dv, cac_muc, _ in NGUON:
        print(f"  {ten} ({dv}):")
        for muc in cac_muc:
            r = [k for k in ket_qua if k["nguon"] == ten and k["muc"] == muc]
            dat = sum(k["dat"] for k in r)
            ly_do = {}
            for k in r:
                if k["ly_do"]:
                    ly_do[k["ly_do"]] = ly_do.get(k["ly_do"], 0) + 1
            dep = sum(int(k.get("dep") or 0) for k in r)
            print(f"    {muc:>6}: đứng {dat:>3}/{len(r)} = {100*dat/len(r):5.1f}%  | đẹp {100*dep/len(r):5.1f}%"
                  f"   {ly_do if ly_do else ''}")


if __name__ == "__main__":
    main()
