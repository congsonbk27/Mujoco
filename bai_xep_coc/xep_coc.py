"""Bài: Xếp 3 cốc thành kim tự tháp bằng cánh tay robot (MuJoCo).

        [ C3 ]
     [ C1 ][ C2 ]

Chạy:
    python xep_coc.py                # chạy headless, in kết quả kiểm tra
    python xep_coc.py --xem          # mở viewer
    python xep_coc.py --gif          # xuất xep_coc.gif (MUJOCO_GL=egl nếu không có màn hình)
    python xep_coc.py --ngau-nhien 5 # 3 cốc ở vị trí ngẫu nhiên

Tái sử dụng từ bài trước (../bai_gap_do/gap_do.py):
    GiaiIK  - động học ngược Damped Least Squares cho site "tcp"
    muot    - nội suy smoothstep bậc 5
    nlerp   - nội suy hướng (vector đơn vị)
Điểm mới của bài này:
    - Lập kế hoạch cho NHIỀU vật, tính vị trí đặt theo hình học kim tự tháp
    - Chọn hướng kẹp để ngón không đụng cốc bên cạnh
    - Đặt vật lên trên vật khác (tầng 2) và kiểm tra độ ổn định của chồng cốc
"""

import argparse
import pathlib
import sys
import time

import mujoco
import numpy as np

THU_MUC = pathlib.Path(__file__).parent
sys.path.insert(0, str(THU_MUC.parent / "bai_gap_do"))
from gap_do import KHOP_TAY, GiaiIK, NhipXem, muot, nlerp  # noqa: E402

MODEL_XML = THU_MUC / "xep_coc.xml"

# --- Kích thước cốc (khớp với mesh trong xep_coc.xml) ---
R_MIENG = 0.035   # bán kính miệng cốc (nằm dưới khi úp)
R_DAY = 0.025     # bán kính đáy cốc (nằm trên khi úp)
CAO = 0.09        # chiều cao cốc

# --- Thông số gắp ---
Z_KEP = 0.045     # TCP cao hơn vành miệng 4.5 cm ~ ngang trọng tâm cốc (4 cm) -> cốc không lật trong kẹp
Z_DI = 0.30       # độ cao TCP khi di chuyển (cốc đang cầm cao hơn đỉnh chồng cốc)
KHE = 0.004       # khe hở giữa 2 cốc tầng dưới (tránh ma sát cọ nhau khi đặt)
THA_CAO = 0.003   # thả cốc khi còn cách mặt đỡ 3 mm
NGON_MO = 0.045
NGON_DONG = 0.0
TEN_COC = ["coc1", "coc2", "coc3"]


def vi_tri_kim_tu_thap(tam, huong_hang):
    """Tính vị trí gốc (tâm vành miệng) của 3 cốc trong kim tự tháp.

    tam        : tâm kim tự tháp trên sàn (x, y)
    huong_hang : vector đơn vị nằm ngang, hướng của hàng cốc tầng dưới
    """
    tam = np.array([tam[0], tam[1], 0.0])
    nua = R_MIENG + KHE / 2           # tâm cốc dưới cách tâm tháp 1 bán kính miệng (+ nửa khe)
    c1 = tam - nua * huong_hang
    c2 = tam + nua * huong_hang
    # Tầng 2: vành miệng C3 (bán kính 3.5 cm) đi qua gần tâm đáy C1, C2 (cách tâm 3.7 cm)
    # -> tựa trọn lên 2 mặt đáy (bán kính 2.5 cm), đứng vững.
    c3 = tam + np.array([0, 0, CAO])
    return [c1, c2, c3]


def lap_ke_hoach(nguon, dich, y_kep):
    """Mỗi cốc: 8 bước như bài gắp đồ. Trả về list (tên, p_tcp, y, ngón, T)."""
    ke_hoach = []
    for i, (p_nguon, p_dich) in enumerate(zip(nguon, dich), start=1):
        len_ = lambda p, z: np.array([p[0], p[1], z])  # noqa: E731
        ke_hoach += [
            (f"C{i} toi_tren", len_(p_nguon, Z_DI),                       y_kep, NGON_MO,   1.5),
            (f"C{i} ha_xuong", len_(p_nguon, p_nguon[2] + Z_KEP),         y_kep, NGON_MO,   1.2),
            (f"C{i} kep",      len_(p_nguon, p_nguon[2] + Z_KEP),         y_kep, NGON_DONG, 0.6),
            (f"C{i} nhac_len", len_(p_nguon, Z_DI),                       y_kep, NGON_DONG, 1.2),
            (f"C{i} di_chuyen", len_(p_dich, Z_DI),                       y_kep, NGON_DONG, 2.0),
            (f"C{i} ha_dat",   len_(p_dich, p_dich[2] + Z_KEP + THA_CAO), y_kep, NGON_DONG, 1.8),
            (f"C{i} tha",      len_(p_dich, p_dich[2] + Z_KEP + THA_CAO), y_kep, NGON_MO,   0.6),
            (f"C{i} rut_len",  len_(p_dich, Z_DI),                        y_kep, NGON_MO,   1.2),
        ]
    return ke_hoach


def chay(xem=False, gif=False, seed=None):
    m = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("san_sang").id)
    qadr = [m.joint(m.body(c).jntadr[0]).qposadr[0] for c in TEN_COC]
    if seed is not None:
        # Rải 3 cốc ngẫu nhiên trong vùng tay với tới, cách nhau đủ xa để ngón không đụng cốc khác
        rng = np.random.default_rng(seed)
        da_dat = []
        while len(da_dat) < 3:
            r, goc = rng.uniform(0.28, 0.5), rng.uniform(-1.3, 0.1)
            p = np.array([r * np.cos(goc), r * np.sin(goc)])
            if all(np.linalg.norm(p - q) > 0.16 for q in da_dat):
                da_dat.append(p)
        for a, p in zip(qadr, da_dat):
            d.qpos[a:a + 2] = p
    mujoco.mj_forward(m, d)

    ik = GiaiIK(m)
    act_tay = np.array([m.actuator(j).id for j in KHOP_TAY])
    act_ngon = m.actuator("kep").id
    tcp = m.site("tcp").id

    # "Cảm biến": vị trí các cốc và tâm kim tự tháp
    nguon = [d.body(c).xpos.copy() for c in TEN_COC]
    tam = d.site_xpos[m.site("dich").id][:2].copy()
    # Hàng cốc dưới nằm theo trục x của thế giới; ngón kẹp đóng theo trục y (vuông góc với hàng)
    # -> khi đặt C2 sát C1, 2 ngón nằm 2 bên (trước/sau) chứ không chen vào giữa 2 cốc.
    huong_hang = np.array([1.0, 0.0, 0.0])
    y_kep = np.array([0.0, 1.0, 0.0])
    dich = vi_tri_kim_tu_thap(tam, huong_hang)
    for c, p in zip(TEN_COC, nguon):
        print(f"{c} ở {np.round(p[:2], 3)}")

    ke_hoach = lap_ke_hoach(nguon, dich, y_kep)

    q = d.qpos[ik.qadr].copy()
    p_truoc = d.site_xpos[tcp].copy()
    y_truoc = d.site_xmat[tcp].reshape(3, 3)[:, 1].copy()
    ngon_truoc = d.ctrl[act_ngon]
    buoc_dk = 10
    khung_hinh = []

    viewer = None
    if xem:
        import mujoco.viewer as mjv
        viewer = mjv.launch_passive(m, d)
    nhip = NhipXem(viewer)
    renderer = mujoco.Renderer(m, 270, 360) if gif else None
    # Nhìn từ phía sau kim tự tháp, vuông góc với hàng cốc -> thấy rõ hình tháp
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.2, 0.05, 0.1]
    cam.distance, cam.azimuth, cam.elevation = 1.3, -105, -25

    def buoc_mo_phong(n):
        for i in range(n):
            mujoco.mj_step(m, d)
            if not nhip(d):
                raise SystemExit(0)
            if renderer is not None and int(round(d.time / m.opt.timestep)) % 40 == 0:
                renderer.update_scene(d, cam)
                khung_hinh.append(renderer.render().copy())

    for ten, p_dich, y_dich, ngon, T in ke_hoach:
        n = int(T / m.opt.timestep)
        for i in range(0, n, buoc_dk):
            s = muot((i + buoc_dk) / n)
            q, _ = ik.giai(q, (1 - s) * p_truoc + s * p_dich, nlerp(y_truoc, y_dich, s))
            d.ctrl[act_tay] = q
            d.ctrl[act_ngon] = (1 - s) * ngon_truoc + s * ngon
            buoc_mo_phong(buoc_dk)
        p_truoc, y_truoc, ngon_truoc = p_dich, y_dich, ngon
        if ten.endswith(("di_chuyen", "rut_len")):
            k = int(ten[1]) - 1
            print(f"  {ten:<13} | cốc C{k+1} z={d.body(TEN_COC[k]).xpos[2]:.3f} m")

    # Tay đã rút lên cao; chờ 2 s: chồng cốc phải tự đứng vững
    print("Chờ 2 s kiểm tra độ vững...")
    buoc_mo_phong(int(2.0 / m.opt.timestep))

    dat = True
    print("\nKết quả:")
    for i, (c, p_mong) in enumerate(zip(TEN_COC, dich), start=1):
        p = d.body(c).xpos
        truc_z = d.body(c).xmat.reshape(3, 3)[:, 2]
        lech = np.linalg.norm(p - p_mong)
        nghieng = np.degrees(np.arccos(np.clip(truc_z[2], -1, 1)))
        ok = lech < 0.006 and nghieng < 3
        dat &= ok
        print(f"  C{i}: ở {np.round(p, 3)}, lệch {lech*1000:4.1f} mm, nghiêng {nghieng:4.1f} độ"
              f" -> {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
    print(f"Kim tự tháp: {'ĐẠT' if dat else 'KHÔNG ĐẠT'}")

    if renderer is not None:
        from PIL import Image
        anh = [Image.fromarray(k) for k in khung_hinh]
        duong_dan = THU_MUC / "xep_coc.gif"
        anh[0].save(duong_dan, save_all=True, append_images=anh[1:], duration=50, loop=0)
        print(f"Đã lưu {duong_dan}")
        # Ảnh cận cảnh kim tự tháp
        r2 = mujoco.Renderer(m, 480, 640)
        cam.lookat[:] = [*dich[2][:2], 0.07]
        cam.distance, cam.azimuth, cam.elevation = 0.45, -100, -12
        r2.update_scene(d, cam)
        Image.fromarray(r2.render()).save(THU_MUC / "kim_tu_thap.png")
        print(f"Đã lưu {THU_MUC / 'kim_tu_thap.png'}")
    if viewer is not None:
        viewer.close()
    return dat


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xem", action="store_true", help="mở viewer tương tác")
    ap.add_argument("--gif", action="store_true", help="xuất xep_coc.gif")
    ap.add_argument("--ngau-nhien", type=int, metavar="SEED", help="rải 3 cốc ngẫu nhiên")
    a = ap.parse_args()
    raise SystemExit(0 if chay(a.xem, a.gif, a.ngau_nhien) else 1)
