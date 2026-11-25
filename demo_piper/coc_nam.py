"""Bài: Dựng cốc nằm đổ rồi xếp kim tự tháp (chỉ cánh tay 6DOF làm được).

Cốc C1 bị đổ, nằm ngang trên sàn. Robot:
  1. Gắp cốc từ trên xuống, 2 ngón kẹp ngang thân cốc.
  2. Nhấc lên, XOAY bàn tay 90°: tay chuyển từ "chúc xuống" sang "nằm ngang" -> cốc dựng đứng.
  3. Hạ cốc xuống sàn (úp, như 2 cốc kia), thả, rút tay ra theo phương ngang.
  4. Xếp kim tự tháp như bài xep_coc.

Vì sao cần 6DOF: bước 2 đòi bàn tay đổi hướng trục z (xuống -> ngang) trong khi vẫn giữ hướng ngón
kẹp. Tay 5DOF chỉ điều khiển được z chúc xuống + 1 góc xoay, không làm được tư thế kẹp ngang tùy ý.

Mẹo hình học: kẹp cốc nằm tại điểm trên trục cốc cách vành miệng Z_KEP (4.5 cm) -> sau khi dựng,
TCP nằm đúng chỗ kẹp chuẩn của cốc đứng (trên trục, cao Z_KEP), dùng lại được thông số bài trước.

Chạy:
    python coc_nam.py                 # headless, in kết quả
    python coc_nam.py --xem           # mở viewer
    python coc_nam.py --ngau-nhien 3  # cốc đổ ở vị trí/hướng ngẫu nhiên
"""

import argparse
import pathlib
import sys
import time

import mujoco
import numpy as np

THU_MUC = pathlib.Path(__file__).parent
sys.path.insert(0, str(THU_MUC))
sys.path.insert(0, str(THU_MUC))
from gap_do import KHOP_TAY, GiaiIK, NhipXem, muot, nlerp  # noqa: E402
from xep_coc import (MODEL_XML, NGON_MO, NGON_DONG, TEN_COC, Z_KEP, Z_DI,  # noqa: E402
                     lap_ke_hoach, vi_tri_kim_tu_thap)

EZ = np.array([0.0, 0.0, 1.0])
XUONG = -EZ
Z_XOAY = 0.22      # độ cao TCP khi xoay cốc (cốc dài 9 cm quay quanh TCP -> không quét sàn)


def dat_coc_nam(m, d, ten, xy, goc_truc):
    """Đặt cốc nằm ngang: trục cốc (vành miệng -> đáy) theo hướng goc_truc trên mặt sàn."""
    a = np.array([np.cos(goc_truc), np.sin(goc_truc), 0.0])
    k = np.cross(EZ, a)                       # quay trục z của cốc 90° quanh k -> trùng a
    adr = m.joint(m.body(ten).jntadr[0]).qposadr[0]
    d.qpos[adr:adr + 3] = [xy[0], xy[1], 0.037]
    d.qpos[adr + 3:adr + 7] = [np.cos(np.pi / 4), *(np.sin(np.pi / 4) * k)]


def ke_hoach_dung_coc(p_mieng, truc):
    """Các bước dựng cốc. Bước: (tên, p_tcp, y, z, ngón, T)."""
    a = truc.copy()
    a[2] = 0.0
    a /= np.linalg.norm(a)                      # hướng trục cốc trên mặt sàn
    p_kep = p_mieng + Z_KEP * truc              # điểm trên trục cách vành Z_KEP
    p_kep[2] += 0.003
    # Lúc gắp: z tay xuống, x tay = ±a (dọc trục cốc), y = z × x (ngang thân cốc).
    # Lúc đặt: x tay = ±ẑ (đáy cốc lên trên), z tay nằm ngang hướng ra xa đế robot (tầm với tự nhiên).
    r = np.array([p_kep[0], p_kep[1], 0.0])
    r /= np.linalg.norm(r)
    tot = None
    for s in (1, -1):
        x0, z0 = s * a, XUONG
        y0 = np.cross(z0, x0)
        x1, z1 = s * EZ, r
        y1 = np.cross(z1, x1)
        R0, R1 = np.column_stack([x0, y0, z0]), np.column_stack([x1, y1, z1])
        goc = np.arccos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1))
        if tot is None or goc < tot[0]:
            tot = (goc, y0, y1, z1)
    _, y0, y1, z1 = tot
    tren = np.array([p_kep[0], p_kep[1], Z_DI])
    xoay = np.array([p_kep[0], p_kep[1], Z_XOAY])
    dat = np.array([p_kep[0], p_kep[1], Z_KEP + 0.004])
    return [
        ("dung toi_tren", tren,    y0, XUONG, NGON_MO,   1.5),
        ("dung ha_xuong", p_kep,   y0, XUONG, NGON_MO,   1.2),
        ("dung kep",      p_kep,   y0, XUONG, NGON_DONG, 0.6),
        ("dung nhac_len", xoay,    y0, XUONG, NGON_DONG, 1.2),
        ("dung xoay",     xoay,    y1, z1,    NGON_DONG, 2.0),
        ("dung ha_dat",   dat,     y1, z1,    NGON_DONG, 1.5),
        ("dung tha",      dat,     y1, z1,    NGON_MO,   0.6),
        # ngón đã mở, đang ôm 2 bên thân cốc -> rút thẳng lên (rút ngang về phía đế dễ đụng thân robot)
        ("dung rut_len",  tren,    y1, z1,    NGON_MO,   1.2),
    ]


def huong(y, z):
    """Trực chuẩn hoá: z đơn vị, y vuông góc với z."""
    z = z / np.linalg.norm(z)
    y = y - (y @ z) * z
    return y / np.linalg.norm(y), z


def chay(xem=False, seed=None):
    m = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("san_sang").id)
    tam = m.site("dich").pos[:2].copy()
    xy, goc = np.array([0.40, -0.22]), -0.4       # mặc định: C1 đổ tại chỗ cũ
    if seed is not None:
        rng = np.random.default_rng(seed)
        khac = [m.body(c).pos[:2] for c in TEN_COC[1:]] + [tam]
        while True:
            r, g = rng.uniform(0.32, 0.45), rng.uniform(-1.2, 0.0)
            xy = np.array([r * np.cos(g), r * np.sin(g)])
            if all(np.linalg.norm(xy - q) > 0.19 for q in khac):
                break
        goc = rng.uniform(-np.pi, np.pi)
    dat_coc_nam(m, d, "coc1", xy, goc)
    mujoco.mj_forward(m, d)
    for _ in range(int(0.5 / m.opt.timestep)):     # cho cốc nằm yên
        mujoco.mj_step(m, d)

    ik = GiaiIK(m)
    act_tay = np.array([m.actuator(j).id for j in KHOP_TAY])
    act_ngon = m.actuator("kep").id
    tcp = m.site("tcp").id

    c1 = d.body("coc1")
    truc = c1.xmat.reshape(3, 3)[:, 2].copy()
    print(f"C1 nằm đổ tại {np.round(c1.xpos[:2], 3)}, trục nghiêng {np.degrees(np.arcsin(abs(truc[2]))):.1f}° so với sàn")
    buoc = ke_hoach_dung_coc(c1.xpos.copy(), truc)

    viewer = None
    if xem:
        import mujoco.viewer as mjv
        viewer = mjv.launch_passive(m, d)
    nhip = NhipXem(viewer)

    def buoc_mo_phong(n):
        for _ in range(n):
            mujoco.mj_step(m, d)
            if not nhip(d):
                raise SystemExit(0)

    q = d.qpos[ik.qadr].copy()
    R = d.site_xmat[tcp].reshape(3, 3)
    trang_thai = [d.site_xpos[tcp].copy(), R[:, 1].copy(), R[:, 2].copy(), d.ctrl[act_ngon]]

    def thuc_hien(ds):
        nonlocal q
        p0, y0, z0, n0 = trang_thai
        for ten, p1, y1, z1, n1, T in ds:
            n = int(T / m.opt.timestep)
            for i in range(0, n, 10):
                s = muot((i + 10) / n)
                y, z = huong(nlerp(y0, y1, s), nlerp(z0, z1, s))
                q, _ = ik.giai(q, (1 - s) * p0 + s * p1, y, z_dich=z)
                d.ctrl[act_tay] = q
                d.ctrl[act_ngon] = (1 - s) * n0 + s * n1
                buoc_mo_phong(10)
            p0, y0, z0, n0 = p1, y1, z1, n1
        trang_thai[:] = [p0, y0, z0, n0]

    thuc_hien(buoc)
    nghieng = np.degrees(np.arccos(np.clip(c1.xmat[8], -1, 1)))
    print(f"Dựng cốc: C1 giờ ở {np.round(c1.xpos, 3)}, nghiêng {nghieng:.1f}° "
          f"-> {'ĐỨNG' if nghieng < 3 else 'CHƯA ĐỨNG'}")
    if nghieng >= 3:
        return False

    # Về tư thế sẵn sàng bằng không gian khớp: sau tư thế kẹp ngang, cổ tay đang ở nhánh nghiệm IK
    # khác hẳn; nội suy Đề-các từ đây sang gắp từ trên làm IK kẹt -> đặt cốc lệch, quệt đổ cốc.
    q_san_sang = m.key("san_sang").qpos[ik.qadr]
    q_dau, n = q.copy(), int(1.5 / m.opt.timestep)
    for i in range(0, n, 10):
        q = q_dau + muot((i + 10) / n) * (q_san_sang - q_dau)
        d.ctrl[act_tay] = q
        buoc_mo_phong(10)
    buoc_mo_phong(100)
    R = d.site_xmat[tcp].reshape(3, 3)
    trang_thai[:3] = [d.site_xpos[tcp].copy(), R[:, 1].copy(), R[:, 2].copy()]

    # Xếp kim tự tháp như bài xep_coc, vị trí cốc lấy từ mô phỏng (cốc C1 vừa dựng)
    nguon = [d.body(c).xpos.copy() for c in TEN_COC]
    y_kep = np.array([0.0, 1.0, 0.0])
    dich = vi_tri_kim_tu_thap(tam, np.array([1.0, 0.0, 0.0]))
    thuc_hien([(t, p, y, XUONG, n, T) for t, p, y, n, T in lap_ke_hoach(nguon, dich, y_kep)])
    buoc_mo_phong(int(2.0 / m.opt.timestep))

    dat = True
    print("\nKết quả:")
    for i, (c, p_mong) in enumerate(zip(TEN_COC, dich), start=1):
        b = d.body(c)
        lech = np.linalg.norm(b.xpos - p_mong)
        ng = np.degrees(np.arccos(np.clip(b.xmat[8], -1, 1)))
        ok = lech < 0.006 and ng < 3
        dat &= ok
        print(f"  C{i}: lệch {lech * 1000:4.1f} mm, nghiêng {ng:4.1f}° -> {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
    print(f"Kim tự tháp: {'ĐẠT' if dat else 'KHÔNG ĐẠT'}")
    if viewer is not None:
        viewer.close()
    return dat


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xem", action="store_true", help="mở viewer tương tác")
    ap.add_argument("--ngau-nhien", type=int, metavar="SEED", help="cốc đổ ở vị trí/hướng ngẫu nhiên")
    a = ap.parse_args()
    raise SystemExit(0 if chay(a.xem, a.ngau_nhien) else 1)
