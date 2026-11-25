"""Bài: Xếp kim tự tháp 3 cốc với cánh tay thật AgileX PiPER (model từ MuJoCo Menagerie).

Cần model PiPER (MIT, ~32 MB, không đưa vào git):
    git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie bai_piper/mujoco_menagerie

Chạy:
    python bai_piper/xep_coc_piper.py                 # headless, in kết quả
    python bai_piper/xep_coc_piper.py --xem           # mở viewer
    python bai_piper/xep_coc_piper.py --ngau-nhien 3  # rải cốc ngẫu nhiên

Khác cánh tay tự dựng (bai_xep_coc):
  - Tay nhỏ: tầm với ngắn hơn -> cốc và tháp đặt gần đế hơn, độ cao di chuyển thấp hơn.
  - Tay kẹp mở tối đa 7 cm, đệm ngón dài 6 cm, lòng bàn tay chỉ cao hơn giữa đệm ngón 1.6 cm
    -> phải kẹp ở GẦN ĐỈNH cốc (TCP cao hơn vành 8.2 cm) để lòng bàn tay không đè lên đáy cốc.
  - Dùng lại nguyên IK (GiaiIK) và hình học kim tự tháp của các bài trước.
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
from gap_do import GiaiIK, NhipXem, muot, nlerp  # noqa: E402
from xep_coc import CAO, R_DAY, R_MIENG, TEN_COC, vi_tri_kim_tu_thap  # noqa: E402

PIPER_XML = THU_MUC / "mujoco_menagerie" / "agilex_piper" / "piper.xml"
KHOP = [f"joint{i}" for i in range(1, 7)]
Q_NHA = np.array([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0])   # keyframe "home" của Menagerie

# --- Thông số gắp cho tay kẹp PiPER ---
TCP_LINK6 = 0.105   # giữa đệm ngón (đệm từ 7.5 đến 13.5 cm dọc trục link6)
Z_KEP = 0.082       # TCP cao hơn vành miệng: lòng bàn tay (TCP + 1.6 cm) cách đáy cốc (9 cm) ~8 mm
Z_DI = 0.21         # TCP khi di chuyển: đáy cốc đang cầm (0.21-0.082) cao hơn đỉnh cốc khác (0.09)
THA_CAO = 0.003
NGON_MO = 0.035     # joint7: 0 = đóng, 0.035 = mở hết (khe 7 cm)
NGON_DONG = 0.0
TAM_THAP = (0.06, 0.30)
# PiPER chỉ chúc bàn tay thẳng xuống được khi TCP thấp hơn ~11 cm so với mặt đế (khớp gập cổ tay joint5
# chỉ ±70°). Xếp tháp cần TCP từ 8 cm (gắp) tới 21 cm (di chuyển) so với mặt bàn -> đặt tay lên BỆ 13 cm.
BE = 0.13
DEBUG = False
KI = 0.3            # hệ số tích phân mỗi chu kỳ điều khiển (20 ms)


def tao_model():
    if not PIPER_XML.exists():
        raise SystemExit("Thiếu model PiPER. Chạy:\n  git clone --depth 1 "
                         "https://github.com/google-deepmind/mujoco_menagerie bai_piper/mujoco_menagerie")
    spec = mujoco.MjSpec.from_file(str(PIPER_XML))
    spec.delete(spec.key("home"))          # keyframe chỉ có qpos của tay; thêm cốc thì không khớp số chiều
    spec.body("link6").add_site(name="tcp", pos=[0, 0, TCP_LINK6], size=[0.006, 0, 0], rgba=[1, 0, 0, 1], group=4)
    # Đệm ngón: ma sát cao + condim 4 (chống xoay cốc quanh trục kẹp), tiếp xúc cứng như các bài trước
    for ten in ("link7", "link8"):
        for g in spec.body(ten).geoms:
            if g.contype:
                g.friction = [1.5, 0.02, 0.001]
                g.condim, g.solref = 4, [0.004, 1]

    w = spec.worldbody
    spec.body("base_link").pos = [0, 0, BE]
    w.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.06, 0.06, BE / 2], pos=[0, 0, BE / 2], rgba=[.3, .3, .32, 1])
    tex = spec.add_texture(name="luoi", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[.85, .85, .85], rgb2=[.7, .7, .7], width=300, height=300)
    mat = spec.add_material(name="luoi", texrepeat=[8, 8])
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    w.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.05], material="luoi")
    w.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    w.add_site(name="dich", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[*TAM_THAP, 0.001], size=[0.09, 0.001, 0],
               rgba=[0, .8, 0, .3])

    # Cốc úp: hình nón cụt (vành r 3.5 cm ở dưới, đáy r 2.5 cm ở trên, cao 9 cm), như bai_xep_coc
    a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    dinh = np.concatenate([np.c_[R_MIENG * np.cos(a), R_MIENG * np.sin(a), np.zeros_like(a)],
                           np.c_[R_DAY * np.cos(a), R_DAY * np.sin(a), np.full_like(a, CAO)]])
    spec.add_mesh(name="coc", uservert=dinh.ravel().tolist())
    for ten, mau, xy in zip(TEN_COC, [[.9, .2, .2, 1], [.2, .5, .9, 1], [.95, .75, .1, 1]],
                            [(0.30, -0.18), (0.34, 0.02), (0.18, -0.28)]):
        b = w.add_body(name=ten, pos=[*xy, 0])
        b.add_freejoint()
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="coc", mass=0.03, rgba=mau,
                   friction=[1, 0.02, 0.001], condim=4, solref=[0.004, 1])
    return spec.compile()


def lap_ke_hoach(nguon, dich, y_kep):
    """8 bước mỗi cốc (như bai_xep_coc). Bước: (tên, p_tcp, y, ngón, T)."""
    kh = []
    o = lambda p, z: np.array([p[0], p[1], z])  # noqa: E731

    def vong(ten, a, b, ngon, v):
        """Đi từ trên a tới trên b, VÒNG qua điểm giữa trên cung tròn quanh đế (bán kính = điểm xa hơn).
        Đường thẳng giữa cốc phía dưới-phải và tháp phía trên đi sát đế (~8 cm): PiPER phải gập hết cỡ,
        cổ tay chạm giới hạn ±70° -> bàn tay nghiêng, cốc lắc tuột khỏi kẹp."""
        ga, gb = np.arctan2(a[1], a[0]), np.arctan2(b[1], b[0])
        r = max(np.linalg.norm(a[:2]), np.linalg.norm(b[:2]))
        g = (ga + gb) / 2
        giua = np.array([r * np.cos(g), r * np.sin(g), Z_DI])
        T = lambda p, q: max(1.0, np.linalg.norm(q[:2] - p[:2]) / v)  # noqa: E731
        return [(f"{ten}/1", giua, y_kep, ngon, T(a, giua)), (f"{ten}/2", o(b, Z_DI), y_kep, ngon, T(giua, b))]

    truoc = None
    for i, (pn, pd) in enumerate(zip(nguon, dich), start=1):
        kh += (vong(f"C{i} toi_tren", truoc, pn, NGON_MO, 0.25) if truoc is not None
               else [(f"C{i} toi_tren", o(pn, Z_DI), y_kep, NGON_MO, 1.5)])
        kh += [
            (f"C{i} ha_xuong", o(pn, pn[2] + Z_KEP),           y_kep, NGON_MO,   1.2),
            (f"C{i} kep",      o(pn, pn[2] + Z_KEP),           y_kep, NGON_DONG, 0.8),
            (f"C{i} nhac_len", o(pn, Z_DI),                    y_kep, NGON_DONG, 1.2),
        ]
        # chở chậm (~12 cm/s): kẹp gần đỉnh cốc (thành thuôn), lực kẹp PiPER chỉ 10 N -> tăng tốc mạnh là cốc trượt
        kh += vong(f"C{i} di_chuyen", pn, pd, NGON_DONG, 0.12)
        truoc = pd
        kh += [
            (f"C{i} ha_dat",   o(pd, pd[2] + Z_KEP + THA_CAO), y_kep, NGON_DONG, 1.8),
            (f"C{i} tha",      o(pd, pd[2] + Z_KEP + THA_CAO), y_kep, NGON_MO,   0.6),
            (f"C{i} rut_len",  o(pd, Z_DI),                    y_kep, NGON_MO,   1.2),
        ]
    return kh


def chay(xem=False, seed=None):
    m = tao_model()
    d = mujoco.MjData(m)
    ik = GiaiIK(m, KHOP)
    d.qpos[ik.qadr] = Q_NHA
    qadr = [m.joint(m.body(c).jntadr[0]).qposadr[0] for c in TEN_COC]
    if seed is not None:
        rng = np.random.default_rng(seed)
        da_dat, lan = [], 0
        while len(da_dat) < 3:
            lan += 1
            if lan % 200 == 0:          # vùng với tới nhỏ: 2 cốc đầu có thể chiếm hết chỗ -> rải lại từ đầu
                da_dat = []
            r, g = rng.uniform(0.22, 0.34), rng.uniform(-1.2, 0.2)
            p = np.array([r * np.cos(g), r * np.sin(g)])
            if all(np.linalg.norm(p - q) > 0.14 for q in da_dat) and np.linalg.norm(p - TAM_THAP) > 0.15:
                da_dat.append(p)
        for a, p in zip(qadr, da_dat):
            d.qpos[a:a + 2] = p
    mujoco.mj_forward(m, d)

    act_tay = np.array([m.actuator(j).id for j in KHOP])
    act_ngon = m.actuator("gripper").id
    d.ctrl[act_tay] = Q_NHA
    tcp = m.site("tcp").id

    nguon = [d.body(c).xpos.copy() for c in TEN_COC]
    dich = vi_tri_kim_tu_thap(np.array(TAM_THAP), np.array([1.0, 0.0, 0.0]))
    y_kep = np.array([0.0, 1.0, 0.0])
    for c, p in zip(TEN_COC, nguon):
        print(f"{c} ở {np.round(p[:2], 3)}")

    viewer = None
    if xem:
        import mujoco.viewer as mjv
        viewer = mjv.launch_passive(m, d)
        viewer.cam.lookat[:] = [0.2, 0.0, 0.1]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.0, -150, -30
    nhip = NhipXem(viewer)

    def buoc_mo_phong(n):
        for _ in range(n):
            mujoco.mj_step(m, d)
            if not nhip(d):
                raise SystemExit(0)

    buoc_mo_phong(200)
    # Tới tư thế bắt đầu (trên cốc đầu tiên) bằng KHÔNG GIAN KHỚP: IK đi thẳng từ "home" (tay duỗi ngang)
    # dễ kẹt ở giới hạn khớp. Thử nhiều nghiệm khởi đầu, lấy nghiệm tốt nhất.
    rng0 = np.random.default_rng(0)
    p_dau = np.array([nguon[0][0], nguon[0][1], Z_DI])
    q = min((ik.giai(ik.lo + (ik.hi - ik.lo) * rng0.random(6), p_dau, y_kep, so_vong=200) for _ in range(30)),
            key=lambda k: k[1])[0]
    n = int(2.0 / m.opt.timestep)
    for i in range(0, n, 10):
        d.ctrl[act_tay] = Q_NHA + muot((i + 10) / n) * (q - Q_NHA)
        d.ctrl[act_ngon] = NGON_MO
        buoc_mo_phong(10)
    buoc_mo_phong(200)
    p0 = d.site_xpos[tcp].copy()
    y0 = d.site_xmat[tcp].reshape(3, 3)[:, 1].copy()
    n0 = NGON_MO
    sai_max = 0.0
    # Khâu tích phân trên lệnh khớp (như bộ PI trong firmware servo của tay thật): actuator của Menagerie
    # mềm (kp 10 ở cổ tay) + ma sát khớp -> cổ tay đứng lệch ~3°, đầu tay lệch ~9 mm, mà khe kẹp chỉ dư 5 mm.
    tich = np.zeros(6)
    for ten, p1, y1, n1, T in lap_ke_hoach(nguon, dich, y_kep):
        n = int(T / m.opt.timestep)
        for i in range(0, n, 10):
            s = muot((i + 10) / n)
            q, sai = ik.giai(q, (1 - s) * p0 + s * p1, nlerp(y0, y1, s))
            tich = np.clip(tich + KI * (q - d.qpos[ik.qadr]), -0.15, 0.15)
            d.ctrl[act_tay] = q + tich
            d.ctrl[act_ngon] = (1 - s) * n0 + s * n1
            buoc_mo_phong(10)
        sai_max = max(sai_max, sai)
        if DEBUG:
            print(f"{ten:<13} lech tcp {np.linalg.norm(d.site_xpos[tcp] - p1) * 1000:5.1f} mm  "
                  f"q_err {np.round(np.degrees(d.qpos[ik.qadr] - q), 1)}  ngon {d.qpos[m.joint('joint7').qposadr[0]] * 1000:.1f}")
        p0, y0, n0 = p1, y1, n1
    print(f"Sai số IK lớn nhất: {sai_max * 1000:.2f} mm")

    buoc_mo_phong(int(2.0 / m.opt.timestep))
    p1, p2, p3 = (d.body(c).xpos for c in TEN_COC)
    dat = True
    print("\nKết quả:")
    for i, (c, p_mong) in enumerate(zip(TEN_COC, [dich[0], dich[1], (p1 + p2) / 2 + [0, 0, CAO]]), start=1):
        b = d.body(c)
        lech = np.linalg.norm(b.xpos - p_mong)
        ng = np.degrees(np.arccos(np.clip(b.xmat[8], -1, 1)))
        ok = lech < 0.008 and ng < 3
        dat &= ok
        print(f"  C{i}: lệch {lech * 1000:4.1f} mm, nghiêng {ng:4.1f}° -> {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
    print(f"Kim tự tháp: {'ĐẠT' if dat else 'KHÔNG ĐẠT'}")
    if viewer is not None:
        viewer.close()
    return dat


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xem", action="store_true", help="mở viewer tương tác")
    ap.add_argument("--ngau-nhien", type=int, metavar="SEED", help="rải 3 cốc ngẫu nhiên")
    a = ap.parse_args()
    raise SystemExit(0 if chay(a.xem, a.ngau_nhien) else 1)
