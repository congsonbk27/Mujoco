"""Sinh ảnh tổng hợp + nhãn phân đoạn (YOLO-seg) từ MuJoCo để train YOLO nhận dạng cốc và marker.

Mỗi ảnh: rải 3 cốc ngẫu nhiên (đứng hoặc nằm đổ), đặt cánh tay ở tư thế ngẫu nhiên, đổi ánh sáng /
màu nền / sắc cốc một chút, rồi chụp bằng đúng camera mà robot dùng. Nhãn lấy từ ảnh phân đoạn của
MuJoCo (mỗi điểm ảnh biết thuộc geom nào) nên đúng tới từng điểm ảnh, không phải vẽ tay.

Ba nguồn ảnh (tỉ lệ ~ 2 : 2 : 1):
    tay 6DOF tự dựng + camera trên cao
    PiPER + camera trên cao
    PiPER + camera cổ tay (bai_piper/xep_coc_piper_cotay.py)

Chạy:
    MUJOCO_GL=egl python bai_yolo/sinh_du_lieu.py --so-anh 2000
Kết quả: bai_yolo/du_lieu/{images,labels}/{train,val}/*.png|txt và bai_yolo/du_lieu/du_lieu.yaml
"""

import argparse
import pathlib
import sys
from multiprocessing import Pool

import mujoco
import numpy as np

THU_MUC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(THU_MUC.parent / "bai_camera"))
sys.path.insert(0, str(THU_MUC.parent / "bai_coc_nam"))
sys.path.insert(0, str(THU_MUC.parent / "bai_piper"))

LOP = ["coc1", "coc2", "coc3", "marker"]          # thứ tự = chỉ số lớp YOLO
DIEN_TICH_TOI_THIEU = 12                          # điểm ảnh: mảnh nhỏ hơn thì bỏ (bị che gần hết)
NGUON = ["6dof", "piper", "piper_cotay"]


def tao_canh(nguon):
    """Model + camera + danh sách khớp tay cho từng nguồn ảnh."""
    import xep_coc_camera as X
    from realsense import CameraRealSense
    if nguon == "6dof":
        m = X.tao_model(nang_cap=True)
        khop = X.KHOP_TAY
        ten_cam = "realsense"
    else:
        import xep_coc_piper_camera as C
        m = C.tao_model()
        khop = C.P.KHOP
        ten_cam = "realsense" if nguon == "piper" else "realsense_cotay"
    return m, CameraRealSense(m, ten_cam), khop


def geom_lop(m):
    """geom id -> chỉ số lớp: cốc theo tên BODY coc1..3 (geom cốc của PiPER không đặt tên),
    marker theo tên geom marker_*."""
    lop = {}
    for g in range(m.ngeom):
        than = m.body(m.geom_bodyid[g]).name
        if than in LOP[:3]:
            lop[g] = LOP.index(than)
        elif m.geom(g).name.startswith("marker"):
            lop[g] = 3
    return lop


def rai_coc(m, d, rng, nguon):
    """Đặt 3 cốc ngẫu nhiên, mỗi cốc 30% nằm đổ."""
    from coc_nam import dat_coc_nam
    r0, r1 = (0.28, 0.5) if nguon == "6dof" else (0.18, 0.36)
    da_dat = []
    for c in LOP[:3]:
        for _ in range(200):
            r, g = rng.uniform(r0, r1), rng.uniform(-1.6, 1.6)
            p = np.array([r * np.cos(g), r * np.sin(g)])
            if all(np.linalg.norm(p - q) > 0.12 for q in da_dat):
                break
        da_dat.append(p)
        a = m.joint(m.body(c).jntadr[0]).qposadr[0]
        if rng.random() < 0.3:
            dat_coc_nam(m, d, c, p, rng.uniform(-np.pi, np.pi))
        else:
            d.qpos[a:a + 7] = [p[0], p[1], 0.0, 1, 0, 0, 0]


def dat_tay(m, d, rng, khop, nguon, ik):
    """Tư thế tay. Camera cổ tay: luôn chúc xuống nhìn một chỗ gần cốc. Camera trên cao: một nửa số ảnh
    tay chúc xuống ở chỗ ngẫu nhiên (lúc tự hiệu chỉnh / chỉnh tay, marker ở mu bàn tay phải nhìn thấy),
    nửa còn lại khớp ngẫu nhiên trong 60% giới hạn (tay che cốc / marker một phần)."""
    qadr = [m.joint(j).qposadr[0] for j in khop]
    if nguon == "piper_cotay" or rng.random() < 0.5:
        if nguon == "piper_cotay":
            xy = d.body(LOP[int(rng.integers(3))]).xpos[:2] + rng.normal(0, 0.04, 2)
            z = rng.uniform(0.18, 0.3)
        else:
            r0, r1 = (0.25, 0.5) if nguon == "6dof" else (0.18, 0.34)
            r, g = rng.uniform(r0, r1), rng.uniform(-1.6, 1.6)
            xy, z = np.array([r * np.cos(g), r * np.sin(g)]), rng.uniform(0.1, 0.45 if nguon == "6dof" else 0.3)
        goc = np.arctan2(xy[1], xy[0]) + rng.normal(0, 0.3)
        y = np.array([-np.sin(goc), np.cos(goc), 0.0])
        q = min((ik.giai(ik.lo + (ik.hi - ik.lo) * rng.random(6), np.array([*xy, z]), y, so_vong=150)
                 for _ in range(8)), key=lambda k: k[1])[0]
        d.qpos[qadr] = q
        return
    for a, j in zip(qadr, khop):
        lo, hi = m.jnt_range[m.joint(j).id]
        giua, nua = (lo + hi) / 2, (hi - lo) / 2 * 0.6
        d.qpos[a] = rng.uniform(giua - nua, giua + nua)


def doi_hinh_thuc(m, rng, goc):
    """Đổi ánh sáng, sắc cốc / marker (chỉ một chút: lớp cốc vẫn phân biệt bằng màu), màu sàn."""
    m.light_diffuse[:] = goc["den"] * rng.uniform(0.6, 1.3)
    m.light_pos[:] = goc["vi_tri_den"] + rng.normal(0, 0.3, goc["vi_tri_den"].shape)
    m.geom_rgba[:] = goc["rgba"]
    m.mat_rgba[:] = goc["mat"]
    for g in goc["geom_mau"]:
        if m.geom_matid[g] >= 0:
            continue                  # geom có material: đổi geom_rgba khỏi mặc định là đè mất màu material
        m.geom_rgba[g, :3] = np.clip(m.geom_rgba[g, :3] * rng.uniform(0.85, 1.15, 3), 0, 1)
    for k in range(m.nmat):
        m.mat_rgba[k, :3] = np.clip(m.mat_rgba[k, :3] * rng.uniform(0.85, 1.15, 3), 0, 1)


def mat_na_sang_da_giac(mask):
    """Mặt nạ bool -> list đa giác (mỗi vùng liền một đa giác), toạ độ điểm ảnh."""
    import cv2
    vien, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [v.reshape(-1, 2) for v in vien if cv2.contourArea(v) >= DIEN_TICH_TOI_THIEU and len(v) >= 3]


def sinh_mot_phan(viec):
    nguon, chi_so, ra, seed = viec
    from PIL import Image
    m, cam, khop = tao_canh(nguon)
    d = mujoco.MjData(m)
    r_seg = mujoco.Renderer(m, cam.H, cam.W)
    r_seg.enable_segmentation_rendering()
    lop = geom_lop(m)
    goc = {"den": m.light_diffuse.copy(), "vi_tri_den": m.light_pos.copy(), "rgba": m.geom_rgba.copy(),
           "mat": m.mat_rgba.copy(), "geom_mau": list(lop)}
    from gap_do import GiaiIK
    ik = GiaiIK(m, khop)
    rng = np.random.default_rng(seed)
    for i in chi_so:
        mujoco.mj_resetData(m, d)
        rai_coc(m, d, rng, nguon)
        dat_tay(m, d, rng, khop, nguon, ik)
        doi_hinh_thuc(m, rng, goc)
        mujoco.mj_forward(m, d)
        color, _ = cam.chup(d)
        r_seg.update_scene(d, cam.cam, cam.tuy_chon)
        seg = r_seg.render()
        geom_id = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)
        dong = []
        for g, k in lop.items():
            for dg in mat_na_sang_da_giac(geom_id == g):
                toa_do = (dg / [cam.W, cam.H]).ravel()
                dong.append(f"{k} " + " ".join(f"{x:.5f}" for x in toa_do))
        tap = "val" if i % 10 == 0 else "train"
        ten = f"{nguon}_{i:05d}"
        Image.fromarray(color).save(ra / "images" / tap / f"{ten}.png")
        (ra / "labels" / tap / f"{ten}.txt").write_text("\n".join(dong) + ("\n" if dong else ""))
    return len(chi_so)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so-anh", type=int, default=2000)
    ap.add_argument("--ra", default=str(THU_MUC / "du_lieu"))
    ap.add_argument("--so-tien-trinh", type=int, default=6)   # >6 hết bộ nhớ GPU render (đã gặp)
    ap.add_argument("--seed", type=int, default=1000, help="đổi seed để sinh tập thử MỚI (khác tập train)")
    a = ap.parse_args()
    ra = pathlib.Path(a.ra)
    for x in ("images", "labels"):
        for t in ("train", "val"):
            (ra / x / t).mkdir(parents=True, exist_ok=True)
    ti_le = {"6dof": 0.4, "piper": 0.4, "piper_cotay": 0.2}
    viec, k = [], 0
    for nguon in NGUON:
        n = int(a.so_anh * ti_le[nguon])
        for phan in np.array_split(np.arange(n), a.so_tien_trinh):
            viec.append((nguon, phan.tolist(), ra, a.seed + k))
            k += 1
    with Pool(a.so_tien_trinh) as p:
        xong = sum(p.map(sinh_mot_phan, viec))
    (ra / "du_lieu.yaml").write_text(
        f"path: {ra}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {t}\n" for i, t in enumerate(LOP)))
    print(f"Đã sinh {xong} ảnh vào {ra}")


if __name__ == "__main__":
    main()
