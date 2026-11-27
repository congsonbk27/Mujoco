"""So mặt nạ HSV và YOLO với nhãn thật trên một tập ảnh MỚI (seed khác lúc train).

Chạy:
    MUJOCO_GL=egl python bai_yolo/sinh_du_lieu.py --so-anh 300 --seed 9000 --ra bai_yolo/du_lieu_thu
    python bai_yolo/danh_gia.py --du-lieu bai_yolo/du_lieu_thu
In ra, theo từng nguồn ảnh và từng lớp: IoU trung bình, tỉ lệ vật bị bỏ sót (IoU < 0.3), thời gian mỗi ảnh.
"""

import argparse
import pathlib
import time

import numpy as np
from PIL import Image, ImageDraw

import yolo_mat_na
from yolo_mat_na import LOP, nhan_dang


def doc_nhan(file_nhan, W, H):
    """Nhãn YOLO-seg -> {lớp: mặt nạ bool} (các vùng cùng lớp gộp lại, như mặt nạ của robot)."""
    kq = {}
    for dong in file_nhan.read_text().split("\n"):
        if not dong:
            continue
        v = dong.split()
        p = np.array(v[1:], float).reshape(-1, 2) * [W, H]
        anh = Image.new("L", (W, H))
        ImageDraw.Draw(anh).polygon([tuple(x) for x in p], fill=1)
        ten = LOP[int(v[0])]
        kq[ten] = kq.get(ten, np.zeros((H, W), bool)) | (np.asarray(anh) > 0)
    return kq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--du-lieu", required=True)
    ap.add_argument("--model", default=str(yolo_mat_na.MODEL_MAC_DINH))
    a = ap.parse_args()
    anh = sorted((pathlib.Path(a.du_lieu) / "images").glob("*/*.png"))
    bo = {"HSV": nhan_dang.mat_na_hsv, "YOLO": yolo_mat_na.MatNaYolo(a.model)}
    bo["YOLO"](np.zeros((480, 640, 3), np.uint8))          # chạy nháp: nạp model / khởi động GPU
    iou, gio = {}, {k: [] for k in bo}
    for f in anh:
        color = np.asarray(Image.open(f).convert("RGB"))
        H, W = color.shape[:2]
        that = doc_nhan(f.parents[2] / "labels" / f.parent.name / (f.stem + ".txt"), W, H)
        nguon = f.stem.rsplit("_", 1)[0]
        for ten_bo, ham in bo.items():
            t0 = time.perf_counter()
            mn = ham(color)
            gio[ten_bo].append(time.perf_counter() - t0)
            for ten, m_that in that.items():
                if m_that.sum() < 30:                       # vật bị che gần hết: bỏ qua
                    continue
                giao, hop = (mn[ten] & m_that).sum(), (mn[ten] | m_that).sum()
                iou.setdefault((nguon, ten, ten_bo), []).append(giao / hop)
    print(f"{len(anh)} ảnh\n")
    print(f"{'nguồn':<12}{'lớp':<8}{'IoU HSV':>9}{'IoU YOLO':>10}{'sót HSV':>9}{'sót YOLO':>10}{'n':>5}")
    for nguon in sorted({k[0] for k in iou}):
        for ten in LOP:
            h, y = iou.get((nguon, ten, "HSV")), iou.get((nguon, ten, "YOLO"))
            if not h:
                continue
            h, y = np.array(h), np.array(y)
            print(f"{nguon:<12}{ten:<8}{h.mean():9.3f}{y.mean():10.3f}"
                  f"{(h < 0.3).mean() * 100:8.0f}%{(y < 0.3).mean() * 100:9.0f}%{len(h):5d}")
    for k, v in gio.items():
        print(f"\nThời gian {k}: {np.median(v) * 1000:.1f} ms / ảnh (trung vị)", end="")
    print()


if __name__ == "__main__":
    main()
