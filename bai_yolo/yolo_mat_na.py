"""Bộ tạo mặt nạ bằng YOLO-seg, thay cho lọc màu HSV trong bai_camera/nhan_dang.py.

Dùng:
    import yolo_mat_na
    yolo_mat_na.bat_yolo()          # từ giờ nhan_dang.mat_na() chạy YOLO
Đầu vào / đầu ra giống mat_na_hsv: ảnh color H x W x 3 -> {coc1, coc2, coc3, marker}: mảng bool H x W.
Mọi bước sau (chiếu 3D, cốc đứng / nằm, gom cụm marker) không đổi.
"""

import pathlib
import sys

import numpy as np

THU_MUC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(THU_MUC.parent / "bai_camera"))
import nhan_dang  # noqa: E402

LOP = ["coc1", "coc2", "coc3", "marker"]       # khớp thứ tự lớp lúc train (sinh_du_lieu.LOP)
NGUONG_TIN_CAY = 0.5
MODEL_MAC_DINH = THU_MUC / "yolo_coc.pt"


class MatNaYolo:
    def __init__(self, duong_dan=MODEL_MAC_DINH):
        import torch
        from ultralytics import YOLO
        self.model = YOLO(str(duong_dan))
        self.thiet_bi = 0 if torch.cuda.is_available() else "cpu"

    def __call__(self, color):
        H, W = color.shape[:2]
        kq = {ten: np.zeros((H, W), bool) for ten in LOP}
        # ultralytics nhận ảnh numpy theo thứ tự BGR (kiểu OpenCV); camera trả RGB -> đảo kênh
        r = self.model.predict(color[..., ::-1], imgsz=640, conf=NGUONG_TIN_CAY, device=self.thiet_bi,
                               retina_masks=True, verbose=False)[0]
        if r.masks is None:
            return kq
        mn = r.masks.data.cpu().numpy() > 0.5          # N x H x W (retina_masks: đúng cỡ ảnh gốc)
        for k, m in zip(r.boxes.cls.cpu().numpy().astype(int), mn):
            kq[LOP[k]] |= m                             # nhiều vùng cùng lớp (2 marker) -> gộp, như HSV
        return kq


def bat_yolo(duong_dan=MODEL_MAC_DINH):
    """Chuyển nhan_dang sang dùng YOLO cho mọi lần nhìn sau đó."""
    nhan_dang.BO_MAT_NA = MatNaYolo(duong_dan)
    print(f"Nhận dạng: YOLO ({pathlib.Path(duong_dan).name})")
