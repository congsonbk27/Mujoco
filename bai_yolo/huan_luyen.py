"""Train YOLOv8n-seg nhận dạng 3 cốc + marker trên ảnh tổng hợp (sinh bằng sinh_du_lieu.py).

Chạy:
    python bai_yolo/huan_luyen.py [--epoch 50] [--batch 16]
Có GPU NVIDIA (PyTorch bản CUDA) thì tự dùng GPU, không có thì chạy CPU (chậm hơn ~5-10 lần).
Model tốt nhất được chép ra bai_yolo/yolo_coc.pt.
"""

import argparse
import pathlib
import shutil

THU_MUC = pathlib.Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)     # RTX 3050 Ti 4 GB: 16 vừa với imgsz 640
    ap.add_argument("--du-lieu", default=str(THU_MUC / "du_lieu" / "du_lieu.yaml"))
    a = ap.parse_args()

    import torch
    from ultralytics import YOLO
    thiet_bi = 0 if torch.cuda.is_available() else "cpu"
    print(f"Train trên {'GPU ' + torch.cuda.get_device_name(0) if thiet_bi == 0 else 'CPU'}")
    model = YOLO("yolov8n-seg.pt")                       # bản nhỏ nhất, pretrained COCO
    kq = model.train(
        data=a.du_lieu, epochs=a.epoch, imgsz=640, batch=a.batch, device=thiet_bi,
        project=str(THU_MUC / "runs"), name="coc", exist_ok=True,
        # Không lật ảnh / không đổi màu mạnh: lớp cốc phân biệt bằng MÀU (đỏ / xanh / vàng),
        # đổi hue là đổi luôn lớp. Lật trái-phải vô hại.
        hsv_h=0.0, hsv_s=0.3, hsv_v=0.3, fliplr=0.5, flipud=0.0, mosaic=1.0,
        patience=15, verbose=False, plots=False)
    tot = pathlib.Path(kq.save_dir) / "weights" / "best.pt"
    shutil.copy(tot, THU_MUC / "yolo_coc.pt")
    print(f"Đã lưu {THU_MUC / 'yolo_coc.pt'}")


if __name__ == "__main__":
    main()
