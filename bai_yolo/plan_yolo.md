# Kế hoạch: nhận dạng cốc và marker bằng YOLO

Nhánh: `yolo-nhan-dien`.

**Mục tiêu:** thay bộ lọc màu HSV bằng mạng YOLO-seg để tìm 3 cốc và marker trên bàn tay. Hệ thống phải vẫn xếp tháp đạt như bản HSV, và sau này chịu được ánh sáng và màu sắc thật tốt hơn khi chuyển sang robot thật.

## 1. Luồng code hiện tại

Robot "nhìn" như sau. Phần đổi trong kế hoạch này là ô **[mặt nạ]**.

```
ảnh color + depth  (realsense.CameraRealSense.chup)
        │
        ▼
[mặt nạ]  nhan_dang.mat_na(color) -> {coc1, coc2, coc3, marker}: mảng bool H x W
        │      mặc định = mat_na_hsv (lọc màu); bản YOLO = mạng YOLO-seg, cùng đầu vào / đầu ra
        │
        ├── cốc   : nhan_dang.tim_coc
        │            chiếu điểm ảnh ra 3D bằng depth
        │            đỉnh thấp -> cốc nằm  -> coc_nam(): trục + tâm vành miệng (PCA)
        │            đỉnh cao  -> cốc đứng -> tâm mặt đáy -> gốc cốc
        │            -> dùng bởi xep_coc_camera.nhin(), xep_coc_piper_cotay.chup()
        │
        └── marker: nhan_dang.tim_marker        (chỉnh tay bằng camera, xep_coc_camera.chinh_tay)
                    tu_hieu_chinh.do_marker_3d  (tự hiệu chỉnh lúc khởi động)
                    gom cụm điểm 3D -> vị trí bàn tay thật
```

Mọi bước sau ô mặt nạ giữ nguyên: chiếu ra 3D, phân biệt cốc đứng hay nằm, gom cụm marker. YOLO chỉ thay việc trả lời câu hỏi "điểm ảnh nào thuộc vật nào". Nhờ vậy:
- So sánh HSV và YOLO là công bằng: cùng phần hình học phía sau.
- Có thể chuyển qua lại giữa hai bộ chỉ bằng một biến `nhan_dang.BO_MAT_NA`.

## 2. Các bước

| # | Bước | File | Kiểm tra | Trạng thái |
|---|---|---|---|---|
| 1 | Gom lọc màu về một hàm `mat_na()`, có biến `BO_MAT_NA` để thay bộ tạo mặt nạ | `bai_camera/nhan_dang.py`, `bai_camera/tu_hieu_chinh.py` | 5 kịch bản cùng seed: log giống hệt trước khi sửa | ✅ Xong |
| 2 | Sinh ảnh tổng hợp + nhãn phân đoạn | `bai_yolo/sinh_du_lieu.py` | Vẽ nhãn lên ảnh để xem; HSV bắt được 91–100% điểm ảnh nhãn (nhãn đúng) | ✅ Xong: 2.000 ảnh trong ~1 phút |
| 3 | Train YOLOv8n-seg trên CPU | `bai_yolo/huan_luyen.py` | mAP50 mặt nạ trên tập val ≥ 0,9 | ✅ Xong: GPU RTX 3050 Ti, 18 phút; mAP50 hộp 0,987, mặt nạ 0,928 |
| 4 | Bộ mặt nạ YOLO, cờ `--yolo` | `bai_yolo/yolo_mat_na.py`; cờ trong `xep_coc_camera.py`, `xep_coc_piper_camera.py`, `demo_live.py` | Chạy các kịch bản với `--yolo` | ✅ Xong |
| 5 | Đánh giá YOLO so với HSV | `bai_yolo/danh_gia.py` | Xem các chỉ số ở mục 4 | 🟡 Đã có kết quả đầu (mục 6), chưa chạy 50 lần |
| 6 | Cập nhật README, CLAUDE.md, đồng bộ `demo_piper` nếu dùng | | | ⏳ |

### Bước 2: sinh dữ liệu

```bash
MUJOCO_GL=egl python bai_yolo/sinh_du_lieu.py --so-anh 2000     # ~1 phút với 6 tiến trình
```

- **Ba nguồn ảnh**, tỉ lệ 40 / 40 / 20%:
  - tay 6DOF + camera trên cao;
  - PiPER + camera trên cao;
  - PiPER + camera cổ tay.
- **Mỗi ảnh:**
  - 3 cốc đặt ngẫu nhiên, mỗi cốc có 30% khả năng nằm đổ.
  - Tay: một nửa số ảnh chúc xuống ở chỗ ngẫu nhiên (lúc tự hiệu chỉnh và chỉnh tay, marker phải nhìn thấy), nửa còn lại ở tư thế khớp ngẫu nhiên (tay che một phần cốc).
  - Đổi ánh sáng và sắc màu ±15%. Lớp cốc vẫn phân biệt bằng màu nên không đổi màu nhiều hơn.
- **Nhãn:** lấy từ ảnh phân đoạn của MuJoCo (mỗi điểm ảnh biết thuộc geom nào), đổi sang đa giác theo định dạng YOLO-seg. Cốc nhận theo tên body `coc1..3`, marker theo tên geom `marker_*`.
- **Kết quả:** `bai_yolo/du_lieu/` gồm `images/`, `labels/`, `du_lieu.yaml`. Thư mục này không đưa lên git; ai cần thì tự sinh lại.

Hai lỗi đã gặp và đã sửa:
- Geom cốc trong model PiPER không có tên, nên phải nhận cốc theo tên body.
- Sửa `geom_rgba` của một geom có material thì màu material bị đè: cốc 6DOF thành màu xám.

### Bước 3: train

```bash
python bai_yolo/huan_luyen.py      # YOLOv8n-seg, imgsz 640, khoảng 50 epoch, CPU (20 nhân): dự kiến 1–2 giờ
```

- 4 lớp: `coc1`, `coc2`, `coc3`, `marker`.
- Model tốt nhất lưu ở `bai_yolo/yolo_coc.pt`, khoảng 7 MB. File này sẽ commit để người khác không phải train lại.

### Bước 4: dùng YOLO trong robot

- `yolo_mat_na.py` có hàm `bat_yolo(duong_dan)`: nạp model rồi đặt `nhan_dang.BO_MAT_NA` bằng hàm chạy YOLO trên ảnh.
- Hàm YOLO trả về mặt nạ đúng kích thước ảnh (`retina_masks=True`). Nhiều vùng cùng lớp được gộp lại, ví dụ 2 marker gộp thành một mặt nạ, giống bản HSV.
- Bỏ các vật có độ tin cậy dưới 0,5.
- Các script nhận thêm cờ `--yolo`. Không có cờ thì vẫn chạy HSV như cũ.

## 3. Thư viện

`ultralytics` 8.4 và PyTorch 2.15 bản CPU, đã cài vào `~/venv-demo`:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
```

Chỉ cần khi dùng `--yolo`. Bản HSV không phụ thuộc hai thư viện này.

## 4. Cách đánh giá (bước 5)

Dùng cùng seed cho HSV và YOLO, rồi so từng ô kết quả:

1. **Độ chính xác mặt nạ:** IoU với nhãn thật, trên khoảng 200 ảnh mới (seed khác lúc train).
2. **Sai số vị trí 3D của cốc** so với vị trí thật trong mô phỏng (mm), tách riêng cốc đứng và cốc nằm.
3. **Tốc độ:** thời gian cho một ảnh. HSV khoảng vài ms; YOLO dự kiến 50–150 ms trên CPU. Chậm hơn nhưng robot chỉ nhìn vài lần mỗi cốc, nên vẫn chấp nhận được.
4. **Tỉ lệ xếp tháp đạt:**
   - `demo_live.py --piper --thu 50` (HSV hiện là 48/50);
   - bài camera và bài PiPER với vài seed.
5. **Thử khó:** đổi ánh sáng, màu sàn, sắc cốc mạnh hơn lúc train. Phần này xem YOLO có thật sự bền hơn HSV không.

**Xong khi:** YOLO đạt tỉ lệ xếp tháp ngang HSV (sai lệch không quá 1–2 lần trên 50), và sai số vị trí cốc không quá HSV cộng 1 mm.

## 5. Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Mép mặt nạ YOLO kém chính xác hơn HSV, nên vị trí cốc và marker lệch vài mm | Hình học phía sau đã lấy trung vị và mặt trên cốc nên ít nhạy với mép. Nếu vẫn lệch: tăng `imgsz` hoặc dùng YOLO chỉ để khoanh vùng, rồi lọc màu trong vùng đó. |
| Marker nhỏ (khoảng 10–20 điểm ảnh), YOLO dễ bỏ sót | Đo tỉ lệ bắt được marker riêng; nếu kém thì giữ HSV cho marker |
| Chỉ train trên ảnh mô phỏng, ảnh thật khác hẳn (sim-to-real) | Khi có camera thật: chụp thêm khoảng 200 ảnh thật để gán nhãn và train tiếp (xem `KE_HOACH_ROBOT_THAT.md`, giai đoạn 3) |
| CPU chậm | Dùng bản `n` (nhỏ nhất); chỉ chạy YOLO khi robot nhìn, không chạy liên tục |

## 6. Kết quả đầu tiên (27/11)

**Mặt nạ**, đo trên 300 ảnh mới (seed 9000, không có trong tập train):

| Nguồn | Lớp | IoU HSV | IoU YOLO | Bỏ sót HSV | Bỏ sót YOLO |
|---|---|---|---|---|---|
| 6DOF | coc1 / coc2 / coc3 | 0,92 / 0,85 / 0,92 | 0,91 / 0,91 / 0,91 | 0 / 8 / 0% | 0 / 1 / 0% |
| 6DOF | marker | 0,79 | 0,69 | 0% | **10%** |
| PiPER | coc1 / coc2 / coc3 | 0,91 / 0,88 / 0,91 | 0,90 / 0,89 / 0,89 | 0% | 0–1% |
| PiPER | marker | 0,81 | 0,75 | 1% | 2% |
| PiPER cổ tay | coc1 / coc2 / coc3 | 0,89 / 0,92 / 0,93 | 0,97 / 0,96 / 0,92 | 4–6% | 0–4% |

- **Tốc độ:** HSV 31 ms/ảnh; YOLO 9 ms/ảnh trên GPU.
- **Cốc:** YOLO ngang HSV, tốt hơn ở cốc xanh (coc2) và ở camera cổ tay.
- **Marker:** YOLO kém hơn HSV. Marker nhỏ, mép mặt nạ YOLO thô hơn.

**Xếp tháp** (cùng seed cho HSV và YOLO):

| Kịch bản | HSV | YOLO |
|---|---|---|
| `demo_live.py --piper --thu 10` | 10/10 | 10/10 |
| `demo_live.py --thu 6` (tay 6DOF) | 6/6 | 6/6 |
| `xep_coc_camera.py --ngau-nhien 5 --nang-cap --pha` | đạt | đạt |
| `xep_coc_piper_camera.py --ngau-nhien 4 --nang-cap --do-coc --lech-encoder 2` | đạt | đạt |
| Sai số vị trí cốc (2 kịch bản trên) | trung bình 2,2 mm, lớn nhất 3,5 mm | trung bình 2,5 mm, lớn nhất 4,3 mm |
| Tự hiệu chỉnh 6DOF, không lệch thật: sai lớn nhất | 0,06° | 0,63° (khớp gập cổ tay) |

**Việc tiếp theo:**
1. Cải thiện marker: thêm ảnh có tay chúc xuống gần camera, tăng `imgsz`, hoặc dùng YOLO khoanh vùng rồi lọc màu trong vùng đó.
2. Chạy `demo_live.py --piper --thu 50 --yolo` để so với 48/50 của HSV.
3. Thử khó: đổi ánh sáng và màu mạnh hơn lúc train.
