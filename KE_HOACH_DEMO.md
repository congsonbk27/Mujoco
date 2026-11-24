# Kế hoạch demo cho khách hàng: robot 6DOF xếp cốc bằng camera

**Hình thức:** chạy mô phỏng MuJoCo **trực tiếp** (live) trước khách hàng
**Thời hạn:** 1 tuần (5 ngày làm việc)
**Cánh tay:** tự dựng 6DOF, nâng cấp từ cánh tay 5DOF hiện tại

---

## 1. Mục tiêu demo (khách hàng sẽ thấy gì)

Trong khoảng 10 phút, khách hàng thấy:

1. **Robot tự hiệu chỉnh lúc khởi động** (khoảng 20 s): tay đi qua các tư thế, màn hình hiện độ lệch encoder và độ lệch camera mà robot tự đo ra.
2. **Xếp 3 cốc thành kim tự tháp** chỉ bằng camera, không dùng vị trí "đọc từ mô phỏng".
3. **Khách hàng tự "phá"**: bấm phím để dời cốc, kéo cốc đi đúng lúc tay đang gắp, thêm nhiễu camera, làm lệch encoder. Robot vẫn hoàn thành.
4. **Điểm mạnh của 6DOF so với 5DOF**: gắp cốc đặt nghiêng/xoay tuỳ ý, tiếp cận từ nhiều hướng.
5. **Số liệu độ bền**: biểu đồ trước/sau nâng cấp, "ngân sách sai số" cho robot thật.

**Tiêu chí sẵn sàng demo:**
- Chạy kịch bản chính 50 lần liên tiếp không lỗi.
- Mỗi thao tác "phá" khi chạy live đều được xử lý.
- Tốc độ thời gian thực trên máy demo.

---

## 2. Chuyển 5DOF → 6DOF

### 2.1 Thiết kế

Hiện tại tay có 5 khớp: xoay đế, vai, khuỷu, cổ tay (gập), xoay tay. Khi đã chọn vị trí TCP, bàn tay chỉ còn chúc thẳng xuống hoặc nghiêng *trong mặt phẳng của cánh tay*. Không thể vừa nghiêng sang bên vừa giữ vị trí.

Đề xuất thêm **khớp xoay cẳng tay** (forearm roll) giữa khuỷu và cổ tay. Kết quả là kiểu "cổ tay cầu" (spherical wrist) quen thuộc của robot công nghiệp (UR, ABB...):

```
xoay_de (z) → vai (y) → khuyu (y) → xoay_cang_tay (trục cẳng tay) → co_tay (y) → xoay_tay (z)
```

- 3 khớp cuối có trục cắt nhau tại một điểm. Nhờ vậy vị trí và hướng của bàn tay gần như tách rời nhau: IK hội tụ tốt, ít điểm kỳ dị.
- Bàn tay đạt được **mọi hướng** trong vùng làm việc. Đây là thứ cần để gắp cốc bị đổ nằm nghiêng hoặc tiếp cận từ bên cạnh.

### 2.2 Phần code phải sửa

| Phần | Việc cần làm | Độ khó |
|---|---|---|
| `bai_gap_do/canh_tay.xml` | thêm body + joint `xoay_cang_tay`, actuator, keyframe | thấp |
| `bai_xep_coc/xep_coc.xml` | cùng thay đổi (model này chép phần cánh tay từ bài gắp đồ) | thấp |
| `KHOP_TAY` + `GiaiIK` | thêm khớp; IK đổi sang ràng buộc **đủ 6 chiều** (vị trí + hướng đầy đủ) | trung bình |
| `giai_ik_moi`, `Q_CHO`, keyframe | chọn nghiệm khởi đầu và tư thế chờ hợp lý cho 6 khớp | trung bình |
| `tu_hieu_chinh.py` | 6 độ lệch encoder; thêm tư thế hiệu chỉnh xoay cẳng tay để khớp mới quan sát được | trung bình |
| `thu_do_ben.py` | `lech_encoder` cho 6 khớp | thấp |
| Kiểm chứng | chạy lại 3 bài cũ + bộ thử độ bền cho 6DOF | máy chạy ~1 giờ |

**Rủi ro chính:** IK 6 chiều có thêm nghiệm (cổ tay "lật"). Cần chặn bằng giới hạn khớp và nghiệm khởi đầu tốt, giống lỗi nhánh nghiệm đã gặp ở bài camera.

---

## 3. Chương trình demo live (`demo/demo_live.py`)

Một file chạy duy nhất, mở viewer MuJoCo:

- **Màn hình:**
  - cảnh chính,
  - ô nhỏ ở góc hiện ảnh camera RealSense có đánh dấu cốc tìm được,
  - dòng trạng thái: đang làm gì (nhìn / gắp / chỉnh tay / đặt), số lần thử, độ lệch đo được.
- **Phím điều khiển** (`key_callback` của `mujoco.viewer`):

| Phím | Tác dụng |
|---|---|
| `Space` | bắt đầu / tạm dừng |
| `R` | rải lại cốc ngẫu nhiên, chạy lại |
| `1` `2` `3` | dời cốc C1/C2/C3 tới chỗ ngẫu nhiên (khách "phá") |
| `N` | bật/tắt nhiễu camera mạnh |
| `E` | làm lệch encoder 2° (rồi xem robot tự hiệu chỉnh) |
| `C` | bật/tắt bản nâng cấp (so sánh cũ/mới ngay trước mặt khách) |
| `T` | đặt 1 cốc nằm nghiêng (chỉ 6DOF gắp được) |

- **Kế hoạch dự phòng:** quay sẵn video toàn bộ kịch bản. Nếu máy demo có sự cố thì chiếu video.

---

## 4. Lịch 5 ngày

| Ngày | Việc | Kết quả kiểm tra được |
|---|---|---|
| **1** | Model 6DOF + IK 6 chiều. Chạy lại bài gắp đồ và xếp cốc. | 2 bài cũ đạt 20/20 với cánh tay 6DOF |
| **2** | Bài camera + tự hiệu chỉnh cho 6 khớp. Chạy lại bộ thử độ bền. | Biểu đồ độ bền 6DOF không kém 5DOF |
| **3** | Tính năng mới của 6DOF: gắp cốc nằm nghiêng / tiếp cận nghiêng | Gắp được cốc nghiêng ≥ 90% |
| **4** | `demo_live.py`: phím điều khiển, ô camera, dòng trạng thái | Chạy được mọi phím; chạy 50 lần liên tiếp không lỗi |
| **5** | Chạy thử trên **máy demo thật**, đo tốc độ, quay video dự phòng, viết kịch bản lời nói | Tổng duyệt 2 lần trọn vẹn |

**Nếu chậm tiến độ thì bỏ theo thứ tự:** gắp cốc nằm nghiêng (ngày 3) → phím `C` so sánh cũ/mới. Giữ nguyên phần lõi: 6DOF + xếp cốc + khách phá + tự hiệu chỉnh.

---

## 5. Rủi ro

| Rủi ro | Ảnh hưởng | Phòng ngừa |
|---|---|---|
| Máy demo không có GPU / chạy chậm | camera ảo render chậm, demo giật | thử máy demo ngay ngày 1; giảm độ phân giải camera; chỉ chụp ảnh khi cần (đang làm vậy) |
| IK 6DOF lật cổ tay | tay chuyển động "lạ" trước khách | giới hạn khớp + nghiệm khởi đầu; thử 50 lần liên tiếp |
| Khách thao tác ngoài kịch bản | robot kẹt | phím `R` reset; giới hạn chỗ dời cốc trong tầm với (đã có `cho_trong`) |
| Kỳ vọng sai về robot thật | khách nghĩ robot thật cũng đạt 100% | trình bày rõ "ngân sách sai số" và giới hạn mô phỏng (README bài độ bền) |

---

## 6. Cần anh/chị xác nhận

1. Máy chạy demo: laptop hay PC? Có card đồ hoạ rời không? Hệ điều hành gì?
2. Có cần giao diện tiếng Anh cho khách không? (Hiện code và chữ trên màn hình đều tiếng Việt.)
3. Kịch bản "cốc nằm nghiêng" có quan trọng với khách không, hay chỉ cần xếp cốc đứng?
