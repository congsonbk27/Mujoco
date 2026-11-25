# Kế hoạch tiếp theo: chuẩn bị demo và chuyển sang robot thật

Kế hoạch 5 ngày ban đầu (`KE_HOACH_DEMO.md`) đã xong. File này liệt kê việc còn lại, theo thứ tự ưu tiên.

## Hiện trạng

| Phần | Trạng thái | Kết quả kiểm tra |
|---|---|---|
| Tay tự dựng 6DOF + camera trên cao (`bai_camera`) | Xong | Bộ thử độ bền: 6DOF bằng hoặc hơn 5DOF ở mọi nguồn sai số (số đo trước các bản sửa ở mục 2) |
| Dựng cốc đổ, tay tự dựng (`bai_coc_nam`) | Xong | 30/30 |
| Demo live, tay tự dựng (`demo/demo_live.py`) | Xong | 44-45/50 khi bị phá ngẫu nhiên, 0 lần treo |
| PiPER, không camera (`bai_piper/xep_coc_piper.py`) | Xong | 20/20 |
| PiPER + camera trên cao (`xep_coc_piper_camera.py`) | Xong | 20/20; cốc đổ giữa chừng 10/10 |
| PiPER + camera cổ tay (`xep_coc_piper_cotay.py`) | Xong | 25/25 |
| Demo live với PiPER (`demo_live.py --piper`) | Xong | 20/20 khi bị phá ngẫu nhiên, 0 lần treo |
| Video dự phòng, kịch bản demo | Xong cho tay tự dựng | Chưa có bản PiPER |
| Code trên GitHub | **Chưa push** | 11 commit chỉ có trên máy Windows |

## 1. Đưa code lên laptop demo (người thực hiện: chủ repo)

Việc này chặn mọi việc phía sau trên laptop demo.

1. Push nhánh `claude/adoring-archimedes-z748jx` bằng tài khoản `congsonbk27`:
   ```bash
   git push https://congsonbk27@github.com/congsonbk27/Mujoco.git claude/adoring-archimedes-z748jx
   ```
2. Trên laptop Ubuntu: `git pull`, rồi tải model PiPER:
   ```bash
   git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie bai_piper/mujoco_menagerie
   ```
3. Cài thư viện: `pip install mujoco numpy pillow matplotlib PyOpenGL imageio[ffmpeg]`.
4. Chạy `python demo/kiem_tra_may.py` với card RTX. Mục 5 phải ghi tên card RTX.
5. Chạy `python demo/demo_live.py` và `python demo/demo_live.py --piper`, bấm thử mọi phím.

**Xong khi:** cả hai demo chạy mượt trên laptop, không lỗi.

## 2. Đo lại độ bền (khoảng 1.5 giờ máy chạy)

Sau lần đo trước, bài camera có các thay đổi ảnh hưởng tới kết quả: dấu nghiệm khởi đầu IK, ngưỡng nhận dạng cốc nằm, thời gian chờ trước khi chụp. Các con số trong kịch bản demo vì vậy là số cũ.

1. Chạy `MUJOCO_GL=egl python bai_do_ben/thu_do_ben.py --nang-cap` và bản không `--nang-cap`.
2. Chạy `python bai_do_ben/ve_bieu_do.py`, so với `ket_qua_nang_cap_5dof.csv`.
3. Cập nhật số liệu trong `bai_do_ben/README.md` và `demo/KICH_BAN_DEMO.md`.

**Xong khi:** mọi con số nói với khách đều lấy từ lần đo này.

Chưa có bộ thử độ bền cho PiPER. Nếu khách chọn PiPER, làm thêm bản `thu_do_ben.py` cho PiPER (khoảng nửa ngày).

## 3. Kịch bản demo cho PiPER (khoảng 30 phút)

Thêm vào `demo/KICH_BAN_DEMO.md`:

- Điểm khác của PiPER: đặt trên bệ 13 cm, kẹp gần đỉnh cốc, dựng cốc bằng cách nghiêng tay ±45° vì khớp cổ tay chỉ quay ±70°.
- Hai ô camera: camera trên cao điều khiển, camera cổ tay để xem.
- Câu trả lời cho câu hỏi "camera trên cao hay camera cổ tay":

| | Camera trên cao | Camera cổ tay |
|---|---|---|
| Nhìn thấy | Cả bàn, mọi lúc | Vùng trước tay, phải quét |
| Sai số encoder | Cần marker + tự hiệu chỉnh | Tự triệt tiêu khi nhìn gần |
| Tốc độ | Nhanh hơn | Chậm hơn (thêm bước quét) |

- Quay video dự phòng cho PiPER: thử chế độ `--quay` với `--piper` (chưa kiểm tra).

**Xong khi:** tổng duyệt hai lần trọn vẹn trên laptop demo, mỗi lần một cánh tay.

## 4. Sau buổi demo: chuyển sang robot thật

Hướng đi tuỳ khách chọn cánh tay nào.

**PiPER thật:**

1. Điều khiển: thay phần gửi lệnh MuJoCo bằng `piper_sdk` qua CAN. Giữ nguyên IK, lập quỹ đạo, nhận dạng.
2. Camera: thay `CameraRealSense` (giả lập) bằng `pyrealsense2`. Hàm `chup()` và `chieu_nguoc()` giữ cùng giao diện.
3. Đo lại các thông số đang là giả định trong mô phỏng:
   - Độ cao bệ (13 cm) và vị trí đặt robot.
   - Lực kẹp thật (mô phỏng dùng 10 N).
   - Độ chính xác servo (mô phỏng dùng khâu tích phân KI = 0.3).
   - Vị trí camera cổ tay (lệch 6 cm, chúc 10°): đo bằng hiệu chỉnh tay-mắt.
4. Chạy trên bàn thật với tốc độ chậm trước, rồi tăng dần.

**Tay tự chế tạo:**

1. Cập nhật `canh_tay.xml` và `xep_coc.xml` theo bản CAD: kích thước khâu, khối lượng, giới hạn khớp.
2. Chạy lại `bai_xep_coc`, `bai_camera`, bộ thử độ bền.
3. Làm phần giao tiếp với driver động cơ của tay.

**Rủi ro chung khi chuyển sang robot thật:** mô phỏng không có độ rơ cơ khí, độ trễ truyền lệnh, và nhiễu camera thật khác nhiễu giả lập. Tỉ lệ thành công phải đo lại trên phần cứng; không dùng số mô phỏng để hứa với khách.
