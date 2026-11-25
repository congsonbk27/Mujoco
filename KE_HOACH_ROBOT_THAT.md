# Kế hoạch đưa xuống robot thật: AgileX PiPER + camera RealSense

Mục tiêu: chạy cùng chương trình xếp kim tự tháp 3 cốc như trong mô phỏng (`bai_piper/xep_coc_piper_camera.py`, `demo_piper/`) trên cánh tay PiPER thật và camera RealSense D435 thật.

Kế hoạch giả định:
- Khách chọn PiPER. Nếu chọn tay tự chế tạo, xem mục cuối.
- Camera đặt trên cao, cố định (eye-to-hand), giống bản mô phỏng chính. Camera cổ tay để sau.

## Nguyên tắc

1. **Giữ nguyên phần "não", chỉ thay phần "tay chân".** IK, lập quỹ đạo, nhận dạng cốc, tự hiệu chỉnh, chỉnh tay bằng camera và xử lý cốc đổ đã kiểm tra kỹ trong mô phỏng. Chỉ thay phần gửi lệnh động cơ và phần chụp ảnh.
2. **Mô phỏng vẫn chạy được sau mỗi bước.** Mọi thay đổi phải giữ `demo_piper.py --thu 50` ra 48/50 như hiện nay. Robot thật hỏng thì vẫn quay về thử trên mô phỏng được.
3. **Chậm trước, nhanh sau.** Lần đầu chạy mỗi chuyển động ở 20–30% tốc độ, tay luôn cầm nút dừng khẩn cấp.
4. **Số đo trên robot thật mới là số nói với khách.** Không dùng tỉ lệ thành công của mô phỏng để hứa.

## Giai đoạn 0: Chuẩn bị phần cứng và an toàn (1–2 ngày, chờ hàng)

| Hạng mục | Ghi chú |
|---|---|
| PiPER + bộ chuyển USB–CAN | Đi kèm PiPER. Cài `piper_sdk` (Python) và `can-utils` trên Ubuntu. |
| Bệ đặt tay cao 13 cm | Mô phỏng tính với bệ 13 cm vì khớp cổ tay chỉ ±70°. Đổi chiều cao thì phải đổi `BE` trong `xep_coc_piper.py`. |
| RealSense D435 + giá treo | Đặt trên bàn khoảng 1 m, nhìn xuống, giống camera `realsense` trong model. Giá phải cứng: camera rung là sai số hiệu chỉnh tay–mắt. |
| 3 cốc thật | Đo đường kính miệng, đường kính đáy, chiều cao, khối lượng. Màu phải nổi trên nền bàn. |
| Marker màu trên bàn tay | Dùng cho tự hiệu chỉnh và chỉnh tay bằng camera. In hoặc dán theo đúng vị trí trong model (`KC_MARKER = 8.5 cm`). |
| Nút dừng khẩn cấp | Bắt buộc trước khi cấp điện cho tay. |
| Vùng làm việc | Rào hoặc đánh dấu bán kính 45 cm quanh đế. Không ai đứng trong vùng khi tay chạy. |

**Xong khi:** tay bật được, đọc được góc khớp qua CAN, camera ra ảnh trong `realsense-viewer`.

## Giai đoạn 1: Tách lớp phần cứng, làm trong mô phỏng (2–3 ngày)

Hiện lớp `RoBot` (`bai_camera/xep_coc_camera.py`) và `TayPiper` (`bai_piper/xep_coc_piper_camera.py`) gọi thẳng vào MuJoCo: `mj_step`, `d.ctrl`, `d.qpos`, `d.time`. Bước này tách các chỗ đó ra hai giao diện.

**Giao diện tay:**

| Hàm | Mô phỏng | Robot thật |
|---|---|---|
| `doc_encoder()` | `d.qpos` (trừ lệch giả lập) | đọc góc khớp qua CAN |
| `gui_lenh_khop(q)` | `d.ctrl` + khâu tích phân giả lập servo | lệnh góc khớp qua CAN |
| `dat_ngon(do_mo)` / `do_mo_ngon()` | actuator `gripper` / `joint7` | lệnh và phản hồi kẹp |
| `cho(giay)` / `bay_gio()` | `mj_step` / `d.time` | ngủ theo đồng hồ thật / `time.monotonic()` |

**Giao diện camera:** `chup()` trả về `(color, depth)`, cộng thêm thông số nội và `chieu_nguoc()`. `CameraRealSense` hiện đã có đúng giao diện này.

**Giữ lại MuJoCo làm mô hình động học.** IK (`GiaiIK`) vẫn chạy trên model PiPER của Menagerie, chỉ không mô phỏng vật lý nữa.

Các phần chỉ có trong mô phỏng đi theo bản mô phỏng: khâu tích phân `KI` (firmware PiPER thật đã tự làm), các hàm "phá" (`xo_do`, `tuot`, phím demo), và ô camera trên viewer.

**Xong khi:** `demo_piper.py --thu 50` vẫn ra 48/50, log giống hệt bản trước khi tách.

## Giai đoạn 2: Điều khiển PiPER thật, chưa dùng camera (3–4 ngày)

1. **Viết driver** theo giao diện tay ở giai đoạn 1, dùng `piper_sdk`: kết nối CAN, bật động cơ, đọc góc khớp, gửi lệnh góc khớp, điều khiển kẹp. Tên hàm và đơn vị (thường là 0,001° và 0,001 mm) phải đối chiếu với phiên bản SDK thật.
2. **Kiểm tra chiều và điểm 0 từng khớp.** Cho từng khớp quay +5°, so chiều với model Menagerie. Sai dấu hay lệch điểm 0 thì sửa trong bảng đổi đơn vị của driver, không sửa model.
3. **Giới hạn an toàn trong driver:**
   - Giới hạn tốc độ khớp, lúc đầu đặt 20%.
   - Giới hạn góc chặt hơn giới hạn của model.
   - Từ chối điểm đích có TCP thấp hơn mặt bàn + 1 cm.
   - Thoát an toàn (tắt động cơ) khi mất kết nối CAN.
4. **Chạy lại quỹ đạo của mô phỏng trên không**, chưa có cốc, ở 20% tốc độ: các tư thế chờ, gắp, đặt, đi vòng quanh đế.
5. **Đo độ chính xác.** Đưa TCP tới 10 điểm đánh dấu trên bàn, đo bằng thước. So với động học thuận của model.
6. **Gắp – đặt 1 cốc, vị trí biết trước** (như `xep_coc_piper.py`, không camera). Sau đó xếp cả tháp 3 cốc.

**Xong khi:**
- Tới cùng một điểm 10 lần, sai lệch nhỏ hơn 2 mm.
- Xếp tháp với vị trí cốc biết trước 10/10 lần, ở tốc độ bình thường.

**Phải đo lại**, vì mô phỏng chỉ giả định:
- Độ cao bệ, vị trí đặt robot.
- Lực kẹp thật (mô phỏng dùng 10 N).
- Độ chính xác servo.
- Thời gian tay đứng yên sau khi dừng (`CHO_THEM`, mô phỏng là 0,8 s).

## Giai đoạn 3: Camera thật (3–4 ngày)

1. **Thay camera giả lập** bằng `pyrealsense2`:
   - Căn depth theo color (`rs.align`).
   - Lấy thông số nội từ thiết bị.
   - Đổi depth sang mét, giữ cùng giao diện `chup()`.
2. **Chỉnh ngưỡng màu HSV** trong `nhan_dang.py` theo ánh sáng phòng thật, cho cốc và marker. Chụp khoảng 30 ảnh ở nhiều chỗ, nhiều giờ trong ngày để chỉnh.
3. **Hiệu chỉnh tay–mắt ban đầu** (vị trí và hướng camera so với đế robot):
   - Cho tay đưa marker tới khoảng 12 điểm, giống `tu_hieu_chinh.py`.
   - Bộ giải hiện có ước lượng được cả lệch encoder lẫn xoay và dịch camera.
   - Đặt camera càng gần vị trí trong model càng tốt, để điểm khởi đầu của bộ giải không quá xa.
4. **Đo sai số nhận dạng.** Đặt cốc ở 20 vị trí đã đo bằng thước, so với vị trí camera tính ra.

**Xong khi:**
- Sai số vị trí cốc nhỏ hơn 5 mm ở cả 20 vị trí.
- Nhận đúng cốc đứng và cốc nằm 20/20 lần.

## Giai đoạn 4: Ghép toàn bộ (3–5 ngày)

Bật dần từng tính năng, chỉ bật cái sau khi cái trước đã ổn định:

1. Xếp tháp với vị trí cốc lấy từ camera, nhìn lại trước mỗi lần gắp.
2. Tự hiệu chỉnh lúc khởi động (`--nang-cap`).
3. Chỉnh tay bằng camera trước khi thả cốc.
4. Phát hiện kẹp hụt và cốc rơi bằng độ mở ngón. Dùng phản hồi thật của kẹp, nên phải đo lại ngưỡng.
5. Dựng cốc đổ, bằng cách nghiêng tay ±45°. Đây là phần nhạy nhất: chạy chậm và thử nhiều lần.

**Xong khi:** 30 lần xếp liên tục, có người dời và làm đổ cốc, đạt từ 25/30 trở lên. Ghi lý do từng lần hỏng.

## Giai đoạn 5: Đo độ bền và chuẩn bị demo trên robot thật (2–3 ngày)

1. Chạy 50 lần, ghi tỉ lệ đạt và lý do hỏng. So với mô phỏng: lỗi nào chỉ có trên robot thật thì xử lý trước.
2. Viết lại kịch bản demo với số đo thật. Chỉnh các ngưỡng thời gian theo tốc độ thật.
3. Tổng duyệt 2 lần trọn vẹn trước ngày gặp khách.

## Tổng thời gian

Khoảng **3–4 tuần làm việc**, chưa tính thời gian chờ hàng. Chậm nhất thường là giai đoạn 3 (ánh sáng, màu thật) và bước dựng cốc đổ ở giai đoạn 4.

## Rủi ro chính

| Rủi ro | Ảnh hưởng | Cách giảm |
|---|---|---|
| Ánh sáng và màu thật khác mô phỏng | Nhận dạng cốc sai, hiệu chỉnh hỏng | Chỉnh HSV trên ảnh thật, cốc màu nổi, đèn cố định |
| Độ rơ cơ khí, servo thật không giống mô hình | Kẹp lệch vài mm, khe kẹp chỉ dư 5 mm | Chỉnh tay bằng camera trước khi thả; đo lại độ chính xác ở giai đoạn 2 |
| Trễ khi gửi lệnh qua CAN | Quỹ đạo giật, rung | Gửi lệnh đều 50–100 Hz, giảm tốc độ |
| Cốc thật trơn hoặc nhẹ hơn mô phỏng | Tuột khi chở, đổ khi đặt | Đo khối lượng và ma sát, chỉnh lực kẹp và tốc độ chở |
| Dựng cốc đổ va vào bệ hoặc bàn | Hỏng cốc, kẹt tay | Chạy chậm, giới hạn độ cao TCP trong driver |
| Camera bị xê dịch sau khi hiệu chỉnh | Mọi vị trí lệch | Giá cứng; tự hiệu chỉnh mỗi lần khởi động (đã có) |

## Cần chốt trước khi bắt đầu

1. Khách chọn PiPER hay tay tự chế tạo?
2. Kẹp của PiPER là loại tiêu chuẩn đi kèm, hay khách thay loại khác?
3. Kích thước và vật liệu cốc thật.
4. Camera đặt trên cao (như mô phỏng) hay gắn cổ tay? Camera cổ tay cần thêm khoảng 1 tuần.
5. Máy tính điều khiển: laptop Ubuntu đang dùng làm demo, hay máy riêng?

## Nếu khách chọn tay tự chế tạo

Các giai đoạn vẫn như trên. Khác ở mấy chỗ:
- Giai đoạn 1: cập nhật `canh_tay.xml`, `xep_coc.xml` theo bản CAD (kích thước khâu, khối lượng, giới hạn khớp). Chạy lại `bai_xep_coc`, `bai_camera` và bộ thử độ bền trên model mới.
- Giai đoạn 2: tự viết driver cho động cơ và encoder của tay thay cho `piper_sdk`. Việc này thường mất thêm 1–2 tuần, tuỳ loại driver.
