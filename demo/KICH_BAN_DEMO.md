# Kịch bản demo cho khách (khoảng 10 phút)

Người trình bày chạy `demo_live.py` và nói theo kịch bản dưới đây. Cột **Bấm** là phím, bấm khi con trỏ đang ở trong cửa sổ MuJoCo.

## Trước giờ demo (30 phút trước)

1. Cắm sạc laptop và tắt thông báo / cập nhật tự động.
2. Kiểm tra máy:
   ```bash
   python demo/kiem_tra_may.py
   ```
   Kết quả phải ra **"MÁY ĐỦ CHO DEMO LIVE"**, và mục 5 phải ghi tên card **RTX**.
3. Chạy thử 1 vòng đầy đủ:
   ```bash
   python demo/demo_live.py
   ```
   Xem robot xếp xong một tháp rồi đóng cửa sổ.
4. Mở sẵn file `demo/video_du_phong.mp4` trong trình xem video, để dự phòng.
5. Mở lại `demo_live.py`, bấm `Space` để tạm dừng, rồi chờ khách vào.

## Kịch bản

| # | Bấm | Robot làm gì | Nói gì |
|---|---|---|---|
| 1 | `Space` | Tự hiệu chỉnh khoảng 20 s: tay đi qua 12 tư thế | "Khi khởi động, robot tự nhìn tay mình qua camera để tìm sai số encoder và sai số lắp camera. Không cần kỹ sư đo tay." |
| 2 | — | Nhìn bằng camera, gắp cốc đầu tiên | "Robot không biết trước cốc ở đâu. Nó tìm cốc bằng camera 3D, giống RealSense thật, có nhiễu thật." Chỉ vào ô camera ở góc dưới phải. |
| 3 | `1`, `2` hoặc `3` | Khách dời một cốc chưa xếp | "Anh/chị thử dời một cốc." Robot sẽ nhìn lại và gắp đúng chỗ mới. |
| 4 | `T` | Một cốc bị đổ nằm. Robot gắp ngang thân cốc, xoay bàn tay 90° để dựng cốc đứng lên | "Đây là điểm cánh tay 6 bậc làm được còn 5 bậc thì không: bàn tay đổi hướng từ chúc xuống sang nằm ngang." |
| 5 | — | Xếp xong tháp | "Tháp phải đứng vững: cốc trên nằm giữa 2 cốc dưới, sai số vài mm." |
| 6 | `R` rồi `E` | Encoder lệch 2° ở mọi khớp. Robot tự hiệu chỉnh lại rồi vẫn xếp được | "Giả sử sau va chạm, encoder bị lệch 2 độ. Robot tự phát hiện và bù lại." |
| 7 | `N` | Nhiễu camera tăng gấp 5 lần, robot vẫn xếp được | "Ánh sáng xấu, camera nhiễu hơn, robot vẫn làm được." |
| 8 *(tuỳ chọn)* | `R`, `C` | Bản cũ, chưa tự hiệu chỉnh, với encoder đang lệch: thường kẹp hụt | "Đây là bản trước khi nâng cấp, để thấy rõ khác biệt." Bấm `C` lần nữa để quay lại bản nâng cấp. |

**Thời điểm bấm phím:**
- `E` và `C`: bấm lúc robot đang đứng chờ hoặc vừa xếp xong một cốc. Robot chỉ biết encoder đã đổi khi tự hiệu chỉnh lại. Nếu bấm lúc đang đặt cốc cuối thì cốc đó dễ hỏng.
- `T`: bấm khi còn cốc chưa xếp. Cốc vừa đổ đôi khi lăn ra ngoài tầm với. Khi đó bấm `R` và nói: "cốc lăn ra ngoài vùng làm việc, mình rải lại."

## Số liệu để trả lời khách

| Câu hỏi | Trả lời |
|---|---|
| Tỉ lệ thành công? | Không bị phá: gần như 100%. Thử 50 lần liên tiếp, mỗi lần bị phá ngẫu nhiên 0–2 lần: tháp đứng vững 44–45/50, không lần nào treo. |
| Chịu sai số tới đâu? | Vẫn ≥ 95% khi encoder lệch tới 3° mỗi khớp, camera lệch vị trí 4 cm hoặc lệch góc 2°, nhiễu độ sâu 10 mm, cốc nặng tới 1,5 kg. Xem `bai_do_ben/README.md`. |
| Có chạy trên robot thật được không? | Đây là mô phỏng. Robot thật còn thêm độ rơ cơ khí, độ trễ, và camera thật nhiễu khác mô phỏng. Phương pháp (tự hiệu chỉnh, chỉnh tay bằng camera) dùng được trên robot thật, nhưng tỉ lệ thành công phải đo lại trên phần cứng. |
| 6 bậc hơn 5 bậc ở đâu? | Bàn tay đổi được hướng tuỳ ý, nên gắp được vật nằm nghiêng hoặc đổ (phím `T`). Độ bền xếp tháp thì 6 bậc bằng hoặc hơn 5 bậc ở mọi loại sai số. |

## Nếu có sự cố

| Sự cố | Xử lý |
|---|---|
| Robot kẹt hoặc làm lạ | Bấm `R`. |
| Cửa sổ đứng hình hoặc crash | Chiếu `demo/video_du_phong.mp4` (2 phút, đủ các tình huống ở bước 1–7). |
| Chạy giật | Kiểm tra cắm sạc. Trên Ubuntu chạy với `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia`. |

Quay lại video dự phòng (khoảng 3 phút, không mở cửa sổ):
```bash
python demo/demo_live.py --quay demo/video_du_phong.mp4
```
