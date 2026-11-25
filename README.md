# Mujoco

Các bài học mô phỏng robot bằng MuJoCo. Mỗi bài chạy được độc lập. Bài sau dùng lại code của bài trước, và kết thúc bằng một demo có thể cho khách xem trực tiếp.

## 1. Cài đặt

Cần Python 3.12; code được phát triển trên MuJoCo 4.0.1.

```bash
git clone https://github.com/congsonbk27/Mujoco.git
cd Mujoco
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install mujoco numpy pillow matplotlib PyOpenGL imageio
```

Các bài PiPER và demo `--piper` cần thêm model cánh tay PiPER (tải một lần):

```bash
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie bai_piper/mujoco_menagerie
```

Kiểm tra máy có đủ mạnh để chạy demo:

```bash
python demo/kiem_tra_may.py
```

## 2. Chạy nhanh

Thêm `--xem` để mở cửa sổ 3D. Nếu không có `--xem`, chương trình chạy ngầm rồi in kết quả `ĐẠT` hoặc `KHÔNG ĐẠT`.

```bash
python demo/demo_live.py                    # demo tay 6DOF, mở cửa sổ, chạy liên tục
python demo/demo_live.py --piper            # demo với cánh tay AgileX PiPER
```

### Phím trong demo (bấm khi cửa sổ viewer đang được chọn)

| Phím | Tác dụng |
|---|---|
| `Space` | Tạm dừng / chạy tiếp |
| `R` | Rải lại cốc ngẫu nhiên, chạy lại từ đầu |
| `1` `2` `3` | Dời cốc C1 / C2 / C3 sang chỗ khác (chỉ cốc chưa xếp) |
| `T` | Làm đổ một cốc, robot tự dựng lại |
| `N` | Bật/tắt nhiễu camera mạnh (10 mm) |
| `E` | Bật/tắt lệch encoder 2°, robot tự hiệu chỉnh lại |
| `C` | Chuyển giữa bản nâng cấp và bản cũ để so sánh |

Mỗi lần chạy, robot mất khoảng 8–10 giây đầu để tự hiệu chỉnh encoder và camera. Sau đó robot mới bắt đầu xếp cốc.

### Tuỳ chọn khác của demo

```bash
python demo/demo_live.py --piper --thu 50   # chạy ngầm 50 lần, tự bấm phím phá ngẫu nhiên, đếm số lần đạt
python demo/demo_live.py --quay demo.mp4    # quay video dự phòng theo kịch bản cố định
python demo/demo_live.py --seed 3           # đổi bộ vị trí cốc ngẫu nhiên
```

Kịch bản trình bày cho khách nằm trong [demo/KICH_BAN_DEMO.md](demo/KICH_BAN_DEMO.md).

## 3. Các bài học

Nên đọc theo thứ tự. Mỗi thư mục có README riêng, ghi kết quả và các bài học rút ra.

| Bài | Nội dung | Lệnh chạy thử |
|---|---|---|
| [bai_gap_do](bai_gap_do/) | Cánh tay 6 bậc tự do gắp và đặt vật: IK, lập quỹ đạo, kẹp bằng ma sát | `python bai_gap_do/gap_do.py --xem` |
| [bai_xep_coc](bai_xep_coc/) | Xếp 3 cốc thành kim tự tháp, chọn hướng kẹp tránh va chạm | `python bai_xep_coc/xep_coc.py --xem --ngau-nhien 5` |
| [bai_camera](bai_camera/) | Tìm cốc bằng camera RealSense giả lập (ảnh màu + độ sâu), điều khiển vòng kín, tự hiệu chỉnh | `python bai_camera/xep_coc_camera.py --xem --nang-cap --pha` |
| [bai_coc_nam](bai_coc_nam/) | Cốc bị đổ: gắp ngang thân, xoay bàn tay 90° để dựng lại | `python bai_coc_nam/coc_nam.py --xem --ngau-nhien 3` |
| [bai_do_ben](bai_do_ben/) | Thử độ bền: bơm 6 loại sai số, đo tỉ lệ thành công qua 1.290 lần chạy | xem [bai_do_ben/README.md](bai_do_ben/README.md) |
| [bai_piper](bai_piper/) | Cánh tay thật AgileX PiPER: xếp cốc, camera trên cao, camera cổ tay | `python bai_piper/xep_coc_piper_camera.py --xem --nang-cap --do-coc` |
| [demo](demo/) | Demo live cho khách | `python demo/demo_live.py` |

Các tuỳ chọn hay dùng:
- `--ngau-nhien SEED`: rải cốc ngẫu nhiên theo một seed.
- `--nang-cap`: robot tự hiệu chỉnh lúc khởi động, chỉnh tay bằng camera trước khi thả cốc, và phát hiện cốc rơi.
- `--do-coc`, `--pha-do`, `--lech-encoder 2`: các tình huống phá (chỉ có ở bài PiPER).

## 4. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| Chạy trên server không có màn hình bị lỗi OpenGL | Đặt `export MUJOCO_GL=egl` trước khi chạy. **Không** đặt biến này khi muốn mở cửa sổ `--xem`. |
| `--piper` báo thiếu file model | Chưa tải Menagerie: chạy lệnh `git clone` ở mục 1. |
| Windows: `import mujoco` lỗi DLL | Dùng Python 3.12 64-bit và cài [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe). |
| Bấm phím mà cảnh 3D bị đổi kiểu hiển thị | Bình thường: các phím demo trùng phím tắt của viewer MuJoCo, demo sẽ tự trả về như cũ. |
| Robot chạy chậm hơn thời gian thực | Chạy `python demo/kiem_tra_may.py` để xem máy có đủ mạnh không. |
