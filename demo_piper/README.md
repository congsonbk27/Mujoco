# Demo PiPER: xếp kim tự tháp 3 cốc bằng camera

Bản demo này mô phỏng cánh tay AgileX PiPER trong MuJoCo. Robot dùng camera RealSense đặt trên cao để tìm 3 cốc, rồi xếp chúng thành kim tự tháp.

Khi chạy, bạn có thể bấm phím để "phá" robot: dời cốc, làm đổ cốc, thêm nhiễu camera, làm lệch encoder. Robot sẽ tự phát hiện và xử lý.

## 1. Cài đặt (một lần)

Cần Python 3.12, 64-bit.

```bash
cd demo_piper
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie mujoco_menagerie
```

Lệnh `git clone` cuối cùng tải model cánh tay PiPER từ MuJoCo Menagerie (giấy phép Apache-2.0). Lệnh này phải chạy bên trong thư mục `demo_piper`.

Kiểm tra máy có đủ mạnh để chạy demo không:

```bash
python kiem_tra_may.py
```

## 2. Chạy demo

```bash
python demo_piper.py
```

Lệnh này mở cửa sổ 3D, robot xếp tháp liên tục. Mỗi lần chạy, robot mất khoảng 8 giây đầu để tự hiệu chỉnh encoder và camera, sau đó mới bắt đầu xếp cốc.

### Phím điều khiển

Bấm phím khi cửa sổ 3D đang được chọn.

| Phím | Tác dụng |
|---|---|
| `Space` | Tạm dừng / chạy tiếp |
| `R` | Rải lại cốc ngẫu nhiên, chạy lại từ đầu |
| `1` `2` `3` | Dời cốc C1 / C2 / C3 sang chỗ khác (chỉ cốc chưa xếp) |
| `T` | Làm đổ một cốc, robot tự dựng lại |
| `N` | Bật/tắt nhiễu camera mạnh (10 mm) |
| `E` | Bật/tắt lệch encoder 2° ở mọi khớp, robot tự hiệu chỉnh lại |
| `C` | Chuyển giữa bản nâng cấp và bản cũ để so sánh |

### Tuỳ chọn khác

```bash
python demo_piper.py --thu 50          # chạy ngầm 50 lần, tự bấm phím phá ngẫu nhiên, in số lần đạt
python demo_piper.py --quay demo.mp4   # quay video theo kịch bản cố định
python demo_piper.py --seed 3          # đổi bộ vị trí cốc ngẫu nhiên
```

Từng phần cũng chạy riêng được, thêm `--xem` để mở cửa sổ 3D:

```bash
python xep_coc_piper.py --xem --ngau-nhien 3                     # xếp tháp, biết trước vị trí cốc (không dùng camera)
python xep_coc_piper_camera.py --xem --nang-cap --do-coc         # dùng camera trên cao, có cốc đổ từ đầu
python xep_coc_piper_cotay.py --xem --ngau-nhien 2               # dùng camera gắn trên cổ tay
```

## 3. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| Báo thiếu file `mujoco_menagerie/agilex_piper/piper.xml` | Chưa tải model: chạy lệnh `git clone` ở mục 1, bên trong thư mục `demo_piper`. |
| Windows: `import mujoco` lỗi DLL | Dùng Python 3.12 64-bit và cài Microsoft Visual C++ Redistributable (x64). |
| Chạy trên máy không có màn hình bị lỗi OpenGL | Đặt `export MUJOCO_GL=egl` trước khi chạy `--thu` hoặc `--quay`. Không đặt biến này khi muốn mở cửa sổ 3D. |
| Bấm phím mà cảnh 3D bị đổi kiểu hiển thị | Bình thường: phím demo trùng phím tắt của MuJoCo, demo sẽ tự trả về như cũ. |
| Robot chạy chậm | Chạy `python kiem_tra_may.py` để xem máy có đủ mạnh không. |

## Các file

| File | Vai trò |
|---|---|
| `demo_piper.py` | Chương trình demo, chạy file này |
| `xep_coc_piper.py`, `xep_coc_piper_camera.py`, `xep_coc_piper_cotay.py` | Điều khiển PiPER: xếp tháp, dùng camera, dựng cốc đổ |
| `xep_coc_camera.py`, `realsense.py`, `nhan_dang.py`, `tu_hieu_chinh.py` | Camera giả lập, nhận dạng cốc, tự hiệu chỉnh |
| `gap_do.py`, `xep_coc.py`, `xep_coc.xml`, `coc_nam.py` | Giải IK, quỹ đạo, cảnh và kích thước cốc |
| `kiem_tra_may.py` | Kiểm tra máy |
