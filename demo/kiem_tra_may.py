"""Kiểm tra máy có đủ nhanh để chạy demo live không.

Chạy trên MÁY DEMO (không cần màn hình phụ, không mở cửa sổ):
    python demo/kiem_tra_may.py

Đo:
  1. Mô phỏng vật lý: bao nhiêu lần nhanh hơn thời gian thực (cần >= 3x để còn dư cho IK + vẽ)
  2. Camera ảo RealSense: thời gian chụp 1 khung color + depth
  3. Nhận dạng cốc trong 1 khung
  4. Bước tự hiệu chỉnh lúc khởi động (bao nhiêu giây thật)
  5. Card đồ hoạ OpenGL đang dùng (laptop 2 GPU hay dùng nhầm GPU tích hợp!)
"""

import contextlib
import io
import pathlib
import sys
import time

import mujoco
import numpy as np

GOC = pathlib.Path(__file__).resolve().parent.parent
for thu_muc in ["bai_gap_do", "bai_xep_coc", "bai_camera"]:
    sys.path.insert(0, str(GOC / thu_muc))

import xep_coc_camera as X  # noqa: E402
from nhan_dang import tim_coc  # noqa: E402


def gio(ham, lap):
    t0 = time.perf_counter()
    for _ in range(lap):
        ham()
    return (time.perf_counter() - t0) / lap


def main():
    print(f"MuJoCo {mujoco.__version__}, Python {sys.version.split()[0]}")
    m = X.tao_model(nang_cap=True)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("san_sang").id)
    mujoco.mj_forward(m, d)

    # 1. Vật lý
    n = 5000
    t = gio(lambda: mujoco.mj_step(m, d), n) * n
    he_so = n * m.opt.timestep / t
    print(f"\n1. Vật lý:        {he_so:6.1f}x thời gian thực   {'OK' if he_so >= 3 else 'CHẬM'}")

    # 2. Camera
    cam = X.CameraRealSense(m)
    cam.cap_nhat_ngoai(d)
    try:
        from OpenGL import GL
        cam.r_mau.update_scene(d, cam.cam)
        cam.r_mau.render()
        gpu = GL.glGetString(GL.GL_RENDERER)
        gpu = gpu.decode() if gpu else "?"
    except Exception:
        gpu = "(không đọc được, cài PyOpenGL để xem)"
    t_cam = gio(lambda: cam.chup(d), 20)
    print(f"2. Camera ảo:     {t_cam*1000:6.1f} ms / khung color+depth   {'OK' if t_cam < 0.1 else 'CHẬM'}")
    color, depth = cam.chup(d)
    t_nd = gio(lambda: tim_coc(cam, color, depth), 20)
    print(f"3. Nhận dạng cốc: {t_nd*1000:6.1f} ms / khung   {'OK' if t_nd < 0.1 else 'CHẬM'}")

    # 4. Tự hiệu chỉnh + 1 lần xếp tháp đầy đủ
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()) as o:
        ok = X.chay(nang_cap=True)
    t_chay = time.perf_counter() - t0
    log = o.getvalue()
    print(f"4. Xếp tháp (bản nâng cấp, không cửa sổ): {t_chay:5.1f} s thật -> "
          f"{'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
    dong = [l for l in log.splitlines() if "Tự hiệu chỉnh" in l]
    if dong:
        print("   " + dong[0][:110])

    print(f"\n5. GPU OpenGL:    {gpu}")
    if any(k in gpu for k in ("Intel", "llvmpipe", "Mesa", "UHD", "Iris")):
        print("   ! Đang dùng GPU tích hợp / phần mềm, KHÔNG phải RTX.")
        print("     Linux:   chạy bằng  prime-run python ...  hoặc  __NV_PRIME_RENDER_OFFLOAD=1 "
              "__GLX_VENDOR_LIBRARY_NAME=nvidia python ...")
        print("     Windows: Settings > Display > Graphics > thêm python.exe > High performance")

    print("\nKết luận:", "MÁY ĐỦ CHO DEMO LIVE" if he_so >= 3 and t_cam < 0.1 else
          "CẦN XEM LẠI (gửi kết quả này cho Claude)")


if __name__ == "__main__":
    main()
