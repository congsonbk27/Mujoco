"""Tìm cốc trong ảnh RealSense: lọc màu (ảnh color) + đo vị trí 3D (ảnh depth).

Ý tưởng:
  1. Mặt nạ màu: chọn các điểm ảnh có màu của cốc (đỏ / xanh / vàng).
  2. Chiếu ngược các điểm ảnh đó ra 3D nhờ depth + intrinsics.
  3. Cốc úp nhìn từ trên xuống thấy mặt đáy (tròn, phẳng, cao nhất) và một phần thành cốc nghiêng.
     Chỉ giữ các điểm cao nhất (cách đỉnh <= 1 cm) -> đó là mặt đáy cốc, tâm của nó nằm đúng trên
     trục cốc. Lấy trung vị (median) để chống nhiễu và điểm lỗi.
"""

import numpy as np

CAO_COC = 0.09

def hsv(color):
    """RGB (uint8) -> H (độ, 0-360), S (0-1), V (0-1). Giống cv2.cvtColor(..., COLOR_RGB2HSV)."""
    c = color.astype(np.float32) / 255.0
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    v = c.max(-1)
    mn = c.min(-1)
    delta = v - mn + 1e-6
    s = np.where(v > 0, (v - mn) / (v + 1e-6), 0)
    h = np.where(v == r, (g - b) / delta % 6,
        np.where(v == g, (b - r) / delta + 2, (r - g) / delta + 4)) * 60
    return h, s, v


# Lọc theo SẮC ĐỘ (hue) chứ không theo giá trị RGB: mặt cốc được đèn chiếu sáng (xanh nhạt) và
# thành cốc trong bóng (xanh đậm) có RGB rất khác nhau nhưng CÙNG hue.
# Độ bão hoà S > 0.35 để loại sàn (xám, S~0) và cánh tay (xám-xanh, S~0.2).
def mat_na_mau(color):
    h, s, v = hsv(color)
    co_mau = (s > 0.35) & (v > 0.12)
    return {
        # Ngưỡng chọn theo hue ĐO ĐƯỢC trên 30 ảnh với cốc ở vị trí ngẫu nhiên:
        #   đỏ 0-37 độ (vùng chói ngả cam), vàng 37-60, xanh 180-205 (mặt chói ngả xanh lơ).
        # Ngưỡng ban đầu (xanh 190-240) cắt mất nửa mặt đáy cốc xanh -> đo lệch 2 cm.
        "coc1": co_mau & ((h < 30) | (h > 340)),      # đỏ
        "coc2": co_mau & (h > 170) & (h < 250),       # xanh dương
        "coc3": co_mau & (h >= 30) & (h < 75),        # vàng
    }


KC_MARKER = 0.052   # mỗi marker cách trục bàn tay 5.2 cm theo trục x


def tim_marker(cam, color, depth, xy_uoc_luong, so_diem_toi_thieu=15):
    """Tìm marker màu hồng (hue ~300 độ) trên mu bàn tay -> xy THẬT của trục bàn tay.

    2 marker nằm 2 bên trục tay, cách trục 5.2 cm theo trục x thế giới (tay luôn kẹp theo trục y
    nên trục x của tay song song trục x thế giới). Cẳng tay hay che mất 1 marker, nên:
      - tách các điểm hồng thành cụm theo x (chỗ hở > 3 cm là ranh giới giữa 2 cụm)
      - mỗi cụm đủ điểm -> suy ra tâm tay = tâm cụm ∓ 5.2 cm (dấu theo cụm nằm bên nào so với
        vị trí tay ƯỚC LƯỢNG; sai số ước lượng vài mm << 5.2 cm nên không thể nhầm bên)
      - thấy 2 cụm thì lấy trung bình 2 kết quả.
    Trả về None nếu không thấy marker nào.
    """
    h, s, v = hsv(color)
    mask = (s > 0.35) & (v > 0.12) & (h > 280) & (h < 325) & (depth > 0)
    vv, uu = np.nonzero(mask)
    if len(uu) < so_diem_toi_thieu:
        return None
    p = cam.chieu_nguoc(uu, vv, depth[vv, uu])
    p = p[np.argsort(p[:, 0])]
    cat = np.nonzero(np.diff(p[:, 0]) > 0.03)[0]
    cum = np.split(p, cat + 1)
    tam = []
    for c in cum:
        if len(c) < so_diem_toi_thieu:
            continue
        xy = np.median(c[:, :2], axis=0)
        dau = 1.0 if xy[0] > xy_uoc_luong[0] else -1.0
        tam.append(xy - [dau * KC_MARKER, 0.0])
    return np.mean(tam, axis=0) if tam else None


def tim_coc(cam, color, depth, so_diem_toi_thieu=40):
    """Trả về dict: tên cốc -> {'goc': vị trí gốc cốc (tâm vành miệng) [x,y,z], 'so_diem': n, 'uv': (u,v)}.
    Cốc không thấy (bị che / ngoài khung hình) sẽ không có trong dict."""
    ket_qua = {}
    for ten, mask in mat_na_mau(color).items():
        mask = mask & (depth > 0)                 # bỏ điểm ảnh không đo được depth
        v, u = np.nonzero(mask)
        if len(u) < so_diem_toi_thieu:
            continue
        p = cam.chieu_nguoc(u, v, depth[v, u])    # N x 3, toạ độ thế giới
        z_dinh = np.percentile(p[:, 2], 95)       # độ cao mặt đáy cốc (chống điểm nhiễu cao bất thường)
        mat_tren = np.abs(p[:, 2] - z_dinh) < 0.01
        if mat_tren.sum() < 10:
            continue
        tam = np.median(p[mat_tren], axis=0)
        ket_qua[ten] = {
            "goc": np.array([tam[0], tam[1], tam[2] - CAO_COC]),
            "so_diem": int(mat_tren.sum()),
            "uv": (float(np.median(u[mat_tren])), float(np.median(v[mat_tren]))),
        }
    return ket_qua
