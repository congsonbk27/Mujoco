"""Vẽ biểu đồ độ bền từ ket_qua.csv -> do_ben.png

    pip install matplotlib
    python ve_bieu_do.py
"""

import csv
import pathlib
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THU_MUC = pathlib.Path(__file__).parent

# (tên trong csv, tiêu đề ô, nhãn trục x)
O = [
    ("nhieu_camera", "Nhiễu depth của camera", "σ ở 1 m (mm)"),
    ("tay_mat_dich", "Hiệu chỉnh tay–mắt: lệch vị trí", "mm"),
    ("tay_mat_xoay", "Hiệu chỉnh tay–mắt: lệch góc", "độ"),
    ("lech_encoder", "Lệch 0 encoder mỗi khớp", "độ"),
    ("ma_sat_kep", "Ma sát ngón–cốc (giảm dần)", "hệ số μ"),
    ("khoi_luong", "Khối lượng mỗi cốc", "gam"),
    ("ket_hop", "Mọi sai số cùng lúc", "1 = tốt · 2 = thường · 3 = xấu"),
]

# Bảng màu đã kiểm tra (validate_palette.js: đạt mọi check, kể cả mù màu), mực/lưới theo palette mặc định
XANH, CAM = "#2a78d6", "#eb6834"
MUC_CHINH, MUC_PHU, MUC_MO = "#0b0b0b", "#52514e", "#898781"
LUOI, TRUC, NEN = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def doc():
    bang = defaultdict(lambda: defaultdict(list))
    thu_tu = defaultdict(list)
    for r in csv.DictReader(open(THU_MUC / "ket_qua.csv")):
        muc = float(r["muc"])
        if muc not in thu_tu[r["nguon"]]:
            thu_tu[r["nguon"]].append(muc)
        bang[r["nguon"]][muc].append((int(r["dat"]), int(r["dep"] or 0)))
    return bang, thu_tu


def nhan(x):
    return f"{x:g}"


def main():
    bang, thu_tu = doc()
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": TRUC, "axes.labelcolor": MUC_PHU,
                         "xtick.color": MUC_MO, "ytick.color": MUC_MO, "text.color": MUC_CHINH})
    fig, axs = plt.subplots(2, 4, figsize=(13, 6.4), facecolor=NEN)
    for ax, (ten, tieu_de, don_vi) in zip(axs.flat, O):
        cac_muc = thu_tu[ten]
        x = range(len(cac_muc))                     # mức không đều -> đặt cách đều, ghi nhãn thật
        dung = [100 * sum(a for a, _ in bang[ten][m]) / len(bang[ten][m]) for m in cac_muc]
        dep = [100 * sum(b for _, b in bang[ten][m]) / len(bang[ten][m]) for m in cac_muc]
        n = len(bang[ten][cac_muc[0]])
        ax.set_facecolor(NEN)
        ax.axhspan(95, 100, color=XANH, alpha=0.07, lw=0)   # vùng "≥ 95%"
        ax.plot(x, dung, color=XANH, lw=2, marker="o", ms=6, mec=NEN, mew=1.5, label="Tháp đứng vững")
        ax.plot(x, dep, color=CAM, lw=2, ls=(0, (4, 2)), marker="s", ms=5.5, mec=NEN, mew=1.5,
                label="Tháp đẹp (C3 lệch giữa ≤ 1 cm)")
        ax.set_xticks(list(x), [nhan(m) for m in cac_muc])
        ax.set_ylim(-4, 104)
        ax.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
        ax.grid(axis="y", color=LUOI, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(tieu_de, loc="left", fontsize=10, fontweight="bold", color=MUC_CHINH)
        ax.set_xlabel(don_vi, fontsize=8.5)
        # Ghi nhãn trực tiếp: mức lớn nhất mà tháp vẫn đứng >= 95%
        # (tính từ mức đầu tới khi lần đầu tụt dưới 95%)
        i = -1
        while i + 1 < len(dung) and dung[i + 1] >= 95:
            i += 1
        if i >= 0 and ten != "ket_hop":
            ax.annotate(f"≥95% tới {nhan(cac_muc[i])}", (i, dung[i]), xytext=(0, -16),
                        textcoords="offset points", ha="center", fontsize=8, color=MUC_PHU)
        ax.text(1, 1.02, f"{n} lần/mức", transform=ax.transAxes, ha="right", fontsize=7.5, color=MUC_MO)

    # ô cuối: chú giải
    leg = axs.flat[-1]
    leg.axis("off")
    h, lbl = axs.flat[0].get_legend_handles_labels()
    leg.legend(h, lbl, loc="center left", frameon=False, fontsize=9.5, handlelength=3)
    leg.text(0.02, 0.18, "Nền xanh nhạt: vùng ≥ 95%\nCốc rải ngẫu nhiên, cùng bộ seed cho mọi mức",
             transform=leg.transAxes, fontsize=8, color=MUC_MO, va="top")
    fig.suptitle("Độ bền hệ thống xếp cốc bằng camera: tỉ lệ thành công khi tăng từng nguồn sai số",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.2, w_pad=1.8)
    fig.savefig(THU_MUC / "do_ben.png", dpi=150, facecolor=NEN)
    print("Đã lưu", THU_MUC / "do_ben.png")


if __name__ == "__main__":
    main()
