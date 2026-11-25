"""Copy code PiPER mới nhất từ các thư mục bai_* sang demo_piper/ (bản gửi khách).

Chạy sau mỗi lần sửa code trong bai_*:
    python dong_bo_demo_piper.py

Đường dẫn tới thư mục bài khác (THU_MUC.parent / "bai_...") được đổi thành THU_MUC, vì trong
demo_piper mọi file nằm chung một thư mục.
Không đụng tới demo_piper.py, kiem_tra_may.py, README.md: các file này đã được sửa riêng cho bản
gửi khách. Sửa demo/demo_live.py thì phải tự sửa demo_piper/demo_piper.py theo.
"""

import pathlib
import re

GOC = pathlib.Path(__file__).resolve().parent
DICH = GOC / "demo_piper"
FILE = ["bai_gap_do/gap_do.py", "bai_xep_coc/xep_coc.py", "bai_xep_coc/xep_coc.xml",
        "bai_camera/nhan_dang.py", "bai_camera/realsense.py", "bai_camera/tu_hieu_chinh.py",
        "bai_camera/xep_coc_camera.py", "bai_coc_nam/coc_nam.py", "bai_piper/xep_coc_piper.py",
        "bai_piper/xep_coc_piper_camera.py", "bai_piper/xep_coc_piper_cotay.py"]


def chuyen(s):
    s = re.sub(r'(THU_MUC|GOC)\.parent / "bai_[a-z_]+"', r"\1", s)
    return s.replace("mujoco_menagerie bai_piper/mujoco_menagerie", "mujoco_menagerie mujoco_menagerie")


def main():
    doi = 0
    for f in FILE:
        moi = chuyen((GOC / f).read_text(encoding="utf-8"))
        dich = DICH / pathlib.Path(f).name
        if not dich.exists() or dich.read_text(encoding="utf-8") != moi:
            dich.write_text(moi, encoding="utf-8")
            print(f"cập nhật {dich.relative_to(GOC)}")
            doi += 1
    print(f"Xong: {doi} file thay đổi" if doi else "demo_piper đã mới nhất")


if __name__ == "__main__":
    main()
