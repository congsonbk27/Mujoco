# Mujoco

Các bài học MuJoCo.

| Bài | Nội dung |
|---|---|
| [bai_gap_do](bai_gap_do/) | Cánh tay robot 6 bậc tự do gắp và đặt vật: IK Damped Least Squares, lập quỹ đạo, kẹp bằng ma sát |
| [bai_xep_coc](bai_xep_coc/) | Xếp 3 cốc thành kim tự tháp: lập kế hoạch nhiều vật, chọn hướng kẹp tránh va chạm, đặt vật lên vật |
| [bai_camera](bai_camera/) | Tìm cốc bằng camera RealSense giả lập (color + depth có nhiễu), điều khiển vòng kín: nhìn lại trước mỗi lần gắp, phát hiện kẹp hụt |
| [bai_do_ben](bai_do_ben/) | Thử độ bền: bơm 6 loại sai số (nhiễu camera, hiệu chỉnh tay–mắt, encoder, ma sát, khối lượng), đo tỉ lệ thành công qua 1.290 lần chạy |
| [bai_coc_nam](bai_coc_nam/) | Cốc bị đổ nằm ngang: gắp ngang thân, xoay bàn tay 90° dựng cốc đứng rồi xếp kim tự tháp (cần 6DOF). `python bai_coc_nam/coc_nam.py --xem --ngau-nhien 3` |
| [bai_piper](bai_piper/) | Cánh tay thật AgileX PiPER (model MuJoCo Menagerie). `xep_coc_piper.py`: xếp kim tự tháp (đặt trên bệ 13 cm, kẹp gần đỉnh cốc, đi vòng quanh đế). `xep_coc_piper_camera.py`: thêm camera, tự hiệu chỉnh, chỉnh tay bằng camera, phát hiện rơi, dựng cốc đổ (nghiêng ±45° vì cổ tay PiPER chỉ ±70°), cốc đổ giữa chừng (`--pha-do`). `xep_coc_piper_cotay.py`: camera gắn trên cổ tay (eye-in-hand): quét + nhìn gần, chịu được lệch encoder mà không cần tự hiệu chỉnh. Cần `git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie bai_piper/mujoco_menagerie` |
| [demo](demo/) | Demo live cho khách: `demo_live.py` (phím phá, ô camera), kịch bản `KICH_BAN_DEMO.md` |
