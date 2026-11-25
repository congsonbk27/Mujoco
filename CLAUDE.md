# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MuJoCo robotics lessons (Vietnamese identifiers, comments, and docs; reply to the user in Vietnamese). Each `bai_*/` folder is a self-contained lesson script; `demo/` is a live customer demo built on top of them. There is no package, build step, linter config, or unit-test suite. "Testing" means running a scenario script and reading its printed result, or running the robustness benchmark.

## Setup and running

```bash
pip install mujoco numpy pillow matplotlib PyOpenGL     # developed on MuJoCo 4.0.1, Python 3.12
```

Every script takes `--xem` (open the interactive viewer) and `--ngau-nhien SEED` (random cup layout). Without `--xem` they run headless and print a result.

```bash
python bai_gap_do/gap_do.py --xem
python bai_xep_coc/xep_coc.py --ngau-nhien 5
python bai_camera/xep_coc_camera.py --ngau-nhien 5 --nang-cap [--pha] [--gif]
python bai_coc_nam/coc_nam.py --ngau-nhien 3
python bai_piper/xep_coc_piper_camera.py --nang-cap [--do-coc] [--lech-encoder DEG] [--pha-do]
python demo/demo_live.py                 # live demo with sabotage keys (see module docstring)
python demo/demo_live.py --thu 50        # headless soak test: 50 runs with random key presses
python demo/kiem_tra_may.py              # demo-machine check (realtime factor, render GPU)
```

- Headless rendering on a server: `export MUJOCO_GL=egl`. Do **not** set it when opening the viewer.
- PiPER lessons need the Menagerie model, which is gitignored: `git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie bai_piper/mujoco_menagerie`.

### Robustness benchmark (`bai_do_ben`)

```bash
cd bai_do_ben
MUJOCO_GL=egl python thu_do_ben.py [--nang-cap] [--nhanh] [--so-lan 30] [--nhan N]
python ve_bieu_do.py    # redraws do_ben.png, so_sanh.png, so_sanh_5dof_6dof.png
```

- A full run is 1,290 episodes, roughly 45 min per variant on a many-core machine. `--nhanh` gives a quick subset.
- It **overwrites** `ket_qua.csv` (or `ket_qua_nang_cap.csv` with `--nang-cap`). `*_5dof.csv` are frozen archives of the old 5DOF arm; never overwrite them.
- To test one case, call `thu_do_ben.mot_lan((nguon, muc, seed, nang_cap))` directly. It returns a dict with `dat` and `ly_do` (failure reason such as `coc_do`, `kep_hut_3_lan`).
- Results use common random numbers (same seeds across variants), so compare variants cell by cell.

## Architecture

Lessons build on each other through `sys.path.insert` of sibling folders and plain imports, not a package. Changing a shared module affects every later lesson, the benchmark, and the demo.

- `bai_gap_do/gap_do.py`: core of everything. `KHOP_TAY` (6 joint names), `GiaiIK` (damped-least-squares IK on site `tcp`, full 6D: position + gripper y-axis with z pointing down), `muot`/`nlerp` trajectory helpers, `NhipXem` (viewer pacing). Model: `canh_tay.xml`.
- `bai_xep_coc/xep_coc.py` + `xep_coc.xml`: pyramid geometry and grasp constants (`vi_tri_kim_tu_thap`, `Z_KEP`, `Z_DI`, `NGON_MO`, `TEN_COC`, `CAO`, `MODEL_XML`).
- `bai_camera/`: perception and closed-loop control.
  - `realsense.py`: `CameraRealSense` simulates color + depth with noise. `R_lech`/`t_lech` are injected hand-eye errors; `R_sua`/`t_sua` are the robot's self-estimated correction.
  - `nhan_dang.py`: HSV-threshold cup and marker detection.
  - `tu_hieu_chinh.py`: kinematic self-calibration (Gauss-Newton over encoder offsets + camera rotation/translation, with per-joint Tikhonov regularization).
  - `xep_coc_camera.py`: `tao_model(nang_cap)` (adds camera and hand markers via `MjSpec`), class `RoBot`, and `chay(..., tuy_chinh, nang_cap)`. `tuy_chinh(m, d, cam)` is the hook the benchmark uses to inject errors.
- `bai_do_ben/thu_do_ben.py`: error sources in `NGUON`; runs `xep_coc_camera.chay` in a spawn-based multiprocessing pool.
- `bai_coc_nam/coc_nam.py`: righting a fallen cup (needs the 6DOF wrist); `dat_coc_nam`, `ke_hoach_dung_coc` are reused by the demo and PiPER.
- `bai_piper/`: the same pipeline on the real AgileX PiPER model (`xep_coc_piper.py` → `_camera.py` → `_cotay.py`, eye-in-hand). Reuses IK and camera modules but with its own grasp constants (PiPER wrist is limited to ±70°, so the arm sits on a 13 cm pedestal).
- `demo/demo_live.py`: wraps `xep_coc_camera` with keyboard sabotage. Its keys collide with MuJoCo viewer shortcuts, so it restores the visual flags after each key.

### `--nang-cap` (upgrade) mode
Self-calibration at startup (12 poses, half with the hand rotated 90° so the wrist-pitch joint is observable), visual servoing on hand markers before release, and drop detection via finger opening. Without it the arm trusts the camera and encoders blindly.

## Conventions and gotchas

- Encoder offset is simulated as `actuator_biasprm[0] = kp·δ` (joint settles at `ctrl + δ`). After changing body masses call `mujoco.mj_setConst`.
- Sites are in a hidden `sitegroup` so the camera does not see them (the red TCP site was once detected as a cup).
- IK must be seeded (`RoBot.giai_ik_moi`) and moves between distant poses go through joint space (`toi_khop`); unseeded IK can land on a flipped-wrist branch.
- Benchmark success criteria are "đứng vững" (tower stands) and "đẹp" (C3 within 1 cm of centered), not absolute position.
- `KE_HOACH_DEMO.md` and `demo/KICH_BAN_DEMO.md` hold the customer-demo plan and script; per-lesson `README.md` files hold results and lessons learned. Update the relevant README when results change.
