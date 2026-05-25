#!/usr/bin/env python3
"""Generate an Excalidraw diagram of rolling flow-matching: training and sampling.

Output: docs/figures/rolling_flow.excalidraw

Open the result at https://excalidraw.com (File > Open) on a machine with a
browser to tweak it and export PNG/SVG. The frame rows are ~200 repetitive
squares, so the diagram is generated rather than hand-placed: edit the CONFIG
constants below and re-run to regenerate.

Run:  uv run python docs/figures/make_rolling_flow_diagram.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(7)  # stable seeds/nonces -> reproducible file

# ---------------------------------------------------------------------------
# CONFIG -- tweak and re-run
# ---------------------------------------------------------------------------
FRAME = 24          # side of one frame square, px
GAP = 4             # gap between frames
PITCH = FRAME + GAP
N = 8               # rolling-window width, in frames

CLEAN_RGB = (25, 113, 194)    # t=1, data        (#1971c2)
NOISE_RGB = (236, 239, 241)   # t=0, pure noise

FRAME_STROKE = "#495057"
INK = "#1e1e1e"
MUTED = "#868e96"
ACCENT = "#1971c2"

ROW_X = 175         # left x of every frame row
BOX_X = 40
BOX_W = 1140

# training scenario
T_TRAIN = 28
FRONT = 9           # the window starts at this frame index
GHOSTS = (2, 17)    # faint window positions for "other steps"

# sampling scenario: 4 snapshots, buffer grows by one frame each step
SMP_START_LEN = 16
SMP_STEPS = 4

OUT = Path(__file__).with_name("rolling_flow.excalidraw")

# ---------------------------------------------------------------------------
# element construction
# ---------------------------------------------------------------------------
elements: list[dict] = []
_counter = 0


def _id() -> str:
    global _counter
    _counter += 1
    return f"el-{_counter}"


def _rand() -> int:
    return random.randint(1, 2**31 - 1)


def _base(kind: str, x, y, w, h, **kw) -> dict:
    el = {
        "id": _id(), "type": kind,
        "x": float(x), "y": float(y),
        "width": float(w), "height": float(h),
        "angle": 0,
        "strokeColor": INK, "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 1, "strokeStyle": "solid",
        "roughness": 1, "opacity": 100,
        "groupIds": [], "frameId": None, "roundness": None,
        "seed": _rand(), "versionNonce": _rand(), "version": 1,
        "isDeleted": False, "boundElements": None,
        "updated": 1, "link": None, "locked": False,
    }
    el.update(kw)
    elements.append(el)
    return el


def box(x, y, w, h, *, fill="transparent", stroke=INK, sw=2):
    return _base("rectangle", x, y, w, h, backgroundColor=fill,
                 strokeColor=stroke, strokeWidth=sw, roundness={"type": 3})


def square(x, y, color, *, opacity=100):
    return _base("rectangle", x, y, FRAME, FRAME, backgroundColor=color,
                 strokeColor=FRAME_STROKE, strokeWidth=1, opacity=opacity)


def text(x, y, s, *, size=15, color=INK, font=1, opacity=100):
    lines = s.split("\n")
    w = max(len(ln) for ln in lines) * size * 0.62
    h = len(lines) * size * 1.25
    return _base("text", x, y, w, h, strokeColor=color, opacity=opacity,
                 text=s, originalText=s, fontSize=size, fontFamily=font,
                 textAlign="left", verticalAlign="top", containerId=None,
                 lineHeight=1.25, autoResize=True)


def linear(kind, pts, *, color=INK, sw=2, dashed=False,
           end=None, start=None, opacity=100):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ox, oy = min(xs), min(ys)
    rel = [[p[0] - ox, p[1] - oy] for p in pts]
    el = _base(kind, ox, oy, max(xs) - ox, max(ys) - oy,
               strokeColor=color, strokeWidth=sw,
               strokeStyle="dashed" if dashed else "solid", opacity=opacity,
               points=rel, lastCommittedPoint=None,
               startBinding=None, endBinding=None,
               startArrowhead=start, endArrowhead=end,
               roundness={"type": 2})
    if kind == "arrow":
        el["elbowed"] = False
    return el


def arrow(x1, y1, x2, y2, **kw):
    kw.setdefault("end", "arrow")
    return linear("arrow", [[x1, y1], [x2, y2]], **kw)


def bracket(x, y, w, *, depth=10, color=MUTED, sw=2, dashed=False, opacity=100):
    """An under-bracket: a flat span whose ends turn up toward the row above."""
    return linear("line", [[x, y], [x, y + depth],
                           [x + w, y + depth], [x + w, y]],
                  color=color, sw=sw, dashed=dashed, opacity=opacity)


def lerp_color(t: float) -> str:
    """t=1 -> clean data colour, t=0 -> pure-noise colour."""
    t = max(0.0, min(1.0, t))
    rgb = tuple(round(NOISE_RGB[i] + (CLEAN_RGB[i] - NOISE_RGB[i]) * t)
                for i in range(3))
    return "#%02x%02x%02x" % rgb


def frame_row(x0, y, t_values, *, opacity=100):
    for i, t in enumerate(t_values):
        square(x0 + i * PITCH, y, lerp_color(t), opacity=opacity)


def window_ramp() -> list[float]:
    """N timesteps from t=1 (clean, left) to t=0 (noise, right)."""
    return [1.0 - j / (N - 1) for j in range(N)]


# ---------------------------------------------------------------------------
# header + legend
# ---------------------------------------------------------------------------
text(BOX_X, 22, "Rolling flow-matching — training & sampling", size=30)
text(BOX_X, 64,
     "RollingFlowSpeaker denoises audio with a sliding window: trained at "
     "random positions, swept at inference.",
     size=15, color=MUTED)

text(BOX_X, 96,
     "Each square = one acoustic frame.   Fill colour = flow timestep t "
     "(how denoised the frame is):", size=14)
LEG_N = 12
LEG_Y = 118
for i in range(LEG_N):
    square(ROW_X + i * PITCH, LEG_Y, lerp_color(1.0 - i / (LEG_N - 1)))
text(ROW_X, LEG_Y + FRAME + 7, "t = 1   clean / data", size=13, color=ACCENT)
_leg_end = ROW_X + (LEG_N - 1) * PITCH + FRAME
text(_leg_end - 118, LEG_Y + FRAME + 7, "t = 0   pure noise",
     size=13, color=MUTED)

# ===========================================================================
# PANEL 1 -- TRAINING
# ===========================================================================
PA_Y = 175
PA_H = 320
box(BOX_X, PA_Y, BOX_W, PA_H)
text(BOX_X + 22, PA_Y + 16, "1  ·   TRAINING STEP", size=20, color=ACCENT)
text(BOX_X + 250, PA_Y + 20,
     "— drop the window at a random position, supervise only inside it",
     size=14, color=MUTED)

X1_Y = PA_Y + 62
X0_Y = X1_Y + 38
XT_Y = X0_Y + 72

# x1 (clean target) and x0 (noise prior)
frame_row(ROW_X, X1_Y, [1.0] * T_TRAIN)
text(BOX_X + 22, X1_Y + 4, "x₁  data", size=14)
frame_row(ROW_X, X0_Y, [0.0] * T_TRAIN)
text(BOX_X + 22, X0_Y + 4, "x₀  noise", size=14)

# interpolation note + arrow down into x_t
arrow(ROW_X - 24, X0_Y + FRAME, ROW_X - 24, XT_Y, sw=2)
text(ROW_X, X0_Y + FRAME + 10,
     "interpolate per-position at t :    "
     "xₜ = (1−t)·x₀  +  t·x₁", size=15)

# x_t (model input) -- the staircase
t_xt = [1.0] * FRONT + window_ramp() + [0.0] * (T_TRAIN - FRONT - N)
frame_row(ROW_X, XT_Y, t_xt)
text(BOX_X + 22, XT_Y + 4, "xₜ  input", size=14)

# ghost windows: where the window lands on other steps
WIN_X = ROW_X + FRONT * PITCH
WIN_W = N * PITCH - GAP
BR_Y = XT_Y + FRAME + 6
for g in GHOSTS:
    bracket(ROW_X + g * PITCH, BR_Y, WIN_W,
            color=MUTED, dashed=True, opacity=40)

# the live window bracket + labels
bracket(WIN_X, BR_Y, WIN_W, color=ACCENT, sw=2)
text(WIN_X, BR_Y + 15,
     f"rolling window  (n = {N})  —  supervised: only these frames "
     "enter the loss  (v_mask)", size=14, color=ACCENT)
text(WIN_X, BR_Y + 33,
     "left edge = front, sampled uniformly at random each step   "
     "(dashed = other steps)", size=13, color=MUTED)

# model flow: x_t window -> transformer -> loss
TB_W, TB_H = 300, 46
TB_X = WIN_X + WIN_W / 2 - TB_W / 2
TB_Y = BR_Y + 52
arrow(WIN_X + WIN_W / 2, BR_Y + 12, WIN_X + WIN_W / 2, TB_Y, sw=2)
box(TB_X, TB_Y, TB_W, TB_H, fill="#e7f5ff", stroke=ACCENT, sw=2)
text(TB_X + 28, TB_Y + 13, "Transformer   ( + text condition )",
     size=15, color=ACCENT)

# text-condition input
TC_W = 124
box(TB_X - TC_W - 56, TB_Y + 6, TC_W, 34, fill="#fff", stroke=MUTED, sw=2)
text(TB_X - TC_W - 44, TB_Y + 15, "text tokens", size=13, color=MUTED)
arrow(TB_X - 56, TB_Y + TB_H / 2, TB_X, TB_Y + TB_H / 2, sw=2, color=MUTED)

# loss
arrow(TB_X + TB_W, TB_Y + TB_H / 2, TB_X + TB_W + 58, TB_Y + TB_H / 2, sw=2)
text(TB_X + TB_W + 70, TB_Y + 6,
     "flow-matching loss,\ncomputed on window frames only", size=14)

# ===========================================================================
# PANEL 2 -- SAMPLING
# ===========================================================================
PB_Y = PA_Y + PA_H + 30
PB_H = 290
box(BOX_X, PB_Y, BOX_W, PB_H)
text(BOX_X + 22, PB_Y + 16, "2  ·   SAMPLING  (inference)",
     size=20, color=ACCENT)
text(BOX_X + 320, PB_Y + 20,
     "— sweep that same window across the sequence, one frame per step",
     size=14, color=MUTED)

R0_Y = PB_Y + 64
ROW_DY = 40
step_labels = ["step  k", "step  k+1", "step  k+2", "step  k+3"]
win_left_x, win_right_x, row_ys = [], [], []
for s in range(SMP_STEPS):
    buf = SMP_START_LEN + s
    row_y = R0_Y + s * ROW_DY
    row_ys.append(row_y)
    # last N frames are the window (clean->noise); the rest are finalised
    t_vals = [1.0] * (buf - N) + window_ramp()
    frame_row(ROW_X, row_y, t_vals)
    text(BOX_X + 22, row_y + 4, step_labels[s], size=13)
    win_left_x.append(ROW_X + (buf - N) * PITCH)
    win_right_x.append(ROW_X + buf * PITCH - GAP)

# window label above the first snapshot
text(win_left_x[0] + 24, PB_Y + 44, f"rolling window  (n = {N})",
     size=13, color=ACCENT)

# dashed band: the channel the window travels through (with sweep arrowheads)
arrow(win_left_x[0], row_ys[0], win_left_x[-1], row_ys[-1] + FRAME,
      color=ACCENT, sw=2, dashed=True)
arrow(win_right_x[0], row_ys[0], win_right_x[-1], row_ys[-1] + FRAME,
      color=ACCENT, sw=2, dashed=True)

# right-gutter annotations
GX = 748
arrow(GX - 6, R0_Y + 6, ROW_X + (SMP_START_LEN - 1) * PITCH + FRAME,
      R0_Y + 4, sw=2, color=MUTED)
text(GX, R0_Y - 32,
     "Each step appends one fresh\npure-noise frame on the right.",
     size=14)

text(GX, R0_Y + ROW_DY + 6,
     "The window sweeps left → right,\none frame per step.",
     size=14, color=ACCENT)

eos_text_y = R0_Y + 3 * ROW_DY - 4
text(GX, eos_text_y,
     "A frame exiting the window is fully\n"
     "denoised (t = 1): checked for the EOS\n"
     "sentinel, then streamed out as audio.",
     size=14)
arrow(GX - 6, eos_text_y + 24, win_left_x[-1] + 9, row_ys[-1] + FRAME + 2,
      sw=2, color=MUTED)

# ===========================================================================
# punchline callout
# ===========================================================================
CB_Y = PB_Y + PB_H + 28
CB_H = 92
box(BOX_X, CB_Y, BOX_W, CB_H, fill="#fff9db", stroke="#f08c00", sw=2)
text(BOX_X + 26, CB_Y + 16, "Same rolling window, two modes.",
     size=18, color="#e8590c")
text(BOX_X + 26, CB_Y + 44,
     "Training drops it at a random position — so the model learns to "
     "denoise every window configuration.\n"
     "Sampling just sweeps it across the sequence: streaming, autoregressive "
     "generation, with no change to the model.",
     size=15)

# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------
doc = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://excalidraw.com",
    "elements": elements,
    "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
    "files": {},
}
OUT.write_text(json.dumps(doc, indent=2))
print(f"wrote {OUT}  ({len(elements)} elements)")
