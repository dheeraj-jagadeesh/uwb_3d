"""
╔══════════════════════════════════════════════════════════════════╗
║        LIVE FULL-BODY UWB MOTION CAPTURE  —  v2.0               ║
║  6 Tags  |  4 Anchors  |  ESP32-S3 DW3000                        ║
║  Changes vs v1:                                                   ║
║    • EMA position smoothing  (eliminates jank)                    ║
║    • Mannequin-style body rendering  (matches reference image)    ║
║    • Purple tag markers with labels                               ║
║    • Yellow anchor markers                                        ║
║    • Dark studio theme + floor grid                               ║
║    • FuncAnimation loop  (smoother than while+pause)             ║
║    • Nelder-Mead early-termination to keep solver fast            ║
╚══════════════════════════════════════════════════════════════════╝
"""

import serial
import json
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import minimize

# ─────────────────────────────────────────────────────────────────
#  ANCHOR POSITIONS  (cm)  —  update to match your real room!
# ─────────────────────────────────────────────────────────────────
ANCHORS = np.array([
    [  0,   0,   0],   # A0  Master  – floor corner
    [225,   0,   0],   # A1          – floor corner
    [225, 310,  115],   # A2          – mid-wall
    [  0, 300,  125],   # A3          – low corner
], dtype=float)
# ─────────────────────────────────────────────────────────────────
#  SKELETON PROPORTIONS
# ─────────────────────────────────────────────────────────────────
SHOULDER_WIDTH = 40   # cm shoulder-to-shoulder
HIP_WIDTH      = 30   # cm hip-to-hip

# ─────────────────────────────────────────────────────────────────
#  SMOOTHING  (Exponential Moving Average)
#  ┌─ Lower alpha → smoother, slightly more latency
#  └─ Higher alpha → more responsive, slightly jankier
#  Recommended range: 0.12 – 0.30
# ─────────────────────────────────────────────────────────────────
EMA_ALPHA = 0.18

# ─────────────────────────────────────────────────────────────────
#  TAG DEFAULT POSITIONS  (cm)
# ─────────────────────────────────────────────────────────────────
_DEFAULTS = {
    0: np.array([ 60.0, 100.0, 110.0]),   # T0 — Left Wrist
    1: np.array([140.0, 100.0, 110.0]),   # T1 — Right Wrist
    2: np.array([ 85.0, 100.0,  50.0]),   # T2 — Left Knee
    3: np.array([115.0, 100.0,  50.0]),   # T3 — Right Knee
    4: np.array([100.0, 100.0, 170.0]),   # T4 — Head
    5: np.array([100.0, 100.0, 100.0]),   # T5 — Belly Button
}

raw_pos      = {k: v.copy() for k, v in _DEFAULTS.items()}
smoothed_pos = {k: v.copy() for k, v in _DEFAULTS.items()}

TAG_LABELS = {
    0: 'T0  L-Wrist',
    1: 'T1  R-Wrist',
    2: 'T2  L-Knee',
    3: 'T3  R-Knee',
    4: 'T4  Head',
    5: 'T5  Belly',
}

# ─────────────────────────────────────────────────────────────────
#  3-D MULTILATERATION  (Nelder-Mead least-squares)
# ─────────────────────────────────────────────────────────────────
def _residual(pt, dists):
    err = 0.0
    for i in range(4):
        if dists[i] <= 0:
            continue
        err += (np.linalg.norm(pt - ANCHORS[i]) - dists[i]) ** 2
    return err

def solve_position(dists, seed):
    res = minimize(
        _residual, seed, args=(dists,),
        method='Nelder-Mead',
        options={'xatol': 0.8, 'fatol': 0.8, 'maxiter': 150},
    )
    return res.x

# ─────────────────────────────────────────────────────────────────
#  SERIAL PORT  (master anchor)
# ─────────────────────────────────────────────────────────────────
COM_PORT = "COM19"
ser = None
try:
    ser = serial.Serial(
        port=COM_PORT, baudrate=115200,
        timeout=0.01, dsrdtr=False, rtscts=False,
    )
    ser.setDTR(False)
    ser.setRTS(False)
    time.sleep(0.5)
    ser.reset_input_buffer()
    print(f"[✓] Connected to UWB master on {COM_PORT}")
except Exception as e:
    print(f"[!] Serial unavailable ({e})  →  running in static preview mode")

# ─────────────────────────────────────────────────────────────────
#  COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────
BG_COL     = '#0e0e1c'   # near-black canvas
BODY_COL   = '#c8a880'   # warm mannequin tone
JOINT_COL  = '#a07850'   # slightly darker for joints
TAG_COL    = '#9b59b6'   # purple — matches reference image dots
ANCHOR_COL = '#f1c40f'   # yellow triangles
GRID_COL   = '#1e1e38'   # subtle floor grid
TEXT_COL   = '#e0e0e0'

# ─────────────────────────────────────────────────────────────────
#  FIGURE SETUP
# ─────────────────────────────────────────────────────────────────
matplotlib.rcParams['toolbar'] = 'None'
fig = plt.figure(figsize=(11, 10), facecolor=BG_COL)
fig.canvas.manager.set_window_title('UWB Full-Body Motion Capture  v2.0')
ax  = fig.add_subplot(111, projection='3d', facecolor=BG_COL)
fig.subplots_adjust(left=0.0, right=1.0, top=0.96, bottom=0.0)

# ─────────────────────────────────────────────────────────────────
#  HELPER — draw one limb segment
# ─────────────────────────────────────────────────────────────────
def _seg(p1, p2, lw, col=BODY_COL):
    ax.plot(
        [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
        color=col, linewidth=lw, solid_capstyle='round',
    )

# ─────────────────────────────────────────────────────────────────
#  KINEMATICS — derive all joints from 6 tracked tags
# ─────────────────────────────────────────────────────────────────
def build_skeleton(sp):
    L_WRIST = sp[0]
    R_WRIST = sp[1]
    L_KNEE  = sp[2]
    R_KNEE  = sp[3]
    HEAD    = sp[4]
    BELLY   = sp[5]

    spine    = HEAD - BELLY
    s_len    = max(np.linalg.norm(spine), 1e-6)
    s_unit   = spine / s_len

    NECK     = BELLY + 0.78 * spine     # 78% of the way up the spine
    PELVIS   = BELLY - 0.18 * spine     # slightly below belly

    # Build the lateral axis from the spine & assumed forward direction (+Y)
    fwd      = np.array([0.0, 1.0, 0.0])
    lateral  = np.cross(s_unit, fwd)
    lat_len  = np.linalg.norm(lateral)
    lateral  = lateral / lat_len if lat_len > 1e-6 else np.array([1.0, 0.0, 0.0])

    L_SHOULDER = NECK   + lateral * (SHOULDER_WIDTH / 2)
    R_SHOULDER = NECK   - lateral * (SHOULDER_WIDTH / 2)
    L_HIP      = PELVIS + lateral * (HIP_WIDTH / 2)
    R_HIP      = PELVIS - lateral * (HIP_WIDTH / 2)

    # Estimated elbows (60% shoulder, 40% wrist)  — creates a natural arm bend
    L_ELBOW    = L_SHOULDER * 0.60 + L_WRIST * 0.40
    R_ELBOW    = R_SHOULDER * 0.60 + R_WRIST * 0.40

    # Feet projected straight down from knees
    L_FOOT     = np.array([L_KNEE[0], L_KNEE[1], 0.0])
    R_FOOT     = np.array([R_KNEE[0], R_KNEE[1], 0.0])

    return dict(
        HEAD=HEAD, NECK=NECK, BELLY=BELLY, PELVIS=PELVIS,
        L_SHOULDER=L_SHOULDER, R_SHOULDER=R_SHOULDER,
        L_HIP=L_HIP, R_HIP=R_HIP,
        L_WRIST=L_WRIST, R_WRIST=R_WRIST,
        L_ELBOW=L_ELBOW, R_ELBOW=R_ELBOW,
        L_KNEE=L_KNEE, R_KNEE=R_KNEE,
        L_FOOT=L_FOOT, R_FOOT=R_FOOT,
    )

# ─────────────────────────────────────────────────────────────────
#  DRAW MANNEQUIN BODY
#  Thick rounded line segments approximate a solid figure, fast
#  to render while still reading as a human silhouette.
# ─────────────────────────────────────────────────────────────────
def draw_body(j):
    TW = 14   # torso line-width  (creates the "block" torso look)
    UL =  9   # upper limb
    LL =  7   # lower limb (thinner below knee/elbow)

    # ── Head ──────────────────────────────────────────────────────
    ax.scatter(*j['HEAD'], s=1100, c=BODY_COL,
               depthshade=True, zorder=5)

    # ── Torso core ────────────────────────────────────────────────
    _seg(j['HEAD'],       j['NECK'],       TW)
    _seg(j['NECK'],       j['BELLY'],      TW)
    _seg(j['BELLY'],      j['PELVIS'],     TW)
    _seg(j['L_SHOULDER'], j['R_SHOULDER'], TW)   # shoulder bar
    _seg(j['L_HIP'],      j['R_HIP'],      TW)   # hip bar

    # ── Left arm ──────────────────────────────────────────────────
    _seg(j['L_SHOULDER'], j['L_ELBOW'],   UL)
    _seg(j['L_ELBOW'],    j['L_WRIST'],   LL)

    # ── Right arm ─────────────────────────────────────────────────
    _seg(j['R_SHOULDER'], j['R_ELBOW'],   UL)
    _seg(j['R_ELBOW'],    j['R_WRIST'],   LL)

    # ── Left leg ──────────────────────────────────────────────────
    _seg(j['L_HIP'],  j['L_KNEE'],  UL + 1)
    _seg(j['L_KNEE'], j['L_FOOT'],  LL)

    # ── Right leg ─────────────────────────────────────────────────
    _seg(j['R_HIP'],  j['R_KNEE'],  UL + 1)
    _seg(j['R_KNEE'], j['R_FOOT'],  LL)

    # ── Joint nodes ───────────────────────────────────────────────
    nodes = np.array([
        j['NECK'],   j['PELVIS'],
        j['L_SHOULDER'], j['R_SHOULDER'],
        j['L_HIP'],      j['R_HIP'],
        j['L_ELBOW'],    j['R_ELBOW'],
        j['L_WRIST'],    j['R_WRIST'],
        j['L_KNEE'],     j['R_KNEE'],
        j['L_FOOT'],     j['R_FOOT'],
    ])
    ax.scatter(nodes[:,0], nodes[:,1], nodes[:,2],
               c=JOINT_COL, s=80, depthshade=True, zorder=6)

# ─────────────────────────────────────────────────────────────────
#  DRAW TAG MARKERS  (purple circles + labels)
# ─────────────────────────────────────────────────────────────────
def draw_tags(j):
    tag_pts = {
        0: j['L_WRIST'],
        1: j['R_WRIST'],
        2: j['L_KNEE'],
        3: j['R_KNEE'],
        4: j['HEAD'],
        5: j['BELLY'],
    }
    for tid, pos in tag_pts.items():
        ax.scatter(*pos,
                   c=TAG_COL, s=180,
                   edgecolors='white', linewidths=1.5,
                   depthshade=False, zorder=10)
        ax.text(pos[0] + 5, pos[1] + 5, pos[2] + 7,
                TAG_LABELS[tid],
                fontsize=7.5, color=TAG_COL,
                fontweight='bold', zorder=11)

# ─────────────────────────────────────────────────────────────────
#  DRAW ANCHORS
# ─────────────────────────────────────────────────────────────────
def draw_anchors():
    labels = ['A0  Master', 'A1', 'A2', 'A3']
    for i, anc in enumerate(ANCHORS):
        ax.scatter(*anc,
                   c=ANCHOR_COL, s=140, marker='^',
                   depthshade=False, zorder=9)
        ax.text(anc[0] + 4, anc[1] + 4, anc[2] + 7,
                labels[i], fontsize=7.5,
                color=ANCHOR_COL, fontweight='bold')

# ─────────────────────────────────────────────────────────────────
#  DRAW FLOOR GRID
# ─────────────────────────────────────────────────────────────────
def draw_floor():
    for v in range(-50, 251, 50):
        ax.plot([v, v], [-50, 250], [0, 0],
                color=GRID_COL, lw=0.8, alpha=0.9)
        ax.plot([-50, 250], [v, v], [0, 0],
                color=GRID_COL, lw=0.8, alpha=0.9)

# ─────────────────────────────────────────────────────────────────
#  STYLE AXES
# ─────────────────────────────────────────────────────────────────
def style_axes():
    ax.set_facecolor(BG_COL)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#1e1e3a')
    ax.grid(True, color='#1e1e3a', alpha=0.5)
    ax.set_xlim([-50, 250])
    ax.set_ylim([-50, 250])
    ax.set_zlim([0, 250])
    ax.set_xlabel('X  Width (cm)',  color=TEXT_COL, labelpad=8)
    ax.set_ylabel('Y  Depth (cm)',  color=TEXT_COL, labelpad=8)
    ax.set_zlabel('Z  Height (cm)', color=TEXT_COL, labelpad=8)
    ax.tick_params(colors='#606090', labelsize=7)
    ax.set_title(
        'Live Full-Body UWB Motion Capture  ·  6 Tags / 4 Anchors',
        color=TEXT_COL, fontsize=11, pad=10,
    )
    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], color=BODY_COL, lw=4, label='Skeleton'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=TAG_COL,
               markersize=8, label='UWB Tag'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=ANCHOR_COL,
               markersize=8, label='UWB Anchor'),
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              facecolor='#1a1a30', edgecolor='#3a3a5a',
              labelcolor=TEXT_COL, fontsize=8)

# ─────────────────────────────────────────────────────────────────
#  ANIMATION UPDATE  (called by FuncAnimation every 33 ms ≈ 30 fps)
# ─────────────────────────────────────────────────────────────────
def update(_frame):
    # ── Drain the serial buffer completely each frame ──────────────
    if ser:
        try:
            while ser.in_waiting > 0:
                raw_line = ser.readline().decode('UTF-8', errors='ignore').strip()
                if not raw_line.startswith('{'):
                    continue
                d      = json.loads(raw_line)
                t_id   = d.get('id', -1)
                ranges = d.get('range', [])
                valid  = sum(1 for r in ranges[:4] if r > 0)
                if valid >= 3 and 0 <= t_id <= 5:
                    computed      = solve_position(ranges[:4], raw_pos[t_id])
                    raw_pos[t_id] = computed
                    # Apply EMA smoothing
                    smoothed_pos[t_id] = (
                        EMA_ALPHA * computed +
                        (1.0 - EMA_ALPHA) * smoothed_pos[t_id]
                    )
        except Exception:
            pass   # silently skip malformed packets

    # ── Rebuild scene every frame ─────────────────────────────────
    ax.cla()
    style_axes()
    draw_floor()
    draw_anchors()

    skel = build_skeleton(smoothed_pos)
    draw_body(skel)
    draw_tags(skel)

    return []   # blit=False, so return value unused

# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Live Full-Body 3D Motion Capture Active — close window to exit.\n")
    ani = animation.FuncAnimation(
        fig, update,
        interval=33,          # ~30 fps
        blit=False,           # 3-D axes don't support blitting
        cache_frame_data=False,
    )
    plt.show()

    if ser:
        ser.close()
        print("[✓] Serial port closed.")