"""
╔══════════════════════════════════════════════════════════════════╗
║      LIVE FULL-BODY UWB MOTION CAPTURE  —  v3.0                 ║
║  6 Tags  |  4 Anchors  |  ESP32-S3 DW3000                        ║
║                                                                   ║
║  Renderer: 3D parametric surfaces (spheres + elliptic cylinders) ║
║  Smoother: Exponential Moving Average on every tag position      ║
║  Loop:     FuncAnimation at 30fps target                         ║
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

# ═══════════════════════════════════════════════════════════════════
#  1.  ANCHOR POSITIONS  (cm)  —  update to your actual room tape!
# ═══════════════════════════════════════════════════════════════════
ANCHORS = np.array([
    [  0,   0,   0],   # A0  Master  – floor corner
    [200,   0,   0],   # A1          – ground reference corner
    [200, 200, 70],   # A2          – mid-height wall
    [  0, 200,62 ],   # A3          – high ceiling corner
])

SHOULDER_WIDTH = 40   # cm, side to side
HIP_WIDTH      = 30   # cm, side to side

# ═══════════════════════════════════════════════════════════════════
#  2.  SMOOTHING — EMA alpha
#      0.10 = very smooth, ~250 ms lag
#      0.18 = recommended balance
#      0.30 = snappier, some residual jitter
# ═══════════════════════════════════════════════════════════════════
EMA_ALPHA = 0.18

# ═══════════════════════════════════════════════════════════════════
#  3.  TAG DEFAULT POSITIONS  (used until first real reading arrives)
# ═══════════════════════════════════════════════════════════════════
_DEFAULTS = {
    0: np.array([ 60.0, 100.0, 115.0]),   # T0 — Left Wrist
    1: np.array([140.0, 100.0, 115.0]),   # T1 — Right Wrist
    2: np.array([ 85.0, 100.0,  52.0]),   # T2 — Left Knee
    3: np.array([115.0, 100.0,  52.0]),   # T3 — Right Knee
    4: np.array([100.0, 100.0, 172.0]),   # T4 — Head
    5: np.array([100.0, 100.0, 103.0]),   # T5 — Belly Button
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

# ═══════════════════════════════════════════════════════════════════
#  4.  MULTILATERATION  (Nelder-Mead least-squares)
# ═══════════════════════════════════════════════════════════════════
def _residual(pt, dists):
    err = 0.0
    for i in range(4):
        if dists[i] > 0:
            err += (np.linalg.norm(pt - ANCHORS[i]) - dists[i]) ** 2
    return err

def solve_position(dists, seed):
    return minimize(
        _residual, seed, args=(dists,), method='Nelder-Mead',
        options={'xatol': 0.8, 'fatol': 0.8, 'maxiter': 150},
    ).x

# ═══════════════════════════════════════════════════════════════════
#  5.  SERIAL  (master anchor on COM19)
# ═══════════════════════════════════════════════════════════════════
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
    print(f"[!] Serial unavailable ({e})  →  static preview mode")

# ═══════════════════════════════════════════════════════════════════
#  6.  3-D GEOMETRY HELPERS
# ═══════════════════════════════════════════════════════════════════
BODY_C   = '#b5b5b5'   # grey mannequin
TAG_C    = '#9b59b6'   # purple dots
ANCHOR_C = '#f1c40f'   # yellow triangles

def _ortho_frame(v):
    """Return two unit vectors orthogonal to v."""
    u = v / np.linalg.norm(v)
    ref = np.array([1, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1, 0])
    e1  = np.cross(u, ref);  e1 /= np.linalg.norm(e1)
    e2  = np.cross(u, e1)
    return e1, e2


def draw_sphere(ax, center, r, n=16, color=BODY_C):
    """Render a sphere as a parametric surface."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi,     n)
    x = center[0] + r * np.outer(np.cos(u), np.sin(v))
    y = center[1] + r * np.outer(np.sin(u), np.sin(v))
    z = center[2] + r * np.outer(np.ones(n), np.cos(v))
    ax.plot_surface(x, y, z, color=color,
                    shade=True, linewidth=0, antialiased=False)


def draw_segment(ax, p1, p2, r1, r2=None, ex=1.0, ey=1.0, n=10, color=BODY_C):
    """
    Render a tapered elliptic cylinder from p1 to p2.
    r1/r2 = radius at base/tip.  ex/ey scale the elliptic cross-section.
    """
    if r2 is None:
        r2 = r1
    v = p2 - p1
    L = np.linalg.norm(v)
    if L < 0.5:
        return
    e1, e2 = _ortho_frame(v)
    theta   = np.linspace(0, 2 * np.pi, n + 1)
    c, s    = np.cos(theta), np.sin(theta)

    bot = p1 + r1 * (ex * c[:, None] * e1 + ey * s[:, None] * e2)
    top = p2 + r2 * (ex * c[:, None] * e1 + ey * s[:, None] * e2)

    X = np.array([bot[:, 0], top[:, 0]])
    Y = np.array([bot[:, 1], top[:, 1]])
    Z = np.array([bot[:, 2], top[:, 2]])

    ax.plot_surface(X, Y, Z, color=color,
                    shade=True, linewidth=0, antialiased=False)

# ═══════════════════════════════════════════════════════════════════
#  7.  KINEMATICS — derive all joints from the 6 raw tag positions
# ═══════════════════════════════════════════════════════════════════
def build_skeleton(sp):
    L_WRIST = sp[0]
    R_WRIST = sp[1]
    L_KNEE  = sp[2]
    R_KNEE  = sp[3]
    HEAD    = sp[4]
    BELLY   = sp[5]

    spine  = HEAD - BELLY
    s_len  = max(np.linalg.norm(spine), 1e-6)
    s_unit = spine / s_len

    NECK   = BELLY + 0.78 * spine   # ~78 % up the spine
    PELVIS = BELLY - 0.18 * spine   # slightly below belly

    # Lateral axis — assumes subject faces the +Y direction
    fwd    = np.array([0.0, 1.0, 0.0])
    lat    = np.cross(s_unit, fwd)
    lat_L  = np.linalg.norm(lat)
    lat    = lat / lat_L if lat_L > 1e-6 else np.array([1.0, 0.0, 0.0])

    L_SHOULDER = NECK   + lat * (SHOULDER_WIDTH / 2)
    R_SHOULDER = NECK   - lat * (SHOULDER_WIDTH / 2)
    L_HIP      = PELVIS + lat * (HIP_WIDTH / 2)
    R_HIP      = PELVIS - lat * (HIP_WIDTH / 2)

    # Estimated elbows (60 % shoulder, 40 % wrist) for a natural arm bend
    L_ELBOW = L_SHOULDER * 0.60 + L_WRIST * 0.40
    R_ELBOW = R_SHOULDER * 0.60 + R_WRIST * 0.40

    # Feet project straight down from knees to floor (z = 0)
    L_FOOT = np.array([L_KNEE[0], L_KNEE[1], 0.0])
    R_FOOT = np.array([R_KNEE[0], R_KNEE[1], 0.0])

    return dict(
        HEAD=HEAD, NECK=NECK, BELLY=BELLY, PELVIS=PELVIS,
        LS=L_SHOULDER, RS=R_SHOULDER,
        LH=L_HIP,      RH=R_HIP,
        LW=L_WRIST,    RW=R_WRIST,
        LE=L_ELBOW,    RE=R_ELBOW,
        LK=L_KNEE,     RK=R_KNEE,
        LF=L_FOOT,     RF=R_FOOT,
    )

# ═══════════════════════════════════════════════════════════════════
#  8.  MANNEQUIN RENDERER
# ═══════════════════════════════════════════════════════════════════
def draw_body(ax, j):
    # ── Head ──────────────────────────────────────────────────────
    draw_sphere(ax, j['HEAD'], 12, n=18)

    # ── Neck ──────────────────────────────────────────────────────
    neck_base = j['HEAD'] - np.array([0, 0, 12])
    draw_segment(ax, neck_base, j['NECK'], 5.5, 6.2, n=8)

    # ── Shoulder bar (joins left & right shoulder caps) ────────────
    draw_segment(ax, j['LS'], j['RS'], 8.0, 8.0, ex=0.55, ey=1.0, n=10)
    draw_sphere(ax, j['LS'], 7.5, n=10)
    draw_sphere(ax, j['RS'], 7.5, n=10)

    # ── Upper torso (chest + abdomen, elliptic cross-section) ─────
    draw_segment(ax, j['NECK'], j['BELLY'], 17, 13, ex=1.0, ey=0.62, n=14)

    # ── Lower torso / hip flare ───────────────────────────────────
    draw_segment(ax, j['BELLY'], j['PELVIS'], 13, 16, ex=1.05, ey=0.65, n=14)

    # ── Hip bar & caps ────────────────────────────────────────────
    draw_segment(ax, j['LH'], j['RH'], 8.5, 8.5, ex=0.55, ey=1.0, n=10)
    draw_sphere(ax, j['LH'], 8.5, n=10)
    draw_sphere(ax, j['RH'], 8.5, n=10)

    # ── Arms ──────────────────────────────────────────────────────
    draw_segment(ax, j['LS'], j['LE'], 6.2, 4.8, n=10)   # upper arm L
    draw_sphere(ax,  j['LE'], 4.5, n=8)                   # elbow L
    draw_segment(ax, j['LE'], j['LW'], 4.5, 3.2, n=10)   # forearm L
    draw_sphere(ax,  j['LW'], 3.8, n=8)                   # hand L

    draw_segment(ax, j['RS'], j['RE'], 6.2, 4.8, n=10)   # upper arm R
    draw_sphere(ax,  j['RE'], 4.5, n=8)                   # elbow R
    draw_segment(ax, j['RE'], j['RW'], 4.5, 3.2, n=10)   # forearm R
    draw_sphere(ax,  j['RW'], 3.8, n=8)                   # hand R

    # ── Legs ──────────────────────────────────────────────────────
    draw_segment(ax, j['LH'], j['LK'], 9.5, 7.5, n=12)   # thigh L
    draw_sphere(ax,  j['LK'], 7.0, n=10)                  # knee L
    draw_segment(ax, j['LK'], j['LF'], 6.5, 4.2, n=12)   # calf L
    draw_sphere(ax,  j['LF'], 5.5, n=8)                   # foot L

    draw_segment(ax, j['RH'], j['RK'], 9.5, 7.5, n=12)   # thigh R
    draw_sphere(ax,  j['RK'], 7.0, n=10)                  # knee R
    draw_segment(ax, j['RK'], j['RF'], 6.5, 4.2, n=12)   # calf R
    draw_sphere(ax,  j['RF'], 5.5, n=8)                   # foot R


def draw_tags(ax, j):
    """Purple marker dots + text labels on each UWB tag site."""
    pts = {
        0: j['LW'], 1: j['RW'],
        2: j['LK'], 3: j['RK'],
        4: j['HEAD'], 5: j['BELLY'],
    }
    for tid, pos in pts.items():
        ax.scatter(*pos, c=TAG_C, s=170,
                   edgecolors='white', linewidths=1.5,
                   depthshade=False, zorder=10)
        ax.text(pos[0] + 5, pos[1] + 5, pos[2] + 7,
                TAG_LABELS[tid],
                fontsize=8, color=TAG_C, fontweight='bold', zorder=11)


def draw_anchors(ax):
    """Yellow triangle markers for the 4 UWB anchors."""
    names = ['A0  Master', 'A1', 'A2', 'A3']
    for i, anc in enumerate(ANCHORS):
        ax.scatter(*anc, c=ANCHOR_C, s=160, marker='^',
                   depthshade=False, zorder=9)
        ax.text(anc[0] + 4, anc[1] + 4, anc[2] + 7,
                names[i], fontsize=8,
                color=ANCHOR_C, fontweight='bold')


def draw_floor(ax):
    """Subtle grid on z = 0."""
    gc = '#cccccc'
    for v in range(-50, 251, 50):
        ax.plot([v, v], [-50, 250], [0, 0], color=gc, lw=0.7, alpha=0.8)
        ax.plot([-50, 250], [v, v], [0, 0], color=gc, lw=0.7, alpha=0.8)

# ═══════════════════════════════════════════════════════════════════
#  9.  FIGURE & AXIS SETUP
# ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(11, 10), facecolor='white')
fig.canvas.manager.set_window_title('UWB Full-Body Motion Capture  v3.0')
ax  = fig.add_subplot(111, projection='3d', facecolor='white')
fig.subplots_adjust(left=0.0, right=1.0, top=0.96, bottom=0.0)

# Set a nice default view angle (can rotate interactively at runtime)
ax.view_init(elev=15, azim=-65)


def _style_axes():
    ax.set_facecolor('white')
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill      = True
        pane.set_facecolor('#f4f4f4')
        pane.set_edgecolor('#cccccc')
    ax.grid(True, color='#dddddd', alpha=0.9)
    ax.set_xlim([-50, 250])
    ax.set_ylim([-50, 250])
    ax.set_zlim([  0, 250])
    ax.set_xlabel('X  Width (cm)',  color='#444444', labelpad=8)
    ax.set_ylabel('Y  Depth (cm)',  color='#444444', labelpad=8)
    ax.set_zlabel('Z  Height (cm)', color='#444444', labelpad=8)
    ax.tick_params(colors='#888888', labelsize=7)
    ax.set_title(
        'Live Full-Body UWB Motion Capture  ·  6 Tags / 4 Anchors',
        color='#222222', fontsize=11, pad=10,
    )
    # Minimal legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=BODY_C,
               markersize=10, label='Body mesh'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=TAG_C,
               markersize=9,  label='UWB tag'),
        Line2D([0],[0], marker='^', color='w', markerfacecolor=ANCHOR_C,
               markersize=9,  label='UWB anchor'),
    ]
    ax.legend(handles=legend_elems, loc='upper left',
              facecolor='white', edgecolor='#cccccc',
              labelcolor='#333333', fontsize=8)

# ═══════════════════════════════════════════════════════════════════
#  10.  ANIMATION UPDATE  (~30 fps via FuncAnimation)
# ═══════════════════════════════════════════════════════════════════
def update(_frame):
    # ── Drain all pending serial packets this frame ────────────────
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
            pass   # silently drop malformed packets

    # ── Redraw ────────────────────────────────────────────────────
    ax.cla()
    _style_axes()
    draw_floor(ax)
    draw_anchors(ax)

    skel = build_skeleton(smoothed_pos)
    draw_body(ax, skel)
    draw_tags(ax, skel)

    return []

# ═══════════════════════════════════════════════════════════════════
#  11.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Live Full-Body 3D Motion Capture Active — close window to exit.\n")
    ani = animation.FuncAnimation(
        fig, update,
        interval=33,             # ~30 fps target
        blit=False,              # 3-D axes don't support blitting
        cache_frame_data=False,
    )
    plt.show()

    if ser:
        ser.close()
        print("[✓] Serial port closed.")