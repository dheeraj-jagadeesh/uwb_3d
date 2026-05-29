"""
═══════════════════════════════════════════════════════════════════
  UWB FULL-BODY MOTION CAPTURE  —  v5.0
  PyVista smooth renderer  |  6 Tags  |  4 Anchors
═══════════════════════════════════════════════════════════════════

  Install (one-time):
      pip install pyvista pyserial scipy numpy

  What changed from v4:
    • Measurement grid restored on all 3 back walls
      (X / Y / Z axes with cm tick labels)
    • Camera pulled back so ALL 4 anchors + full body are visible
    • Coordinate-axis widget (XYZ arrows) in bottom-left corner
    • show_bounds() fixed for current PyVista API
      (xtitle / ytitle / ztitle  instead of deprecated xlabel etc.)
═══════════════════════════════════════════════════════════════════
"""

import os
os.environ.setdefault('PYVISTA_OFF_SCREEN', 'false')

try:
    import pyvista as pv
except ImportError:
    raise SystemExit("PyVista not found.  Run:  pip install pyvista")

import numpy as np
import serial
import json
import time
from scipy.optimize import minimize

pv.global_theme.smooth_shading = True

# ═══════════════════════════════════════════════════════════════════
#  1.  ANCHOR POSITIONS  (cm)  — your measured values
# ═══════════════════════════════════════════════════════════════════
ANCHORS = np.array([
    [  0,   0,   0],   # A0  Master  – floor corner
    [225,   0,   0],   # A1          – floor corner
    [225, 310,  115],   # A2          – mid-wall
    [  0, 300,  125],   # A3          – low corner
], dtype=float)

# ═══════════════════════════════════════════════════════════════════
#  2.  SKELETON CONSTANTS
# ═══════════════════════════════════════════════════════════════════
SHOULDER_WIDTH = 40    # cm  shoulder span
HIP_WIDTH      = 30    # cm  hip span
LOWER_LEG_LEN  = 43    # cm  knee→ankle  (knee at 50 cm → foot ≈ 7 cm)
EMA_ALPHA      = 0.18  # smoothing  (0.10 = smooth, 0.30 = snappy)

# ═══════════════════════════════════════════════════════════════════
#  3.  SERIAL
# ═══════════════════════════════════════════════════════════════════
COM_PORT = "COM19"
ser = None
try:
    ser = serial.Serial(
        COM_PORT, 115200, timeout=0.01, dsrdtr=False, rtscts=False,
    )
    ser.setDTR(False)
    ser.setRTS(False)
    time.sleep(0.5)
    ser.reset_input_buffer()
    print(f"[✓] Connected on {COM_PORT}")
except Exception as e:
    print(f"[!] Serial unavailable ({e})  —  static preview mode")

# ═══════════════════════════════════════════════════════════════════
#  4.  TAG DEFAULT POSITIONS  (from your body measurements)
# ═══════════════════════════════════════════════════════════════════
_DEFAULTS = {
    0: np.array([ 60., 100., 115.]),   # T0  L-Wrist
    1: np.array([140., 100., 115.]),   # T1  R-Wrist
    2: np.array([ 85., 100.,  50.]),   # T2  L-Knee
    3: np.array([115., 100.,  50.]),   # T3  R-Knee
    4: np.array([100., 100., 172.]),   # T4  Head
    5: np.array([100., 100., 103.]),   # T5  Belly
}
raw_pos      = {k: v.copy() for k, v in _DEFAULTS.items()}
smoothed_pos = {k: v.copy() for k, v in _DEFAULTS.items()}

TAG_LABELS = {
    0: 'T0  L-Wrist', 1: 'T1  R-Wrist',
    2: 'T2  L-Knee',  3: 'T3  R-Knee',
    4: 'T4  Head',    5: 'T5  Belly',
}

# ═══════════════════════════════════════════════════════════════════
#  5.  MULTILATERATION
# ═══════════════════════════════════════════════════════════════════
def _residual(pt, dists):
    return sum(
        (np.linalg.norm(pt - ANCHORS[i]) - dists[i]) ** 2
        for i in range(4) if dists[i] > 0
    )

def solve_position(dists, seed):
    return minimize(
        _residual, seed, args=(dists,), method='Nelder-Mead',
        options={'xatol': 0.8, 'fatol': 0.8, 'maxiter': 150},
    ).x

def read_serial():
    if not ser:
        return
    try:
        while ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line.startswith('{'):
                continue
            d   = json.loads(line)
            tid = d.get('id', -1)
            rng = d.get('range', [])
            if sum(1 for r in rng[:4] if r > 0) >= 3 and 0 <= tid <= 5:
                c = solve_position(rng[:4], raw_pos[tid])
                raw_pos[tid]      = c
                smoothed_pos[tid] = EMA_ALPHA * c + (1 - EMA_ALPHA) * smoothed_pos[tid]
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════════
#  6.  KINEMATICS
# ═══════════════════════════════════════════════════════════════════
def build_skeleton(sp):
    LW = sp[0]; RW = sp[1]; LK = sp[2]
    RK = sp[3]; HD = sp[4]; BL = sp[5]

    spine  = HD - BL
    s_unit = spine / max(np.linalg.norm(spine), 1e-6)
    NECK   = BL + 0.78 * spine
    PELVIS = BL - 0.18 * spine

    lat   = np.cross(s_unit, np.array([0., 1., 0.]))
    lat_L = np.linalg.norm(lat)
    lat   = lat / lat_L if lat_L > 1e-6 else np.array([1., 0., 0.])

    LS = NECK   + lat * (SHOULDER_WIDTH / 2)
    RS = NECK   - lat * (SHOULDER_WIDTH / 2)
    LH = PELVIS + lat * (HIP_WIDTH / 2)
    RH = PELVIS - lat * (HIP_WIDTH / 2)
    LE = LS * 0.60 + LW * 0.40
    RE = RS * 0.60 + RW * 0.40

    def est_foot(hip, knee):
        v    = knee - hip
        n    = max(np.linalg.norm(v), 1e-6)
        foot = (knee + (v / n) * LOWER_LEG_LEN).copy()
        foot[2] = max(foot[2], 0.0)
        return foot

    return dict(
        HEAD=HD, NECK=NECK, BELLY=BL, PELVIS=PELVIS,
        LS=LS, RS=RS, LH=LH, RH=RH,
        LE=LE, RE=RE, LW=LW, RW=RW,
        LK=LK, RK=RK,
        LF=est_foot(LH, LK),
        RF=est_foot(RH, RK),
    )

# ═══════════════════════════════════════════════════════════════════
#  7.  SMOOTH CAPSULE
# ═══════════════════════════════════════════════════════════════════
def make_capsule(p1, p2, radius, res=20):
    v = p2 - p1
    L = np.linalg.norm(v)
    if L < 0.5:
        return None
    ctr = (p1 + p2) / 2
    return pv.merge([
        pv.Cylinder(
            center=ctr.tolist(), direction=v.tolist(),
            radius=radius, height=L, resolution=res, capping=True,
        ),
        pv.Sphere(radius=radius, center=p1.tolist(),
                  theta_resolution=res, phi_resolution=res),
        pv.Sphere(radius=radius, center=p2.tolist(),
                  theta_resolution=res, phi_resolution=res),
    ])

# ═══════════════════════════════════════════════════════════════════
#  8.  BODY MESH
# ═══════════════════════════════════════════════════════════════════
def build_body_mesh(j):
    parts = []

    def cap(p1, p2, r, res=20):
        m = make_capsule(p1, p2, r, res)
        if m is not None:
            parts.append(m)

    # Head
    parts.append(pv.Sphere(
        radius=12, center=j['HEAD'].tolist(),
        theta_resolution=30, phi_resolution=30,
    ))
    # Neck
    cap(j['HEAD'] - np.array([0., 0., 12.]), j['NECK'], 5.5, 16)
    # Torso — 3 zones for chest / waist / hip taper
    chest_mid = j['NECK'] * 0.55 + j['BELLY'] * 0.45
    cap(j['NECK'],   chest_mid,   16, 22)
    cap(chest_mid,   j['BELLY'],  13, 22)
    cap(j['BELLY'],  j['PELVIS'], 15, 22)
    # Shoulder bar
    cap(j['LS'], j['RS'], 8, 18)
    # Hip bar
    cap(j['LH'], j['RH'], 9, 18)
    # Left arm
    cap(j['LS'], j['LE'], 6.0, 18)
    cap(j['LE'], j['LW'], 4.5, 18)
    # Right arm
    cap(j['RS'], j['RE'], 6.0, 18)
    cap(j['RE'], j['RW'], 4.5, 18)
    # Left leg
    cap(j['LH'], j['LK'], 9.0, 20)
    cap(j['LK'], j['LF'], 7.5, 20)
    # Right leg
    cap(j['RH'], j['RK'], 9.0, 20)
    cap(j['RK'], j['RF'], 7.5, 20)

    return pv.merge(parts)

# ═══════════════════════════════════════════════════════════════════
#  9.  PLOTTER — static scene elements built once
# ═══════════════════════════════════════════════════════════════════
BODY_COLOR   = '#c8c8c8'
TAG_COLOR    = '#8e44ad'   # purple
ANCHOR_COLOR = '#e67e22'   # amber

plotter = pv.Plotter(
    window_size=[1100, 900],
    title='UWB Full-Body Motion Capture  v5.0',
)
plotter.background_color = '#f5f5f5'

# ── Lighting ────────────────────────────────────────────────────
plotter.add_light(pv.Light(
    position=(400, -300, 500),
    focal_point=(100, 100, 100),
    intensity=0.75,
))
plotter.add_light(pv.Light(
    position=(-100, 300, 300),
    focal_point=(100, 100, 100),
    intensity=0.45,
))

# ── Measurement grid on all 3 back walls  ───────────────────────
# Room runs 0–200 cm in X & Y, body reaches ~220 cm in Z
plotter.show_bounds(
    bounds=[0, 200, 0, 200, 0, 220],
    grid='back',           # grid lines on back faces
    location='outer',      # rulers on the outer edges
    ticks='outside',       # tick marks outside the bounding box
    n_xlabels=5,           # tick count along X
    n_ylabels=5,           # tick count along Y
    n_zlabels=6,           # tick count along Z
    xtitle='X  Width (cm)',
    ytitle='Y  Depth (cm)',
    ztitle='Z  Height (cm)',
    font_size=11,
    bold=False,
    color='#333333',
    padding=0.03,          # small gap between scene and ruler
)

# ── XYZ orientation arrows (bottom-left corner) ─────────────────
plotter.add_axes(
    line_width=4,
    interactive=False,     # not draggable — just visual reference
)

# ── Anchor markers (static — added once, never removed) ─────────
for i, anc in enumerate(ANCHORS):
    plotter.add_mesh(
        pv.Sphere(radius=5, center=anc.tolist()),
        color=ANCHOR_COLOR, smooth_shading=True,
    )
    plotter.add_point_labels(
        pv.PolyData((anc + np.array([0., 0., 12.])).reshape(1, 3)),
        [f'A{i}  ({int(anc[2])} cm)'],
        font_size=10, text_color=ANCHOR_COLOR,
        fill_shape=False, shape_opacity=0, always_visible=True,
    )

# ── Camera: pull back so all 4 anchors + full body are in frame ─
# Room corners span X 0–200, Y 0–200.  A3 is at Y=200 far-left.
# Camera placed at negative Y (in front of the person) and
# to the left so the wide Y=200 wall is fully visible.
plotter.camera_position = [
    (-200, -450, 330),   # eye position
    ( 100,  100, 110),   # focal point  (centre of body / room)
    (   0,    0,   1),   # up direction
]

# ═══════════════════════════════════════════════════════════════════
#  10.  ANIMATION LOOP
# ═══════════════════════════════════════════════════════════════════
print("\nLive Full-Body 3D Motion Capture Active — close window to exit.\n")
plotter.show(auto_close=False, interactive_update=True)

body_actor = None
tag_actor  = None
lbl_actors = []

try:
    while True:
        t0 = time.time()

        # 1. Read all pending UWB packets
        read_serial()

        # 2. Build skeleton + mesh
        skel      = build_skeleton(smoothed_pos)
        body_mesh = build_body_mesh(skel)

        # 3. Swap body actor
        if body_actor is not None:
            plotter.remove_actor(body_actor)
        body_actor = plotter.add_mesh(
            body_mesh,
            color=BODY_COLOR,
            smooth_shading=True,
            lighting=True,
        )

        # 4. Swap tag markers + labels
        if tag_actor is not None:
            plotter.remove_actor(tag_actor)
        for a in lbl_actors:
            plotter.remove_actor(a)
        lbl_actors.clear()

        tag_pts = np.array([
            skel['LW'], skel['RW'],
            skel['LK'], skel['RK'],
            skel['HEAD'], skel['BELLY'],
        ])
        tag_actor = plotter.add_points(
            pv.PolyData(tag_pts),
            color=TAG_COLOR,
            point_size=22,
            render_points_as_spheres=True,
        )
        for pos, lbl in zip(tag_pts, TAG_LABELS.values()):
            a = plotter.add_point_labels(
                pv.PolyData((pos + np.array([0., 0., 9.])).reshape(1, 3)),
                [lbl],
                font_size=10, text_color=TAG_COLOR,
                fill_shape=False, shape_opacity=0, always_visible=True,
            )
            lbl_actors.append(a)

        # 5. Render
        plotter.update(1, force_redraw=True)

        # 6. Pace to ≈ 20 fps max
        dt = time.time() - t0
        if dt < 0.05:
            time.sleep(0.05 - dt)

except Exception as e:
    print(f"[Loop ended]: {e}")

finally:
    if ser:
        ser.close()
        print("[✓] Serial closed.")