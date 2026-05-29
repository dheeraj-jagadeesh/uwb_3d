"""
═══════════════════════════════════════════════════════════════════
  UWB FULL-BODY MOTION CAPTURE  —  v4.0
  PyVista smooth mesh renderer  |  6 Tags  |  4 Anchors
═══════════════════════════════════════════════════════════════════

  Install (one-time):
      pip install pyvista pyserial scipy numpy

  What changed from v3:
    • Renderer swapped from matplotlib → PyVista (OpenGL via VTK)
      This gives smooth, shaded, solid geometry — not cylinders
      joined with visible seams.
    • Foot calculation fixed: foot now follows the hip→knee
      direction vector instead of projecting straight to z=0.
    • Anchor heights (70 cm, 62 cm) baked in from your measurements.
    • Real knee height (50 cm) and belly (103 cm) defaults.
    • Two-light rig gives the same plastic-mannequin look as
      the reference image.
═══════════════════════════════════════════════════════════════════
"""

import os
# Required before importing pyvista in headless / remote sessions.
# On a normal desktop Windows machine this line does nothing.
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

pv.global_theme.smooth_shading = True   # default smooth for every mesh

# ═══════════════════════════════════════════════════════════════════
#  1.  ANCHOR POSITIONS  (cm)  — YOUR ACTUAL MEASUREMENTS
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
LOWER_LEG_LEN  = 43    # cm  knee-to-ankle (knee at 50 cm → foot at ~7 cm)
EMA_ALPHA      = 0.18  # EMA smoothing  (0.10=smooth, 0.30=snappy)

# ═══════════════════════════════════════════════════════════════════
#  3.  SERIAL PORT
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
    print(f"[✓] Connected to UWB master on {COM_PORT}")
except Exception as e:
    print(f"[!] Serial unavailable ({e})  —  static preview mode")

# ═══════════════════════════════════════════════════════════════════
#  4.  TAG DEFAULTS  (based on your real body measurements)
# ═══════════════════════════════════════════════════════════════════
_DEFAULTS = {
    0: np.array([ 60., 100., 115.]),   # T0  L-Wrist   (arms at sides)
    1: np.array([140., 100., 115.]),   # T1  R-Wrist
    2: np.array([ 85., 100.,  50.]),   # T2  L-Knee    (your measurement)
    3: np.array([115., 100.,  50.]),   # T3  R-Knee
    4: np.array([100., 100., 172.]),   # T4  Head      (your measurement)
    5: np.array([100., 100., 103.]),   # T5  Belly     (your measurement)
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
    """Drain all pending serial packets and update smoothed positions."""
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
#  6.  KINEMATICS  —  derive all joints from the 6 tracked tags
# ═══════════════════════════════════════════════════════════════════
def build_skeleton(sp):
    LW = sp[0]; RW = sp[1]; LK = sp[2]
    RK = sp[3]; HD = sp[4]; BL = sp[5]

    spine  = HD - BL
    s_unit = spine / max(np.linalg.norm(spine), 1e-6)
    NECK   = BL + 0.78 * spine    # 78% up the spine
    PELVIS = BL - 0.18 * spine    # slightly below belly

    # Lateral axis (person assumed to face +Y)
    lat   = np.cross(s_unit, np.array([0., 1., 0.]))
    lat_L = np.linalg.norm(lat)
    lat   = lat / lat_L if lat_L > 1e-6 else np.array([1., 0., 0.])

    LS = NECK   + lat * (SHOULDER_WIDTH / 2)
    RS = NECK   - lat * (SHOULDER_WIDTH / 2)
    LH = PELVIS + lat * (HIP_WIDTH / 2)
    RH = PELVIS - lat * (HIP_WIDTH / 2)

    # Estimated elbows: 60% from shoulder, 40% from wrist
    LE = LS * 0.60 + LW * 0.40
    RE = RS * 0.60 + RW * 0.40

    # ── FIXED FOOT CALCULATION ────────────────────────────────────
    # Extend the hip→knee direction vector below the knee.
    # This means when you lift your knee, the foot lifts & swings
    # with it rather than staying glued to z = 0.
    def est_foot(hip, knee):
        v = knee - hip
        n = max(np.linalg.norm(v), 1e-6)
        foot = (knee + (v / n) * LOWER_LEG_LEN).copy()
        foot[2] = max(foot[2], 0.0)   # never underground
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
#  7.  SMOOTH CAPSULE PRIMITIVE
# ═══════════════════════════════════════════════════════════════════
def make_capsule(p1, p2, radius, res=20):
    """
    Smooth capsule = cylinder + two hemisphere end-caps.
    High res + smooth_shading makes seams invisible.
    """
    v = p2 - p1
    L = np.linalg.norm(v)
    if L < 0.5:
        return None
    ctr = (p1 + p2) / 2
    cyl = pv.Cylinder(
        center=ctr.tolist(), direction=v.tolist(),
        radius=radius, height=L, resolution=res, capping=True,
    )
    s1 = pv.Sphere(radius=radius, center=p1.tolist(),
                   theta_resolution=res, phi_resolution=res)
    s2 = pv.Sphere(radius=radius, center=p2.tolist(),
                   theta_resolution=res, phi_resolution=res)
    return pv.merge([cyl, s1, s2])

# ═══════════════════════════════════════════════════════════════════
#  8.  BODY MESH ASSEMBLER
# ═══════════════════════════════════════════════════════════════════
def build_body_mesh(j):
    """
    Assemble all body-part capsules into one merged mesh.
    Merging into a single PolyData lets PyVista compute
    unified smooth normals → no visible seams.
    """
    parts = []

    def cap(p1, p2, r, res=20):
        m = make_capsule(p1, p2, r, res)
        if m is not None:
            parts.append(m)

    # ── Head (high-res sphere) ───────────────────────────────────
    parts.append(pv.Sphere(
        radius=12, center=j['HEAD'].tolist(),
        theta_resolution=30, phi_resolution=30,
    ))

    # ── Neck ────────────────────────────────────────────────────
    cap(j['HEAD'] - np.array([0., 0., 12.]), j['NECK'], 5.5, 16)

    # ── Torso — 3 zones for chest / waist / hip taper ───────────
    chest_mid = j['NECK'] * 0.55 + j['BELLY'] * 0.45
    cap(j['NECK'],    chest_mid,   16, 22)   # pectoral / upper chest
    cap(chest_mid,    j['BELLY'],  13, 22)   # waist / abdomen
    cap(j['BELLY'],   j['PELVIS'], 15, 22)   # hip flare

    # ── Shoulder crossbar ───────────────────────────────────────
    cap(j['LS'], j['RS'], 8, 18)

    # ── Hip crossbar ────────────────────────────────────────────
    cap(j['LH'], j['RH'], 9, 18)

    # ── Left arm ────────────────────────────────────────────────
    cap(j['LS'], j['LE'], 6.0, 18)   # upper arm
    cap(j['LE'], j['LW'], 4.5, 18)   # forearm

    # ── Right arm ───────────────────────────────────────────────
    cap(j['RS'], j['RE'], 6.0, 18)
    cap(j['RE'], j['RW'], 4.5, 18)

    # ── Left leg ────────────────────────────────────────────────
    cap(j['LH'], j['LK'], 9.0, 20)   # thigh
    cap(j['LK'], j['LF'], 7.5, 20)   # calf + foot

    # ── Right leg ───────────────────────────────────────────────
    cap(j['RH'], j['RK'], 9.0, 20)
    cap(j['RK'], j['RF'], 7.5, 20)

    return pv.merge(parts)

# ═══════════════════════════════════════════════════════════════════
#  9.  PLOTTER SETUP
# ═══════════════════════════════════════════════════════════════════
BODY_COLOR   = '#c8c8c8'   # smooth light-grey mannequin
TAG_COLOR    = '#8e44ad'   # purple tag spheres
ANCHOR_COLOR = '#e67e22'   # amber anchor markers

plotter = pv.Plotter(
    window_size=[900, 900],
    title='UWB Full-Body Motion Capture  v4.0',
)
plotter.background_color = 'white'

# Camera — elevated, slightly off-axis (matches reference image feel)
plotter.camera_position = [
    (100, -380, 210),   # camera XYZ (in front of & above the person)
    (100,  100, 100),   # focal point (centre of room / body)
    (  0,    0,   1),   # up direction
]

# Two-light rig for the smooth plastic look of the reference image
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

# Floor grid (subtle)
floor = pv.Plane(
    center=(100, 100, 0), direction=(0, 0, 1),
    i_size=320, j_size=320, i_resolution=8, j_resolution=8,
)
plotter.add_mesh(floor, color='#dddddd', style='wireframe', opacity=0.55)

# Static anchor markers (added once)
for i, anc in enumerate(ANCHORS):
    plotter.add_mesh(
        pv.Sphere(radius=5, center=anc.tolist()),
        color=ANCHOR_COLOR, smooth_shading=True,
    )
    plotter.add_point_labels(
        pv.PolyData((anc + np.array([0., 0., 10.])).reshape(1, 3)),
        [f'A{i}  ({int(anc[2])} cm)'],
        font_size=10, text_color=ANCHOR_COLOR,
        fill_shape=False, shape_opacity=0, always_visible=True,
    )

print("\nLive Full-Body 3D Motion Capture Active — close window to exit.\n")
plotter.show(auto_close=False, interactive_update=True)

# ═══════════════════════════════════════════════════════════════════
#  10.  MAIN ANIMATION LOOP
# ═══════════════════════════════════════════════════════════════════
body_actor  = None
tag_actor   = None
lbl_actors  = []

try:
    while True:
        t0 = time.time()

        # ── Step 1: Read all pending UWB packets ──────────────────
        read_serial()

        # ── Step 2: Build skeleton & mesh ─────────────────────────
        skel      = build_skeleton(smoothed_pos)
        body_mesh = build_body_mesh(skel)

        # ── Step 3: Swap body actor ───────────────────────────────
        if body_actor is not None:
            plotter.remove_actor(body_actor)
        body_actor = plotter.add_mesh(
            body_mesh,
            color=BODY_COLOR,
            smooth_shading=True,
            lighting=True,
        )

        # ── Step 4: Swap tag markers & labels ─────────────────────
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

        # ── Step 5: Render frame ──────────────────────────────────
        plotter.update(1, force_redraw=True)

        # ── Step 6: Pace to ≈ 20 fps (leave CPU for serial reads) ─
        dt = time.time() - t0
        if dt < 0.05:
            time.sleep(0.05 - dt)

except Exception as e:
    print(f"[Loop ended]: {e}")

finally:
    if ser:
        ser.close()
        print("[✓] Serial port closed.")