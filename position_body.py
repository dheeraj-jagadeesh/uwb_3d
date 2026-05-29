import serial
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

# --- PHYSICAL ROOM ANCHOR CONFIGURATION (IN CM) ---
# Ensure these match the exact tape measurements of your 4 modules in the room!
ANCHORS = np.array([
    [  0,   0,   0],   # A0  Master  – floor corner
    [225,   0,   0],   # A1          – floor corner
    [225, 310,  115],   # A2          – mid-wall
    [  0, 300,  125],   # A3          – low corner
], dtype=float)
# Structural Skeletal Offsets (in cm)
SHOULDER_WIDTH = 40
HIP_WIDTH = 30

# Initialize all 6 Tracked Nodes to a default standing position at the center of the room
tag_positions = {
    0: np.array([60.0,  100.0, 110.0]), # Tag 0: Left Wrist
    1: np.array([140.0, 100.0, 110.0]), # Tag 1: Right Wrist
    2: np.array([85.0,  100.0, 50.0]),  # Tag 2: Left Knee
    3: np.array([115.0, 100.0, 50.0]),  # Tag 3: Right Knee
    4: np.array([100.0, 100.0, 170.0]), # Tag 4: Head (Absolute Moving Anchor)
    5: np.array([100.0, 100.0, 100.0])  # Tag 5: Belly Button (Absolute Moving Anchor)
}

# 3D Multilateration Least-Squares Optimization Solver
def error_distance_function(point, distances):
    error = 0
    for i in range(4):
        if distances[i] <= 0:
            continue
        calculated_dist = np.linalg.norm(point - ANCHORS[i])
        error += (calculated_dist - distances[i]) ** 2
    return error

def calculate_3d_position(distances, last_known_pos):
    # Use last known position as the optimization seed to speed up convergence
    result = minimize(error_distance_function, last_known_pos, args=(distances,), method='Nelder-Mead')
    return result.x

# Establish direct link to Master Anchor Module
com_port = "COM19"
try:
    ser = serial.Serial(port=com_port, baudrate=115200, timeout=0.01, dsrdtr=False, rtscts=False)
    ser.setDTR(False)
    ser.setRTS(False)
    time.sleep(0.5)
    ser.reset_input_buffer()
    print(f"Connected to UWB Master Array on port: {com_port}")
except Exception as e:
    print(f"CRITICAL: Could not open connection to {com_port}: {e}")
    ser = None

# Initialize Interactive 3D Matplotlib Plotting Space
plt.ion()
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

print("Live Full-Body 3D Motion Capture Active. Close the window to exit.")

while plt.fignum_exists(fig.number):
    if ser and ser.in_waiting > 0:
        try:
            line = ser.readline().decode('UTF-8', errors='ignore').strip()
            if line.startswith("{"):
                data = json.loads(line)
                t_id = data['id']
                ranges = data['range']
                
                # Ensure at least 3 anchors provide clean range information
                valid_ranges = sum(1 for r in ranges[:4] if r > 0)
                if valid_ranges >= 3 and 0 <= t_id <= 5:
                    # Update the respective joint coordinates dynamically
                    computed_xyz = calculate_3d_position(ranges[:4], tag_positions[t_id])
                    tag_positions[t_id] = computed_xyz
        except Exception:
            pass

    # Clear previous visualization frame
    ax.cla()
    
    # Extract calculated coordinates for clearer kinematic linking
    L_WRIST  = tag_positions[0]
    R_WRIST  = tag_positions[1]
    L_KNEE   = tag_positions[2]
    R_KNEE   = tag_positions[3]
    HEAD     = tag_positions[4]
    BELLY    = tag_positions[5]

    # --- KINEMATIC SKELETON VECTOR MATH ---
    # Construct spine vector orientation
    spine_vector = HEAD - BELLY
    spine_unit = spine_vector / np.linalg.norm(spine_vector)
    
    # Calculate joint junctions along the spine vector
    NECK = BELLY + 0.75 * spine_vector
    PELVIS = BELLY - 0.15 * spine_vector

    # Generate lateral offsets assuming the user faces forward along the Y-axis
    left_offset = np.array([-1, 0, 0])
    right_offset = np.array([1, 0, 0])

    # Map moving Shoulder and Hip boundaries
    L_SHOULDER = NECK + (left_offset * (SHOULDER_WIDTH / 2))
    R_SHOULDER = NECK + (right_offset * (SHOULDER_WIDTH / 2))
    L_HIP      = PELVIS + (left_offset * (HIP_WIDTH / 2))
    R_HIP      = PELVIS + (right_offset * (HIP_WIDTH / 2))
    
    # Extrapolate feet locations to drop to floor baseline naturally
    L_FOOT     = np.array([L_KNEE[0], L_KNEE[1], 0])
    R_FOOT     = np.array([R_KNEE[0], R_KNEE[1], 0])

    # --- RENDER SKELETON SEGMENTS ---
    # Draw Torso Core Structure
    ax.plot([HEAD[0], NECK[0]], [HEAD[1], NECK[1]], [HEAD[2], NECK[2]], color='blue', linewidth=4)
    ax.plot([L_SHOULDER[0], R_SHOULDER[0]], [L_SHOULDER[1], R_SHOULDER[1]], [L_SHOULDER[2], R_SHOULDER[2]], color='blue', linewidth=4)
    ax.plot([NECK[0], PELVIS[0]], [NECK[1], PELVIS[1]], [NECK[2], PELVIS[2]], color='blue', linewidth=4)
    ax.plot([L_HIP[0], R_HIP[0]], [L_HIP[1], R_HIP[1]], [L_HIP[2], R_HIP[2]], color='blue', linewidth=4)

    # Draw Arms (Shoulders connected to live moving Wrists T0 and T1)
    ax.plot([L_SHOULDER[0], L_WRIST[0]], [L_SHOULDER[1], L_WRIST[1]], [L_SHOULDER[2], L_WRIST[2]], color='red', linewidth=3, label='Left Arm')
    ax.plot([R_SHOULDER[0], R_WRIST[0]], [R_SHOULDER[1], R_WRIST[1]], [R_SHOULDER[2], R_WRIST[2]], color='green', linewidth=3, label='Right Arm')

    # Draw Legs (Hips connected to live moving Knees T2 and T3 downwards to feet)
    ax.plot([L_HIP[0], L_KNEE[0]], [L_HIP[1], L_KNEE[1]], [L_HIP[2], L_KNEE[2]], color='orange', linewidth=3, label='Left Leg')
    ax.plot([L_KNEE[0], L_FOOT[0]], [L_KNEE[1], L_FOOT[1]], [L_KNEE[2], L_FOOT[2]], color='orange', linewidth=3)
    ax.plot([R_HIP[0], R_KNEE[0]], [R_HIP[1], R_KNEE[1]], [R_HIP[2], R_KNEE[2]], color='purple', linewidth=3, label='Right Leg')
    ax.plot([R_KNEE[0], R_FOOT[0]], [R_KNEE[1], R_FOOT[1]], [R_KNEE[2], R_FOOT[2]], color='purple', linewidth=3)

    # Plot joint marker nodes
    all_joints = np.array([HEAD, NECK, PELVIS, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_WRIST, R_WRIST, L_KNEE, R_KNEE, L_FOOT, R_FOOT])
    ax.scatter(all_joints[:,0], all_joints[:,1], all_joints[:,2], c='blue', s=40)
    
    # Highlight wearable tag trackers explicitly
    trackers = np.array([L_WRIST, R_WRIST, L_KNEE, R_KNEE, HEAD, BELLY])
    ax.scatter(trackers[:,0], trackers[:,1], trackers[:,2], c='cyan', s=100, edgecolors='darkblue', marker='o')

    # Bounding Box Boundaries
    ax.set_xlim([-50, 250])
    ax.set_ylim([-50, 250])
    ax.set_zlim([0, 250])
    ax.set_xlabel('X Width (cm)')
    ax.set_ylabel('Y Length (cm)')
    ax.set_zlabel('Z Height (cm)')
    ax.set_title('Live 6-Tag Dynamic Full-Body Mocap View')
    
    plt.draw()
    plt.pause(0.01)

if ser:
    ser.close()