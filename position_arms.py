import serial
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

# --- PHYSICAL SETUP CONFIGURATION (IN CM) ---
# Update these coordinates to match where you placed your anchors in the room!
ANCHORS = np.array([
    [  0,   0,   0],   # A0  Master  – floor corner
    [225,   0,   0],   # A1          – floor corner
    [225, 310,  115],   # A2          – mid-wall
    [  0, 300,  125],   # A3          – low corner
], dtype=float)

# Fixed Human Skeleton Dimensions (in cm) for Reference Body Frame
# This positions a mock torso in the center of your tracking zone
TORSO_CENTER_X = 100
TORSO_CENTER_Y = 100
FLOOR_Z = 0

SHOULDER_WIDTH = 40
SPINE_HEIGHT = 60

# Derive structural joint anchors
HEAD = np.array([TORSO_CENTER_X, TORSO_CENTER_Y, FLOOR_Z + SPINE_HEIGHT + 20])
NECK = np.array([TORSO_CENTER_X, TORSO_CENTER_Y, FLOOR_Z + SPINE_HEIGHT])
PELVIS = np.array([TORSO_CENTER_X, TORSO_CENTER_Y, FLOOR_Z + 20])
L_SHOULDER = np.array([TORSO_CENTER_X - (SHOULDER_WIDTH / 2), TORSO_CENTER_Y, FLOOR_Z + SPINE_HEIGHT])
R_SHOULDER = np.array([TORSO_CENTER_X + (SHOULDER_WIDTH / 2), TORSO_CENTER_Y, FLOOR_Z + SPINE_HEIGHT])

# Dynamic Target Positions Initial State 
tag_positions = {
    0: np.array([TORSO_CENTER_X - 40, TORSO_CENTER_Y, 40]), # Tag 0: Left Hand
    1: np.array([TORSO_CENTER_X + 40, TORSO_CENTER_Y, 40])  # Tag 1: Right Hand
}

# 3D Multilateration Optimization Solver Optimization Step
def error_distance_function(point, distances):
    error = 0
    for i in range(4):
        if distances[i] <= 0:
            continue
        calculated_dist = np.linalg.norm(point - ANCHORS[i])
        error += (calculated_dist - distances[i]) ** 2
    return error

def calculate_3d_position(distances):
    # Start guess at center of tracking envelope
    initial_guess = np.array([100.0, 100.0, 100.0])
    result = minimize(error_distance_function, initial_guess, args=(distances,), method='Nelder-Mead')
    return result.x

# Establish explicit pipeline to Master Module
com_port = "COM19"
try:
    ser = serial.Serial(port=com_port, baudrate=115200, timeout=0.01, dsrdtr=False, rtscts=False)
    ser.setDTR(False)
    ser.setRTS(False)
    time.sleep(0.5)
    ser.reset_input_buffer()
    print(f"Direct link to active array established on {com_port}")
except Exception as e:
    print(f"CRITICAL: Failed to bind to target interface {com_port}: {e}")
    ser = None

# Initialize Interactive Matplotlib Canvas Interface
plt.ion()
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subnet(111, projection='3d') if hasattr(fig, 'add_subnet') else fig.add_subplot(111, projection='3d')

print("Starting Kinematic Skeleton Projection Loop. Close plot frame to exit.")

while plt.fignum_exists(fig.number):
    if ser and ser.in_waiting > 0:
        try:
            line = ser.readline().decode('UTF-8', errors='ignore').strip()
            if line.startswith("{"):
                data = json.loads(line)
                t_id = data['id']
                ranges = data['range']
                
                # Verify at least 3 anchor frames provide safe distances
                valid_ranges = sum(1 for r in ranges[:4] if r > 0)
                if valid_ranges >= 3 and (t_id == 0 or t_id == 1):
                    # Compute 3D Point Coordinates
                    computed_xyz = calculate_3d_position(ranges[:4])
                    # Update active track coordinates
                    tag_positions[t_id] = computed_xyz
        except Exception:
            pass

    # Clear axes and draw updated frame vectors
    ax.cla()
    
    # 1. Plot Base Reference Stations (Anchors)
    ax.scatter(ANCHORS[:, 0], ANCHORS[:, 1], ANCHORS[:, 2], c='black', marker='^', s=100, label='Anchors')
    for i, txt in enumerate(['A0', 'A1', 'A2', 'A3']):
        ax.text(ANCHORS[i, 0], ANCHORS[i, 1], ANCHORS[i, 2] + 10, txt, color='black', fontsize=10)

    # 2. Render Stable Core Spine Structures
    ax.plot([HEAD[0], NECK[0]], [HEAD[1], NECK[1]], [HEAD[2], NECK[2]], color='blue', linewidth=4)
    ax.plot([L_SHOULDER[0], R_SHOULDER[0]], [L_SHOULDER[1], R_SHOULDER[1]], [L_SHOULDER[2], R_SHOULDER[2]], color='blue', linewidth=4)
    ax.plot([NECK[0], PELVIS[0]], [NECK[1], PELVIS[1]], [NECK[2], PELVIS[2]], color='blue', linewidth=4)
    ax.scatter([HEAD[0], PELVIS[0], L_SHOULDER[0], R_SHOULDER[0]], 
               [HEAD[1], PELVIS[1], L_SHOULDER[1], R_SHOULDER[1]], 
               [HEAD[2], PELVIS[2], L_SHOULDER[2], R_SHOULDER[2]], c='blue', s=50)

    # 3. Graph Dynamic Extremity Vectors linked to Tag inputs
    # Left Arm: Shoulder to Left Wrist (Tag 0)
    ax.plot([L_SHOULDER[0], tag_positions[0][0]], 
            [L_SHOULDER[1], tag_positions[0][1]], 
            [L_SHOULDER[2], tag_positions[0][2]], color='red', linewidth=3, label='Left Arm')
    ax.scatter(tag_positions[0][0], tag_positions[0][1], tag_positions[0][2], c='red', marker='o', s=120)
    ax.text(tag_positions[0][0], tag_positions[0][1], tag_positions[0][2] + 10, "Left Hand (T0)", color='red')

    # Right Arm: Shoulder to Right Wrist (Tag 1)
    ax.plot([R_SHOULDER[0], tag_positions[1][0]], 
            [R_SHOULDER[1], tag_positions[1][1]], 
            [R_SHOULDER[2], tag_positions[1][2]], color='green', linewidth=3, label='Right Arm')
    ax.scatter(tag_positions[1][0], tag_positions[1][1], tag_positions[1][2], c='green', marker='o', s=120)
    ax.text(tag_positions[1][0], tag_positions[1][1], tag_positions[1][2] + 10, "Right Hand (T1)", color='green')

    # Enforce constant limits on bounding frame boundaries
    ax.set_xlim([-50, 250])
    ax.set_ylim([-50, 250])
    ax.set_zlim([0, 250])
    ax.set_xlabel('X Dimension (cm)')
    ax.set_ylabel('Y Dimension (cm)')
    ax.set_zlabel('Z Elevation (cm)')
    ax.set_title('Real-Time UWB 3D Motion Capture Kinematics')
    
    plt.draw()
    plt.pause(0.01)

if ser:
    ser.close()