import serial
import json
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

# --- PHYSICAL ROOM ANCHOR CONFIGURATION (IN CM) ---
ANCHORS = np.array([
    [  0.0, 222.9, 155.0],   # A0 – Back-Left Corner (155cm High)
    [155.2, 153.4, 155.0],   # A1 – Back-Right Corner (155cm High)
    [193.3,   0.0,  80.0],   # A2 – Front-Right Corner (80cm High)
    [  0.0,   0.0, 185.0],   # A3 – Front-Left Corner (185cm High)
], dtype=float)

# Dual Tracker Point Initialization (Tag 0 and Tag 1)
tag0_position = np.array([80.0, 110.0, 120.0])  
tag1_position = np.array([120.0, 110.0, 120.0]) 

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
    result = minimize(error_distance_function, last_known_pos, args=(distances,), method='Nelder-Mead')
    return result.x

# Establish CSV Data Logging File
csv_filename = "dual_tag_mocap_depth_enhanced.csv"
csv_file = open(csv_filename, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "Timestamp (s)", "Tag ID", 
    "Range_A0 (cm)", "Range_A1 (cm)", "Range_A2 (cm)", "Range_A3 (cm)", 
    "Computed_X (cm)", "Computed_Y (cm)", "Computed_Z (cm)"
])

# Establish direct link to Master Anchor Module
com_port = "COM19"
try:
    ser = serial.Serial(port=com_port, baudrate=115200, timeout=0.01, dsrdtr=False, rtscts=False)
    ser.setDTR(False)
    ser.setRTS(False)
    time.sleep(0.5)
    ser.reset_input_buffer()
    print(f"Connected to UWB Master Array on port: {com_port}")
    print(f"Data logging initiated. Saving sessions to: {csv_filename}")
except Exception as e:
    print(f"CRITICAL: Could not open connection to {com_port}: {e}")
    ser = None

# Initialize Interactive 3D Matplotlib Plotting Space
plt.ion()
fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection='3d')

print("Live Depth-Enhanced Dual-Tag 3D Tracking Active. Close window to exit.")
start_time = time.time()

try:
    while plt.fignum_exists(fig.number):
        if ser and ser.in_waiting > 0:
            try:
                line = ser.readline().decode('UTF-8', errors='ignore').strip()
                if line.startswith("{"):
                    data = json.loads(line)
                    t_id = data['id']
                    ranges = data['range']
                    
                    if t_id in [0, 1]:
                        valid_ranges = sum(1 for r in ranges[:4] if r > 0)
                        if valid_ranges >= 3:
                            if t_id == 0:
                                tag0_position = calculate_3d_position(ranges[:4], tag0_position)
                                current_pos = tag0_position
                            else:
                                tag1_position = calculate_3d_position(ranges[:4], tag1_position)
                                current_pos = tag1_position
                            
                            raw_ranges = ranges[:4] + [0] * (4 - len(ranges[:4]))
                            elapsed_time = time.time() - start_time
                            csv_writer.writerow([
                                f"{elapsed_time:.3f}", t_id,
                                raw_ranges[0], raw_ranges[1], raw_ranges[2], raw_ranges[3],
                                f"{current_pos[0]:.2f}", f"{current_pos[1]:.2f}", f"{current_pos[2]:.2f}"
                            ])
                            csv_file.flush() 
                            
            except Exception:
                pass

        # --- RENDERING GENERATION ---
        ax.cla() # Wipe previous frame state buffer
        
        # 1. Plot Clockwise Calibrated Anchors (Red Triangles)
        ax.scatter(ANCHORS[:, 0], ANCHORS[:, 1], ANCHORS[:, 2], color='red', s=130, marker='^', label='Anchors (A0-A3)')
        for i, anchor in enumerate(ANCHORS):
            ax.text(anchor[0] + 5, anchor[1] + 5, anchor[2] + 5, 
                    f"A{i} ({int(anchor[0])}, {int(anchor[1])}, {int(anchor[2])})", 
                    color='red', fontsize=9, weight='bold')

        # 2. Plot Moving Target Tag 0 (Cyan Sphere) + Depth Enhancements
        # Dynamic Drop line to floor
        ax.plot([tag0_position[0], tag0_position[0]], [tag0_position[1], tag0_position[1]], [0, tag0_position[2]], 
                color='cyan', linestyle='--', linewidth=1.5)
        # Floor Shadow Projection
        ax.scatter(tag0_position[0], tag0_position[1], 0, color='gray', s=40, alpha=0.4, marker='o')
        # Active Tag 3D Node
        ax.scatter(tag0_position[0], tag0_position[1], tag0_position[2], color='cyan', s=160, edgecolors='darkblue', marker='o', label='Tag 0')
        ax.text(tag0_position[0] + 5, tag0_position[1] + 5, tag0_position[2] + 5, 
                f"Tag 0\n[{tag0_position[0]:.1f}, {tag0_position[1]:.1f}, {tag0_position[2]:.1f}]", 
                color='darkblue', fontsize=9, weight='bold')

        # 3. Plot Moving Target Tag 1 (Magenta Sphere) + Depth Enhancements
        # Dynamic Drop line to floor
        ax.plot([tag1_position[0], tag1_position[0]], [tag1_position[1], tag1_position[1]], [0, tag1_position[2]], 
                color='magenta', linestyle='--', linewidth=1.5)
        # Floor Shadow Projection
        ax.scatter(tag1_position[0], tag1_position[1], 0, color='gray', s=40, alpha=0.4, marker='o')
        # Active Tag 3D Node
        ax.scatter(tag1_position[0], tag1_position[1], tag1_position[2], color='magenta', s=160, edgecolors='darkmagenta', marker='o', label='Tag 1')
        ax.text(tag1_position[0] + 5, tag1_position[1] - 15, tag1_position[2] - 5, 
                f"Tag 1\n[{tag1_position[0]:.1f}, {tag1_position[1]:.1f}, {tag1_position[2]:.1f}]", 
                color='darkmagenta', fontsize=9, weight='bold')

        # 4. Viewport Env Boundaries Optimization
        ax.set_xlim([-50, 250])
        ax.set_ylim([-50, 300])
        ax.set_zlim([0, 250])
        
        ax.set_xlabel('X Width (cm)')
        ax.set_ylabel('Y Length (cm)')
        ax.set_zlabel('Z Height (cm)')
        ax.set_title('Live UWB Dual-Tag Depth-Enhanced 3D Space')
        ax.legend(loc='upper left')
        ax.grid(True)
        
        # Explicitly set camera elevation and azimuth to force optimal 3D perspective depth perception
        ax.view_init(elev=22, azim=-55)
        
        plt.draw()
        plt.pause(0.01)

finally:
    if ser:
        ser.close()
    csv_file.close()
    print("Serial port connection closed safely. Telemetry file compiled.")