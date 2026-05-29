import serial
import json
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

# --- PHYSICAL ROOM ANCHOR CONFIGURATION (IN CM) ---
# Coordinates of your 4 fixed anchor modules
ANCHORS = np.array([
    [  0,   0,   0],   # A0 Master – floor corner
    [225,   0,   0],   # A1        – floor corner
    [225, 310, 115],   # A2        – mid-wall
    [  0, 300, 125],   # A3        – low corner
], dtype=float)

# Targeted single tag tracker initialization (Tag 0)
TARGET_TAG_ID = 0
tag_position = np.array([100.0, 100.0, 100.0]) # Seed position (Center of the room)

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

# Establish CSV Data Logging File
csv_filename = "single_tag_mocap.csv"
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
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

print(f"Live Single-Tag (ID: {TARGET_TAG_ID}) 3D Tracking Active. Close the window to exit.")
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
                    
                    # Track only the specified Target Tag
                    if t_id == TARGET_TAG_ID:
                        # Ensure at least 3 anchors provide clean range information
                        valid_ranges = sum(1 for r in ranges[:4] if r > 0)
                        if valid_ranges >= 3:
                            # Update coordinates dynamically
                            tag_position = calculate_3d_position(ranges[:4], tag_position)
                            
                            # Pad ranges array if incoming stream has less than 4 elements
                            raw_ranges = ranges[:4] + [0] * (4 - len(ranges[:4]))
                            
                            # Save telemetry frame to CSV
                            elapsed_time = time.time() - start_time
                            csv_writer.writerow([
                                f"{elapsed_time:.3f}", t_id,
                                raw_ranges[0], raw_ranges[1], raw_ranges[2], raw_ranges[3],
                                f"{tag_position[0]:.2f}", f"{tag_position[1]:.2f}", f"{tag_position[2]:.2f}"
                            ])
                            csv_file.flush() 
                            
            except Exception:
                pass

        # --- RENDERING GENERATION ---
        ax.cla() # Clear previous frame
        
        # 1. Plot Fixed Anchors (Red Triangles)
        ax.scatter(ANCHORS[:, 0], ANCHORS[:, 1], ANCHORS[:, 2], color='red', s=120, marker='^', label='Anchors (A0-A3)')
        
        # Add labels to the Anchors
        for i, anchor in enumerate(ANCHORS):
            ax.text(anchor[0] + 5, anchor[1] + 5, anchor[2] + 5, f"A{i} ({int(anchor[0])},{int(anchor[1])},{int(anchor[2])})", color='red', fontsize=9, weight='bold')

        # 2. Plot Moving Tag (Cyan Circle with Blue Border)
        ax.scatter(tag_position[0], tag_position[1], tag_position[2], color='cyan', s=150, edgecolors='darkblue', marker='o', label=f'Tag {TARGET_TAG_ID}')
        
        # Add live coordinate text tracking just above the moving tag
        ax.text(tag_position[0] + 5, tag_position[1] + 5, tag_position[2] + 5, f"Tag {TARGET_TAG_ID}\n[{tag_position[0]:.1f}, {tag_position[1]:.1f}, {tag_position[2]:.1f}]", color='blue', fontsize=10, weight='bold')

        # 3. Configure Grid Boundaries and Space Environment
        ax.set_xlim([-50, 250])
        ax.set_ylim([-50, 350])
        ax.set_zlim([0, 250])
        
        ax.set_xlabel('X Width (cm)')
        ax.set_ylabel('Y Length (cm)')
        ax.set_zlabel('Z Height (cm)')
        ax.set_title(f'Live 3D UWB Positioning Demo (Tag {TARGET_TAG_ID})')
        ax.legend(loc='upper left')
        ax.grid(True)
        
        plt.draw()
        plt.pause(0.01)

finally:
    if ser:
        ser.close()
    csv_file.close()
    print("Serial port closed and tracking file successfully saved safely.")