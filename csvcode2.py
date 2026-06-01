import serial
import json
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from mpl_toolkits.mplot3d import Axes3D

# --- PHYSICAL ROOM ANCHOR CONFIGURATION (IN CM) ---
ANCHORS = np.array([
    [  0,   0,   0],   # A0  Master  – floor corner
    [200,   0,   0],   # A1          – floor corner
    [200, 200,  70],   # A2          – mid-wall
    [  0, 200,  62],   # A3          – low corner
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
    4: np.array([100.0, 100.0, 170.0]), # Tag 4: Head
    5: np.array([100.0, 100.0, 100.0])  # Tag 5: Belly Button
}

# Mapping Tag IDs to friendly names
TAG_NAMES = {
    0: "Left Wrist",
    1: "Right Wrist",
    2: "Left Knee",
    3: "Right Knee",
    4: "Head",
    5: "Belly Button"
}

# Residual function for bounded least_squares tracking
def residuals_function(point, distances):
    valid = distances > 0
    calculated_dists = np.linalg.norm(ANCHORS[valid] - point, axis=1)
    return calculated_dists - distances[valid]

def calculate_3d_position(distances, last_known_pos):
    distances = np.array(distances[:4], dtype=float)
    valid = distances > 0
    
    # Require at least 3 valid anchors to calculate a reliable 3D position
    if np.sum(valid) < 3:
        return last_known_pos

    # Room boundaries (with safety margins) to keep the tags inside physical space
    lower_bounds = [-100.0, -100.0, -100.0]
    upper_bounds = [400.0, 400.0, 300.0]

    try:
        # Bounded optimization utilizing robust soft_l1 loss for UWB outlier rejection
        result = least_squares(
            residuals_function,
            x0=last_known_pos,
            args=(distances,),
            bounds=(lower_bounds, upper_bounds),
            loss='soft_l1',
            f_scale=5.0  # Softens penalty for errors beyond 5cm (outliers)
        )
        return result.x
    except Exception:
        return last_known_pos

# Establish CSV Data Logging File
csv_filename = "mocap_data.csv"
csv_file = open(csv_filename, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "Timestamp (Local)", "Tag ID", "Tag Name", 
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
    print(f"[✓] Connected to UWB master on {com_port}")
except Exception as e:
    print(f"CRITICAL: Could not open connection to {com_port}: {e}")
    ser = None

# Initialize Interactive 3D Matplotlib Plotting Space
plt.ion()
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Force-render window framework so fignum_exists condition passes safely
plt.show(block=False)
plt.pause(0.5)  

print("Live Full-Body 3D Motion Capture Active — close window to exit.")

try:
    while plt.fignum_exists(fig.number):
        data_updated = False
        
        # DRAINING LOOP: Read ALL available lines in the serial buffer at once.
        while ser and ser.in_waiting > 0:
            try:
                line = ser.readline().decode('UTF-8', errors='ignore').strip()
                if line.startswith("{"):
                    data = json.loads(line)
                    t_id = data['id']
                    ranges = data['range']
                    
                    valid_ranges = sum(1 for r in ranges[:4] if r > 0)
                    if valid_ranges >= 3 and 0 <= t_id <= 5:
                        # Compute coordinates using robust least_squares method
                        computed_xyz = calculate_3d_position(ranges[:4], tag_positions[t_id])
                        tag_positions[t_id] = computed_xyz
                        
                        raw_ranges = ranges[:4] + [0] * (4 - len(ranges[:4]))
                        
                        # Local time formatting transformation
                        local_time_str = time.strftime("%H:%M:%S")
                        
                        # Save telemetry frame to CSV RAM buffer instantly
                        csv_writer.writerow([
                            local_time_str, t_id, TAG_NAMES.get(t_id, "Unknown"),
                            raw_ranges[0], raw_ranges[1], raw_ranges[2], raw_ranges[3],
                            f"{computed_xyz[0]:.2f}", f"{computed_xyz[1]:.2f}", f"{computed_xyz[2]:.2f}"
                        ])
                        data_updated = True
            except Exception:
                pass

        # Only redraw graphics canvas frame when fresh telemetry updates are verified
        if data_updated:
            csv_file.flush()  # Commit background buffer lines to physical disk memory
            
            # Clear previous visualization frame
            ax.cla()
            
            # --- RENDER PHYSICAL ROOM ANCHORS ---
            # Plots anchors as distinctive large red triangles with black outlines
            ax.scatter(ANCHORS[:, 0], ANCHORS[:, 1], ANCHORS[:, 2], c='red', s=120, marker='^', edgecolors='black', label='Fixed Anchors')
            for idx, anchor in enumerate(ANCHORS):
                ax.text(anchor[0] + 5, anchor[1] + 5, anchor[2] + 5, f"A{idx}", color='darkred', fontsize=10, fontweight='bold')
            
            # Extract calculated coordinates
            L_WRIST = tag_positions[0]
            R_WRIST = tag_positions[1]
            L_KNEE  = tag_positions[2]
            R_KNEE  = tag_positions[3]
            HEAD    = tag_positions[4]
            BELLY   = tag_positions[5]

            # --- KINEMATIC SKELETON VECTOR MATH ---
            spine_vector = HEAD - BELLY
            norm_spine = np.linalg.norm(spine_vector)
            
            if norm_spine < 1e-5:
                spine_unit = np.array([0.0, 0.0, 1.0])
            else:
                spine_unit = spine_vector / norm_spine
            
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
            # Torso
            ax.plot([HEAD[0], NECK[0]], [HEAD[1], NECK[1]], [HEAD[2], NECK[2]], color='blue', linewidth=4)
            ax.plot([L_SHOULDER[0], R_SHOULDER[0]], [L_SHOULDER[1], R_SHOULDER[1]], [L_SHOULDER[2], R_SHOULDER[2]], color='blue', linewidth=4)
            ax.plot([NECK[0], PELVIS[0]], [NECK[1], PELVIS[1]], [NECK[2], PELVIS[2]], color='blue', linewidth=4)
            ax.plot([L_HIP[0], R_HIP[0]], [L_HIP[1], R_HIP[1]], [L_HIP[2], R_HIP[2]], color='blue', linewidth=4)

            # Arms
            ax.plot([L_SHOULDER[0], L_WRIST[0]], [L_SHOULDER[1], L_WRIST[1]], [L_SHOULDER[2], L_WRIST[2]], color='red', linewidth=3)
            ax.plot([R_SHOULDER[0], R_WRIST[0]], [R_SHOULDER[1], R_WRIST[1]], [R_SHOULDER[2], R_WRIST[2]], color='green', linewidth=3)

            # Legs
            ax.plot([L_HIP[0], L_KNEE[0]], [L_HIP[1], L_KNEE[1]], [L_HIP[2], L_KNEE[2]], color='orange', linewidth=3)
            ax.plot([L_KNEE[0], L_FOOT[0]], [L_KNEE[1], L_FOOT[1]], [L_KNEE[2], L_FOOT[2]], color='orange', linewidth=3)
            ax.plot([R_HIP[0], R_KNEE[0]], [R_HIP[1], R_KNEE[1]], [R_HIP[2], R_KNEE[2]], color='purple', linewidth=3)
            ax.plot([R_KNEE[0], R_FOOT[0]], [R_KNEE[1], R_FOOT[1]], [R_KNEE[2], R_FOOT[2]], color='purple', linewidth=3)

            # Plot joint marker nodes
            all_joints = np.array([HEAD, NECK, PELVIS, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_WRIST, R_WRIST, L_KNEE, R_KNEE, L_FOOT, R_FOOT])
            ax.scatter(all_joints[:,0], all_joints[:,1], all_joints[:,2], c='blue', s=40)
            
            # Highlight tracking tags
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
            
            # Add legend to distinguish Anchors from Body Nodes
            ax.legend(loc='upper left')
            
            plt.draw()
            
        # Keep GUI message queues operating cleanly
        plt.pause(0.001)

finally:
    if ser:
        ser.close()
    csv_file.close()
    print("\n[✓] Serial port closed.")