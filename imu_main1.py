import serial
import time
import sys
from datetime import datetime, timedelta
import numpy as np

# High-speed throughput pipeline to prevent serial logging bottlenecks
BAUD_RATE = 921600
SERIAL_PORT = "COM17" 

# Explicit mapping directory for expanding to all 6 Wearable Body Tags
TAG_NAMES = {
    0: "Left Wrist",
    1: "Right Wrist",
    2: "Left Knee",
    3: "Right Knee",
    4: "Head",
    5: "Belly Button"
}

# Dynamic Anchor Coordinate Matrix (Modify these values to update CSV headers automatically)
ANCHORS = np.array([
    [  0,   0,   0],   # A0  Master  – floor corner
    [245,   0,   0],   # A1          – floor corner
    [245, 245,  95],   # A2          – mid-wall
    [  0, 245,  62],   # A3          – low corner
], dtype=float)

def get_serial_connection():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2) # Stabilization window
        return ser
    except Exception as e:
        print(f"[ERROR]: Unable to reach Master Anchor on {SERIAL_PORT}: {e}")
        sys.exit()

def run_status_check(ser, targets):
    # targets is a list of strings, e.g., ['0', '1', '2'] or ['all']
    display_targets = "all" if "all" in targets else ", ".join(targets)
    print(f"[INFO]: Querying hardware diagnostics for Targets: {display_targets}...")
    
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    
    # Optimization: If querying multiple specific tags, broadcast 'all' to save time
    if "all" in targets or len(targets) > 1:
        ser.write(f"STATUS:all:{timestamp}\n".encode())
    else:
        ser.write(f"STATUS:{targets[0]}:{timestamp}\n".encode())
    
    start_time = time.time()
    responses = {}
    
    # Listen for 1.5 seconds to catch all over-the-air incoming status responses
    while time.time() - start_time < 1.5:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line and line.startswith("[STATUS]"):
            parts = line.split(":")
            if len(parts) >= 3:
                tag_id = parts[1]
                msg = parts[2]
                responses[tag_id] = msg
        time.sleep(0.01)

    print("\n==============================================")
    print("         BNO055 SENSOR HEALTH REPORT          ")
    print("==============================================")
    
    if 'all' in targets:
        for i in range(6):
            status = responses.get(str(i), "OFFLINE / UNREACHABLE")
            color_format = "✔ ACTIVE" if "ACTIVE" in status else "✘ FAULT"
            print(f"  -> Tag {i}: {color_format} ({status})")
    else:
        # Filter the print block to output only the specific indices requested
        for t in targets:
            status = responses.get(str(t), "OFFLINE / UNREACHABLE")
            color_format = "✔ ACTIVE" if "ACTIVE" in status else "✘ FAULT"
            print(f"  -> Tag {t}: {color_format} ({status})")
    print("==============================================\n")

def run_pull_cycle(ser, tag_id):
    print(f"[INFO]: Initializing remote download pipeline for Tag {tag_id}...")
    
    # File naming reflective of the exact run timestamp to prevent overwriting
    file_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tag_{tag_id}_telemetry_{file_ts}.csv"
    
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    ser.write(f"PULL:{tag_id}:{timestamp}\n".encode())
    
    file_open = False
    f = None
    start_time = time.time()
    
    base_time = None
    initial_micros = None
    
    while True:
        line = ser.readline().decode('utf-8', errors='ignore')
        if not line:
            if time.time() - start_time > 6.0:
                print(f"[TIMEOUT]: Tag {tag_id} did not respond.")
                break
            continue
        
        start_time = time.time()
        stripped_line = line.strip()
        
        if stripped_line.startswith("[STATUS]"):
            parts = stripped_line.split(":")
            if len(parts) >= 3 and parts[1] == str(tag_id):
                print(f"\n[REMOTE ERROR] Tag {tag_id} reported: {parts[2]}")
            continue
            
        if stripped_line.startswith("[EOF]"):
            parts = stripped_line.split(":")
            if len(parts) > 1 and int(parts[1]) == tag_id:
                print(f"[SUCCESS]: Data saved successfully to '{filename}'")
                break
            continue

        if not stripped_line:
            continue

        if stripped_line.startswith("[START_TIME]"):
            ts_str = stripped_line.split(":")[1]
            base_time = datetime.strptime(ts_str, "%Y_%m_%d_%H_%M_%S_%f")
            continue

        if not file_open:
            f = open(filename, "w", encoding="utf-8")
            file_open = True
            
        data_parts = stripped_line.split(",")
        
        if "relative_us" in stripped_line:
            remaining_headers = data_parts[1:]
            updated_headers = []
            
            # Map anchor coordinates dynamically alongside dist_A columns
            for header in remaining_headers:
                header_clean = header.strip()
                if header_clean.startswith("dist_A") and header_clean[6:].isdigit():
                    anchor_idx = int(header_clean[6:])
                    if anchor_idx < len(ANCHORS):
                        coord = ANCHORS[anchor_idx]
                        # Format floats to clean strings (removes decimal if it's a whole number)
                        cx = int(coord[0]) if coord[0].is_integer() else coord[0]
                        cy = int(coord[1]) if coord[1].is_integer() else coord[1]
                        cz = int(coord[2]) if coord[2].is_integer() else coord[2]
                        header_clean = f"{header_clean}_[{cx}_{cy}_{cz}]"
                updated_headers.append(header_clean)
                
            new_header = "Timestamp,Tag ID,Tag Name," + ",".join(updated_headers) + "\n"
            f.write(new_header)
        else:
            if base_time is None:
                base_time = datetime.now()

            try:
                current_micros = int(data_parts[0])
                if initial_micros is None:
                    initial_micros = current_micros
                
                elapsed_us = current_micros - initial_micros
                if elapsed_us < 0:
                    elapsed_us = 0
                
                row_time = base_time + timedelta(microseconds=elapsed_us)
                timestamp_str = row_time.strftime("%Y_%m_%d_%H_%M_%S_%f")
            except Exception:
                timestamp_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
                
            remaining_data = data_parts[1:]
            tag_name = TAG_NAMES.get(tag_id, "Unknown Target")
            
            new_data_line = f"{timestamp_str},{tag_id},{tag_name}," + ",".join(remaining_data) + "\n"
            f.write(new_data_line)
            
    if f and not f.closed:
        f.close()

def main():
    ser = get_serial_connection()
    print("====================================================")
    print("      UWB / IMU ESP-NOW Tracking Array Online       ")
    print("====================================================")
    
    while True:
        try:
            print("Available Commands: [start] | [stop] | [pull] | [status] | [exit]")
            user_input = input("Enter command (<cmd> <tag_id1 tag_id2 ... / all>): ").strip().lower()
            if not user_input:
                continue
                
            parts = user_input.split()
            cmd = parts[0]
            
            if cmd == 'exit':
                print("Closing data collection interface.")
                break
                
            # Capture all trailing arguments as target values
            targets = parts[1:] if len(parts) > 1 else ['all']
            
            if cmd in ['start', 'stop']:
                if "all" in targets:
                    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
                    ser.write(f"{cmd.upper()}:all:{timestamp}\n".encode())
                    print(f"[SENT]: {cmd.upper()} command issued to Target: all\n")
                else:
                    # Loop and fire sequential transmissions for each manual index parsed
                    for t in targets:
                        try:
                            tag_id = int(t)
                            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
                            ser.write(f"{cmd.upper()}:{tag_id}:{timestamp}\n".encode())
                            print(f"[SENT]: {cmd.upper()} command issued to Target: {tag_id}")
                            time.sleep(0.1)  # Safe recovery gap for the RF buffer queue
                        except ValueError:
                            print(f"[INVALID]: Target '{t}' is not a recognized integer. Skipping.")
                    print()
                
            elif cmd == 'status':
                run_status_check(ser, targets)
                
            elif cmd == 'pull':
                if "all" in targets:
                    for i in range(6):
                        run_pull_cycle(ser, i)
                        time.sleep(0.5)
                else:
                    for t in targets:
                        try:
                            tag_id = int(t)
                            run_pull_cycle(ser, tag_id)
                            time.sleep(0.5)
                        except ValueError:
                            print(f"[INVALID]: Target '{t}' must be an integer. Skipping.")
            else:
                print("[INVALID]: Command structure unrecognized.\n")
                
        except KeyboardInterrupt:
            print("\nInterface execution halted.")
            break
            
    ser.close()

if __name__ == "__main__":
    main()