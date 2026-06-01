import serial
import json
import time
import csv
from datetime import datetime

# Mapping Tag IDs to friendly names
TAG_NAMES = {
    0: "Left Wrist",
    1: "Right Wrist",
    2: "Left Knee",
    3: "Right Knee",
    4: "Head",
    5: "Belly Button"
}

# Establish CSV Data Logging File
csv_filename = "mocap_raw_ranges.csv"
csv_file = open(csv_filename, mode='w', newline='')
csv_writer = csv.writer(csv_file)

# Write streamlined header columns (Removed X, Y, Z fields)
csv_writer.writerow([
    "Timestamp", "Tag ID", "Tag Name", 
    "Range_A0 (cm)", "Range_A1 (cm)", "Range_A2 (cm)", "Range_A3 (cm)"
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

print("UWB Logging Active — Press Ctrl+C to terminate data collection safely.")

try:
    while ser and ser.is_open:
        # Drain loop: read streaming serial lines as they land in the buffer
        while ser.in_waiting > 0:
            try:
                line = ser.readline().decode('UTF-8', errors='ignore').strip()
                if line.startswith("{"):
                    data = json.loads(line)
                    t_id = data['id']
                    ranges = data['range']
                    
                    # Ensure range array has exactly 4 slots (padded with 0 if short)
                    raw_ranges = ranges[:4] + [0] * (4 - len(ranges[:4]))
                    
                    # Apply your microsecond time format string
                    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
                    
                    # Log telemetry packet directly to CSV
                    csv_writer.writerow([
                        timestamp, 
                        t_id, 
                        TAG_NAMES.get(t_id, "Unknown"),
                        raw_ranges[0], 
                        raw_ranges[1], 
                        raw_ranges[2], 
                        raw_ranges[3]
                    ])
                    
                    # Instantly push data from RAM buffer onto the physical disk
                    csv_file.flush()
                    
            except Exception:
                # Discard corrupted serial lines silently to avoid dropping packets
                pass
        
        # Yield CPU runtime briefly if buffer is empty
        time.sleep(0.001)

except KeyboardInterrupt:
    print("\n[!] Data logging paused by user manual override.")

finally:
    if ser:
        ser.close()
    csv_file.close()
    print("[✓] Serial pipeline cleanly disconnected. Target log file sealed.")