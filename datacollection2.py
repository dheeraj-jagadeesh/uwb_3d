import serial
import json
import time
import csv
from datetime import datetime

# -----------------------------
# User-configurable parameters
# -----------------------------

COM_PORT = "COM19"
BAUDRATE = 115200

# Flush CSV data to disk after every N rows
FLUSH_EVERY_N_ROWS = 2000

# Mapping Tag IDs to friendly names
TAG_NAMES = {
    0: "Left Wrist",
    1: "Right Wrist",
    2: "Left Knee",
    3: "Right Knee",
    4: "Head",
    5: "Belly Button"
}

# Create a unique CSV filename for each data collection run
run_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
csv_filename = f"mocap_raw_ranges_{run_timestamp}.csv"

ser = None
csv_file = None
rows_since_flush = 0

try:
    # Establish direct link to Master Anchor Module
    ser = serial.Serial(
        port=COM_PORT,
        baudrate=BAUDRATE,
        timeout=0.01,
        dsrdtr=False,
        rtscts=False
    )

    ser.setDTR(False)
    ser.setRTS(False)
    time.sleep(0.5)
    ser.reset_input_buffer()

    print(f"[✓] Connected to UWB master on {COM_PORT}")
    print(f"[✓] Logging data to {csv_filename}")

    # Establish CSV data logging file
    csv_file = open(csv_filename, mode="w", newline="")
    csv_writer = csv.writer(csv_file)

    # Write CSV header
    csv_writer.writerow([
        "Timestamp",
        "Tag ID",
        "Tag Name",
        "Range_A0 (cm)",
        "Range_A1 (cm)",
        "Range_A2 (cm)",
        "Range_A3 (cm)"
    ])

    print("UWB Logging Active — Press Ctrl+C to terminate data collection safely.")

    while ser and ser.is_open:
        # Drain loop: read streaming serial lines as they land in the buffer
        while ser.in_waiting > 0:
            try:
                line = ser.readline().decode("UTF-8", errors="ignore").strip()

                if not line.startswith("{"):
                    continue

                data = json.loads(line)

                if "id" not in data or "range" not in data:
                    continue

                t_id = data["id"]
                ranges = data["range"]

                # Ensure range array has exactly 4 slots
                # Missing values are stored as empty fields instead of 0
                raw_ranges = ranges[:4] + [""] * (4 - len(ranges[:4]))

                timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

                csv_writer.writerow([
                    timestamp,
                    t_id,
                    TAG_NAMES.get(t_id, "Unknown"),
                    raw_ranges[0],
                    raw_ranges[1],
                    raw_ranges[2],
                    raw_ranges[3]
                ])

                rows_since_flush += 1

                # Flush after every N rows
                if rows_since_flush >= FLUSH_EVERY_N_ROWS:
                    csv_file.flush()
                    rows_since_flush = 0

            except Exception:
                # Ignore malformed serial lines
                # For debugging, replace this with: print(e)
                pass

        # Yield CPU runtime briefly if buffer is empty
        time.sleep(0.001)

except KeyboardInterrupt:
    print("\n[!] Data logging paused by user manual override.")

except Exception as e:
    print(f"CRITICAL: Data collection failed: {e}")

finally:
    if csv_file:
        csv_file.flush()
        csv_file.close()

    if ser:
        ser.close()

    print("[✓] Serial pipeline cleanly disconnected. Target log file sealed.")