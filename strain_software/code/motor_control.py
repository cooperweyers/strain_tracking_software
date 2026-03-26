import serial
import time
import cv2
import os
from datetime import datetime

# ================= USER CONFIG =================

ARDUINO_PORT = "COM3"       # Change if needed (e.g. COM4)
BAUD_RATE = 9600

NUM_CYCLES = 3             # Number of motor moves
STEPS_PER_MOVE = 10      # Must be <= 9999
STEP_MODE = 'm'             # 's', 'd', 'i', or 'm'
DIRECTIONS = ['b']          # ['f'], ['b'], or ['f','b']

CAMERA_INDEX = 0            # Camera ID
CAMERA_DELAY = 0.5          # Seconds to wait before each capture

# =================================================


# ---------- Helper Functions ----------

def format_command(direction, steps, mode):
    """Create Arduino-compatible command like f0100m"""
    steps_str = f"{steps:04d}"
    return f"{direction}{steps_str}{mode}\n"


def capture_image(cap, save_dir, index):
    """Capture and save an image with indexed filename"""
    ret, frame = cap.read()
    if not ret:
        print(f"ERROR: Failed to capture image {index}")
        return
    filename = os.path.join(save_dir, f"img_{index:03d}.png")
    cv2.imwrite(filename, frame)
    print(f"Saved image: {filename}")


# ---------- Directory Setup ----------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_IMAGE_DIR = os.path.join(SCRIPT_DIR, "images")

timestamp = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
IMAGE_SAVE_DIR = os.path.join(BASE_IMAGE_DIR, timestamp)

os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

print("Images will be saved to:")
print(IMAGE_SAVE_DIR)


# ---------- Serial Setup ----------

print("Opening serial connection...")
ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # Allow Arduino reset
print("Serial connection established")


# ---------- Camera Setup ----------

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    ser.close()
    raise RuntimeError("ERROR: Could not open camera")

print("Camera opened successfully")


# ---------- Initial Image (Before Motion) ----------

print("\nCapturing initial image (before motion)...")
time.sleep(CAMERA_DELAY)
capture_image(cap, IMAGE_SAVE_DIR, 1)


# ---------- Main Loop ----------

for cycle in range(1, NUM_CYCLES + 1):
    print(f"\nCycle {cycle} / {NUM_CYCLES}")

    direction = DIRECTIONS[(cycle - 1) % len(DIRECTIONS)]
    command = format_command(direction, STEPS_PER_MOVE, STEP_MODE)

    print(f"Sending command to Arduino: {command.strip()}")
    ser.write(command.encode())

    # Wait for Arduino to finish motion
    while True:
        response = ser.readline().decode().strip()
        if response == "DONE":
            print("Motor move complete")
            break

    # Capture image AFTER movement
    time.sleep(CAMERA_DELAY)
    capture_image(cap, IMAGE_SAVE_DIR, cycle + 1)


# ---------- Cleanup ----------

cap.release()
ser.close()

print("\nExperiment complete.")
print(f"Total images captured: {NUM_CYCLES + 1}")
