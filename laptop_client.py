import cv2
import socketio
import time
import csv
import threading
import numpy as np
import requests

# --- CONFIGURATION ---
BASE_URL = "https://amr.nlpamr.pp.ua"
APPLY_COLOR_CORRECTION = True  # Set to True to swap Red and Blue channels

# Globals for shared state
latest_telemetry = {}
sio = socketio.Client()
logging_active = False
log_start_time = 0

# Video writers
out_cam0 = None
out_cam1 = None
csv_file = None
csv_writer = None

# --- WEBSOCKET CLIENT ---
@sio.event
def connect():
    print("[INFO] Connected to AMR Telemetry via Cloudflare/Domain")

@sio.on('telemetry')
def on_telemetry(data):
    global latest_telemetry, logging_active, csv_writer
    latest_telemetry = data
    
    if logging_active and csv_writer is not None:
        # Check if nested keys exist to prevent errors
        mpu = data.get('mpu6050', {})
        accel = mpu.get('accel', {'x':0, 'y':0, 'z':0})
        gyro = mpu.get('gyro', {'z':0})
        
        csv_writer.writerow([
            time.time(),
            accel.get('x', 0),
            accel.get('y', 0),
            accel.get('z', 0),
            gyro.get('z', 0),
            data.get('ultrasonic_cm', -1),
            data.get('state', 'UNKNOWN')
        ])

def connect_socket():
    try:
        sio.connect(BASE_URL)
        sio.wait()
    except Exception as e:
        print(f"[ERROR] WebSocket Connection Failed: {e}")

threading.Thread(target=connect_socket, daemon=True).start()

import requests # Make sure this is imported at the top

# --- CAMERA CAPTURE THREADS ---
class CameraStream:
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            try:
                # stream=True keeps the HTTPS connection open
                res = requests.get(self.url, stream=True, timeout=10)
                if res.status_code == 200:
                    bytes_data = b''
                    for chunk in res.iter_content(chunk_size=4096):
                        if not self.running:
                            break
                        bytes_data += chunk
                        
                        # Look for JPEG start (0xff 0xd8) and end (0xff 0xd9) bytes
                        a = bytes_data.find(b'\xff\xd8')
                        b = bytes_data.find(b'\xff\xd9')
                        
                        if a != -1 and b != -1:
                            jpg = bytes_data[a:b+2]
                            bytes_data = bytes_data[b+2:] # clear the processed bytes
                            
                            # Decode the raw JPEG bytes into an OpenCV image
                            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            
                            if frame is not None:
                                if APPLY_COLOR_CORRECTION:
                                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                self.frame = frame
                else:
                    print(f"[WARN] Stream {self.url} returned status {res.status_code}. Retrying...")
                    time.sleep(2)
            except Exception as e:
                print(f"[WARN] Stream {self.url} dropped ({e}). Reconnecting...")
                time.sleep(2)

    def read(self):
        return self.frame

    def stop(self):
        self.running = False
print("[INFO] Opening remote camera streams...")
# Cam 1 is Front, Cam 0 is Rear
cam1_front = CameraStream(f"{BASE_URL}/video_feed_1")
cam0_rear  = CameraStream(f"{BASE_URL}/video_feed_0")

time.sleep(3) # Give remote streams a moment to buffer over the internet

# --- MAIN DISPLAY & LOGGING LOOP ---
print("[INFO] Ready. Press 'L' to toggle logging, 'Q' to quit.")

while True:
    f_front = cam1_front.read()
    f_rear = cam0_rear.read()

    # Create a blank canvas if frames aren't ready
    display_front = f_front.copy() if f_front is not None else np.zeros((480, 640, 3), dtype=np.uint8)
    display_rear = f_rear.copy() if f_rear is not None else np.zeros((480, 640, 3), dtype=np.uint8)

    # Overlay Telemetry on Front Camera
    if latest_telemetry:
        cv2.putText(display_front, f"State: {latest_telemetry.get('state', 'N/A')}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_front, f"Sonar: {latest_telemetry.get('ultrasonic_cm', -1)} cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        gyro_z = latest_telemetry.get('mpu6050', {}).get('gyro', {}).get('z', 0)
        cv2.putText(display_front, f"Gyro Z: {gyro_z}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Overlay Logging Status
    if logging_active:
        cv2.circle(display_front, (610, 30), 10, (0, 0, 255), -1)
        cv2.putText(display_front, "REC", (550, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Write frames to video
        if out_cam1 is not None and f_front is not None:
            out_cam1.write(f_front)
        if out_cam0 is not None and f_rear is not None:
            out_cam0.write(f_rear)

    # Show images (Side by Side)
    combined = cv2.hconcat([display_front, display_rear])
    cv2.imshow("AMR VSLAM Logger (Left: Front Cam 1 | Right: Rear Cam 0)", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('l'):
        logging_active = not logging_active
        if logging_active:
            print("[INFO] Logging STARTED")
            timestamp = int(time.time())
            
            # Setup Video Writers
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out_cam1 = cv2.VideoWriter(f'front_cam_{timestamp}.avi', fourcc, 15.0, (640, 480))
            out_cam0 = cv2.VideoWriter(f'rear_cam_{timestamp}.avi', fourcc, 15.0, (640, 480))
            
            # Setup CSV Writer
            csv_file = open(f'imu_data_{timestamp}.csv', 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['timestamp_sec', 'accel_x', 'accel_y', 'accel_z', 'gyro_z', 'sonar_cm', 'cmd_state'])
        else:
            print("[INFO] Logging STOPPED")
            if out_cam1: out_cam1.release()
            if out_cam0: out_cam0.release()
            if csv_file: csv_file.close()

# Cleanup
cam1_front.stop()
cam0_rear.stop()
sio.disconnect()
cv2.destroyAllWindows()
if out_cam1: out_cam1.release()
if out_cam0: out_cam0.release()
if csv_file: csv_file.close()