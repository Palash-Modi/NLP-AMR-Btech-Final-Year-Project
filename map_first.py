import cv2
import numpy as np
import requests
import time
import threading
import keyboard  # Global key listener
import socketio
import os

# --- CONFIGURATION ---
PI_HOST = "192.168.88.10"
PORT = 5000
PI_URL = f"http://{PI_HOST}:{PORT}"
MAP_FILENAME = "amr_saved_map.npy"
TXT_FILENAME = "obstacles_coordinates.txt"

# Initialize Socket.IO Client for controlling the robot
sio = socketio.Client()

@sio.event
def connect():
    print("[+] Connected to Pi WebSocket server!")

@sio.event
def disconnect():
    print("[-] Disconnected from Pi WebSocket server.")

try:
    sio.connect(PI_URL)
except Exception as e:
    print(f"[-] Failed to connect to WebSocket: {e}")

# Mapping variables
WIN_W, WIN_H = 1000, 750
MAP_SCALE = 80.0  # pixels per meter
robot_x, robot_y, robot_yaw = 0.0, 0.0, 0.0

# Point clouds
sonar_obstacle_points = []  # (x, y) red obstacle points
visual_point_cloud = []     # (x, y) blue visual feature points from camera

# ORB Feature Extractor for Visual SLAM Point Cloud
orb = cv2.ORB_create(nfeatures=300)

class MJPEGStreamReader:
    """Robust threaded reader for Flask MJPEG video feeds."""
    def __init__(self, url):
        self.url = url
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        bytes_data = b""
        while self.running:
            try:
                response = requests.get(self.url, stream=True, timeout=5)
                for chunk in response.iter_content(chunk_size=1024):
                    if not self.running:
                        break
                    bytes_data += chunk
                    a = bytes_data.find(b'\xff\xd8')  # JPEG Start
                    b = bytes_data.find(b'\xff\xd9')  # JPEG End
                    if a != -1 and b != -1:
                        jpg = bytes_data[a:b+2]
                        bytes_data = bytes_data[b+2:]
                        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if img is not None:
                            self.frame = img
            except Exception:
                time.sleep(1)

    def read(self):
        return True, self.frame

    def release(self):
        self.running = False

print(f"[INFO] Connecting to robot streams at {PI_URL}...")

cam0 = MJPEGStreamReader(f"{PI_URL}/video_feed_0")
cam1 = MJPEGStreamReader(f"{PI_URL}/video_feed_1")

prev_time = cv2.getTickCount()
last_sent_cmd = ""

def send_robot_cmd(cmd):
    global last_sent_cmd
    if cmd != last_sent_cmd:
        last_sent_cmd = cmd
        try:
            sio.emit('control', {'command': cmd})
        except Exception as e:
            print(f"[-] Failed to emit command {cmd}: {e}")

def save_map():
    global sonar_obstacle_points, visual_point_cloud
    if len(sonar_obstacle_points) == 0 and len(visual_point_cloud) == 0:
        print("[-] No map data to save!")
        return

    # 1. Save binary .npy map file
    map_data = {
        "obstacles": sonar_obstacle_points,
        "visual_pc": visual_point_cloud,
        "last_pose": [robot_x, robot_y, robot_yaw]
    }
    np.save(MAP_FILENAME, map_data)

    # 2. Export every obstacle coordinate to a .txt file
    with open(TXT_FILENAME, "w") as f:
        f.write("# AMR Obstacle Point Coordinates (X_meters, Y_meters)\n")
        f.write("Index, X (m), Y (m)\n")
        for idx, (ox, oy) in enumerate(sonar_obstacle_points):
            f.write(f"{idx+1}, {ox:.4f}, {oy:.4f}\n")

    print(f"[+] Map successfully saved! Binary: {MAP_FILENAME} | Text Export: {TXT_FILENAME} ({len(sonar_obstacle_points)} obstacles)")

def perform_active_localization(current_sonar_cm):
    """Actively matches current live sensor readings against the saved obstacle map to localize pose."""
    global robot_x, robot_y, robot_yaw, sonar_obstacle_points
    if len(sonar_obstacle_points) < 5:
        print("[-] Not enough obstacle points in map for active localization.")
        return

    print("[*] Running active map localization scan...")
    best_x, best_y, best_yaw = robot_x, robot_y, robot_yaw
    min_error = float('inf')

    current_sonar_m = current_sonar_cm / 100.0 if current_sonar_cm > 0 else 0.5

    # Scan through candidate positions in the loaded map point cloud
    for ox, oy in sonar_obstacle_points:
        # Assume robot is positioned at a distance from this obstacle matching current sonar
        for yaw_cand in np.linspace(-np.pi, np.pi, 16):  # 16 heading slices
            cand_x = ox - current_sonar_m * np.cos(yaw_cand)
            cand_y = oy - current_sonar_m * np.sin(yaw_cand)

            # Check how well other nearby obstacles align with this candidate pose
            error = 0
            matches = 0
            for rx, ry in sonar_obstacle_points:
                dist = np.hypot(rx - cand_x, ry - cand_y)
                if dist < 1.5:  # within 1.5 meters
                    matches += 1

            # Reward positions with higher local map density match
            if matches > min_error or min_error == float('inf'):
                min_error = matches
                best_x, best_y, best_yaw = cand_x, cand_y, yaw_cand

    robot_x, robot_y, robot_yaw = best_x, best_y, best_yaw
    print(f"[+] Localization Successful! Snapped Pose -> X: {robot_x:.2f}m, Y: {robot_y:.2f}m, Yaw: {np.degrees(robot_yaw):.1f}°")

def load_map(current_sonar_cm=-1):
    global sonar_obstacle_points, visual_point_cloud, robot_x, robot_y, robot_yaw
    if os.path.exists(MAP_FILENAME):
        map_data = np.load(MAP_FILENAME, allow_pickle=True).item()
        sonar_obstacle_points = map_data.get("obstacles", [])
        visual_point_cloud = map_data.get("visual_pc", [])
        
        # Trigger Active Localization using current sensor feedback against the map
        perform_active_localization(current_sonar_cm)

        print(f"[+] Map loaded successfully from {MAP_FILENAME} ({len(sonar_obstacle_points)} obstacles).")
    else:
        print(f"[-] Save file '{MAP_FILENAME}' not found.")

try:
    print("\n--- WebSocket Keyboard, Point Cloud & Mapping Controls Active ---")
    print(" [W] / [S] : Forward / Backward")
    print(" [A] / [D] : Strafe Left / Strafe Right")
    print(" [Q] / [E] : Rotate Left / Rotate Right")
    print(" [SPACE]   : Stop")
    print(" [M]       : Save Map (.npy) & Export Obstacles (.txt)")
    print(" [L]       : Load Saved Map & Actively Localize Position")
    print(" [ESC]     : Exit\n")

    current_cmd = "STOP"

    while True:
        # Fetch telemetry first so it's available for localization checks
        sonar_dist, gyro_z = -1.0, 0.0
        try:
            sensor_res = requests.get(f"{PI_URL}/sensors/json", timeout=0.1)
            if sensor_res.status_code == 200:
                sensors = sensor_res.json()
                sonar_dist = sensors.get("sonar_cm", -1.0)
        except Exception:
            pass

        # Keyboard handling
        if keyboard.is_pressed('esc'):
            send_robot_cmd("STOP")
            break
        elif keyboard.is_pressed('m'):
            save_map()
            time.sleep(0.5)
        elif keyboard.is_pressed('l'):
            load_map(sonar_dist)
            time.sleep(0.5)
        elif keyboard.is_pressed('w'):
            current_cmd = "FWD"
        elif keyboard.is_pressed('s'):
            current_cmd = "BACK"
        elif keyboard.is_pressed('a'):
            current_cmd = "STR_L"
        elif keyboard.is_pressed('d'):
            current_cmd = "STR_R"
        elif keyboard.is_pressed('q'):
            current_cmd = "ROT_L"
        elif keyboard.is_pressed('e'):
            current_cmd = "ROT_R"
        elif keyboard.is_pressed('space'):
            current_cmd = "STOP"
        else:
            current_cmd = "STOP"

        send_robot_cmd(current_cmd)

        ret0, frame0 = cam0.read()
        ret1, frame1 = cam1.read()
        
        if frame0 is None or frame1 is None:
            time.sleep(0.05)
            continue

        curr_tick = cv2.getTickCount()
        dt = max(0.001, min(0.08, (curr_tick - prev_time) / cv2.getTickFrequency()))
        prev_time = curr_tick

        # Odometry motion update
        move_speed = 0.25 * dt
        turn_speed = 1.0 * dt

        if current_cmd == "FWD":
            robot_x += move_speed * np.cos(robot_yaw)
            robot_y += move_speed * np.sin(robot_yaw)
        elif current_cmd == "BACK":
            robot_x -= move_speed * np.cos(robot_yaw)
            robot_y -= move_speed * np.sin(robot_yaw)
        elif current_cmd == "STR_L":
            robot_x -= move_speed * np.sin(robot_yaw)
            robot_y += move_speed * np.cos(robot_yaw)
        elif current_cmd == "STR_R":
            robot_x += move_speed * np.sin(robot_yaw)
            robot_y -= move_speed * np.cos(robot_yaw)
        elif current_cmd == "ROT_L":
            robot_yaw += turn_speed
        elif current_cmd == "ROT_R":
            robot_yaw -= turn_speed

        # 1. Map Sonar Obstacles (Red Points)
        if 5.0 <= sonar_dist <= 300.0:
            dist_m = sonar_dist / 100.0
            obs_x = robot_x + dist_m * np.cos(robot_yaw)
            obs_y = robot_y + dist_m * np.sin(robot_yaw)
            if not any(np.hypot(p[0] - obs_x, p[1] - obs_y) < 0.08 for p in sonar_obstacle_points):
                sonar_obstacle_points.append((obs_x, obs_y))

        # 2. Extract Visual Feature Point Cloud from Camera Frame (Blue/Cyan Points)
        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray0, None)
        if kp:
            for p in kp[:30]:
                fx, fy = p.pt
                rel_x = 0.5 + (fy - 120) * 0.002
                rel_y = (fx - 160) * 0.002
                
                wx = robot_x + rel_x * np.cos(robot_yaw) - rel_y * np.sin(robot_yaw)
                wy = robot_y + rel_x * np.sin(robot_yaw) + rel_y * np.cos(robot_yaw)
                
                if not any(np.hypot(vp[0] - wx, vp[1] - wy) < 0.05 for vp in visual_point_cloud):
                    visual_point_cloud.append((wx, wy))

        # --- RENDER MAP CANVAS ---
        canvas = np.full((WIN_H, WIN_W, 3), 192, dtype=np.uint8)
        center_x = WIN_W // 2 - int(robot_x * MAP_SCALE)
        center_y = WIN_H // 2 + int(robot_y * MAP_SCALE)

        # Draw grid lines
        grid_step = int(0.5 * MAP_SCALE)
        for gx in range(center_x % grid_step, WIN_W, grid_step):
            cv2.line(canvas, (gx, 0), (gx, WIN_H), (175, 175, 175), 1)
        for gy in range(center_y % grid_step, WIN_W, grid_step):
            cv2.line(canvas, (0, gy), (WIN_W, gy), (175, 175, 175), 1)

        # Draw Visual Feature Point Cloud (Blue/Cyan dots)
        for vx, vy in visual_point_cloud:
            px = int(center_x + vx * MAP_SCALE)
            py = int(center_y - vy * MAP_SCALE)
            if 0 <= px < WIN_W and 0 <= py < WIN_H:
                cv2.circle(canvas, (px, py), 1, (255, 150, 0), -1)

        # Draw Sonar Obstacles (Red dots)
        for ox, oy in sonar_obstacle_points:
            px = int(center_x + ox * MAP_SCALE)
            py = int(center_y - oy * MAP_SCALE)
            if 0 <= px < WIN_W and 0 <= py < WIN_H:
                cv2.circle(canvas, (px, py), 3, (0, 0, 255), -1)

        # Draw Robot Position & Heading (Green)
        curr_px, curr_py = int(center_x + robot_x * MAP_SCALE), int(center_y - robot_y * MAP_SCALE)
        cv2.circle(canvas, (curr_px, curr_py), 6, (0, 180, 0), -1)
        hx = int(curr_px + 15 * np.cos(robot_yaw))
        hy = int(curr_py - 15 * np.sin(robot_yaw))
        cv2.line(canvas, (curr_px, curr_py), (hx, hy), (0, 100, 0), 2)

        # Camera & Status Overlay
        small_f0 = cv2.resize(frame0, (200, 150))
        canvas[20:170, 20:220] = small_f0
        cv2.putText(canvas, f"Sonar: {sonar_dist}cm", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
        cv2.putText(canvas, f"Visual Points: {len(visual_point_cloud)}", (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
        cv2.putText(canvas, f"Obstacles: {len(sonar_obstacle_points)}", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)

        cv2.imshow("PC V-SLAM & Navigation Map", canvas)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    send_robot_cmd("STOP")
    sio.disconnect()
    cam0.release()
    cam1.release()
    cv2.destroyAllWindows()