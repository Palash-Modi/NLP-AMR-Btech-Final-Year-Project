import cv2
import numpy as np
import requests
import time
import socketio
import os
import json
import heapq
import logging
from collections import deque
from ultralytics import YOLO
import threading

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("AMR_NAV")

# --- CONFIGURATION ---
PI_HOST = "10.66.245.150"  # Ensure this matches your Pi's IP
PORT = 5000
PI_URL = f"http://{PI_HOST}:{PORT}"
MAP_FILENAME = "amr_saved_map.npy"
TARGETS_FILENAME = "amr_targets.npy"
GOAL_FILENAME = "current_goal.json"

WIN_W, WIN_H = 1000, 750
MAP_SCALE = 80.0  # pixels per meter
nav_mode = False            

# --- PID & VELOCITY TUNING CONSTANTS ---
# Visual Servoing PID (keeps target cx centered in camera frame)
KP_VIS = 0.28             # Proportional gain for pixel centering error
KI_VIS = 0.01             # Integral gain (eliminates steady-state offset)
KD_VIS = 0.06             # Derivative gain (dampens overshoot/oscillation)

# World Heading PID (used when target is outside camera FOV)
KP_YAW = 0.35             # Proportional gain for yaw angle error (rad)
KD_YAW = 0.05             # Derivative gain for yaw rate damping

MAX_OMEGA = 0.24          # Hard cap on rotation speed so camera never blurs
MIN_OMEGA = 0.12          # Minimum omega to overcome Mecanum roller static friction
MAX_VX = 0.26             # Max forward cruising speed
MIN_VX = 0.14             # Minimum forward creep speed when approaching target
CENTER_DEADBAND_PX = 18   # Pixel tolerance around center (640px width -> +-18px)
PID_UPDATE_RATE = 0.065   # ~15 Hz continuous PID update interval

# --- PHYSICAL SENSOR & CAMERA CALIBRATION (Pi Camera 3 / IMX708) ---
FOCAL_DISTANCE_NUMERATOR = 1880
FOCAL_LENGTH_PX = 470.2

# --- ODOMETRY, IMU & SENSOR VARIABLES ---
robot_x, robot_y, robot_yaw = 0.0, 0.0, 0.0
raw_sonar_dist = -1.0
sonar_history = deque(maxlen=100)
sonar_dist = -1.0

imu_gz_dps = 0.0
imu_motion_state = "STOPPED"
last_imu_time = time.time()

sonar_obstacle_points = []  
known_targets = {}          
target_depth_cache = {}
last_command_time = 0.0      
last_sent_cmd = ""
current_cmd = "STOP"
cmd_vx, cmd_omega = 0.0, 0.0
current_path = []           
active_llm_goal = None
active_goal_name = None
arrival_counter = 0

# --- PID STATE VARIABLES ---
pid_prev_error = 0.0
pid_integral = 0.0
pid_last_time = time.time()

def reset_pid():
    global pid_prev_error, pid_integral, pid_last_time
    pid_prev_error = 0.0
    pid_integral = 0.0
    pid_last_time = time.time()

sio = socketio.Client()

@sio.event
def connect():
    logger.info("Connected to Pi WebSocket server successfully.")

@sio.event
def disconnect():
    logger.warning("Disconnected from Pi WebSocket server.")

@sio.on('telemetry')
def on_telemetry(data):
    """Processes 25Hz telemetry and computes 100-sample running average on PC."""
    global raw_sonar_dist, sonar_dist, imu_gz_dps, imu_motion_state, robot_yaw, last_imu_time
    now = time.time()
    dt_imu = now - last_imu_time
    last_imu_time = now

    raw_val = float(data.get('ultrasonic_cm', -1.0))
    if 2.0 <= raw_val <= 350.0:
        sonar_history.append(raw_val)
        sonar_dist = round(sum(sonar_history) / len(sonar_history), 1)
    else:
        raw_sonar_dist = -1.0

    imu_motion_state = data.get('state', 'STOPPED')
    mpu = data.get('mpu6050', {})
    gyro = mpu.get('gyro', {})
    imu_gz_dps = float(gyro.get('z', 0.0))

    if 0.0 < dt_imu < 0.5:
        robot_yaw += np.deg2rad(imu_gz_dps) * dt_imu
        robot_yaw = (robot_yaw + np.pi) % (2 * np.pi) - np.pi

try:
    sio.connect(PI_URL, wait_timeout=5)
except Exception as e:
    logger.error(f"Failed to connect to WebSocket: {e}")

def clear_goal_file():
    """Clears or resets current_goal.json."""
    try:
        if os.path.exists(GOAL_FILENAME):
            empty_data = {"status": "idle", "destination": None, "coordinates": None}
            with open(GOAL_FILENAME, "w") as f:
                json.dump(empty_data, f, indent=4)
            logger.info("Cleared target in current_goal.json.")
    except Exception as e:
        logger.error(f"Failed to clear goal file: {e}")

logger.info("Loading YOLO model (best.pt)...")
yolo_model = YOLO("best.pt")
logger.info("YOLO model loaded successfully.")

class FreshFrameReader:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.frame = None
        self.ret = False
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.ret = ret
                self.frame = frame

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.cap.release()

cap0 = FreshFrameReader(f"{PI_URL}/video_feed_0")
cap1 = FreshFrameReader(f"{PI_URL}/video_feed_1")

prev_time = cv2.getTickCount()

def send_robot_cmd(cmd):
    """Emits discrete string command to Pi (used for STOP or manual keys)."""
    global last_sent_cmd, current_cmd, cmd_vx, cmd_omega
    current_cmd = cmd
    if cmd == "STOP":
        cmd_vx, cmd_omega = 0.0, 0.0
    if not sio.connected:
        return
    if cmd != last_sent_cmd:
        last_sent_cmd = cmd
        try:
            sio.emit('control', {'command': cmd})
        except Exception as e:
            logger.error(f"Failed to emit command {cmd}: {e}")

def send_pid_velocity(vx, omega):
    """Emits continuous float velocities (vx, omega) to Pi's PID handler."""
    global current_cmd, last_sent_cmd, cmd_vx, cmd_omega
    if not sio.connected:
        return
    cmd_vx = float(vx)
    cmd_omega = float(omega)
    current_cmd = "PID_DRIVING" if (abs(vx) > 0.01 or abs(omega) > 0.01) else "STOP"
    last_sent_cmd = current_cmd
    try:
        sio.emit('control', {'vx': round(cmd_vx, 3), 'omega': round(cmd_omega, 3)})
    except Exception as e:
        logger.error(f"Failed to emit PID velocity: {e}")

def detect_objects_yolo(frame):
    """Detects objects using best.pt and calculates physical depth via Pi Cam 3 optics."""
    results = yolo_model(frame, verbose=False)
    detected_objects = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            cls_name = yolo_model.names[cls_id].lower().replace(" ", "_")
            
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            aspect_ratio = w / h
            
            if conf > 0.45 and (0.60 <= aspect_ratio <= 1.50):
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                dimension_px = w
                est_dist_cm = FOCAL_DISTANCE_NUMERATOR / dimension_px
                
                detected_objects.append({
                    "class": cls_name,
                    "cx": cx,
                    "cy": cy,
                    "box": (int(x1), int(y1), int(w), int(h)),
                    "dist_cm": est_dist_cm,
                    "conf": conf
                })
    return detected_objects

def plan_path_a_star(start_wx, start_wy, goal_wx, goal_wy, obstacles, grid_resolution=0.1, grid_size=100):
    def world_to_grid(wx, wy):
        gx = int(round((wx - start_wx) / grid_resolution)) + grid_size // 2
        gy = int(round((wy - start_wy) / grid_resolution)) + grid_size // 2
        return gx, gy

    def grid_to_world(gx, gy):
        wx = start_wx + (gx - grid_size // 2) * grid_resolution
        wy = start_wy + (gy - grid_size // 2) * grid_resolution
        return wx, wy

    start_g = world_to_grid(start_wx, start_wy)
    goal_g = world_to_grid(goal_wx, goal_wy)

    if not (0 <= start_g[0] < grid_size and 0 <= start_g[1] < grid_size):
        return [(goal_wx, goal_wy)]

    grid = np.zeros((grid_size, grid_size), dtype=np.int8)
    for ox, oy in obstacles:
        if np.hypot(ox - goal_wx, oy - goal_wy) < 0.30:
            continue
        gx, gy = world_to_grid(ox, oy)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    grid[ny, nx] = 1

    open_set = []
    heapq.heappush(open_set, (0.0, start_g))
    came_from = {}
    g_score = {start_g: 0.0}
    f_score = {start_g: np.hypot(goal_g[0] - start_g[0], goal_g[1] - start_g[1])}

    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    while open_set:
        _, current = heapq.heappop(open_set)

        if np.hypot(current[0] - goal_g[0], current[1] - goal_g[1]) <= 1.5:
            path = []
            while current in came_from:
                path.append(grid_to_world(current[0], current[1]))
                current = came_from[current]
            path.reverse()
            path.append((goal_wx, goal_wy))
            return path

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            if not (0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size):
                continue
            if grid[neighbor[1], neighbor[0]] == 1:
                continue

            tentative_g = g_score[current] + np.hypot(dx, dy)
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h = np.hypot(goal_g[0] - neighbor[0], goal_g[1] - neighbor[1])
                f_score[neighbor] = tentative_g + h
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return [(goal_wx, goal_wy)]

cv2.namedWindow("Point Cloud Navigation Map", cv2.WINDOW_NORMAL)

try:
    logger.info("--- AMR Autonomous Navigation (PID Visual Servoing & Centering) Active ---")
    logger.info("Controls: Focus Map Window -> [W/S/A/D/Q/E] Drive | [SPACE] Stop | [N] Toggle Autonomy | [ESC] Exit")

    last_goal_check = 0.0
    last_coord_log_time = 0.0

    while True:
        now = time.time()

        # 1. Non-blocking OpenCV key detection
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            logger.info("Escape key pressed. Exiting.")
            active_llm_goal = None
            nav_mode = False
            clear_goal_file()
            send_robot_cmd("STOP")
            break
        elif key == ord('n') or key == ord('N'):
            nav_mode = not nav_mode
            current_path = []
            arrival_counter = 0
            reset_pid()
            if nav_mode:
                logger.info("AUTONOMOUS PURSUIT MODE ENGAGED via [N] key...")
            else:
                logger.info("AUTONOMOUS MODE DISENGAGED via [N] key...")
                send_robot_cmd("STOP")

        manual_cmd = None
        if key == ord('w'): manual_cmd = "FWD"
        elif key == ord('s'): manual_cmd = "BACK"
        elif key == ord('a'): manual_cmd = "STR_L"
        elif key == ord('d'): manual_cmd = "STR_R"
        elif key == ord('q'): manual_cmd = "ROT_L"
        elif key == ord('e'): manual_cmd = "ROT_R"
        elif key == ord(' ') or key == ord('x'): manual_cmd = "STOP"

        if manual_cmd is not None:
            if nav_mode or active_llm_goal is not None:
                logger.info("Manual override triggered. Disengaging autonomous mode.")
                nav_mode = False
                active_llm_goal = None
                arrival_counter = 0
                reset_pid()
                clear_goal_file()
            if manual_cmd == "ROT_L":
                send_pid_velocity(0.0, -0.20)
            elif manual_cmd == "ROT_R":
                send_pid_velocity(0.0, 0.20)
            else:
                send_robot_cmd(manual_cmd)

        # 2. Check for LLM goal updates
        if now - last_goal_check > 0.4:
            last_goal_check = now
            if os.path.exists(GOAL_FILENAME):
                try:
                    with open(GOAL_FILENAME, "r") as f:
                        goal_data = json.load(f)
                    
                    status = goal_data.get("status")
                    new_coords = goal_data.get("coordinates")
                    new_dest = goal_data.get("destination")
                    
                    if status == "navigating" and new_coords:
                        if (active_llm_goal is None or 
                            active_goal_name != new_dest or
                            np.hypot(active_llm_goal[0] - new_coords[0], active_llm_goal[1] - new_coords[1]) > 0.25):
                            
                            active_llm_goal = [float(new_coords[0]), float(new_coords[1])]
                            active_goal_name = new_dest
                            nav_mode = True
                            current_path = []
                            arrival_counter = 0
                            reset_pid()
                            logger.info(f"New LLM Goal Activated: {new_dest} -> Destination: ({active_llm_goal[0]:.2f}, {active_llm_goal[1]:.2f})")
                except Exception:
                    pass

        # 3. Read Camera Feeds with Color Correction
        ret0, f0_raw = cap0.read()
        ret1, f1_raw = cap1.read()
        
        if not ret0 or not ret1 or f0_raw is None or f1_raw is None:
            time.sleep(0.01)
            continue

        frame0 = cv2.cvtColor(f0_raw.copy(), cv2.COLOR_RGB2BGR)
        frame1 = cv2.cvtColor(f1_raw.copy(), cv2.COLOR_RGB2BGR)

        # 4. Integrate Local Odometry (supports both PID float driving and manual commands)
        curr_tick = cv2.getTickCount()
        dt = max(0.001, min(0.08, (curr_tick - prev_time) / cv2.getTickFrequency()))
        prev_time = curr_tick

        if current_cmd == "PID_DRIVING":
            # Scale odometry step proportionally to commanded vx
            lin_step = (cmd_vx * 0.45) * dt
            robot_x += lin_step * np.cos(robot_yaw)
            robot_y += lin_step * np.sin(robot_yaw)
        elif current_cmd == "FWD":
            robot_x += (0.20 * dt) * np.cos(robot_yaw)
            robot_y += (0.20 * dt) * np.sin(robot_yaw)
        elif current_cmd == "BACK":
            robot_x -= (0.20 * dt) * np.cos(robot_yaw)
            robot_y -= (0.20 * dt) * np.sin(robot_yaw)
        elif current_cmd == "STR_L":
            robot_x -= (0.20 * dt) * np.sin(robot_yaw)
            robot_y += (0.20 * dt) * np.cos(robot_yaw)
        elif current_cmd == "STR_R":
            robot_x += (0.20 * dt) * np.sin(robot_yaw)
            robot_y -= (0.20 * dt) * np.cos(robot_yaw)

        # 5. Build Point Cloud using 100-sample averaged Sonar Feed
        if 8.0 < sonar_dist < 180.0:
            dist_m = sonar_dist / 100.0
            ox = robot_x + dist_m * np.cos(robot_yaw)
            oy = robot_y + dist_m * np.sin(robot_yaw)
            if not any(np.hypot(p[0] - ox, p[1] - oy) < 0.12 for p in sonar_obstacle_points):
                sonar_obstacle_points.append((ox, oy))

        # 6. Object Detection & Visual Target Locking
        detected_objects = detect_objects_yolo(frame1)
        frame_center_x = frame1.shape[1] // 2  
        visible_goal_obj = None

        # Draw center crosshair on front camera feed for visual servo reference
        cv2.line(frame1, (frame_center_x, 0), (frame_center_x, frame1.shape[0]), (255, 255, 0), 1)

        for obj in detected_objects:
            cls_name = obj['class']
            cx, cy = obj['cx'], obj['cy']
            dist_m = obj['dist_cm'] / 100.0
            
            if cls_name in target_depth_cache:
                target_depth_cache[cls_name] = 0.6 * target_depth_cache[cls_name] + 0.4 * obj['dist_cm']
            else:
                target_depth_cache[cls_name] = obj['dist_cm']
            
            # Project angle (right of center is negative yaw in standard CCW coordinates)
            azimuth = -np.arctan2((cx - frame_center_x), FOCAL_LENGTH_PX)
            abs_angle = robot_yaw + azimuth
            obj_wx = robot_x + dist_m * np.cos(abs_angle)
            obj_wy = robot_y + dist_m * np.sin(abs_angle)
            
            known_targets[cls_name] = [obj_wx, obj_wy]
            
            # Check if this detected object is our active goal
            if active_goal_name and cls_name == active_goal_name:
                active_llm_goal = [obj_wx, obj_wy]
                visible_goal_obj = obj
            elif active_goal_name is None and nav_mode and visible_goal_obj is None:
                visible_goal_obj = obj
                active_goal_name = cls_name

            is_current_goal = False
            if active_llm_goal:
                is_current_goal = np.hypot(active_llm_goal[0] - obj_wx, active_llm_goal[1] - obj_wy) < 0.40
            
            if not is_current_goal:
                if not any(np.hypot(p[0] - obj_wx, p[1] - obj_wy) < 0.25 for p in sonar_obstacle_points):
                    sonar_obstacle_points.append((obj_wx, obj_wy))

            x, y, w, h = obj['box']
            box_color = (0, 0, 255) if (active_goal_name == cls_name) else (0, 255, 0)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(frame1, f"{cls_name} {target_depth_cache[cls_name]:.1f}cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 2)

        np.save(TARGETS_FILENAME, known_targets)

        # 7. Continuous PID Visual Centering & Navigation
        if nav_mode:
            target_pos = active_llm_goal if active_llm_goal else (list(known_targets.values())[0] if known_targets else None)
            
            if target_pos:
                goal_x, goal_y = target_pos[0], target_pos[1]
                dist_to_goal = np.hypot(goal_x - robot_x, goal_y - robot_y)
                visual_depth_cm = target_depth_cache.get(active_goal_name, 999.0) if active_goal_name else 999.0

                if now - last_coord_log_time > 1.0:
                    mode_str = "VISUAL-LOCK" if visible_goal_obj else "ODOM-SEARCH"
                    logger.info(
                        f"[{mode_str}] Coords: ({robot_x:.2f}, {robot_y:.2f}) Yaw: {np.rad2deg(robot_yaw):.1f}° | "
                        f"Dest: ({goal_x:.2f}, {goal_y:.2f}) | Dist: {dist_to_goal:.2f}m | VisDepth: {visual_depth_cm:.1f}cm | Sonar: {sonar_dist:.1f}cm"
                    )
                    last_coord_log_time = now
                VISUAL_STOP_CM = 10.0  # Adjust based on how close you want the 4cm ball detection to trigger
                SONAR_STOP_CM = 12.0
                # Debounced Arrival Condition
                visually_close = (visual_depth_cm <= VISUAL_STOP_CM and dist_to_goal <= 0.40)
                sonar_close = (2.0 < sonar_dist <= SONAR_STOP_CM and dist_to_goal <= 0.40)
                if visually_close or sonar_close:
                    arrival_counter += 1
                else:
                    arrival_counter = 0

                if arrival_counter >= 3:
                    send_robot_cmd("STOP")
                    nav_mode = False
                    active_llm_goal = None
                    active_goal_name = None
                    arrival_counter = 0
                    reset_pid()
                    clear_goal_file()
                    logger.info(
                        f"SUCCESS: Target reached and centered! (VisDepth: {visual_depth_cm:.1f}cm, Dist: {dist_to_goal:.2f}m, Sonar: {sonar_dist:.1f}cm)."
                    )
                elif now - last_command_time >= PID_UPDATE_RATE:
                    dt_pid = max(0.01, min(0.25, now - pid_last_time))
                    pid_last_time = now

                    # --- CASE A: TARGET IS VISIBLE IN CAMERA -> VISUAL SERVOING PID ---
                    if visible_goal_obj is not None:
                        cx = visible_goal_obj['cx']
                        px_error = cx - frame_center_x  # Positive = ball is to the right -> need ROT_R (+omega)

                        # Draw error vector line from center crosshair to ball center
                        cv2.line(frame1, (frame_center_x, visible_goal_obj['cy']), (cx, visible_goal_obj['cy']), (0, 0, 255), 2)

                        if abs(px_error) <= CENTER_DEADBAND_PX:
                            # Target is locked in the center deadband -> zero rotation, smooth forward drive
                            omega_out = 0.0
                            pid_integral = 0.0
                            pid_prev_error = 0.0
                        else:
                            norm_err = px_error / float(frame_center_x)  # [-1.0, 1.0]
                            pid_integral = np.clip(pid_integral + norm_err * dt_pid, -0.5, 0.5)
                            derivative = (norm_err - pid_prev_error) / dt_pid
                            pid_prev_error = norm_err

                            raw_omega = (KP_VIS * norm_err) + (KI_VIS * pid_integral) + (KD_VIS * derivative)
                            sign = 1.0 if raw_omega > 0 else -1.0
                            omega_out = sign * np.clip(abs(raw_omega), MIN_OMEGA, MAX_OMEGA)

                        # Modulate forward speed (vx): slow down as we get close or if heavily off-center
                        alignment_factor = max(0.0, 1.0 - abs(px_error) / 220.0)
                        depth_factor = np.clip((visual_depth_cm - 15.0) / 60.0, 0.45, 1.0)
                        
                        if abs(px_error) > 160:
                            # Ball is near edge of frame -> center it in place first before driving forward
                            vx_out = 0.0
                        else:
                            vx_out = np.clip(MAX_VX * alignment_factor * depth_factor, MIN_VX, MAX_VX)

                        send_pid_velocity(vx_out, omega_out)

                    # --- CASE B: TARGET OUT OF CAMERA VIEW -> WORLD HEADING PID ---
                    else:
                        dx = goal_x - robot_x
                        dy = goal_y - robot_y
                        desired_yaw = np.arctan2(dy, dx)
                        angle_diff = (desired_yaw - robot_yaw + np.pi) % (2 * np.pi) - np.pi

                        # Note: angle_diff > 0 means need CCW left turn (negative omega on Pi)
                        yaw_err = -angle_diff
                        derivative = (yaw_err - pid_prev_error) / dt_pid
                        pid_prev_error = yaw_err

                        if abs(angle_diff) <= 0.18:
                            omega_out = 0.0
                            vx_out = MIN_VX + 0.04
                        else:
                            raw_omega = (KP_YAW * yaw_err) + (KD_YAW * derivative)
                            sign = 1.0 if raw_omega > 0 else -1.0
                            omega_out = sign * np.clip(abs(raw_omega), MIN_OMEGA, MAX_OMEGA)
                            # Creep forward only if roughly facing the target (< ~35 deg)
                            vx_out = MIN_VX if abs(angle_diff) < 0.60 else 0.0

                        send_pid_velocity(vx_out, omega_out)

                    last_command_time = now

        # 8. Render Canvas
        canvas = np.full((WIN_H, WIN_W, 3), 192, dtype=np.uint8)
        center_x = WIN_W // 2 - int(robot_x * MAP_SCALE)
        center_y = WIN_H // 2 + int(robot_y * MAP_SCALE)

        for ox, oy in sonar_obstacle_points:
            px = int(center_x + ox * MAP_SCALE)
            py = int(center_y - oy * MAP_SCALE)
            if 0 <= px < WIN_W and 0 <= py < WIN_H:
                cv2.circle(canvas, (px, py), 3, (80, 80, 80), -1)

        for pt in current_path:
            px = int(center_x + pt[0] * MAP_SCALE)
            py = int(center_y - pt[1] * MAP_SCALE)
            if 0 <= px < WIN_W and 0 <= py < WIN_H:
                cv2.circle(canvas, (px, py), 3, (0, 255, 0), -1)

        for name, (bx, by) in known_targets.items():
            px = int(center_x + bx * MAP_SCALE)
            py = int(center_y - by * MAP_SCALE)
            if 0 <= px < WIN_W and 0 <= py < WIN_H:
                cv2.circle(canvas, (px, py), 6, (0, 165, 255), -1)
                cv2.putText(canvas, name, (px + 8, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        curr_px, curr_py = int(center_x + robot_x * MAP_SCALE), int(center_y - robot_y * MAP_SCALE)
        arrow_x = int(curr_px + 15 * np.cos(robot_yaw))
        arrow_y = int(curr_py - 15 * np.sin(robot_yaw))
        cv2.circle(canvas, (curr_px, curr_py), 6, (0, 180, 0), -1)
        cv2.arrowedLine(canvas, (curr_px, curr_py), (arrow_x, arrow_y), (0, 100, 0), 2)

        small_f1 = cv2.resize(frame1, (250, 180))
        small_f0 = cv2.resize(frame0, (250, 180))
        canvas[20:200, 20:270] = small_f1
        canvas[215:395, 20:270] = small_f0

        cv2.imshow("Point Cloud Navigation Map", canvas)

finally:
    logger.info("Shutting down system. Sending final STOP command.")
    send_robot_cmd("STOP")
    active_llm_goal = None
    clear_goal_file()
    if sio.connected:
        sio.disconnect()
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()