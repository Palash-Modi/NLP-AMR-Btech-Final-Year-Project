import cv2
import numpy as np
import glob

# --- CONFIGURATION ---
VIDEO_FILE = 'front_cam_1790537857.avi'  # Replace with your recorded video file
CHECKERBOARD = (7,5) # Number of INTERIOR corners (width, height)
SQUARE_SIZE = 0.025 # Size of a square in meters (e.g., 25mm = 0.025)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

cap = cv2.VideoCapture(VIDEO_FILE)
frame_count = 0
success_count = 0

print(f"Processing {VIDEO_FILE} for calibration...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process every 10th frame to get distinct angles
    if frame_count % 10 == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            success_count += 1
            
            # Draw and display the corners
            cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)
            cv2.imshow('Calibration (Press Q to skip to calculation)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    frame_count += 1

cap.release()
cv2.destroyAllWindows()

if success_count >=3:
    print(f"\n[INFO] Calculating intrinsics from {success_count} valid frames...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    print("\n--- CALIBRATION RESULTS ---")
    print("Camera Matrix (Intrinsic):\n", mtx)
    print("\nDistortion Coefficients:\n", dist)
    
    # Save to file for the VSLAM node to load later
    np.savez('cam_intrinsics.npz', mtx=mtx, dist=dist)
    print("\n[SUCCESS] Saved to cam_intrinsics.npz")
else:
    print(f"\n[ERROR] Not enough frames with a visible checkerboard ({success_count}/10 needed).")