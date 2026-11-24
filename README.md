# Autonomous Mobile Robot (AMR) with Natural Language Processing & Operational Dashboard

A lightweight, native-Python **Autonomous Mobile Robot (AMR)** system combining **Visual SLAM (VSLAM)**, **YOLOv8 object detection**, a **fine-tuned Qwen3-1.7B LoRA NLP command pipeline**, and a **real-time operational telemetry dashboard**—built without ROS/ROS 2 overhead for low-latency edge-to-laptop execution.

---

## Table of Contents

1. [Key Features](#key-features)
2. [System Architecture](#system-architecture)
3. [Repository Structure](#repository-structure)
4. [Prerequisites & Hardware](#prerequisites--hardware)
5. [Dependencies](#dependencies)
6. [Installation](#installation)
7. [How to Run](#how-to-run)
8. [Troubleshooting](#troubleshooting)
9. [Contributors](#contributors)

---

## Key Features

* **Natural Language Command Interface (`AMR_NLP_Integration/`)**: Translates unstructured human instructions into structured robot navigation goals using a fine-tuned **Qwen3-1.7B LoRA** adapter (`qwen3-1.7b-amr-lora`), backed by fuzzy target matching (`fuzzyMatch.py`) and a browser UI (`index.html`).
* **Visual SLAM & Environment Mapping (`vslam_map.py`, `map_first.py`)**: Performs monocular camera-based localization and mapping, persisting maps (`amr_saved_map.npy`) and named waypoint targets (`amr_targets.npy`).
* **Computer Vision & Camera Calibration (`calibri.py`, `best.pt`, `yolov8n.pt`)**: Supports intrinsic camera calibration (`cam_intrinsics.npz`) and real-time object/obstacle detection using pre-trained (`yolov8n.pt`) and custom-trained (`best.pt`) YOLOv8 weights.
* **Edge-to-Laptop Control Client (`laptop_client.py`)**: Offloads heavy inference (LLM, YOLOv8, VSLAM) to a host laptop while streaming velocity/navigation commands and logging IMU telemetry (`imu_data_*.csv`).
* **Live Operational Dashboard (`amr_dashboard.py`)**: Visualizes real-time robot pose, active goals (`current_goal.json`), map state, and sensor telemetry.

---

## System Architecture

```text
[User Browser: index.html]
           │  Natural language command
           ▼
[AMR_NLP_Integration/main.py] ──► Qwen3-1.7B LoRA + fuzzyMatch.py
           │  Writes structured destination
           ▼
   [current_goal.json]
           │
           ▼
   [laptop_client.py] ◄──► [vslam_map.py + YOLOv8 (best.pt / yolov8n.pt)]
           │                (Uses cam_intrinsics.npz, amr_saved_map.npy, amr_targets.npy)
           │  Control commands & IMU/state telemetry
           ▼
 [Mobile Robot Hardware] ──► [amr_dashboard.py] (Live Telemetry & Monitoring)
```

---

## Repository Structure

```text
.
├── AMR_NLP_Integration/
│   ├── model/
│   │   └── qwen3-1.7b-amr-lora/    # Fine-tuned Qwen3-1.7B LoRA adapter weights & config
│   ├── fuzzyMatch.py               # Fuzzy string matching for command/waypoint resolution
│   ├── index.html                  # Web frontend for natural language commands
│   ├── main.py                     # NLP backend API server & goal writer
│   ├── requirements.txt            # NLP module Python dependencies
│   └── README.md                   # NLP module-specific documentation
│
├── amr_dashboard.py                # Real-time operational & telemetry dashboard
├── laptop_client.py                # Host client for robot communication & control
├── vslam_map.py                    # Live Visual SLAM localization & mapping
├── map_first.py                    # Initial environment mapping & waypoint capture
├── calibri.py                      # Camera calibration script
│
├── cam_intrinsics.npz              # Saved camera matrix & distortion coefficients
├── amr_saved_map.npy               # Persisted environment map array
├── amr_targets.npy                 # Saved waypoint/target coordinates
├── current_goal.json               # Active navigation goal shared between modules
├── imu_data_1790537857.csv         # Sample logged IMU telemetry data
│
├── best.pt                         # Custom-trained YOLOv8 model weights
├── yolov8n.pt                      # Pre-trained YOLOv8 nano weights
├── Start_Commands.txt              # Quick-reference startup commands
└── temp.ipynb                      # Experimental Jupyter notebook
```

---

## Prerequisites & Hardware

* **OS**: Windows 10/11 or Ubuntu 22.04 / 24.04
* **Python**: 3.10 – 3.13
* **GPU (Recommended)**: NVIDIA GPU with CUDA support (4 GB+ VRAM recommended for concurrent Qwen3-1.7B LoRA + YOLOv8 inference)
* **Git LFS**: Required if cloning large model weights (`.safetensors`, `.pt`)

---

## Dependencies

### 1. NLP & LLM Pipeline (`AMR_NLP_Integration/`)
| Package | Purpose |
| :--- | :--- |
| `torch` | Deep learning backend (CUDA recommended) |
| `transformers` | Loading base Qwen3-1.7B model & tokenizer |
| `peft` | Loading the fine-tuned `qwen3-1.7b-amr-lora` adapter |
| `accelerate` | Automatic device placement (`device_map="auto"`) |
| `safetensors` | Fast, safe loading of `adapter_model.safetensors` |
| `bitsandbytes` | Optional 4-bit/8-bit quantization for lower VRAM usage |
| `fastapi` & `uvicorn` *(or `flask`)* | Local HTTP API serving `main.py` and `index.html` |
| `rapidfuzz` *(or `thefuzz`)* | Fuzzy string matching in `fuzzyMatch.py` |

### 2. Computer Vision, VSLAM & Calibration
| Package | Purpose |
| :--- | :--- |
| `ultralytics` | YOLOv8 inference (`yolov8n.pt`, `best.pt`) |
| `opencv-python` | Camera capture, calibration (`calibri.py`), and VSLAM feature tracking |
| `numpy` | Map/target arrays (`.npy`), camera intrinsics (`.npz`), and geometry |
| `scipy` | Spatial transformations and filtering |

### 3. Dashboard, Telemetry & Client Communication
| Package | Purpose |
| :--- | :--- |
| `pandas` | Reading/logging IMU CSV telemetry (`imu_data_*.csv`) |
| `matplotlib` / `streamlit` / `plotly` | Live map, pose, and telemetry visualization (`amr_dashboard.py`) |
| `requests` / `websockets` / `paho-mqtt` | Host-to-robot network communication (`laptop_client.py`) |

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Palash-Modi/NLP-AMR-Btech-Final-Year-Project.git
cd NLP-AMR-Btech-Final-Year-Project
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```

### 3. Install PyTorch (Choose GPU or CPU)
* **NVIDIA GPU (CUDA 12.x):**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  ```
* **CPU Only:**
  ```bash
  pip install torch torchvision
  ```

### 4. Install Project Dependencies
```bash
# Install NLP module requirements
pip install -r AMR_NLP_Integration/requirements.txt

# Install Vision, VSLAM, and Dashboard dependencies
pip install ultralytics opencv-python numpy scipy pandas matplotlib requests
```

---

## How to Run

> **Tip:** Check `Start_Commands.txt` in the root folder for any project-specific flags, IP addresses, or port numbers used during live hardware runs.

### Step 1: Camera Calibration *(First-Time Setup Only)*
Skip this step if `cam_intrinsics.npz` is already calibrated for your current camera.
```bash
python calibri.py
```
*Capture checkerboard frames as prompted. This generates/updates `cam_intrinsics.npz`.*

### Step 2: Map the Environment *(First-Time Setup or New Arena)*
Skip this step if you want to reuse the existing `amr_saved_map.npy` and `amr_targets.npy`.
```bash
python map_first.py
```
*Drive the robot through the environment to build the map and register named target locations.*

### Step 3: Launch the Full Autonomous Stack
Open **4 separate terminals** (with the virtual environment activated in each) and run the modules in order:

**Terminal 1 — NLP Command Server:**
```bash
cd AMR_NLP_Integration
python main.py
```
*Once running, open `AMR_NLP_Integration/index.html` in your browser (or visit the local server URL printed in the terminal, e.g., `http://127.0.0.1:8000`) to enter natural language commands.*

**Terminal 2 — Visual SLAM & Perception:**
```bash
python vslam_map.py
```
*Loads `cam_intrinsics.npz`, `amr_saved_map.npy`, and YOLOv8 weights (`best.pt` / `yolov8n.pt`) for real-time localization and obstacle detection.*

**Terminal 3 — Robot Control Client:**
```bash
python laptop_client.py
```
*Monitors `current_goal.json` and `amr_targets.npy`, connects to the robot over the local network, and streams navigation commands while logging IMU telemetry.*

**Terminal 4 — Operational Dashboard:**
```bash
python amr_dashboard.py
```
*(If `amr_dashboard.py` is built with Streamlit, launch it using `streamlit run amr_dashboard.py` instead.)*

---

## Troubleshooting

* **`CUDA out of memory` when running Qwen3-1.7B + YOLOv8 together:** Close unused GPU apps or enable 4-bit quantization (`bitsandbytes`) in `AMR_NLP_Integration/main.py`.
* **`FileNotFoundError` for `cam_intrinsics.npz` or `amr_saved_map.npy`:** Ensure you are running the root scripts from the repository root directory, not from inside `AMR_NLP_Integration/`.
* **Robot not responding to `laptop_client.py`:** Verify that your laptop and the robot controller are on the same Wi-Fi/LAN subnet and that the target IP/port in `laptop_client.py` matches the onboard controller.
* **Large model files not loading after clone:** Run `git lfs pull` to ensure `.safetensors` and `.pt` files were downloaded completely rather than as Git LFS pointers.

---

## Contributors

* **Palash Modi**
* **Aditya Roy**
* **Sanvi Pathare**

*B.Tech in Robotics and Automation — Symbiosis Institute of Technology, Pune*