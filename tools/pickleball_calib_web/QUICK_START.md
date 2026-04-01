# 🎾 Pickleball Calibration Tool - Quick Start

## ⚡ 2-Minute Setup

### First Time (installation):
```bash
cd /home/ubuntu/test_work/judex-web/tools/pickleball_calib_web
bash install.sh
```

### Every Time After (run):
```bash
cd /home/ubuntu/test_work/judex-web/tools/pickleball_calib_web
bash run.sh
```

Then open: **http://localhost:5000**

---

## 📐 How to Calibrate

### **Intrinsic Calibration (Tab 1)**
1. Select camera (source or sink)
2. Click **🚀 Run Intrinsic Calibration**
3. Chessboard detection happens automatically
4. Results saved to `calibration_1512/{camera}/camera_object.yaml`

### **Extrinsic Calibration (Tab 2)**
1. Select camera (source or sink)
2. Select a world point from dropdown
3. Click the corresponding point in the image
4. Repeat for 4+ points
5. Click **🔧 Compute Pose**
6. Results saved to `calibration_1512/{camera}/extrinsic_pose.yaml`

### **View Results (Tab 3)**
1. Select camera
2. Click **Load Results**
3. See intrinsic and extrinsic parameters

---

## 📁 File Locations

| Item | Path |
|------|------|
| Web Tool | `/tools/pickleball_calib_web/` |
| Calibration Results | `/tools/pickleball_calib/calibration_1512/` |
| Chessboard Images | `/tools/pickleball_calib/3.6mm_checkerboard/` |
| World Points | `/Camera-Testing-Web-App-new_calibration/backend/calibration/world_coordinates/worldpickleball.txt` |

---

## 🔧 Troubleshooting

**Port 5000 already in use?**
- Edit `app.py` line at bottom: change `port=5000` to `port=5001`

**World points not loading?**
- Check: `/Camera-Testing-Web-App-new_calibration/backend/calibration/world_coordinates/worldpickleball.txt` exists

**Chessboard images not found?**
- Ensure images in: `/tools/pickleball_calib/3.6mm_checkerboard/`

**Flask errors?**
- Check conda environment: `conda activate pickleball_web`
- Then: `bash run.sh`

---

## 💾 Output Format

All results saved as **YAML + JSON**:

```yaml
# camera_object.yaml
camera_matrix:
  - [fx, 0, cx]
  - [0, fy, cy]
  - [0, 0, 1]
dist_coeffs: [k1, k2, p1, p2, ...]
reprojection_error: 0.815
is_fisheye: false
```

```yaml
# extrinsic_pose.yaml
rvec: [...]  # Rotation vector
tvec: [...]  # Translation vector
rotation_matrix: [[...], [...], [...]]
reprojection_error: 0.234
```

---

## 🌐 Access URL

- **Local machine**: http://localhost:5000
- **Remote SSH**: http://<your-server-ip>:5000

---

## ❌ Stop the Server

Press `Ctrl+C` in the terminal running `bash run.sh`

---

Ready! 🚀
