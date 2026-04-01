# ✅ Installation Complete!

The Pickleball Camera Calibration Web Tool has been successfully created and installed!

## 📁 Files Created

```
/home/ubuntu/test_work/judex-web/tools/pickleball_calib_web/
├── app.py                 # Flask backend with API endpoints
├── requirements.txt       # Python dependencies
├── install.sh            # Installation script (uses conda)
├── run.sh                # Run script
├── README.md             # Documentation
└── static/
    └── index.html        # Web interface
```

## 🚀 Quick Start

### Step 1: Install (first time only)
```bash
cd /home/ubuntu/test_work/judex-web/tools/pickleball_calib_web
bash install.sh
```

This will:
- Create a new conda environment called `pickleball_web` (Python 3.10)
- Install Flask, OpenCV, NumPy, PyYAML, and other dependencies
- Load 16 world coordinate points from worldpickleball.txt

### Step 2: Run the Server
```bash
cd /home/ubuntu/test_work/judex-web/tools/pickleball_calib_web
bash run.sh
```

The web tool will start on: **http://localhost:5000**

## 📊 What's Included

### 1. **Intrinsic Calibration** (📐 Tab)
- Upload chessboard images
- Configure chessboard size (default: 6×8 corners, 25mm squares)
- Automatically computes camera matrix and distortion coefficients
- Saves results to: `calibration_1512/{source|sink}/camera_object.yaml`

### 2. **Extrinsic Calibration** (🎯 Tab)
- Interactive image with point marking
- Select world points from 16 pre-loaded pickleball court landmarks
- Click corresponding image points
- Compute camera pose (rotation + translation)
- Saves results to: `calibration_1512/{source|sink}/extrinsic_pose.yaml`

### 3. **Results Viewer** (📊 Tab)
- View all calibration results
- Export as YAML or JSON
- Download intrinsic and extrinsic parameters

## 🔧 Configuration

Everything is pre-configured for pickleball calibration:

| Setting | Value |
|---------|-------|
| Camera Folders | `source`, `sink` |
| Calibration Base | `/tools/pickleball_calib/calibration_1512/` |
| World Coordinates | 16 pickleball court landmarks |
| Chessboard Path | `/tools/pickleball_calib/3.6mm_checkerboard/` |
| Chessboard Size | 6×8 internal corners, 25mm squares |

## 📦 Dependencies Installed

- **Flask 3.0.0** - Web framework
- **Flask-CORS 4.0.0** - Cross-origin requests
- **OpenCV 4.8.1** - Computer vision (calibration, image processing)
- **NumPy 1.26.4** - Numerical computing
- **PyYAML 6.0.1** - Configuration file format
- **Werkzeug 3.0.1** - WSGI utilities

## 🎯 Environment

- **Conda Environment**: `pickleball_web`
- **Python Version**: 3.10
- **Location**: `/home/ubuntu/anaconda3/envs/pickleball_web/`

## ✨ Verification

All components verified working:
- ✅ Conda environment created
- ✅ All dependencies installed
- ✅ Flask server starts correctly
- ✅ 16 world points loaded from worldpickleball.txt
- ✅ Configuration paths accessible
- ✅ Calibration folders ready

## 🌐 Accessing the Tool

Once running, open in your browser:
- **Local**: http://localhost:5000
- **Remote**: http://<your-ip>:5000

## 🛑 Stopping the Server

Press `Ctrl+C` in the terminal where `bash run.sh` is running

## 📝 Tips

1. **First time**: Run `bash install.sh` once
2. **Every time after**: Just run `bash run.sh`
3. **Check status**: The startup message shows if world points loaded
4. **Multiple cameras**: Switch between source/sink in each tab
5. **Results saved automatically** to YAML and JSON formats

## 🔗 Related Files

- Previous calibrations: `/tools/pickleball_calib/calibration_1512/`
- Chessboard images: `/tools/pickleball_calib/3.6mm_checkerboard/`
- Frame extraction: `/tools/pickleball_calib/frame_extractor.py`
- Intrinsic CLI tool: `/tools/pickleball_calib/intrinsic.py`
- Extrinsic CLI tool: `/tools/pickleball_calib/extrinsic.py`

---

**Ready to start calibrating! 🎾**
