# 🎾 Pickleball Camera Calibration Tool

A web-based tool for intrinsic and extrinsic camera calibration using chessboard images and point marking.

## Features

- **Intrinsic Calibration**: Calibrate camera matrix and distortion coefficients from chessboard images
- **Extrinsic Calibration**: Compute camera pose by marking corresponding points between image and world coordinates
- **Multi-Camera Support**: Calibrate both source and sink cameras
- **Interactive Interface**: Web-based UI for easy point marking and visualization
- **Results Export**: Save calibration results as YAML and JSON

## Quick Start

### 1. Installation

```bash
cd /home/ubuntu/test_work/judex-web/tools/pickleball_calib_web
bash install.sh
```

### 2. Run the Server

```bash
bash run.sh
```

The tool will start on `http://localhost:5000`

## Usage

### Intrinsic Calibration

1. Navigate to the **📐 Intrinsic Calibration** tab
2. Select a camera (Source or Sink)
3. Configure chessboard parameters:
   - Width: 6 corners (default)
   - Height: 8 corners (default)
   - Square Size: 25 mm (default)
4. Click **🚀 Run Intrinsic Calibration**
5. Results are automatically saved to the calibration folder

### Extrinsic Calibration

1. Navigate to the **🎯 Extrinsic Calibration** tab
2. Select a camera (Source or Sink)
3. Select a world point from the dropdown
4. Click on the corresponding point in the image
5. Repeat for at least 4 different points
6. Click **🔧 Compute Pose** to compute camera position and orientation
7. Results are automatically saved

### View Results

1. Navigate to the **📊 Results** tab
2. Select a camera
3. Click **Load Results** to view calibration data in YAML format

## Configuration

Configuration is handled automatically based on:
- Camera intrinsics: `/tools/pickleball_calib/calibration_1512/{source|sink}/camera_object.yaml`
- World coordinates: `/Camera-Testing-Web-App-new_calibration/backend/calibration/world_coordinates/worldpickleball.txt`
- Chessboard images: `/tools/pickleball_calib/3.6mm_checkerboard/`

## Output Files

Calibration results are saved in:
- `/tools/pickleball_calib/calibration_1512/{source|sink}/camera_object.yaml` (Intrinsic)
- `/tools/pickleball_calib/calibration_1512/{source|sink}/camera_object.json` (Intrinsic)
- `/tools/pickleball_calib/calibration_1512/{source|sink}/extrinsic_pose.yaml` (Extrinsic)
- `/tools/pickleball_calib/calibration_1512/{source|sink}/extrinsic_pose.json` (Extrinsic)

## API Reference

### GET /api/config
Returns configuration, world points, and calibration status

### POST /api/intrinsic/calibrate
Runs intrinsic calibration with specified parameters

**Request:**
```json
{
  "folder": "source",
  "chessboard_size": [6, 8],
  "square_size": 25
}
```

### POST /api/extrinsic/mark-points
Computes camera pose from marked point correspondences

**Request:**
```json
{
  "folder": "source",
  "points": [
    {"world_name": "p1", "image_x": 100, "image_y": 150},
    ...
  ]
}
```

### GET /api/image/<folder>
Returns base64-encoded image for a camera

### GET /api/results/<folder>
Returns saved calibration results (intrinsic and extrinsic)

## Requirements

- Python 3.7+
- Flask 3.0+
- OpenCV 4.8+
- NumPy 1.24+
- PyYAML 6.0+

See `requirements.txt` for full list.

## Troubleshooting

### Virtual environment not found
```bash
bash install.sh
```

### Port 5000 already in use
Edit `run.sh` or `app.py` to use a different port

### Chessboard images not found
Ensure chessboard images are in `/tools/pickleball_calib/3.6mm_checkerboard/`

### World coordinates not loading
Verify `worldpickleball.txt` exists at the configured path

## Development

To modify the tool:
1. Edit `app.py` for backend logic
2. Edit `static/index.html` for frontend UI
3. Run `bash run.sh` to test changes (Flask debug mode is enabled)

## License

Same as parent project
