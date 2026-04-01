from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import yaml
import json
import os
from datetime import datetime
import base64
from io import BytesIO

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Configuration for pickleball calibration
CONFIG = {
    'calib_base': '/home/ubuntu/test_work/judex-web/tools/pickleball_calib/calibration_1512',
    'world_coords': '/home/ubuntu/test_work/judex-web/Camera-Testing-Web-App-new_calibration/backend/calibration/world_coordinates/worldpickleball.txt',
    'chessboard_path': '/home/ubuntu/test_work/judex-web/tools/pickleball_calib/3.6mm_checkerboard',
}

# Global state
camera_intrinsics = {}
world_points_dict = {}

def load_world_coordinates(filepath):
    """Load world coordinates from file."""
    world_points = {}
    try:
        with open(filepath, 'r') as f:
            point_id = 0
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                # Handle both formats: with name or just coordinates
                if len(parts) >= 3:
                    try:
                        # Try to parse as x y z (numeric)
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        # If successful, auto-generate name
                        point_id += 1
                        name = f"p{point_id}"
                        world_points[name] = [x, y, z]
                    except ValueError:
                        # If first part is not a number, assume it's a name
                        if len(parts) >= 4:
                            name = parts[0]
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            world_points[name] = [x, y, z]
    except Exception as e:
        print(f"Error loading world coordinates: {e}")
    return world_points

def load_camera_intrinsics(yaml_path):
    """Load camera intrinsics from YAML."""
    try:
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading intrinsics: {e}")
        return None

# Load world coordinates on startup
world_points_dict = load_world_coordinates(CONFIG['world_coords'])
print(f"Loaded {len(world_points_dict)} world points")

@app.route('/')
def index():
    """Serve the main page."""
    return send_from_directory('static', 'index.html')

@app.route('/api/config')
def get_config():
    """Get configuration and available calibration folders."""
    folders = ['source', 'sink']
    configs = {}
    
    for folder in folders:
        yaml_path = os.path.join(CONFIG['calib_base'], folder, 'camera_object.yaml')
        if os.path.exists(yaml_path):
            calib = load_camera_intrinsics(yaml_path)
            configs[folder] = {
                'path': yaml_path,
                'camera_matrix': calib.get('camera_matrix'),
                'dist_coeffs': calib.get('dist_coeffs'),
                'image_size': calib.get('image_size'),
                'is_fisheye': calib.get('is_fisheye', False),
                'reprojection_error': calib.get('reprojection_error')
            }
    
    return jsonify({
        'calib_folders': configs,
        'world_points': world_points_dict,
        'chessboard_path': CONFIG['chessboard_path']
    })

@app.route('/api/intrinsic/chessboard-images')
def get_chessboard_images():
    """Get list of chessboard images."""
    chessboard_dir = CONFIG['chessboard_path']
    if not os.path.exists(chessboard_dir):
        return jsonify({'error': f'Chessboard directory not found: {chessboard_dir}'}), 404
    
    images = [f for f in os.listdir(chessboard_dir) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    return jsonify({'images': sorted(images)})

@app.route('/api/intrinsic/calibrate', methods=['POST'])
def intrinsic_calibrate():
    """Perform intrinsic calibration on chessboard images."""
    data = request.json
    folder = data.get('folder', 'source')
    chessboard_size = tuple(data.get('chessboard_size', [6, 8]))
    square_size = data.get('square_size', 25)
    
    chessboard_dir = CONFIG['chessboard_path']
    
    # Get chessboard images
    images = [f for f in os.listdir(chessboard_dir) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not images:
        return jsonify({'error': 'No chessboard images found'}), 400
    
    # Calibrate
    objpoints = []
    imgpoints = []
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[1], 0:chessboard_size[0]].T.reshape(-1, 2)
    objp *= square_size
    
    img_shape = None
    
    for img_file in images:
        img = cv2.imread(os.path.join(chessboard_dir, img_file))
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
            img_shape = gray.shape[::-1]
    
    if len(objpoints) < 3:
        return jsonify({'error': 'Not enough valid chessboard images'}), 400
    
    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    
    # Compute reprojection error
    mean_error = 0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                         camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
        total_points += len(imgpoints[i])
    mean_error /= len(objpoints)
    
    # Save calibration
    output_dir = os.path.join(CONFIG['calib_base'], folder)
    os.makedirs(output_dir, exist_ok=True)
    
    calib_result = {
        'calibration_type': 'pinhole',
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.flatten().tolist(),
        'image_size': list(img_shape),
        'is_fisheye': False,
        'reprojection_error': float(mean_error),
        'num_images': len(objpoints),
        'date': datetime.now().isoformat()
    }
    
    # Save YAML
    yaml_path = os.path.join(output_dir, 'camera_object.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(calib_result, f)
    
    # Save JSON
    json_path = os.path.join(output_dir, 'camera_object.json')
    with open(json_path, 'w') as f:
        json.dump(calib_result, f, indent=2)
    
    return jsonify({
        'success': True,
        'message': f'Calibration complete for {folder}',
        'reprojection_error': float(mean_error),
        'num_images': len(objpoints),
        'saved_to': output_dir
    })

@app.route('/api/extrinsic/mark-points', methods=['POST'])
def mark_extrinsic_points():
    """Process marked points for extrinsic calibration."""
    data = request.json
    folder = data.get('folder', 'source')
    marked_points = data.get('points', [])  # [{world_name, image_x, image_y}, ...]
    
    if len(marked_points) < 4:
        return jsonify({'error': 'Need at least 4 point pairs'}), 400
    
    # Load camera intrinsics
    yaml_path = os.path.join(CONFIG['calib_base'], folder, 'camera_object.yaml')
    calib = load_camera_intrinsics(yaml_path)
    camera_matrix = np.array(calib['camera_matrix'])
    dist_coeffs = np.array(calib['dist_coeffs']).flatten()
    
    # Prepare point arrays
    world_points = []
    image_points = []
    point_names = []
    
    for point in marked_points:
        world_name = point['world_name']
        if world_name in world_points_dict:
            world_points.append(world_points_dict[world_name])
            image_points.append([point['image_x'], point['image_y']])
            point_names.append(world_name)
    
    if len(world_points) < 4:
        return jsonify({'error': 'Need at least 4 valid world point correspondences'}), 400
    
    world_points = np.array(world_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)
    
    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(
        world_points, image_points, camera_matrix, dist_coeffs,
        useExtrinsicGuess=False, flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    # Note: Refinement with solvePnPRefineVVS has compatibility issues
    # The iterative solver above is already good quality for calibration purposes
    
    # Rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    
    # Reprojection error
    reprojected, _ = cv2.projectPoints(world_points, rvec, tvec, camera_matrix, dist_coeffs)
    reprojection_error = np.sqrt(np.mean(
        (image_points - reprojected.reshape(-1, 2))**2
    ))
    
    # Save result
    output_dir = os.path.join(CONFIG['calib_base'], folder)
    os.makedirs(output_dir, exist_ok=True)
    
    result = {
        'rvec': rvec.flatten().tolist(),
        'tvec': tvec.flatten().tolist(),
        'rotation_matrix': rotation_matrix.tolist(),
        'reprojection_error': float(reprojection_error),
        'num_points': len(world_points),
        'point_names': point_names,
        'timestamp': datetime.now().isoformat(),
        'folder': folder
    }
    
    # Save YAML
    yaml_path = os.path.join(output_dir, 'extrinsic_pose.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(result, f)
    
    # Save JSON
    json_path = os.path.join(output_dir, 'extrinsic_pose.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    return jsonify({
        'success': True,
        'message': f'Extrinsic calibration complete for {folder}',
        'rvec': result['rvec'],
        'tvec': result['tvec'],
        'reprojection_error': float(reprojection_error),
        'num_points': len(world_points),
        'saved_to': output_dir
    })

@app.route('/api/image/<folder>')
def get_image(folder):
    """Get image for a folder."""
    img_path = os.path.join(CONFIG['calib_base'], folder, f'{folder}.jpg')
    if not os.path.exists(img_path):
        return jsonify({'error': 'Image not found'}), 404
    
    img = cv2.imread(img_path)
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode()
    
    return jsonify({'image': f'data:image/jpeg;base64,{img_base64}'})

@app.route('/api/results/<folder>')
def get_results(folder):
    """Get calibration results for a folder."""
    output_dir = os.path.join(CONFIG['calib_base'], folder)
    
    results = {}
    
    # Intrinsic
    intrinsic_path = os.path.join(output_dir, 'camera_object.yaml')
    if os.path.exists(intrinsic_path):
        with open(intrinsic_path, 'r') as f:
            results['intrinsic'] = yaml.safe_load(f)
    
    # Extrinsic
    extrinsic_path = os.path.join(output_dir, 'extrinsic_pose.yaml')
    if os.path.exists(extrinsic_path):
        with open(extrinsic_path, 'r') as f:
            results['extrinsic'] = yaml.safe_load(f)
    
    return jsonify(results)

@app.route('/api/extrinsic/save-marked-points', methods=['POST'])
def save_marked_points():
    """Save marked points to config.json for verification"""
    data = request.get_json()
    folder = data.get('folder', 'source')
    points = data.get('points', [])
    canvas_width = data.get('canvas_width', 1200)
    canvas_height = data.get('canvas_height', 800)
    
    # Load config
    config_path = os.path.join(CONFIG['calib_base'], folder, 'config.json')
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Scale image points to original image coordinates
    # Get the image to determine original dimensions
    image_path = os.path.join(CONFIG['calib_base'], folder, f'{folder}.jpg')
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            orig_height, orig_width = img.shape[:2]
            
            # Scale factor from canvas to original image
            scale_x = orig_width / canvas_width
            scale_y = orig_height / canvas_height
            
            # Convert points and scale
            scaled_points = []
            for point in points:
                scaled_point = {
                    'world_name': point['world_name'],
                    'image_x': float(point['image_x']) * scale_x,
                    'image_y': float(point['image_y']) * scale_y,
                    'canvas_x': float(point['image_x']),
                    'canvas_y': float(point['image_y'])
                }
                scaled_points.append(scaled_point)
            
            # Add to config
            if 'extrinsic_marked_points' not in config:
                config['extrinsic_marked_points'] = []
            config['extrinsic_marked_points'] = scaled_points
            
            # Save config
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            return jsonify({
                'success': True,
                'message': f'Saved {len(scaled_points)} marked points to config.json',
                'scaled_points': scaled_points,
                'original_dimensions': {'width': orig_width, 'height': orig_height}
            })
    
    return jsonify({'success': False, 'error': 'Image not found'})

@app.route('/api/extrinsic/load-marked-points', methods=['GET'])
def load_marked_points():
    """Load marked points from config.json"""
    folder = request.args.get('folder', 'source')
    
    config_path = os.path.join(CONFIG['calib_base'], folder, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        points = config.get('extrinsic_marked_points', [])
        return jsonify({
            'success': True,
            'points': points,
            'count': len(points)
        })
    
    return jsonify({'success': False, 'points': [], 'count': 0})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  PICKLEBALL CALIBRATION WEB TOOL")
    print("="*60)
    print(f"Config base: {CONFIG['calib_base']}")
    print(f"World coords: {CONFIG['world_coords']}")
    print(f"Chessboard: {CONFIG['chessboard_path']}")
    print(f"World points loaded: {len(world_points_dict)}")
    print("\nStarting Flask server...")
    print("Open http://localhost:5000 in your browser")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
