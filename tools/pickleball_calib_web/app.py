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
import pickle
from types import SimpleNamespace

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Configuration for pickleball calibration
_TOOLS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
CONFIG = {
    'calib_base': '/home/ubuntu/test_work/judex-web/tools/pickleball_calib/calibration_1512',
    # Legacy path (optional). Prefer bundled file under tools/pickleball_calib/ when missing.
    'world_coords': os.path.join(
        _TOOLS_DIR,
        'Camera-Testing-Web-App-new_calibration',
        'backend',
        'calibration',
        'world_coordinates',
        'worldpickleball.txt',
    ),
    'chessboard_path': '/home/ubuntu/test_work/judex-web/tools/pickleball_calib/3.6mm_checkerboard',
    'world_coords_bundled': os.path.join(_TOOLS_DIR, 'pickleball_calib', 'worldpickleball.txt'),
}

# Global state
camera_intrinsics = {}
world_points_dict = {}
WORLD_COORDS_RESOLVED = None

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


def _world_coord_candidates():
    """Search order: env override, bundled repo file, CONFIG legacy path."""
    candidates = []
    env = os.environ.get('JUDEX_WORLD_COORDS', '').strip()
    if env:
        candidates.append(os.path.abspath(env))
    if CONFIG.get('world_coords_bundled'):
        candidates.append(os.path.abspath(CONFIG['world_coords_bundled']))
    if CONFIG.get('world_coords'):
        candidates.append(os.path.abspath(CONFIG['world_coords']))
    seen = set()
    out = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def resolve_world_points():
    """Load world landmark file from first existing candidate path."""
    for path in _world_coord_candidates():
        if os.path.isfile(path):
            d = load_world_coordinates(path)
            if len(d) > 0:
                return d, path
    return {}, None


def load_camera_intrinsics(yaml_path):
    """Load camera intrinsics from YAML."""
    try:
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading intrinsics: {e}")
        return None


# Load world coordinates on startup (bundled fallback if legacy path missing)
world_points_dict, WORLD_COORDS_RESOLVED = resolve_world_points()
print(f"Loaded {len(world_points_dict)} world points from {WORLD_COORDS_RESOLVED or '(none)'}")


def _load_reference_image(folder, undistort=False):
    """Load source image, optionally undistorted using saved intrinsics."""
    image_path = os.path.join(CONFIG['calib_base'], folder, f'{folder}.jpg')
    image = cv2.imread(image_path)
    if image is None:
        return None

    if undistort:
        yaml_path = os.path.join(CONFIG['calib_base'], folder, 'camera_object.yaml')
        calib = load_camera_intrinsics(yaml_path)
        if calib and calib.get('camera_matrix') is not None:
            camera_matrix = np.array(calib['camera_matrix'], dtype=np.float64)
            dist_coeffs = np.array(calib.get('dist_coeffs', []), dtype=np.float64).flatten()
            image = cv2.undistort(image, camera_matrix, dist_coeffs)
    return image


def _project_points_with_projection_matrix(projection_matrix, world_points):
    """Project Nx3 world points with 3x4 projection matrix."""
    world_h = np.hstack((world_points.astype(np.float64), np.ones((world_points.shape[0], 1))))
    projected_h = (projection_matrix @ world_h.T).T
    z = projected_h[:, 2:3]
    z[z == 0] = 1e-9
    return projected_h[:, :2] / z


def _build_homography_data(world_points, image_points):
    """
    Build planar homography using the dominant Z-plane from selected world points.
    Returns dict with matrix and diagnostics, or None if not enough planar points.
    """
    if len(world_points) < 4:
        return None

    rounded_z = np.round(world_points[:, 2], 3)
    unique_z, counts = np.unique(rounded_z, return_counts=True)
    dominant_z = unique_z[np.argmax(counts)]
    plane_mask = np.isclose(rounded_z, dominant_z)
    plane_world = world_points[plane_mask]
    plane_img = image_points[plane_mask]

    if len(plane_world) < 4:
        return None

    h_matrix, inlier_mask = cv2.findHomography(
        plane_world[:, :2].astype(np.float32),
        plane_img.astype(np.float32),
        cv2.RANSAC
    )
    if h_matrix is None:
        return None

    reproj = cv2.perspectiveTransform(
        plane_world[:, :2].astype(np.float32).reshape(-1, 1, 2),
        h_matrix
    ).reshape(-1, 2)
    homography_errors = np.linalg.norm(plane_img - reproj, axis=1)

    return {
        'homography_matrix': h_matrix,
        'dominant_z': float(dominant_z),
        'plane_world': plane_world,
        'plane_img': plane_img,
        'inlier_mask': [] if inlier_mask is None else inlier_mask.flatten().astype(int).tolist(),
        'mean_reproj_error': float(np.mean(homography_errors)),
        'max_reproj_error': float(np.max(homography_errors)),
    }


def _write_legacy_outputs(output_dir, folder, world_points, image_points, point_names,
                          camera_matrix, rvec, tvec, reprojection_error, image_mode):
    """Write legacy parity artifacts used by downstream calibration flows."""
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    extrinsic_matrix = np.hstack((rotation_matrix, tvec.reshape(3, 1)))
    projection_matrix = camera_matrix @ extrinsic_matrix

    # Build a correlation-compatible camera object pickle (attribute-based).
    intrinsic_yaml_path = os.path.join(output_dir, 'camera_object.yaml')
    intrinsic_data = {}
    if os.path.exists(intrinsic_yaml_path):
        loaded = load_camera_intrinsics(intrinsic_yaml_path)
        if isinstance(loaded, dict):
            intrinsic_data = loaded

    dist_coeffs = intrinsic_data.get('distortion_coefficients')
    if dist_coeffs is None:
        dist_coeffs = intrinsic_data.get('dist_coeffs', [])
    dist_coeffs = np.array(dist_coeffs, dtype=np.float64).flatten()
    if dist_coeffs.size == 0:
        dist_coeffs = np.zeros(5, dtype=np.float64)

    image_size = intrinsic_data.get('image_size', None)
    if image_size and len(image_size) == 2:
        width, height = int(image_size[0]), int(image_size[1])
    else:
        ref_img = _load_reference_image(folder, undistort=False)
        if ref_img is not None:
            height, width = ref_img.shape[:2]
        else:
            width, height = 1440, 1080

    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix.astype(np.float64),
        dist_coeffs,
        (width, height),
        0.2,
        (width, height)
    )
    camera_obj = SimpleNamespace(
        camera_matrix=camera_matrix.astype(np.float64),
        rotation_matrix=rotation_matrix.astype(np.float64),
        translation_vectors=tvec.reshape(3, 1).astype(np.float64),
        calibration_rotation_vectors=rvec.reshape(3, 1).astype(np.float64),
        calibration_translation_vectors=tvec.reshape(3, 1).astype(np.float64),
        projection_matrix=projection_matrix.astype(np.float64),
        distortion_coefficients=dist_coeffs.astype(np.float64),
        calibration_type=intrinsic_data.get('calibration_type', 'pinhole'),
        final_dimensions=[int(width), int(height)],
        new_scaled_camera_matrix=new_camera_matrix.astype(np.float64),
        image_size=[int(width), int(height)],
        scale_factor=1.0,
        threshold=float(intrinsic_data.get('reprojection_error', reprojection_error)),
        new_camera_matrix=new_camera_matrix.astype(np.float64),
        # Keep aliases for newer dict-like consumers.
        dist_coeffs=dist_coeffs.astype(np.float64),
        rvecs=[rvec.reshape(3, 1).astype(np.float64)],
        tvecs=[tvec.reshape(3, 1).astype(np.float64)],
        date=datetime.now().isoformat(),
    )
    camera_pickle_path = os.path.join(output_dir, 'camera_object.pkl')
    with open(camera_pickle_path, 'wb') as f:
        pickle.dump(camera_obj, f)
    generated_files.append(camera_pickle_path)

    # Core text artifacts
    world_path = os.path.join(output_dir, 'world_coordinates_mapped.txt')
    image_pts_path = os.path.join(output_dir, 'undistorted_img_pt.txt')
    projection_path = os.path.join(output_dir, 'projection_matrix.txt')
    reproj_errors_path = os.path.join(output_dir, 'reprojection_errors.txt')

    np.savetxt(world_path, world_points, fmt='%.8f')
    np.savetxt(image_pts_path, image_points, fmt='%.8f')
    np.savetxt(projection_path, projection_matrix, fmt='%.12f')

    projected_points = _project_points_with_projection_matrix(projection_matrix, world_points)
    point_errors = np.linalg.norm(image_points - projected_points, axis=1)
    with open(reproj_errors_path, 'w') as f:
        for idx, err in enumerate(point_errors, 1):
            f.write(f"point_{idx}: {float(err):.6f}\n")
        f.write(f"mean_error: {float(np.mean(point_errors)):.6f}\n")
        f.write(f"api_reprojection_error: {float(reprojection_error):.6f}\n")

    generated_files.extend([world_path, image_pts_path, projection_path, reproj_errors_path])

    # Homography + court metadata artifacts
    homography_data = _build_homography_data(world_points, image_points)
    if homography_data is not None:
        h_matrix = homography_data['homography_matrix']
        homography_payload = {
            'homography_matrix': h_matrix.tolist(),
            'dominant_plane_z': homography_data['dominant_z'],
            'num_planar_points': int(len(homography_data['plane_world'])),
            'inlier_mask': homography_data['inlier_mask'],
            'mean_reprojection_error': homography_data['mean_reproj_error'],
            'max_reprojection_error': homography_data['max_reproj_error'],
            'timestamp': datetime.now().isoformat()
        }
        homography_path = os.path.join(output_dir, 'homography_matrix.json')
        with open(homography_path, 'w') as f:
            json.dump(homography_payload, f, indent=2)
        generated_files.append(homography_path)

        court_payload = {
            'folder': folder,
            'image_mode': image_mode,
            'dominant_plane_z': homography_data['dominant_z'],
            'point_names': point_names,
            'world_points': world_points.tolist(),
            'image_points': image_points.tolist(),
            'homography_matrix': h_matrix.tolist(),
            'projection_matrix': projection_matrix.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        court_json_path = os.path.join(output_dir, 'court_info.json')
        court_yaml_path = os.path.join(output_dir, 'court_info.yaml')
        with open(court_json_path, 'w') as f:
            json.dump(court_payload, f, indent=2)
        with open(court_yaml_path, 'w') as f:
            yaml.dump(court_payload, f)
        generated_files.extend([court_json_path, court_yaml_path])

    # Visualization artifacts
    base_image = _load_reference_image(folder, undistort=True)
    if base_image is not None:
        # 1) projection_comparison.png
        comparison_image = base_image.copy()
        for i, (expected, projected) in enumerate(zip(image_points, projected_points)):
            ex = tuple(np.round(expected).astype(int))
            pr = tuple(np.round(projected).astype(int))
            cv2.circle(comparison_image, ex, 8, (0, 255, 0), -1)
            cv2.circle(comparison_image, pr, 8, (0, 0, 255), 2)
            cv2.line(comparison_image, ex, pr, (255, 255, 0), 2)
            cv2.putText(comparison_image, str(i + 1), (ex[0] + 8, ex[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        projection_img_path = os.path.join(output_dir, 'projection_comparison.png')
        cv2.imwrite(projection_img_path, comparison_image)
        generated_files.append(projection_img_path)

        # 2) homography_visualization.png
        homography_image = base_image.copy()
        if homography_data is not None:
            h_matrix = homography_data['homography_matrix']
            plane_world = homography_data['plane_world'][:, :2].astype(np.float32).reshape(-1, 1, 2)
            warped = cv2.perspectiveTransform(plane_world, h_matrix).reshape(-1, 2)
            for idx, pt in enumerate(warped):
                x, y = tuple(np.round(pt).astype(int))
                cv2.circle(homography_image, (x, y), 6, (255, 0, 0), -1)
                cv2.putText(homography_image, f"H{idx + 1}", (x + 6, y + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        else:
            cv2.putText(homography_image, "Homography unavailable (insufficient planar points)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        homography_img_path = os.path.join(output_dir, 'homography_visualization.png')
        cv2.imwrite(homography_img_path, homography_image)
        generated_files.append(homography_img_path)

    return generated_files

@app.route('/')
def index():
    """Serve the main page."""
    return send_from_directory('static', 'index.html')

@app.route('/api/config')
def get_config():
    """Get configuration and available calibration folders."""
    wp, path_used = resolve_world_points()
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
        'world_points': wp,
        'world_coords_path': path_used,
        'world_point_count': len(wp),
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
    canvas_width = float(data.get('canvas_width', 0) or 0)
    canvas_height = float(data.get('canvas_height', 0) or 0)
    image_mode = data.get('image_mode', 'raw')

    wp_dict, coords_path = resolve_world_points()
    if len(wp_dict) == 0:
        return jsonify({
            'error': 'No world coordinates loaded. Add worldpickleball.txt or set JUDEX_WORLD_COORDS.',
            'world_coords_path': coords_path,
            'tried_paths': _world_coord_candidates(),
        }), 400
    
    if len(marked_points) < 4:
        return jsonify({'error': 'Need at least 4 point pairs'}), 400
    
    # Load camera intrinsics
    yaml_path = os.path.join(CONFIG['calib_base'], folder, 'camera_object.yaml')
    calib = load_camera_intrinsics(yaml_path)
    camera_matrix = np.array(calib['camera_matrix'], dtype=np.float64)
    dist_coeffs = np.array(calib['dist_coeffs']).flatten()
    
    # Prepare point arrays
    world_points = []
    image_points = []
    point_names = []
    
    image_path = os.path.join(CONFIG['calib_base'], folder, f'{folder}.jpg')
    image = cv2.imread(image_path)
    if image is None:
        return jsonify({'error': 'Reference image not found'}), 404

    orig_height, orig_width = image.shape[:2]

    if canvas_width <= 0 or canvas_height <= 0:
        # Fallback for UI timing/layout edge cases where canvas reports 0 size.
        canvas_width = float(orig_width)
        canvas_height = float(orig_height)

    scale_x = orig_width / canvas_width
    scale_y = orig_height / canvas_height

    for point in marked_points:
        world_name = point['world_name']
        if world_name in wp_dict:
            world_points.append(wp_dict[world_name])
            image_points.append([
                float(point['image_x']) * scale_x,
                float(point['image_y']) * scale_y
            ])
            point_names.append(world_name)
    
    if len(world_points) < 4:
        unknown = [p.get('world_name') for p in marked_points if p.get('world_name') not in wp_dict]
        known_keys_sample = sorted(wp_dict.keys())[:24]
        return jsonify({
            'error': 'Need at least 4 valid world point correspondences',
            'detail': f'Only {len(world_points)} marks matched landmark names. Check world point names vs loaded file.',
            'world_coords_path': coords_path,
            'unknown_world_names': unknown,
            'available_landmarks': known_keys_sample,
            'landmark_count': len(wp_dict),
        }), 400
    
    world_points = np.array(world_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)

    # Points marked on an undistorted image should be solved with zero distortion.
    if image_mode == 'undistorted':
        dist_coeffs = np.zeros_like(dist_coeffs)
    
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

    generated_files = [
        yaml_path,
        json_path
    ] + _write_legacy_outputs(
        output_dir=output_dir,
        folder=folder,
        world_points=world_points.astype(np.float64),
        image_points=image_points.astype(np.float64),
        point_names=point_names,
        camera_matrix=camera_matrix.astype(np.float64),
        rvec=rvec.astype(np.float64),
        tvec=tvec.astype(np.float64),
        reprojection_error=float(reprojection_error),
        image_mode=image_mode
    )
    
    return jsonify({
        'success': True,
        'message': f'Extrinsic calibration complete for {folder}',
        'rvec': result['rvec'],
        'tvec': result['tvec'],
        'reprojection_error': float(reprojection_error),
        'num_points': len(world_points),
        'saved_to': output_dir,
        'generated_files': [os.path.relpath(path, CONFIG['calib_base']) for path in generated_files]
    })

@app.route('/api/image/<folder>')
def get_image(folder):
    """Get image for a folder, optionally undistorted."""
    img_path = os.path.join(CONFIG['calib_base'], folder, f'{folder}.jpg')
    if not os.path.exists(img_path):
        return jsonify({'error': 'Image not found'}), 404
    
    img = cv2.imread(img_path)
    if img is None:
        return jsonify({'error': 'Failed to load image'}), 500

    undistort = request.args.get('undistort', '0').lower() in ('1', 'true', 'yes')
    image_kind = 'raw'

    if undistort:
        yaml_path = os.path.join(CONFIG['calib_base'], folder, 'camera_object.yaml')
        calib = load_camera_intrinsics(yaml_path)
        if calib and calib.get('camera_matrix') is not None:
            camera_matrix = np.array(calib['camera_matrix'], dtype=np.float64)
            dist_coeffs = np.array(calib.get('dist_coeffs', []), dtype=np.float64).flatten()
            img = cv2.undistort(img, camera_matrix, dist_coeffs)
            image_kind = 'undistorted'
        else:
            image_kind = 'raw_no_intrinsics'

    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode()
    
    return jsonify({
        'image': f'data:image/jpeg;base64,{img_base64}',
        'image_kind': image_kind,
        'width': int(img.shape[1]),
        'height': int(img.shape[0])
    })

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
