#!/usr/bin/env python3
"""
Automated Extrinsic Calibration Tool for Pickleball
Uses predefined point correspondences (no manual marking needed).
"""

import cv2
import numpy as np
import yaml
import json
import os
import argparse
from datetime import datetime


def load_world_points(filepath):
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
                if len(parts) >= 3:
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        point_id += 1
                        name = f"p{point_id}"
                        world_points[name] = [x, y, z]
                    except ValueError:
                        if len(parts) >= 4:
                            name = parts[0]
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            world_points[name] = [x, y, z]
    except Exception as e:
        print(f"Error loading world coordinates: {e}")
    return world_points


def load_intrinsic(yaml_path):
    """Load intrinsic calibration from YAML."""
    with open(yaml_path, 'r') as f:
        calib = yaml.safe_load(f)
    
    camera_matrix = np.array(calib['camera_matrix'])
    dist_coeffs = np.array(calib['dist_coeffs']).flatten()
    
    return {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'is_fisheye': calib.get('is_fisheye', False),
        'reprojection_error': calib.get('reprojection_error', None)
    }


def auto_detect_points_on_court(image, world_points_dict):
    """
    Try to automatically detect court corners/landmarks on the image.
    This is a simplified version - you can enhance it with feature detection.
    
    Returns a mapping of world point names to image coordinates.
    """
    h, w = image.shape[:2]
    
    # For now, use approximate court corners based on image dimensions
    # These are rough estimates - you should refine based on your specific setup
    auto_correspondences = {
        'p1': (int(w*0.1), int(h*0.1)),      # Top-left corner
        'p3': (int(w*0.9), int(h*0.1)),      # Top-right corner
        'p7': (int(w*0.1), int(h*0.9)),      # Bottom-left corner
        'p9': (int(w*0.9), int(h*0.9)),      # Bottom-right corner
        'p2': (int(w*0.5), int(h*0.1)),      # Top-middle
        'p8': (int(w*0.5), int(h*0.9)),      # Bottom-middle
    }
    
    return auto_correspondences


def get_predefined_correspondences(camera_type='source'):
    """
    Get predefined point correspondences for each camera.
    These should be calibrated/measured points.
    Format: {world_point_name: (image_x, image_y)}
    """
    
    # These are example correspondences - UPDATE WITH YOUR ACTUAL MARKED POINTS!
    # You should mark these points manually first and record the coordinates here
    
    if camera_type == 'source':
        return {
            # Format: 'p{number}': (x, y)
            # Example - please replace with actual coordinates from marking tool
            'p1': (100, 150),
            'p2': (250, 140),
            'p3': (400, 160),
            'p4': (100, 400),
            'p5': (250, 380),
            'p6': (400, 390),
        }
    elif camera_type == 'sink':
        return {
            'p1': (120, 160),
            'p2': (270, 150),
            'p3': (420, 170),
            'p4': (120, 420),
            'p5': (270, 400),
            'p6': (420, 410),
        }
    
    return {}


def compute_extrinsic_from_correspondences(world_points_dict, image_correspondences, 
                                          camera_matrix, dist_coeffs, image):
    """
    Compute extrinsic calibration from world-to-image point correspondences.
    """
    
    if len(image_correspondences) < 4:
        print(f"❌ Need at least 4 points, have {len(image_correspondences)}")
        return None
    
    # Build world and image point arrays
    world_pts = []
    img_pts = []
    point_names = []
    
    for world_name, img_coord in image_correspondences.items():
        if world_name in world_points_dict:
            world_pts.append(world_points_dict[world_name])
            img_pts.append(img_coord)
            point_names.append(world_name)
    
    if len(world_pts) < 4:
        print(f"❌ Only {len(world_pts)} valid correspondences (need 4+)")
        return None
    
    world_points = np.array(world_pts, dtype=np.float32)
    image_points = np.array(img_pts, dtype=np.float32)
    
    print(f"\n{'='*60}")
    print(f"COMPUTING POSE ({len(world_pts)} points)")
    print(f"{'='*60}\n")
    
    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(
        world_points, image_points,
        camera_matrix, dist_coeffs,
        useExtrinsicGuess=False,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        print("❌ solvePnP failed")
        return None
    
    print(f"✓ Initial solve successful")
    
    # Refine with VVS
    rvec, tvec = cv2.solvePnPRefineVVS(
        world_points, image_points,
        camera_matrix, dist_coeffs,
        rvec, tvec,
        useExtrinsicGuess=True,
        criteria=cv2.TermCriteria(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.01)
    )
    
    print(f"✓ Refinement complete")
    
    # Rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    
    # Reprojection error
    reprojected, _ = cv2.projectPoints(world_points, rvec, tvec, 
                                      camera_matrix, dist_coeffs)
    reprojection_error = np.sqrt(np.mean((image_points - reprojected.reshape(-1, 2))**2))
    
    print(f"\n✓ Rotation Vector:\n{rvec.flatten()}")
    print(f"\n✓ Translation Vector:\n{tvec.flatten()}")
    print(f"\n✓ Reprojection Error: {reprojection_error:.4f} pixels")
    print(f"✓ Used {len(world_pts)} point correspondences: {point_names}\n")
    
    # Visualize reprojection
    print("📊 Visualizing reprojection...")
    img_reproj = image.copy()
    
    for i, point in enumerate(image_points):
        # Marked point (blue)
        cv2.circle(img_reproj, tuple(map(int, point)), 8, (255, 0, 0), 2)
        
        # Reprojected point (green)
        reproj_x, reproj_y = int(reprojected[i][0][0]), int(reprojected[i][0][1])
        cv2.circle(img_reproj, (reproj_x, reproj_y), 8, (0, 255, 0), 2)
        
        # Connection line (red)
        cv2.line(img_reproj, tuple(map(int, point)), (reproj_x, reproj_y), (0, 0, 255), 2)
        
        # Label
        cv2.putText(img_reproj, point_names[i], 
                   (int(point[0])+10, int(point[1])+10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Add legend
    cv2.circle(img_reproj, (30, 30), 5, (255, 0, 0), 2)
    cv2.putText(img_reproj, "Marked", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    cv2.circle(img_reproj, (30, 60), 5, (0, 255, 0), 2)
    cv2.putText(img_reproj, "Reprojected", (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imshow('Extrinsic Calibration - Reprojection Verification', img_reproj)
    print("Press any key to close visualization...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return {
        'rvec': rvec.flatten().tolist(),
        'tvec': tvec.flatten().tolist(),
        'rotation_matrix': rotation_matrix.tolist(),
        'reprojection_error': float(reprojection_error),
        'num_points': len(world_pts),
        'point_names': point_names,
        'timestamp': datetime.now().isoformat(),
        'camera_matrix': camera_matrix.tolist(),
        'is_fisheye': False
    }


def save_result(result, output_dir):
    """Save extrinsic calibration result."""
    os.makedirs(output_dir, exist_ok=True)
    
    # YAML
    yaml_path = os.path.join(output_dir, 'extrinsic_pose.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(result, f, default_flow_style=False)
    print(f"✓ Saved: {yaml_path}")
    
    # JSON
    json_path = os.path.join(output_dir, 'extrinsic_pose.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"✓ Saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Automated extrinsic calibration')
    parser.add_argument('--image', required=True, help='Image file path')
    parser.add_argument('--intrinsic', required=True, help='Intrinsic YAML file')
    parser.add_argument('--world-coords', required=True, help='World coordinates file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--camera', default='source', help='Camera type (source/sink)')
    parser.add_argument('--points', help='JSON file with point correspondences')
    
    args = parser.parse_args()
    
    try:
        print("="*60)
        print("AUTOMATED EXTRINSIC CALIBRATION")
        print("="*60)
        print()
        
        # Load data
        print("Loading data...")
        image = cv2.imread(args.image)
        if image is None:
            raise FileNotFoundError(f"Image not found: {args.image}")
        print(f"✓ Image: {args.image} ({image.shape[1]}x{image.shape[0]})")
        
        intrinsic_data = load_intrinsic(args.intrinsic)
        print(f"✓ Intrinsics loaded (error: {intrinsic_data['reprojection_error']:.4f}px)")
        
        world_points_dict = load_world_points(args.world_coords)
        print(f"✓ World points: {len(world_points_dict)}")
        
        # Get point correspondences
        if args.points:
            # Load from JSON file if provided
            with open(args.points, 'r') as f:
                correspondences = json.load(f)
            print(f"✓ Loaded {len(correspondences)} correspondences from file")
        else:
            # Use predefined correspondences
            correspondences = get_predefined_correspondences(args.camera)
            if not correspondences:
                print("⚠ No correspondences defined. Using auto-detection...")
                correspondences = auto_detect_points_on_court(image, world_points_dict)
            print(f"✓ Using {len(correspondences)} predefined correspondences")
        
        print()
        
        # Compute extrinsic
        result = compute_extrinsic_from_correspondences(
            world_points_dict,
            correspondences,
            intrinsic_data['camera_matrix'],
            intrinsic_data['dist_coeffs'],
            image
        )
        
        if result is None:
            print("❌ Extrinsic calibration failed")
            return
        
        # Save
        save_result(result, args.output)
        
        print(f"\n✅ Extrinsic calibration complete!")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
