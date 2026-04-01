#!/usr/bin/env python3
"""
Extrinsic Calibration Tool for Pickleball
Computes camera pose from world-to-image point correspondences.
"""

import cv2
import numpy as np
import yaml
import json
import os
import argparse
from datetime import datetime


class ExtrinsicCalibrator:
    """Interactive extrinsic calibration using solvePnP."""
    
    def __init__(self, image_path, intrinsic_yaml, world_coords_file):
        """Initialize calibrator."""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load intrinsic parameters
        with open(intrinsic_yaml, 'r') as f:
            calib_data = yaml.safe_load(f)
        
        self.camera_matrix = np.array(calib_data['camera_matrix'])
        self.dist_coeffs = np.array(calib_data['dist_coeffs']).flatten()
        self.is_fisheye = calib_data.get('is_fisheye', False)
        
        # Load world coordinates
        self.world_points_dict = self._load_world_points(world_coords_file)
        self.world_points_list = list(self.world_points_dict.items())
        
        # State for marking
        self.marked_points = []
        self.current_point_idx = 0
        self.img_display = self.image.copy()
        
        print(f"✓ Image loaded: {image_path}")
        print(f"✓ Intrinsics loaded: {calib_data.get('num_images', '?')} images used")
        print(f"✓ World points loaded: {len(self.world_points_dict)} points")
        print()
    
    def _load_world_points(self, filepath):
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
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for marking points."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_point_idx < len(self.world_points_list):
                point_name = self.world_points_list[self.current_point_idx][0]
                world_coord = self.world_points_list[self.current_point_idx][1]
                
                self.marked_points.append({
                    'world_name': point_name,
                    'world_coord': world_coord,
                    'image_x': x,
                    'image_y': y
                })
                
                print(f"✓ Marked: {point_name} {world_coord} → ({x}, {y})")
                
                self.current_point_idx += 1
                self._redraw_image()
    
    def _redraw_image(self):
        """Redraw image with marked points."""
        self.img_display = self.image.copy()
        
        # Draw marked points
        for i, point in enumerate(self.marked_points):
            cv2.circle(self.img_display, (point['image_x'], point['image_y']), 8, (0, 255, 0), -1)
            cv2.circle(self.img_display, (point['image_x'], point['image_y']), 8, (255, 255, 255), 2)
            cv2.putText(self.img_display, str(i+1), 
                       (point['image_x']-3, point['image_y']+3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show status
        h = self.img_display.shape[0]
        status = f"Marked: {len(self.marked_points)}/{len(self.world_points_list)} points"
        if self.current_point_idx < len(self.world_points_list):
            next_point = self.world_points_list[self.current_point_idx][0]
            status += f" | Next: {next_point}"
        
        cv2.putText(self.img_display, status, (10, h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Extrinsic Calibration - Click to mark points', self.img_display)
    
    def interactive_marking(self):
        """Interactive point marking loop."""
        print("="*60)
        print("EXTRINSIC CALIBRATION - INTERACTIVE MARKING")
        print("="*60)
        print()
        print("Instructions:")
        print("  • Left Click: Mark point on image")
        print("  • 'u': Undo last point")
        print("  • 'c': Compute pose (need 4+ points)")
        print("  • 'r': Reset all points")
        print("  • 'q': Quit without saving")
        print()
        
        cv2.namedWindow('Extrinsic Calibration - Click to mark points')
        cv2.setMouseCallback('Extrinsic Calibration - Click to mark points', self._mouse_callback)
        
        self._redraw_image()
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                print("\n✗ Exiting without saving")
                cv2.destroyAllWindows()
                return False
            
            elif key == ord('u'):  # Undo
                if self.marked_points:
                    removed = self.marked_points.pop()
                    self.current_point_idx -= 1
                    print(f"↶ Undid: {removed['world_name']}")
                    self._redraw_image()
            
            elif key == ord('r'):  # Reset
                self.marked_points = []
                self.current_point_idx = 0
                print("🔄 Reset all points")
                self._redraw_image()
            
            elif key == ord('c'):  # Compute
                if len(self.marked_points) >= 4:
                    cv2.destroyAllWindows()
                    return True
                else:
                    print(f"⚠ Need 4+ points, have {len(self.marked_points)}")
    
    def compute_pose(self):
        """Compute camera pose using solvePnP."""
        if len(self.marked_points) < 4:
            print(f"❌ Need 4+ points, have {len(self.marked_points)}")
            return None
        
        # Prepare arrays
        world_points = np.array([p['world_coord'] for p in self.marked_points], dtype=np.float32)
        image_points = np.array([[p['image_x'], p['image_y']] for p in self.marked_points], dtype=np.float32)
        
        print(f"\n{'='*60}")
        print(f"COMPUTING POSE ({len(self.marked_points)} points)")
        print(f"{'='*60}\n")
        
        # First solve
        success, rvec, tvec = cv2.solvePnP(
            world_points, image_points,
            self.camera_matrix, self.dist_coeffs,
            useExtrinsicGuess=False,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            print("❌ solvePnP failed")
            return None
        
        # Refine with VVS
        rvec, tvec = cv2.solvePnPRefineVVS(
            world_points, image_points,
            self.camera_matrix, self.dist_coeffs,
            rvec, tvec,
            useExtrinsicGuess=True,
            criteria=cv2.TermCriteria(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.01)
        )
        
        # Convert rotation vector to matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        # Calculate reprojection error
        reprojected, _ = cv2.projectPoints(world_points, rvec, tvec, 
                                          self.camera_matrix, self.dist_coeffs)
        reprojection_error = np.sqrt(np.mean((image_points - reprojected.reshape(-1, 2))**2))
        
        print(f"✓ Rotation Vector: {rvec.flatten()}")
        print(f"✓ Translation Vector: {tvec.flatten()}")
        print(f"✓ Reprojection Error: {reprojection_error:.4f} pixels")
        print()
        
        # Visualize reprojection
        img_reproj = self.image.copy()
        for i, point in enumerate(self.marked_points):
            # Draw marked point (blue)
            cv2.circle(img_reproj, (point['image_x'], point['image_y']), 6, (255, 0, 0), -1)
            
            # Draw reprojected point (green)
            reproj_x, reproj_y = int(reprojected[i][0][0]), int(reprojected[i][0][1])
            cv2.circle(img_reproj, (reproj_x, reproj_y), 6, (0, 255, 0), -1)
            
            # Draw line between them
            cv2.line(img_reproj, (point['image_x'], point['image_y']), 
                    (reproj_x, reproj_y), (0, 0, 255), 2)
        
        cv2.imshow('Reprojection Visualization (Blue=Marked, Green=Reprojected)', img_reproj)
        print("📊 Showing reprojection visualization (press any key to close)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return {
            'rvec': rvec.flatten().tolist(),
            'tvec': tvec.flatten().tolist(),
            'rotation_matrix': rotation_matrix.tolist(),
            'reprojection_error': float(reprojection_error),
            'num_points': len(self.marked_points),
            'point_names': [p['world_name'] for p in self.marked_points],
            'timestamp': datetime.now().isoformat(),
            'camera_matrix': self.camera_matrix.tolist(),
            'is_fisheye': self.is_fisheye
        }
    
    def save_result(self, result, output_dir):
        """Save extrinsic calibration result."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save YAML
        yaml_path = os.path.join(output_dir, 'extrinsic_pose.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(result, f, default_flow_style=False)
        print(f"✓ Saved: {yaml_path}")
        
        # Save JSON
        json_path = os.path.join(output_dir, 'extrinsic_pose.json')
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"✓ Saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Extrinsic calibration using point marking')
    parser.add_argument('--image', required=True, help='Image file path')
    parser.add_argument('--intrinsic', required=True, help='Intrinsic YAML file path')
    parser.add_argument('--world-coords', required=True, help='World coordinates file path')
    parser.add_argument('--output', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    try:
        # Initialize
        calib = ExtrinsicCalibrator(args.image, args.intrinsic, args.world_coords)
        
        # Interactive marking
        if not calib.interactive_marking():
            return
        
        # Compute pose
        result = calib.compute_pose()
        if result is None:
            return
        
        # Save
        calib.save_result(result, args.output)
        
        print(f"\n✅ Extrinsic calibration complete!")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
