Simple pickleball calibration tools

Two scripts:

- intrinsic.py - run intrinsic calibration. Supports automatic chessboard detection (default) and manual point marking.
- extrinsic.py - compute extrinsics (solvePnP) given an image, camera YAML/pickle and matched image->world points (clicking in a window).

Defaults match the original repo: chessboard internal corners (6,8) and square size 25 mm. Configure fisheye vs pinhole by editing `config.json`.

Usage examples

1) Intrinsic from chessboard images:

python intrinsic.py --images-dir /path/to/chessboard_images --out /path/to/output_dir

2) Extrinsic using image and world points (click to mark points):

python extrinsic.py --image /path/to/image.jpg --camera /path/to/camera_object.yaml --world /path/to/worldpickleball.txt

Both scripts use OpenCV windows for point picking when manual mode is required.
