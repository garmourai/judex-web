# Pairwise Correlation Module

This module provides functionality for performing pairwise shuttle matching between two camera views and generating 3D triangulation.

## Modular Structure

The code has been refactored into a modular structure for better maintainability, testability, and extensibility:

```
pairwise_correlation/
├── __init__.py                 # Main package exports
├── main.py                     # Main entry point
├── perform_correlation.py      # Backward compatibility wrapper
├── utils.py                    # Backward compatibility wrapper
├── core/                       # Core mathematical algorithms
│   ├── __init__.py
│   ├── triangulation.py        # DLT triangulation logic
│   ├── matching.py             # Shuttle matching algorithms
│   └── camera_utils.py         # Camera calibration and distortion functions
├── data/                       # Data handling and loading
│   ├── __init__.py
│   ├── frame_sync.py           # Frame synchronization logic
│   ├── coordinate_processor.py # Coordinate processing and CSV handling
│   └── data_loader.py          # Data loading and validation
├── visualization/              # Visualization and drawing
│   ├── __init__.py
│   ├── video_generator.py      # Video creation and annotation
│   ├── plotting.py             # Matplotlib plotting functions
│   └── drawing_utils.py        # OpenCV drawing functions
├── processing/                 # Main processing logic
│   ├── __init__.py
│   ├── correlation_engine.py   # Main correlation processing logic
│   ├── segment_processor.py    # Frame segment processing
│   └── cost_analyzer.py        # Cost matrix analysis and thresholding
├── output/                     # Output and file management
│   ├── __init__.py
│   ├── csv_writer.py           # CSV output handling
│   ├── file_manager.py         # File and directory management
│   └── metrics_writer.py       # Cost matrix and metrics writing
└── config/                     # Configuration and settings
    ├── __init__.py
    └── settings.py             # Configuration and constants
```

## Usage

### Basic Usage

```python
from pairwise_correlation import do_pairwise_correlation

# Run correlation
do_pairwise_correlation(
    camera_1_cam_path="path/to/camera1.pkl",
    camera_2_cam_path="path/to/camera2.pkl",
    camera_1_id="cam1",
    camera_2_id="cam2",
    camera_1_video_path="path/to/video1.mp4",
    camera_2_video_path="path/to/video2.mp4",
    output_dir="path/to/output",
    frame_segments=[[100, 200], [300, 400]],
    frame_sync_info_path="path/to/sync.txt",
    create_video=True
)
```

### Advanced Usage with Configuration

```python
from pairwise_correlation import CorrelationEngine, CorrelationConfig

# Customize configuration
config = CorrelationConfig()
config.DEFAULT_ALPHA = 0.5
config.DEFAULT_BETA = 1.5
config.MAX_SEGMENT_LENGTH = 500

# Create engine and run
engine = CorrelationEngine(config)
engine.do_pairwise_correlation(...)
```

## Key Benefits

### 1. **Separation of Concerns**
- **Core**: Mathematical algorithms (triangulation, matching)
- **Data**: Input/output data handling
- **Visualization**: All visual output generation
- **Processing**: Main business logic
- **Output**: File writing and results management
- **Config**: Centralized configuration

### 2. **Error Handling Strategy**
- Removed unnecessary try-catch blocks
- Let meaningful errors propagate up
- Added validation at data boundaries
- Use assertions for critical invariants

### 3. **Modularity Benefits**
- Each module has a single responsibility
- Easy to test individual components
- Clear dependencies between modules
- Easy to extend or modify specific functionality

### 4. **Backward Compatibility**
- Original `do_pairwise_correlation` function interface preserved
- All original functions available through `utils.py`
- No changes required to existing code that imports from this module

## Testing

Each module can be tested independently:

```python
# Test triangulation
from pairwise_correlation.core.triangulation import triangulate_dlt

# Test matching
from pairwise_correlation.core.matching import match_shuttles

# Test data loading
from pairwise_correlation.data.data_loader import DataLoader
```

## Configuration

All configuration parameters are centralized in `config/settings.py`:

- Matching parameters (alpha, beta)
- Frame processing parameters
- Visualization parameters
- File extensions and paths
- Plot parameters

## Error Handling

The modular structure implements proper error handling:

1. **Validation at data boundaries**: Check file existence, data format
2. **Assertions for critical invariants**: Validate array shapes, coordinate counts
3. **Let meaningful errors propagate**: Don't hide important errors with broad try-catch
4. **Clear error messages**: Descriptive error messages for debugging

## Migration from Original Code

The original code has been preserved in:
- `perform_correlation_original.py`
- `utils_original.py`

The new modular structure maintains the same external interface, so existing code should work without changes.
