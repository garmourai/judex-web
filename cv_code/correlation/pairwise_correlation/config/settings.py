"""
Configuration settings for pairwise correlation.
"""

class CorrelationConfig:
    """Configuration class for pairwise correlation parameters."""
    
    # Default matching parameters
    DEFAULT_ALPHA = 0
    DEFAULT_BETA = 1.0
    
    # Frame processing parameters
    MAX_SEGMENT_LENGTH = 10000000
    COST_PERCENTILE_THRESHOLD = 30
    
    # Visualization parameters
    DEFAULT_COLORS = [
        (0, 0, 255),     # Red
        (0, 255, 0),     # Green
        (255, 0, 0),     # Blue
        (255, 255, 0),   # Cyan
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Yellow
        (255, 255, 255), # White
    ]
    
    DEFAULT_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    
    # Video parameters
    VIDEO_CODEC = "mp4v"
    FONT = "FONT_HERSHEY_SIMPLEX"
    FONT_SCALE = 1.2
    FONT_THICKNESS = 2
    FONT_COLOR = (0, 255, 255)  # Bright yellow
    
    # File extensions
    COST_MATRIX_FILE = "cost_matrix.txt"
    EPIPOLAR_VALUES_FILE = "epi_polar_values.txt"
    REPROJ_VALUES_FILE = "reproj_values.txt"
    TEMPORAL_VALUES_FILE = "temporal_values.txt"
    TRACKER_CSV_FILE = "tracker.csv"
    MATCH_DECISIONS_CSV_FILE = "match_decisions.csv"
    CORRELATION_VIDEO_FILE = "correlation_video.mp4"
    
    # Plot parameters
    HISTOGRAM_BINS = 30
    ALL_COSTS_BINS = 50
    PLOT_DPI = 300
    FIGURE_SIZE = (6, 5)
