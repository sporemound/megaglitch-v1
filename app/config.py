"""
Configuration for MegaGlitch Node0.6
====================================

Global configuration settings for the application.
"""

# =============================================================================
# OpenCV Backend Configuration
# =============================================================================

# Enable OpenCV backend (auto-detects if cv2 available)
OPENCV_ENABLED = True

# Prefer OpenCV over PIL for image I/O (faster)
OPENCV_PREFERRED_FOR_IMAGES = True

# Prefer OpenCV over imageio for video I/O (faster, better codec support)
OPENCV_PREFERRED_FOR_VIDEO = True

# Default video codec for exports (FourCC codes)
# Common options: "mp4v", "avc1", "H264", "XVID", "MJPG"
OPENCV_VIDEO_CODEC = "mp4v"

# Video export quality (0-100, higher = better quality but larger file)
VIDEO_EXPORT_QUALITY = 95

# Image export quality for JPEG (0-100)
IMAGE_EXPORT_QUALITY_JPEG = 95

# PNG compression level (0-9, higher = smaller file but slower)
IMAGE_EXPORT_QUALITY_PNG = 9

# =============================================================================
# Performance Settings
# =============================================================================

# Use CUDA if available (requires cv2.cuda)
USE_CUDA_IF_AVAILABLE = True

# Number of worker processes for parallel processing
NUM_WORKER_PROCESSES = 1

# Queue size for multiprocessing (0 = unlimited)
QUEUE_MAX_SIZE = 2

# =============================================================================
# Display Settings
# =============================================================================

# Default display resolution
DEFAULT_DISPLAY_WIDTH = 1280
DEFAULT_DISPLAY_HEIGHT = 720

# FPS cap for display (0 = unlimited)
DISPLAY_FPS_CAP = 60

# =============================================================================
# Logging Settings
# =============================================================================

# Log level: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
LOG_LEVEL = "INFO"

# Log file path (None = no file logging)
LOG_FILE = "glitch_app.log"

# Log file max size in bytes (10 MB default)
LOG_FILE_MAX_SIZE = 10 * 1024 * 1024

# Number of log file backups to keep
LOG_FILE_BACKUP_COUNT = 3

# =============================================================================
# Effect Settings
# =============================================================================

# Default effect intensity
DEFAULT_EFFECT_INTENSITY = 0.5

# Enable effect presets
ENABLE_EFFECT_PRESETS = True

# Preset directory
PRESET_DIRECTORY = "presets"

# Node preset directory
NODE_PRESET_DIRECTORY = "node_presets"

# =============================================================================
# Audio/MIDI Settings
# =============================================================================

# Enable audio reactive mode
ENABLE_AUDIO = True

# Enable MIDI control
ENABLE_MIDI = True

# Audio buffer size (samples)
AUDIO_BUFFER_SIZE = 1024

# Audio sample rate (Hz)
AUDIO_SAMPLE_RATE = 44100

# =============================================================================
# Advanced Settings
# =============================================================================

# Enable experimental features (may be unstable)
ENABLE_EXPERIMENTAL = False

# Enable GPU operations (requires compatible GPU + drivers)
ENABLE_GPU = True

# Enable quantum gestalt AI (experimental)
ENABLE_QUANTUM_GESTALT = False

# Enable profiling (performance measurement)
ENABLE_PROFILING = True

# Profiling report interval (frames)
PROFILING_REPORT_INTERVAL = 120


# =============================================================================
# Helper Functions
# =============================================================================

def get_opencv_settings():
    """Get OpenCV-related settings as dict."""
    return {
        "enabled": OPENCV_ENABLED,
        "prefer_images": OPENCV_PREFERRED_FOR_IMAGES,
        "prefer_video": OPENCV_PREFERRED_FOR_VIDEO,
        "video_codec": OPENCV_VIDEO_CODEC,
        "video_quality": VIDEO_EXPORT_QUALITY,
        "jpeg_quality": IMAGE_EXPORT_QUALITY_JPEG,
        "png_compression": IMAGE_EXPORT_QUALITY_PNG,
    }

def get_performance_settings():
    """Get performance-related settings as dict."""
    return {
        "use_cuda": USE_CUDA_IF_AVAILABLE,
        "num_workers": NUM_WORKER_PROCESSES,
        "queue_size": QUEUE_MAX_SIZE,
        "enable_gpu": ENABLE_GPU,
        "enable_profiling": ENABLE_PROFILING,
    }

def get_display_settings():
    """Get display-related settings as dict."""
    return {
        "width": DEFAULT_DISPLAY_WIDTH,
        "height": DEFAULT_DISPLAY_HEIGHT,
        "fps_cap": DISPLAY_FPS_CAP,
    }

def print_config():
    """Print current configuration (for debugging)."""
    print("=" * 70)
    print("MegaGlitch Configuration")
    print("=" * 70)
    
    print("\nOpenCV Backend:")
    for key, val in get_opencv_settings().items():
        print(f"  {key}: {val}")
    
    print("\nPerformance:")
    for key, val in get_performance_settings().items():
        print(f"  {key}: {val}")
    
    print("\nDisplay:")
    for key, val in get_display_settings().items():
        print(f"  {key}: {val}")
    
    print("=" * 70)


if __name__ == "__main__":
    # Print config when run directly
    print_config()





