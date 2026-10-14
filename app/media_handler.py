import cv2
import mss
import pygetwindow as gw
import imageio
import os
import re
from PIL import Image, ImageSequence
import logging
import dearpygui.dearpygui as dpg
import numpy as np

_display_texture_cache = {}


def _safe_text(value):
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


def _mojibake_candidates(path):
    text = str(path)
    candidates = [text]
    for source_encoding in ("cp1252", "latin1"):
        try:
            repaired = text.encode(source_encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if repaired and repaired not in candidates:
            candidates.append(repaired)
    return candidates


def resolve_existing_media_path(path):
    """Return an existing path, repairing common UTF-8/cp1252 mojibake."""
    for candidate in _mojibake_candidates(path):
        if os.path.exists(candidate):
            if candidate != str(path):
                logging.info(
                    "Repaired media path mojibake: %s -> %s",
                    _safe_text(path),
                    _safe_text(candidate),
                )
            return candidate
    for candidate in _mojibake_candidates(path):
        recovered = _recover_media_path_by_video_id(candidate)
        if recovered:
            logging.info(
                "Recovered media path by video id: %s -> %s",
                _safe_text(path),
                _safe_text(recovered),
            )
            return recovered
    return str(path)


def _recover_media_path_by_video_id(path):
    directory = os.path.dirname(str(path))
    if not directory or not os.path.isdir(directory):
        return ""
    match = re.search(r"\[([A-Za-z0-9_-]{6,})\]", os.path.basename(str(path)))
    if not match:
        return ""
    video_id = match.group(1)
    preferred_extensions = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v")
    try:
        matches = [
            os.path.join(directory, name)
            for name in os.listdir(directory)
            if video_id in name
        ]
    except OSError:
        return ""
    if not matches:
        return ""
    matches.sort(
        key=lambda item: (
            os.path.splitext(item)[1].lower() not in preferred_extensions,
            len(os.path.basename(item)),
        )
    )
    return matches[0]

# Import optional OpenCV backend for optimized I/O
try:
    import opencv_backend as ocv
    OPENCV_AVAILABLE = ocv.is_cv2_available()
except ImportError:
    OPENCV_AVAILABLE = False
    ocv = None

def open_image_file(path, prefer_opencv=True):
    """
    Load image file with optional OpenCV backend.
    
    Args:
        path: Image file path
        prefer_opencv: Use OpenCV if available (default True)
    
    Returns:
        PIL.Image.Image in RGBA mode or None on failure
    """
    path = resolve_existing_media_path(path)
    try:
        # Try OpenCV first if preferred and available
        if prefer_opencv and OPENCV_AVAILABLE:
            frame = ocv.read_image(path, as_format="rgba")
            if frame is not None:
                # Convert numpy array to PIL Image for compatibility
                img = Image.fromarray(frame)
                print(f"Loaded image (OpenCV): {_safe_text(path)}")
                return img
        
        # Fallback to PIL
        img = Image.open(path).convert("RGBA")
        print(f"Loaded image: {_safe_text(path)}")
        return img
    except Exception as e:
        print(f"Error loading image: {_safe_text(e)}")
        logging.error("Error loading image: %s", _safe_text(e))
        return None

def open_gif_file(path):
    path = resolve_existing_media_path(path)
    try:
        gif = Image.open(path)
        gif_frames = [frame.copy().convert('RGBA') for frame in ImageSequence.Iterator(gif)]
        gif_durations = [frame.info.get('duration', 100) for frame in ImageSequence.Iterator(gif)]
        print(f"Loaded GIF: {_safe_text(path)}")
        return gif_frames, gif_durations
    except Exception as e:
        print(f"Error loading GIF: {_safe_text(e)}")
        logging.error("Error loading GIF: %s", _safe_text(e))
        return [], []

def open_video_file(path, prefer_opencv=True):
    """
    Open video file with optional OpenCV backend.
    
    Args:
        path: Video file path
        prefer_opencv: Use OpenCV if available (default True)
    
    Returns:
        Video reader object or None on failure
    """
    path = resolve_existing_media_path(path)
    video_reader = None
    try:
        # Try OpenCV first if preferred and available
        if prefer_opencv and OPENCV_AVAILABLE:
            try:
                video_reader = ocv.VideoReader(path)
                print(f"Loaded video (OpenCV): {_safe_text(path)}")
                return video_reader
            except Exception as e:
                logging.warning("OpenCV video load failed, trying imageio: %s", _safe_text(e))
        
        # Fallback to imageio
        video_reader = imageio.get_reader(path)
        video_reader.set_image_index(0)
        print(f"Loaded video: {_safe_text(path)}")
        return video_reader
    except Exception as e:
        print(f"Error loading video: {_safe_text(e)}")
        logging.error("Error loading video: %s", _safe_text(e))
        if video_reader:
            try:
                video_reader.close()
            except:
                pass
        return None

def get_window_titles():
    try:
        windows = gw.getAllTitles()
        windows = [w for w in windows if w.strip()]
        if not windows:
            print("No windows found.")
        return windows
    except Exception as e:
        print(f"Error retrieving windows: {e}")
        logging.error(f"Error retrieving windows: {e}")
        return []

def capture_window_frame(selected_window, display_width, display_height, apply_glitch_effect_func, previous_frame, feedback_image, trail_image, frame_buffer, recording_queue=None):
    if not selected_window:
        print("No window selected for capture.")
        return None, previous_frame, feedback_image, trail_image
    wins = gw.getWindowsWithTitle(selected_window)
    if not wins:
        print(f"Window {selected_window} not found.")
        return None, previous_frame, feedback_image, trail_image
    w = wins[0]
    if w.isMinimized:
        w.restore()
        import time
        time.sleep(0.5)

    if w.width <= 0 or w.height <= 0:
        print(f"Window '{selected_window}' has invalid dimensions (width: {w.width}, height: {w.height}). Cannot capture.")
        return None, previous_frame, feedback_image, trail_image

    bbox = {"top": w.top, "left": w.left, "width": w.width, "height": w.height}
    with mss.mss() as sct:
        try:
            shot = sct.grab(bbox)
            frame_img = Image.frombytes('RGB', shot.size, shot.rgb).convert('RGBA')
            # Apply glitch effect here, passing necessary state/params
            # This function will need to be passed from the main app or have access to app state
            # For now, a placeholder call:
            glitched_img, new_prev_frame, new_feedback_image, new_trail_image = apply_glitch_effect_func(frame_img, previous_frame, feedback_image, trail_image, frame_buffer)
            if recording_queue:
                recording_queue.put(glitched_img)
            return glitched_img, new_prev_frame, new_feedback_image, new_trail_image
        except mss.exception.ScreenShotError as e:
            print(f"Failed to capture window '{selected_window}': {e}")
            logging.error(f"Failed to capture window '{selected_window}': {e}")
            return None, previous_frame, feedback_image, trail_image

def update_display_texture(img, texture_tag, display_width, display_height):
    """
    Update DearPyGui texture with image data.
    OPTIMIZED: Accepts both PIL.Image and numpy array.
    
    Args:
        img: PIL.Image OR numpy array (H,W,3) uint8
        texture_tag: DearPyGui texture tag
        display_width: Target display width
        display_height: Target display height
    """
    try:
        if isinstance(img, np.ndarray):
            arr = img
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[..., :3]
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.asarray(img.convert('RGB'), dtype=np.uint8)

        disp_w, disp_h = int(display_width), int(display_height)
        h, w = arr.shape[:2]
        if h <= 0 or w <= 0 or disp_w <= 0 or disp_h <= 0:
            return

        aspect = w / h
        target_aspect = disp_w / disp_h
        if aspect > target_aspect:
            new_w = disp_w
            new_h = max(1, int(new_w / aspect))
        else:
            new_h = disp_h
            new_w = max(1, int(new_h * aspect))

        if new_w != w or new_h != h:
            interpolation = cv2.INTER_AREA if (new_w < w or new_h < h) else cv2.INTER_LINEAR
            resized = cv2.resize(arr, (new_w, new_h), interpolation=interpolation)
        else:
            resized = arr

        cache_key = (texture_tag, disp_w, disp_h)
        cached = _display_texture_cache.get(cache_key)
        if cached is None:
            canvas_u8 = np.zeros((disp_h, disp_w, 4), dtype=np.uint8)
            canvas_f32 = np.empty((disp_h, disp_w, 4), dtype=np.float32)
            _display_texture_cache[cache_key] = (canvas_u8, canvas_f32)
        else:
            canvas_u8, canvas_f32 = cached

        canvas_u8.fill(0)
        canvas_u8[..., 3] = 255

        px = (disp_w - new_w) // 2
        py = (disp_h - new_h) // 2
        canvas_u8[py:py + new_h, px:px + new_w, :3] = resized

        np.multiply(canvas_u8, 1.0 / 255.0, out=canvas_f32, casting='unsafe')
        dpg.set_value(texture_tag, canvas_f32.ravel())
    except Exception as e:
        print(f"Error in update_display: {e}")
        logging.error(f"Error in update_display: {e}", exc_info=True)
