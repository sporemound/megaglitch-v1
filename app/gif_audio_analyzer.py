import numpy as np
from PIL import Image, ImageStat, ImageSequence
import cv2
import logging

def extract_brightness_from_gif(gif_path, num_samples, framerate):
    try:
        gif = Image.open(gif_path)
        frame_brightness = []
        frame_durations = []

        for frame in ImageSequence.Iterator(gif):
            brightness = ImageStat.Stat(frame.convert("L")).mean[0] / 255.0
            duration = frame.info.get('duration', 100)  # Duration in milliseconds
            frame_brightness.append(brightness)
            frame_durations.append(duration / 1000.0)  # Convert to seconds

        total_duration = sum(frame_durations)
        num_total_samples = int(total_duration * framerate)

        cumulative_durations = np.cumsum([0] + frame_durations)
        brightness_values = np.interp(
            np.linspace(0, total_duration, num_total_samples),
            cumulative_durations[:-1],
            frame_brightness
        )

        if num_total_samples > num_samples:
            brightness_values = brightness_values[:num_samples]
        else:
            brightness_values = np.resize(brightness_values, num_samples)

        return np.clip(brightness_values, 0, 1)
    except Exception as e:
        logging.error(f"Error extracting brightness from GIF: {e}")
        return np.zeros(num_samples)

def extract_color_data_from_gif(gif_path, num_samples):
    try:
        gif = Image.open(gif_path)
        color_data = []
        frame_durations = []

        for frame in ImageSequence.Iterator(gif):
            rgb_mean = ImageStat.Stat(frame.convert("RGB")).mean[:3]
            normalized_rgb = [val / 255 for val in rgb_mean]  # Normalize RGB to [0, 1]
            duration = frame.info.get('duration', 100)  # Duration in milliseconds
            color_data.append(normalized_rgb)
            frame_durations.append(duration / 1000.0)  # Convert to seconds

        total_duration = sum(frame_durations)
        num_total_samples = num_samples # We want to interpolate to the target num_samples

        cumulative_durations = np.cumsum([0] + frame_durations)
        interpolated_data = np.zeros((num_total_samples, 3))
        for i in range(3):
            interpolated_data[:, i] = np.interp(
                np.linspace(0, total_duration, num_total_samples),
                cumulative_durations[:-1],
                [cd[i] for cd in color_data]
            )

        return np.clip(interpolated_data, 0, 1)
    except Exception as e:
        logging.error(f"Error extracting color data from GIF: {e}")
        return np.zeros((num_samples, 3))

def extract_motion_from_gif(gif_path, num_samples, framerate):
    try:
        gif = Image.open(gif_path)
        frames = []
        frame_durations = []

        for frame in ImageSequence.Iterator(gif):
            gray_frame = cv2.cvtColor(np.array(frame.convert('RGB')), cv2.COLOR_RGB2GRAY)
            frames.append(gray_frame)
            duration = frame.info.get('duration', 100)  # Duration in milliseconds
            frame_durations.append(duration / 1000.0)  # Convert to seconds

        motion_magnitudes = []
        for i in range(len(frames) - 1):
            if frames[i].shape != frames[i+1].shape:
                frames[i+1] = cv2.resize(frames[i+1], (frames[i].shape[1], frames[i].shape[0]))

            flow = cv2.calcOpticalFlowFarneback(
                frames[i],
                frames[i+1],
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_magnitude = np.mean(magnitude)
            motion_magnitudes.append(mean_magnitude)

        total_duration = sum(frame_durations[:-1]) if frame_durations else 0
        if not motion_magnitudes:
            return np.zeros(num_samples)

        cumulative_durations = np.cumsum([0] + frame_durations[:-1])
        motion_data = np.interp(
            np.linspace(0, total_duration, num_samples),
            cumulative_durations[:-1],
            motion_magnitudes
        )
        motion_data = motion_data / np.max(motion_data) if np.max(motion_data) != 0 else motion_data
        return motion_data
    except Exception as e:
        logging.error(f"Error extracting motion from GIF: {e}")
        return np.zeros(num_samples)
