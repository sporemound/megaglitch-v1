import threading
import queue
import ffmpeg
import numpy as np
import dearpygui.dearpygui as dpg
import subprocess
import os
import logging
import time

class RecordingManager:
    def __init__(self, display_width, display_height):
        self.recording = False
        self.recording_process = None
        self.recording_queue = queue.Queue()
        self.processed_frame_queue = queue.Queue(maxsize=2) # For thread communication
        self.recording_thread = None
        self.display_width = display_width
        self.display_height = display_height

    def start_recording(self, fmt, path, framerate_tag, display_width, display_height, record_duration):
        if self.recording:
            print("Already recording.")
            return
        self.recording = True
        self.recording_thread = threading.Thread(target=self.record_output, args=(fmt, path, framerate_tag, display_width, display_height, record_duration), daemon=True)
        self.recording_thread.start()
        print("Recording started.")

    def stop_recording(self):
        if not self.recording:
            print("Not recording.")
            return
        self.recording_queue.put(None) # Signal to stop the recording thread
        self.recording = False
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join()
        print("Recording stopped.")

    def record_output(self, fmt, path, framerate_tag, display_width, display_height, record_duration):
        if fmt == "MP4":
            framerate = dpg.get_value(framerate_tag)
            total_frames_to_record = int(record_duration * framerate)
            frames_recorded = 0

            print(f"FFmpeg input dimensions: {display_width}x{display_height}, framerate: {framerate}, duration: {record_duration}s ({total_frames_to_record} frames)")
            logging.info(f"FFmpeg input dimensions: {display_width}x{display_height}, framerate: {framerate}, duration: {record_duration}s ({total_frames_to_record} frames)")
            logging.info(f"Recording thread started for format: {fmt}, path: {path}")

            process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgba', s=f"{display_width}x{display_height}", r=framerate)
            .output(path, pix_fmt='yuv420p', vcodec='libx264', loglevel='debug') # Added loglevel
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stderr=True)
        )
            while self.recording and frames_recorded < total_frames_to_record:
                frame = self.recording_queue.get()
                if frame is None:
                    logging.info("Received stop signal (None frame) in recording thread.")
                    break
                
                # Ensure the frame is in RGBA mode before getting bytes
                if frame.mode != 'RGBA':
                    frame = frame.convert('RGBA')
                
                # Get raw RGBA bytes directly from PIL Image
                raw_bytes = frame.tobytes()
                
                process.stdin.write(raw_bytes)
                frames_recorded += 1
                time.sleep(0.001) # Small delay to prevent pipe overload
                if frames_recorded % (framerate * 5) == 0: # Log every 5 seconds of frames
                    logging.info(f"Recorded {frames_recorded} of {total_frames_to_record} frames.")

            if frames_recorded >= total_frames_to_record:
                logging.info(f"Reached total frames limit ({total_frames_to_record}). Stopping recording.")
            else:
                logging.info("Recording stopped by external signal.")

            process.stdin.close()
            print(f"FFmpeg command: {' '.join(process.args)}") # Print the full command
            logging.info(f"FFmpeg command: {' '.join(process.args)}")
            stdout, stderr = process.communicate() # Use communicate to get all output
            if stdout:
                print(f"FFmpeg Stdout:\n{stdout.decode('utf-8')}")
                logging.info(f"FFmpeg Stdout:\n{stdout.decode('utf-8')}")
            if stderr:
                print(f"FFmpeg Stderr:\n{stderr.decode('utf-8')}")
                logging.error(f"FFmpeg Stderr:\n{stderr.decode('utf-8')}")
            self.recording = False # Explicitly stop recording after loop
            print("Recording finished.") # Add this line for feedback
        else:
            frame_count = 0
            frames_recorded = 0
            total_frames_to_record = int(record_duration * dpg.get_value(framerate_tag))
            while self.recording and frames_recorded < total_frames_to_record:
                frame = self.recording_queue.get()
                if frame is None:
                    logging.info("Received stop signal (None frame) in recording thread for image sequence.")
                    break
                frame.save(f"{path}_{frame_count}.png")
                frame_count += 1
                frames_recorded += 1
            if frames_recorded >= total_frames_to_record:
                logging.info(f"Reached total frames limit ({total_frames_to_record}) for image sequence. Stopping recording.")
            else:
                logging.info("Image sequence recording stopped by external signal.")
            self.recording = False # Explicitly stop recording after loop
            print("Recording finished.") # And also here for image sequences

    def put_frame(self, frame):
        if self.recording:
            self.recording_queue.put(frame)
        # Also put frames into processed_frame_queue for display/export capture
        try:
            self.processed_frame_queue.put_nowait(frame) # Use put_nowait to avoid blocking if queue is full
        except queue.Full:
            pass # Drop frame if queue is full (e.g., display can't keep up)

    def record_frames_to_temp_video(self, frames, output_path, framerate):
        if not frames:
            logging.error("No frames provided to record_frames_to_temp_video.")
            return

        width, height = frames[0].size
        logging.info(f"Recording {len(frames)} frames to temporary video: {output_path} at {width}x{height}@{framerate}fps")
        print(f"FFmpeg command (temp video): {' '.join(process.args)}") # Print the full command
        logging.info(f"FFmpeg command (temp video): {' '.join(process.args)}")

        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgba', s=f"{width}x{height}", r=framerate)
            .output(output_path, pix_fmt='yuv420p', vcodec='libx264', loglevel='debug') # Added loglevel
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stderr=True)
        )
        try:
            for frame in frames:
                if frame.mode != 'RGBA':
                    frame = frame.convert('RGBA')
                raw_bytes = frame.tobytes()
                process.stdin.write(raw_bytes)
                time.sleep(0.001) # Small delay to prevent pipe overload
            process.stdin.close()
            stdout, stderr = process.communicate() # Use communicate to get all output
            if stdout:
                logging.info(f"FFmpeg Stdout (temp video):\n{stdout.decode('utf-8')}")
            if stderr:
                logging.error(f"FFmpeg Stderr (temp video):\n{stderr.decode('utf-8')}")
        except Exception as e:
            logging.error(f"Error writing frames to temporary video: {e}", exc_info=True)
            if process.stdin:
                process.stdin.close()
            if process.poll() is None: # If process is still running, terminate it
                process.terminate()
            process.wait()
        except Exception as e:
            logging.error(f"Error writing frames to temporary video: {e}", exc_info=True)
            if process.stdin:
                process.stdin.close()
            if process.poll() is None: # If process is still running, terminate it
                process.terminate()
            process.wait()

def combine_audio_and_gif_to_mp4(gif_path, audio_path, output_mp4_path, duration):
    command = [
        'ffmpeg',
        '-y',  # Overwrite output files without asking
        '-i', video_path,      # Input video (processed frames)
        '-i', audio_path,      # Input audio
        '-c:v', 'libx264',     # Video codec
        '-c:a', 'aac',         # Audio codec
        '-map', '0:v:0',       # Map video stream from first input
        '-map', '1:a:0',       # Map audio stream from second input
        '-t', str(duration),   # Output duration (controlled by audio duration)
        '-pix_fmt', 'yuv420p', # Pixel format for compatibility
        output_mp4_path
    ]

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if stdout:
            print(f"FFmpeg Stdout (combine audio/video):\n{stdout.decode('utf-8')}")
            logging.info(f"FFmpeg Stdout (combine audio/video):\n{stdout.decode('utf-8')}")
        if stderr:
            print(f"FFmpeg Stderr (combine audio/video):\n{stderr.decode('utf-8')}")
            logging.error(f"FFmpeg Stderr (combine audio/video):\n{stderr.decode('utf-8')}")

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, output=stdout, stderr=stderr)
        print(f"Successfully created {output_mp4_path}")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"Deleted temporary audio file: {audio_path}")