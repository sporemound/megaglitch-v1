import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

try:
    from megaglitchnode.opencv_cuda_bootstrap import bootstrap_for_dev_defaults
    bootstrap_for_dev_defaults()
except Exception:
    pass

from glitch_effects import apply_glitch_effect
from effects_registry import apply_node_effect
from frame_state import FrameState
from node_pipeline_runtime import execute_flat_node_pipeline
import multiprocessing as mp
import logging
from PIL import Image
import numpy as np


def _get_configured_worker_count() -> int:
    """Get the configured worker count, falling back to CPU count."""
    try:
        import config
        configured = int(getattr(config, "NUM_WORKER_PROCESSES", 0) or 0)
        if configured > 0:
            return configured
    except Exception:
        pass
    return max(1, mp.cpu_count() - 1)


def process_frames_in_parallel(source_frames, glitch_params, frame_buffer):
    """
    Processes a list of frames in parallel using a thread pool.

    Args:
        source_frames (list): A list of PIL Image objects to be processed.
        glitch_params (dict): A dictionary of glitch parameters.
        frame_buffer (list): The frame buffer for temporal effects.

    Returns:
        list: A list of processed PIL Image objects.
    """
    processed_frames = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Create a future for each frame processing task
        futures = [executor.submit(apply_glitch_effect, frame.copy(), glitch_params, None, None, None, frame_buffer) for frame in source_frames]
        
        # Process results as they complete, with a progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(source_frames), desc="Rendering frames in parallel"):
            try:
                # The result from apply_glitch_effect is a tuple (glitched_img, prev, feed, trail)
                glitched_img, _, _, _ = future.result()
                processed_frames.append(glitched_img)
            except Exception as e:
                print(f"Error processing a frame in parallel: {e}")

    return processed_frames


def _process_single_frame_worker(frame_data):
    """Worker function for processing a single frame in a separate process."""
    try:
        frame_idx = frame_data['index']
        source_image = frame_data['image']
        secondary_image = frame_data.get('secondary_image')
        pipeline = frame_data['pipeline']
        extra_params = frame_data.get('extra_params', {})
        use_cuda = frame_data.get('use_cuda', False)
        
        # Configure GPU for this worker
        try:
            import gpu_ops as _gpu_ops
            _gpu_ops.set_use_cuda(use_cuda)
        except Exception:
            pass

        worker_state = FrameState()
        try:
            worker_state.frame_count = int(extra_params.get("frame_num", frame_idx) or frame_idx)
        except Exception:
            pass
        
        current_glitched_frame, _, _, _ = execute_flat_node_pipeline(
            pipeline,
            source_image,
            secondary_frame=secondary_image,
            extra_params=extra_params,
            previous_frame=None,
            feedback_image=None,
            trail_image=None,
            frame_buffer=[],
            worker_state=worker_state,
        )
        
        return (frame_idx, current_glitched_frame)
    except Exception as e:
        logging.error(f"Error in parallel frame worker: {e}")
        return (frame_data.get('index', -1), None)


def batch_process_frames_parallel(source_frames, pipeline, extra_params_list, use_cuda=False, max_workers=None,
                                  has_temporal_effects=False, progress_bar=None):
    """
    Process frames in parallel using multiple processes.
    
    Args:
        source_frames (list): List of PIL Image objects (source frames to process)
        pipeline (list): Processing pipeline (list of effect configs)
        extra_params_list (list): List of extra_params dicts for each frame (e.g., time)
        use_cuda (bool): Whether to use CUDA acceleration
        max_workers (int): Number of parallel workers (default: CPU count)
        has_temporal_effects (bool): If True, uses sequential processing to maintain temporal state
        
    Returns:
        list: Processed frames in order
    """
    if max_workers is None:
        max_workers = _get_configured_worker_count()
    
    num_frames = len(source_frames)
    
    # Check if pipeline has temporal effects
    if has_temporal_effects:
        # Temporal effects detected - using sequential processing
        return _batch_process_sequential(
            source_frames,
            pipeline,
            extra_params_list,
            use_cuda,
            progress_bar=progress_bar,
        )
    
    # Starting parallel processing
    
    # Prepare frame data for workers
    frame_data_list = []
    for i, source_frame in enumerate(source_frames):
        extra_params = extra_params_list[i] if i < len(extra_params_list) else {}
        frame_data_list.append({
            'index': i,
            'image': source_frame,
            'pipeline': pipeline,
            'extra_params': extra_params,
            'use_cuda': use_cuda
        })
    
    # Process frames in parallel
    processed_frames = [None] * num_frames
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(_process_single_frame_worker, data): data['index'] 
                      for data in frame_data_list}
            
            # Collect results with progress bar
            progress_iter = as_completed(futures)
            if progress_bar is None:
                progress_iter = tqdm(progress_iter, total=num_frames, desc="Processing frames", unit="frame")

            for future in progress_iter:
                try:
                    frame_idx, processed_frame = future.result()
                    if processed_frame is not None:
                        processed_frames[frame_idx] = processed_frame
                except Exception as e:
                    frame_idx = futures[future]
                    logging.error(f"Error processing frame {frame_idx}: {e}")
                    processed_frames[frame_idx] = source_frames[frame_idx].copy()  # Fallback to unprocessed
                finally:
                    if progress_bar is not None:
                        progress_bar.update(1)
    except Exception as e:
        logging.error(f"Error in parallel batch processing: {e}")
        # Fallback to sequential
        return _batch_process_sequential(
            source_frames,
            pipeline,
            extra_params_list,
            use_cuda,
            progress_bar=progress_bar,
        )
    
    # Batch processing complete
    return processed_frames


def _batch_process_sequential(source_frames, pipeline, extra_params_list, use_cuda, progress_bar=None):
    """Sequential processing for temporal effects."""
    # Processing frames sequentially...
    
    # Configure GPU
    try:
        import gpu_ops as _gpu_ops
        _gpu_ops.set_use_cuda(use_cuda)
    except Exception:
        pass
    
    # Initialize state
    state = FrameState()
    processed_frames = []
    
    frame_iter = source_frames
    if progress_bar is None:
        frame_iter = tqdm(source_frames, desc="Processing frames sequentially", unit="frame")

    for i, source_frame in enumerate(frame_iter):
        extra_params = extra_params_list[i] if i < len(extra_params_list) else {}
        
        current_glitched_frame, current_previous_frame, current_feedback_image, current_trail_image = execute_flat_node_pipeline(
            pipeline,
            source_frame,
            secondary_frame=None,
            extra_params=extra_params,
            previous_frame=state.previous_frame,
            feedback_image=state.feedback_image,
            trail_image=state.trail_image,
            frame_buffer=state.frame_buffer,
            worker_state=state,
        )
        
        # Update state for next frame
        state.previous_frame = current_previous_frame
        state.feedback_image = current_feedback_image
        state.trail_image = current_trail_image
        
        # Update frame buffer
        try:
            state.frame_buffer.append(current_glitched_frame.copy())
            while len(state.frame_buffer) > 100:  # Limit buffer size
                state.frame_buffer.pop(0)
        except Exception:
            pass
        
        processed_frames.append(current_glitched_frame)
        if progress_bar is not None:
            progress_bar.update(1)
    
    # Sequential processing complete
    return processed_frames


def analyze_temporal_effects(pipeline):
    """
    Analyze the pipeline for temporal/stateful effects.

    Returns:
        (is_temporal: bool, reasons: list[str])
    """
    if not pipeline:
        return False, []

    temporal_labels = {
        "Row Temporal Stutter",
        "P-Frame Glitches",
        "CUDA Flow Warp",
        "CUDA Flow Pixel Sort",
        "CUDA Bitplane Drift",
        "CUDA Temporal Median",
        "CUDA Audio Flow",
        "CUDA Persistent Feedback",
        "CUDA GPU Feedback Mixer",
        "Faultline Possession",
    }

    def _num(value, default=0.0):
        try:
            return float(value)
        except Exception:
            return float(default)

    reasons = []
    for effect_config in pipeline:
        label = str(effect_config.get("effect_type") or "").strip()
        params = effect_config.get("params", {}) or {}

        if label == "Display":
            continue

        if label in temporal_labels:
            reasons.append(f"{label} is stateful")
            continue

        if label == "Colony Reef Gen":
            reasons.append(f"{label} keeps persistent generator memory")
            continue

        if label == "Feedback / Trail":
            if _num(params.get("feedback_intensity")) > 0 or _num(params.get("trailing_intensity")) > 0:
                reasons.append(f"{label} uses feedback/trails")
            continue

        if label == "FrameDrop / RGB Split":
            if _num(params.get("frame_drop_chance")) > 0:
                reasons.append(f"{label} uses previous-frame drops")
            continue

        if label == "Data Moshing":
            if _num(params.get("slit_scan_intensity")) > 0:
                reasons.append(f"{label} uses slit-scan state")
            continue

        if label == "Frame Buffer":
            if int(_num(params.get("frame_buffer_size"), 1)) > 1:
                reasons.append(f"{label} keeps a frame buffer > 1")
            continue

        if _num(params.get("row_stutter")) > 0:
            reasons.append(f"{label} enables row stutter")
            continue

        if _num(params.get("temporal_bleeding")) > 0:
            reasons.append(f"{label} enables temporal bleeding")
            continue

        if _num(params.get("motion_vector_corruption")) > 0:
            reasons.append(f"{label} enables motion-vector corruption")
            continue

        if _num(params.get("prediction_error_amplify")) > 0:
            reasons.append(f"{label} enables prediction-error amplification")
            continue

        if _num(params.get("macroblock_scramble")) > 0:
            reasons.append(f"{label} enables macroblock scrambling history")
            continue

        if _num(params.get("residual_injection")) > 0:
            reasons.append(f"{label} enables residual injection history")
            continue

        if _num(params.get("reference_corruption")) > 0:
            reasons.append(f"{label} enables reference corruption")
            continue

    return bool(reasons), reasons


def detect_temporal_effects(pipeline):
    """Backward-compatible boolean temporal check."""
    is_temporal, _ = analyze_temporal_effects(pipeline)
    return is_temporal


def _batch_process_parallel_modulated(source_frames, secondary_source_frames, pipelines_per_frame, extra_params_list, use_cuda,
                                      progress_bar=None):
    """
    Process frames in parallel where each frame has its own modulated pipeline.
    
    Args:
        source_frames: List of source images
        pipelines_per_frame: List of pipelines (one per frame) with modulated parameters
        extra_params_list: List of extra params for each frame
        use_cuda: Whether to use CUDA
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    
    num_frames = len(source_frames)
    max_workers = _get_configured_worker_count()
    
    # Processing frames in parallel with per-frame modulation...
    
    processed_frames = [None] * num_frames
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i in range(num_frames):
                extra_params = extra_params_list[i] if i < len(extra_params_list) else {}
                pipeline = pipelines_per_frame[i] if i < len(pipelines_per_frame) else pipelines_per_frame[0]
                
                future = executor.submit(
                    _process_single_frame_worker,
                    {
                        'index': i,
                        'image': source_frames[i],
                        'secondary_image': secondary_source_frames[i] if i < len(secondary_source_frames) else None,
                        'pipeline': pipeline,
                        'extra_params': extra_params,
                        'use_cuda': use_cuda
                    }
                )
                futures[future] = i
            
            # Collect with progress
            progress_iter = as_completed(futures)
            if progress_bar is None:
                progress_iter = tqdm(progress_iter, total=num_frames, desc="Processing frames", unit="frame")

            for future in progress_iter:
                try:
                    frame_idx, processed_frame = future.result()
                    if processed_frame is not None:
                        processed_frames[frame_idx] = processed_frame
                except Exception as e:
                    frame_idx = futures[future]
                    logging.error(f"Error processing frame {frame_idx}: {e}")
                    # Try to use source frame or create blank frame
                    try:
                        processed_frames[frame_idx] = source_frames[frame_idx].copy()
                    except Exception:
                        # If copy fails, create a blank frame
                        try:
                            w, h = source_frames[0].size if source_frames else (800, 600)
                            from PIL import Image
                            processed_frames[frame_idx] = Image.new('RGBA', (w, h), (0, 0, 0, 255))
                        except Exception:
                            pass
                finally:
                    if progress_bar is not None:
                        progress_bar.update(1)
    except Exception as e:
        logging.error(f"Error in parallel modulated processing: {e}")
        raise
    
    return processed_frames


def _batch_process_sequential_modulated(source_frames, secondary_source_frames, pipelines_per_frame, extra_params_list, use_cuda,
                                        state=None, return_state=False, progress_bar=None):
    """
    Process frames sequentially where each frame has its own modulated pipeline.
    Used for temporal effects that maintain state across frames.
    """
    from frame_state import FrameState
    
    num_frames = len(source_frames)
    # Processing frames sequentially with per-frame modulation...
    
    # Configure GPU
    try:
        import gpu_ops as _gpu_ops
        _gpu_ops.set_use_cuda(use_cuda)
    except Exception:
        pass
    
    if state is None:
        state = FrameState()
    processed_frames = []
    
    frame_iter = range(num_frames)
    if progress_bar is None:
        frame_iter = tqdm(frame_iter, desc="Processing frames sequentially", unit="frame")

    for i in frame_iter:
        source_frame = source_frames[i]
        secondary_source_frame = secondary_source_frames[i] if i < len(secondary_source_frames) else None
        pipeline = pipelines_per_frame[i] if i < len(pipelines_per_frame) else pipelines_per_frame[0]
        extra_params = extra_params_list[i] if i < len(extra_params_list) else {}

        current_glitched_frame, current_previous_frame, current_feedback_image, current_trail_image = execute_flat_node_pipeline(
            pipeline,
            source_frame,
            secondary_frame=secondary_source_frame,
            extra_params=extra_params,
            previous_frame=state.previous_frame,
            feedback_image=state.feedback_image,
            trail_image=state.trail_image,
            frame_buffer=state.frame_buffer,
            worker_state=state,
        )
        
        # Update state for next frame
        state.previous_frame = current_previous_frame
        state.feedback_image = current_feedback_image
        state.trail_image = current_trail_image
        
        try:
            state.frame_buffer.append(current_glitched_frame.copy())
            while len(state.frame_buffer) > 100:
                state.frame_buffer.pop(0)
        except Exception:
            pass

        processed_frames.append(current_glitched_frame)
        if progress_bar is not None:
            progress_bar.update(1)
    
    if return_state:
        return processed_frames, state
    return processed_frames
