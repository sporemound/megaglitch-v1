import cv2
import numpy as np
import time

try:
    from megaglitchnode.opencv_cuda_bootstrap import bootstrap_for_dev_defaults
    bootstrap_for_dev_defaults()
except Exception as e:
    print(f"Bootstrap failed: {e}")

from gpu_ops import set_use_cuda
set_use_cuda(True)

from node_pipeline_runtime import execute_flat_node_pipeline

# Create dummy image
img = np.zeros((1080, 1920, 3), dtype=np.uint8)
cv2.circle(img, (960, 540), 200, (255, 255, 255), -1)

pipeline = [
    {
        "effect_type": "CUDA Channel Drift",
        "params": {
            "enabled": True,
            "b_shift_x": 50.0,
            "drift_speed": 1.0,
            "rot_var": 0.5,
            "scale_var": 0.1
        }
    },
    {
        "effect_type": "CUDA GPU Feedback Mixer",
        "params": {
            "enabled": True,
            "decay": 0.9,
            "mix": 0.5
        }
    }
]

from frame_state import FrameState
state = FrameState()
current_glitched_frame = img.copy()

for i in range(10):
    extra_params = {'time_accum': i / 60.0}
    (
        current_glitched_frame,
        state.previous_frame,
        state.feedback_image,
        state.trail_image,
    ) = execute_flat_node_pipeline(
        pipeline,
        current_glitched_frame,
        extra_params=extra_params,
        previous_frame=state.previous_frame,
        feedback_image=state.feedback_image,
        trail_image=state.trail_image,
        frame_buffer=state.frame_buffer,
        worker_state=state,
        fallback_size=(1920, 1080),
    )
    
    # Analyze output
    diff = np.sum(np.abs(current_glitched_frame.astype(np.int32) - img.astype(np.int32)))
    print(f"Frame {i}: diff={diff}, shape={current_glitched_frame.shape}")
