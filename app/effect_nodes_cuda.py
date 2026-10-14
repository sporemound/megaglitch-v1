"""
GPU-Accelerated CUDA Nodes for MegaGlitch

These nodes leverage CUDA for real-time performance at 1080p60.
All nodes use the megaglitchnode core pipeline for actual processing.
"""

from effect_nodes import GlitchEffectNode
import numpy as np
import logging

# Import GPU pipeline (will be available after bootstrap)
try:
    from megaglitchnode.gpu import Frame, Context, Pipeline
    from megaglitchnode.gpu.nodes import GpuUpload, GpuWarpRemap, GpuFeedbackMixer, GpuBloomGhost
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)

# Shared GPU context for all GPU nodes
_gpu_context = None

def get_gpu_context():
    """Get or create shared GPU context."""
    global _gpu_context
    if _gpu_context is None and GPU_AVAILABLE:
        _gpu_context = Context(use_stream=False)
    return _gpu_context


class CUDAFlowWarpNode(GlitchEffectNode):
    """GPU optical flow warping with motion-based distortion."""
    def __init__(self, parent_node_editor):
        params = [
            {"name": "flow_gain", "type": "float", "label": "Flow Gain", "min_value": 0.0, "max_value": 10.0, "default_value": 2.0},
            {"name": "nonlinearity", "type": "float", "label": "Nonlinearity", "min_value": 0.0, "max_value": 4.0, "default_value": 1.2},
            {"name": "motion_threshold", "type": "float", "label": "Motion Threshold", "min_value": 0.0, "max_value": 50.0, "default_value": 3.0},
            {"name": "temporal_smooth", "type": "float", "label": "Temporal Smooth", "min_value": 0.0, "max_value": 1.0, "default_value": 0.2},
            {"name": "border_mode", "type": "combo", "label": "Border Mode", "items": ["reflect", "wrap", "constant"], "default_value": "reflect"},
        ]
        super().__init__("CUDA Flow Warp", parent_node_editor, params)


class CUDASpectralDatamoshNode(GlitchEffectNode):
    """GPU DFT-based spectral manipulation for shimmer effects."""
    def __init__(self, parent_node_editor):
        params = [
            {"name": "band_kill_low", "type": "float", "label": "Kill Low Freq", "min_value": 0.0, "max_value": 1.0, "default_value": 0.05},
            {"name": "band_kill_high", "type": "float", "label": "Kill High Freq", "min_value": 0.0, "max_value": 1.0, "default_value": 0.85},
            {"name": "notch_count", "type": "int", "label": "Notch Count", "min_value": 0, "max_value": 64, "default_value": 8},
            {"name": "notch_width", "type": "float", "label": "Notch Width", "min_value": 0.0, "max_value": 0.2, "default_value": 0.03},
            {"name": "mask_lfo_rate", "type": "float", "label": "LFO Rate", "min_value": 0.0, "max_value": 10.0, "default_value": 0.7},
            {"name": "mix", "type": "float", "label": "Mix", "min_value": 0.0, "max_value": 1.0, "default_value": 0.7},
        ]
        super().__init__("CUDA Spectral Datamosh", parent_node_editor, params)


class CUDAFlowPixelSortNode(GlitchEffectNode):
    """Hybrid GPU/CPU pixel sorting guided by motion flow."""
    def __init__(self, parent_node_editor):
        params = [
            {"name": "sort_axis", "type": "combo", "label": "Sort Axis", "items": ["x", "y"], "default_value": "x"},
            {"name": "region_mode", "type": "combo", "label": "Region Mode", "items": ["motion", "edges", "luma"], "default_value": "motion"},
            {"name": "motion_thresh", "type": "float", "label": "Motion Threshold", "min_value": 0.0, "max_value": 50.0, "default_value": 8.0},
            {"name": "streak_len", "type": "int", "label": "Streak Length", "min_value": 2, "max_value": 2048, "default_value": 256},
            {"name": "jitter", "type": "float", "label": "Jitter", "min_value": 0.0, "max_value": 1.0, "default_value": 0.1},
            {"name": "mix", "type": "float", "label": "Mix", "min_value": 0.0, "max_value": 1.0, "default_value": 0.8},
        ]
        super().__init__("CUDA Flow Pixel Sort", parent_node_editor, params)


class CUDANonlinearRemapNode(GlitchEffectNode):
    """GPU nonlinear displacement field for liquid warping."""
    def __init__(self, parent_node_editor):
        params = [
            {"name": "field_scale", "type": "float", "label": "Field Scale", "min_value": 0.01, "max_value": 1.0, "default_value": 0.25},
            {"name": "amplitude_px", "type": "float", "label": "Amplitude (px)", "min_value": 0.0, "max_value": 200.0, "default_value": 35.0},
            {"name": "turbulence", "type": "float", "label": "Turbulence", "min_value": 0.0, "max_value": 4.0, "default_value": 1.5},
            {"name": "curl", "type": "float", "label": "Curl", "min_value": 0.0, "max_value": 2.0, "default_value": 0.6},
            {"name": "evolve_rate", "type": "float", "label": "Evolve Rate", "min_value": 0.0, "max_value": 10.0, "default_value": 1.0},
            {"name": "octaves", "type": "int", "label": "Octaves", "min_value": 1, "max_value": 5, "default_value": 3},
        ]
        super().__init__("CUDA Nonlinear Remap", parent_node_editor, params)


class CUDASingularityEngineNode(GlitchEffectNode):
    """Single CUDA remap vortex prototype for SingularityEngine."""
    def __init__(self, parent_node_editor):
        params = [
            {"name": "position_x", "type": "float", "label": "Position X", "min_value": 0.0, "max_value": 1.0, "default_value": 0.5},
            {"name": "position_y", "type": "float", "label": "Position Y", "min_value": 0.0, "max_value": 1.0, "default_value": 0.5},
            {"name": "orbit_x", "type": "float", "label": "Orbit X", "min_value": 0.0, "max_value": 0.5, "default_value": 0.06},
            {"name": "orbit_y", "type": "float", "label": "Orbit Y", "min_value": 0.0, "max_value": 0.5, "default_value": 0.04},
            {"name": "mass", "type": "float", "label": "Mass", "min_value": 0.0, "max_value": 4.0, "default_value": 1.0},
            {"name": "radius", "type": "float", "label": "Radius", "min_value": 0.01, "max_value": 1.5, "default_value": 0.42},
            {"name": "spin", "type": "float", "label": "Spin", "min_value": -12.0, "max_value": 12.0, "default_value": 3.5},
            {"name": "softness", "type": "float", "label": "Softness", "min_value": 0.05, "max_value": 4.0, "default_value": 1.8},
            {"name": "pull", "type": "float", "label": "Pull", "min_value": 0.0, "max_value": 0.35, "default_value": 0.08},
            {"name": "animation_speed", "type": "float", "label": "Anim Speed", "min_value": 0.0, "max_value": 5.0, "default_value": 1.0},
            {"name": "blur_sigma", "type": "float", "label": "Blur Sigma", "min_value": 0.0, "max_value": 8.0, "default_value": 2.0},
            {"name": "blur_mix", "type": "float", "label": "Blur Mix", "min_value": 0.0, "max_value": 1.0, "default_value": 0.35},
            {"name": "mix", "type": "float", "label": "Mix", "min_value": 0.0, "max_value": 1.0, "default_value": 1.0},
            {"name": "border_mode", "type": "combo", "label": "Border", "items": ["reflect", "replicate", "wrap", "constant"], "default_value": "reflect"},
        ]
        super().__init__("CUDA SingularityEngine", parent_node_editor, params)


class CUDABitplaneDriftNode(GlitchEffectNode):
    """Bitplane temporal manipulation for digital shimmer."""
    def __init__(self, parent_node_editor):
        params = [
            {"name": "recombine_mix", "type": "float", "label": "Recombine Mix", "min_value": 0.0, "max_value": 1.0, "default_value": 0.7},
            {"name": "max_history", "type": "int", "label": "Max History", "min_value": 3, "max_value": 50, "default_value": 30},
        ]
        super().__init__("CUDA Bitplane Drift", parent_node_editor, params)


class CUDATemporalMedianNode(GlitchEffectNode):
    """Temporal median filter for motion smearing."""
    def __init__(self, parent_node_editor):
        params = [
            {"name": "window", "type": "int", "label": "Window Size", "min_value": 3, "max_value": 15, "default_value": 5},
            {"name": "mode", "type": "combo", "label": "Mode", "items": ["approx", "full"], "default_value": "approx"},
            {"name": "mix", "type": "float", "label": "Mix", "min_value": 0.0, "max_value": 1.0, "default_value": 0.8},
        ]
        super().__init__("CUDA Temporal Median", parent_node_editor, params)


class CUDAAudioFlowNode(GlitchEffectNode):
    """Audio-driven flow field injection."""
    def __init__(self, parent_node_editor):
        params = [
            {"name": "bass_force", "type": "float", "label": "Bass Force", "min_value": 0.0, "max_value": 10.0, "default_value": 2.5},
            {"name": "treble_ripple", "type": "float", "label": "Treble Ripple", "min_value": 0.0, "max_value": 10.0, "default_value": 1.0},
            {"name": "feature_smoothing", "type": "float", "label": "Feature Smoothing", "min_value": 0.0, "max_value": 1.0, "default_value": 0.8},
            {"name": "force_scale", "type": "float", "label": "Force Scale", "min_value": 0.0, "max_value": 200.0, "default_value": 50.0},
            {"name": "ripple_frequency", "type": "float", "label": "Ripple Frequency", "min_value": 0.1, "max_value": 10.0, "default_value": 4.0},
        ]
        super().__init__("CUDA Audio Flow", parent_node_editor, params)


class CUDAPersistentFeedbackNode(GlitchEffectNode):
    """GPU-accelerated persistent frame feedback with decay."""
    def __init__(self, parent_node_editor):
        params = [
            {"name": "feedback_gain", "type": "float", "label": "Feedback Gain", "min_value": 0.0, "max_value": 1.0, "default_value": 0.92},
            {"name": "inject_gain", "type": "float", "label": "Inject Gain", "min_value": 0.0, "max_value": 1.0, "default_value": 0.25},
            {"name": "decay_curve", "type": "combo", "label": "Decay Curve", "items": ["linear", "exp"], "default_value": "exp"},
            {"name": "clamp", "type": "int", "label": "Clamp", "min_value": 0, "max_value": 255, "default_value": 255},
            {"name": "gate_motion", "type": "float", "label": "Gate Motion", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
        ]
        super().__init__("CUDA Persistent Feedback", parent_node_editor, params)


class CUDAVectorMoshNode(GlitchEffectNode):
    """Authentic H.264 datamoshing using CUDA motion vectors."""
    def __init__(self, parent_node_editor):
        params = [
            {"name": "flow_sensitivity", "type": "float", "label": "Flow Sensitivity", "min_value": 0.0, "max_value": 5.0, "default_value": 1.0},
            {"name": "i_frame_drop", "type": "float", "label": "I-Frame Drop Chance", "min_value": 0.0, "max_value": 1.0, "default_value": 0.5},
            {"name": "vector_decay", "type": "float", "label": "Vector Decay", "min_value": 0.0, "max_value": 1.0, "default_value": 0.9},
            {"name": "x_bias", "type": "float", "label": "X Bias", "min_value": -50.0, "max_value": 50.0, "default_value": 0.0},
            {"name": "y_bias", "type": "float", "label": "Y Bias", "min_value": -50.0, "max_value": 50.0, "default_value": 0.0},
        ]
        super().__init__("CUDA Vector Mosh", parent_node_editor, params)


class CUDATemporalSlitScanNode(GlitchEffectNode):
    """Temporal Slit-Scan mapping depth to history buffer."""
    def __init__(self, parent_node_editor):
        params = [
            {"name": "buffer_size", "type": "int", "label": "Buffer Size", "min_value": 2, "max_value": 90, "default_value": 30},
            {"name": "orientation", "type": "combo", "label": "Orientation", "items": ["y_axis", "x_axis", "luma"], "default_value": "y_axis"},
            {"name": "mix", "type": "float", "label": "Mix", "min_value": 0.0, "max_value": 1.0, "default_value": 1.0},
            {"name": "direction", "type": "combo", "label": "Direction", "items": ["backward", "forward"], "default_value": "backward"},
            {"name": "time_scale", "type": "float", "label": "Time Scale", "min_value": 0.1, "max_value": 5.0, "default_value": 1.0},
            {"name": "time_offset", "type": "float", "label": "Time Offset", "min_value": -30.0, "max_value": 30.0, "default_value": 0.0},
            {"name": "depth_curve", "type": "combo", "label": "Depth Curve", "items": ["linear", "ease_in", "ease_out", "sine", "center_peak"], "default_value": "linear"},
            {"name": "luma_scale", "type": "float", "label": "Luma Depth", "min_value": 0.1, "max_value": 2.0, "default_value": 1.0},
            {"name": "luma_gamma", "type": "float", "label": "Luma Gamma", "min_value": 0.2, "max_value": 4.0, "default_value": 1.0},
            {"name": "luma_invert", "type": "checkbox", "label": "Invert Luma", "default_value": False},
            {"name": "luma_blur", "type": "int", "label": "Luma Blur", "min_value": 0, "max_value": 31, "default_value": 0},
            {"name": "wave_amount", "type": "float", "label": "Wave Amount", "min_value": -1.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "wave_frequency", "type": "float", "label": "Wave Frequency", "min_value": 0.1, "max_value": 16.0, "default_value": 4.0},
            {"name": "jitter_amount", "type": "float", "label": "Jitter Amount", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0},
            {"name": "jitter_scale", "type": "float", "label": "Jitter Scale", "min_value": 1.0, "max_value": 64.0, "default_value": 16.0},
            {"name": "posterize_steps", "type": "int", "label": "Posterize Steps", "min_value": 0, "max_value": 32, "default_value": 0},
            {"name": "freeze_history", "type": "checkbox", "label": "Freeze History", "default_value": False},
        ]
        super().__init__("CUDA Temporal Slit-Scan", parent_node_editor, params)


class CUDARecursiveFractalCorrupterNode(GlitchEffectNode):
    """GPU remap driven by iterated Julia/Mandelbrot-style coordinate fields."""
    def __init__(self, parent_node_editor):
        params = [
            {"name": "iterations", "type": "int", "label": "Iterations", "min_value": 1, "max_value": 64, "default_value": 18},
            {"name": "zoom", "type": "float", "label": "Zoom", "min_value": 0.2, "max_value": 8.0, "default_value": 1.4},
            {"name": "seed_x", "type": "float", "label": "Seed X", "min_value": -1.5, "max_value": 1.5, "default_value": -0.72},
            {"name": "seed_y", "type": "float", "label": "Seed Y", "min_value": -1.5, "max_value": 1.5, "default_value": 0.27},
            {"name": "warp_intensity", "type": "float", "label": "Warp Intensity", "min_value": 0.0, "max_value": 160.0, "default_value": 34.0},
            {"name": "evolve_rate", "type": "float", "label": "Evolve Rate", "min_value": 0.0, "max_value": 5.0, "default_value": 0.35},
            {"name": "mix", "type": "float", "label": "Mix", "min_value": 0.0, "max_value": 1.0, "default_value": 1.0},
        ]
        super().__init__("CUDA Recursive Fractal Corrupter", parent_node_editor, params)


class CUDAVoronoiShatterNode(GlitchEffectNode):
    """GPU remap that fractures the frame into drifting Voronoi shards."""
    def __init__(self, parent_node_editor):
        params = [
            {"name": "shard_count", "type": "int", "label": "Shard Count", "min_value": 2, "max_value": 128, "default_value": 32},
            {"name": "drift_speed", "type": "float", "label": "Drift Speed", "min_value": 0.0, "max_value": 5.0, "default_value": 0.8},
            {"name": "rotation_variance", "type": "float", "label": "Rotation Variance", "min_value": 0.0, "max_value": 45.0, "default_value": 8.0},
            {"name": "translation", "type": "float", "label": "Translation", "min_value": 0.0, "max_value": 140.0, "default_value": 28.0},
            {"name": "feature_tracking", "type": "checkbox", "label": "Feature Tracking", "default_value": False},
            {"name": "map_resolution", "type": "int", "label": "Map Resolution", "min_value": 96, "max_value": 512, "default_value": 256},
            {"name": "mix", "type": "float", "label": "Mix", "min_value": 0.0, "max_value": 1.0, "default_value": 1.0},
        ]
        super().__init__("CUDA Voronoi Shatter", parent_node_editor, params)


class CUDAReactionDiffusionNode(GlitchEffectNode):
    """Gray-Scott reaction-diffusion growth that consumes and warps the video."""
    def __init__(self, parent_node_editor):
        params = [
            {"name": "feed_rate", "type": "float", "label": "Feed Rate", "min_value": 0.0, "max_value": 0.10, "default_value": 0.036},
            {"name": "kill_rate", "type": "float", "label": "Kill Rate", "min_value": 0.0, "max_value": 0.10, "default_value": 0.060},
            {"name": "diffusion_a", "type": "float", "label": "Diffusion A", "min_value": 0.0, "max_value": 2.0, "default_value": 1.0},
            {"name": "diffusion_b", "type": "float", "label": "Diffusion B", "min_value": 0.0, "max_value": 1.0, "default_value": 0.5},
            {"name": "iterations_per_frame", "type": "int", "label": "Iterations", "min_value": 1, "max_value": 20, "default_value": 4},
            {"name": "video_influence", "type": "float", "label": "Video Influence", "min_value": 0.0, "max_value": 1.0, "default_value": 0.35},
            {"name": "seed_amount", "type": "float", "label": "Seed Amount", "min_value": 0.0, "max_value": 1.0, "default_value": 0.08},
            {"name": "sim_resolution", "type": "int", "label": "Sim Resolution", "min_value": 64, "max_value": 512, "default_value": 192},
            {"name": "warp_strength", "type": "float", "label": "Warp Strength", "min_value": 0.0, "max_value": 80.0, "default_value": 18.0},
            {"name": "growth_contrast", "type": "float", "label": "Growth Contrast", "min_value": 0.1, "max_value": 4.0, "default_value": 1.6},
            {"name": "color_mode", "type": "combo", "label": "Color Mode", "items": ["biofilm", "burn", "luma_mask"], "default_value": "biofilm"},
            {"name": "reset_sim", "type": "checkbox", "label": "Reset Sim", "default_value": False},
            {"name": "mix", "type": "float", "label": "Mix", "min_value": 0.0, "max_value": 1.0, "default_value": 0.85},
        ]
        super().__init__("CUDA Reaction-Diffusion", parent_node_editor, params)


class CUDALumaKeyedFluidDynamicsNode(GlitchEffectNode):
    """Luma-keyed Eulerian fluid warp driven by video motion and optical flow."""
    def __init__(self, parent_node_editor):
        params = [
            {"name": "viscosity", "type": "float", "label": "Viscosity", "min_value": 0.0, "max_value": 4.0, "default_value": 0.8},
            {"name": "flow_dissipation", "type": "float", "label": "Flow Dissipation", "min_value": 0.80, "max_value": 1.0, "default_value": 0.94},
            {"name": "velocity_inject_strength", "type": "float", "label": "Inject Strength", "min_value": 0.0, "max_value": 8.0, "default_value": 2.5},
            {"name": "color_decay", "type": "float", "label": "Color Decay", "min_value": 0.80, "max_value": 1.0, "default_value": 0.92},
            {"name": "grid_resolution", "type": "int", "label": "Grid Resolution", "min_value": 64, "max_value": 512, "default_value": 192},
            {"name": "luma_threshold", "type": "float", "label": "Luma Threshold", "min_value": 0.0, "max_value": 1.0, "default_value": 0.42},
            {"name": "luma_softness", "type": "float", "label": "Luma Softness", "min_value": 0.01, "max_value": 0.5, "default_value": 0.16},
            {"name": "projection_iterations", "type": "int", "label": "Projection Iterations", "min_value": 0, "max_value": 30, "default_value": 8},
            {"name": "vorticity", "type": "float", "label": "Vorticity", "min_value": 0.0, "max_value": 4.0, "default_value": 0.6},
            {"name": "warp_strength", "type": "float", "label": "Warp Strength", "min_value": 0.0, "max_value": 120.0, "default_value": 36.0},
            {"name": "dye_mix", "type": "float", "label": "Dye Mix", "min_value": 0.0, "max_value": 1.0, "default_value": 0.35},
            {"name": "reset_fluid", "type": "checkbox", "label": "Reset Fluid", "default_value": False},
            {"name": "mix", "type": "float", "label": "Mix", "min_value": 0.0, "max_value": 1.0, "default_value": 0.9},
        ]
        super().__init__("CUDA Luma-Keyed Fluid Dynamics", parent_node_editor, params)


# ===== NEW GPU PIPELINE NODES =====

class GPUUploadNode(GlitchEffectNode):
    """Upload frame to GPU memory (first node in GPU pipeline)."""
    def __init__(self, parent_node_editor):
        params = []
        super().__init__("CUDA GPU Upload", parent_node_editor, params)
        self.gpu_node = GpuUpload("gpu_upload") if GPU_AVAILABLE else None
    
    def apply_effect(self, img_array, params, prev_frame, time_val):
        """Process frame through GPU upload node."""
        if not GPU_AVAILABLE or self.gpu_node is None:
            return img_array, prev_frame
        
        try:
            ctx = get_gpu_context()
            frame = Frame.from_numpy(img_array, fmt="bgr8")
            frame = self.gpu_node.process(frame, ctx)
            return frame.ensure_cpu(ctx), prev_frame
        except Exception as e:
            logger.error(f"GPU Upload error: {e}")
            return img_array, prev_frame


class GPUWarpRemapNode(GlitchEffectNode):
    """GPU geometric warping (swirl, ripple, blockshift)."""
    def __init__(self, parent_node_editor):
        params = [
            {"name": "mode", "type": "combo", "label": "Mode", "items": ["swirl", "ripple", "blockshift"], "default_value": "swirl"},
            {"name": "strength", "type": "float", "label": "Strength", "min_value": 0.0, "max_value": 0.01, "default_value": 0.001},
            {"name": "block_size", "type": "int", "label": "Block Size", "min_value": 8, "max_value": 128, "default_value": 32},
            {"name": "border_mode", "type": "combo", "label": "Border Mode", "items": ["reflect", "replicate", "constant"], "default_value": "reflect"},
            {"name": "interpolation", "type": "combo", "label": "Interpolation", "items": ["linear", "nearest"], "default_value": "linear"},
            {"name": "animate", "type": "checkbox", "label": "Animate", "default_value": True},
        ]
        super().__init__("CUDA GPU Warp Remap", parent_node_editor, params)
        self.gpu_node = GpuWarpRemap("gpu_warp") if GPU_AVAILABLE else None
    
    def apply_effect(self, img_array, params, prev_frame, time_val):
        """Process frame through GPU warp node."""
        if not GPU_AVAILABLE or self.gpu_node is None:
            return img_array, prev_frame
        
        try:
            # Update node parameters
            self.gpu_node.set_params(**params)
            
            ctx = get_gpu_context()
            frame = Frame.from_numpy(img_array, fmt="bgr8")
            frame.ensure_gpu(ctx)  # Ensure on GPU
            frame = self.gpu_node.process(frame, ctx)
            return frame.ensure_cpu(ctx), prev_frame
        except Exception as e:
            logger.error(f"GPU Warp error: {e}")
            return img_array, prev_frame


class GPUFeedbackMixerNode(GlitchEffectNode):
    """GPU feedback trails with persistent buffer."""
    def __init__(self, parent_node_editor):
        params = [
            {"name": "decay", "type": "float", "label": "Decay", "min_value": 0.0, "max_value": 1.0, "default_value": 0.85},
            {"name": "mix", "type": "float", "label": "Mix", "min_value": 0.0, "max_value": 1.0, "default_value": 0.25},
            {"name": "warp_prev", "type": "checkbox", "label": "Warp Previous", "default_value": False},
            {"name": "warp_strength", "type": "float", "label": "Warp Strength", "min_value": 0.0, "max_value": 0.01, "default_value": 0.0005},
        ]
        super().__init__("CUDA GPU Feedback Mixer", parent_node_editor, params)
        self.gpu_node = GpuFeedbackMixer("gpu_feedback") if GPU_AVAILABLE else None
    
    def apply_effect(self, img_array, params, prev_frame, time_val):
        """Process frame through GPU feedback node."""
        if not GPU_AVAILABLE or self.gpu_node is None:
            return img_array, prev_frame
        
        try:
            # Update node parameters
            self.gpu_node.set_params(**params)
            
            ctx = get_gpu_context()
            frame = Frame.from_numpy(img_array, fmt="bgr8")
            frame.ensure_gpu(ctx)  # Ensure on GPU
            frame = self.gpu_node.process(frame, ctx)
            return frame.ensure_cpu(ctx), prev_frame
        except Exception as e:
            logger.error(f"GPU Feedback error: {e}")
            return img_array, prev_frame


class GPUBloomGhostNode(GlitchEffectNode):
    """GPU Gaussian bloom effect."""
    def __init__(self, parent_node_editor):
        params = [
            {"name": "ksize", "type": "int", "label": "Kernel Size", "min_value": 5, "max_value": 51, "default_value": 21},
            {"name": "sigma", "type": "float", "label": "Sigma", "min_value": 0.5, "max_value": 20.0, "default_value": 5.0},
            {"name": "amount", "type": "float", "label": "Amount", "min_value": 0.0, "max_value": 1.0, "default_value": 0.35},
        ]
        super().__init__("CUDA GPU Bloom Ghost", parent_node_editor, params)
        self.gpu_node = GpuBloomGhost("gpu_bloom") if GPU_AVAILABLE else None
    
    def apply_effect(self, img_array, params, prev_frame, time_val):
        """Process frame through GPU bloom node."""
        if not GPU_AVAILABLE or self.gpu_node is None:
            return img_array, prev_frame
        
        try:
            self.gpu_node.set_params(**params)
            ctx = get_gpu_context()
            frame = Frame.from_numpy(img_array, fmt="bgr8")
            frame.ensure_gpu(ctx)
            frame = self.gpu_node.process(frame, ctx)
            return frame.ensure_cpu(ctx), prev_frame
        except Exception as e:
            logger.error(f"GPU Bloom error: {e}")
            return img_array, prev_frame

class CUDAEdgeSlicerNode(GlitchEffectNode):
    """GPU-accelerated tearing driven by edge detection."""
    def __init__(self, parent_node_editor):
        params = [
            {"name": "edge_thresh", "type": "float", "label": "Edge Threshold", "min_value": 0.0, "max_value": 1.0, "default_value": 0.2},
            {"name": "shift_x", "type": "float", "label": "Shift X", "min_value": -200.0, "max_value": 200.0, "default_value": 50.0},
            {"name": "shift_y", "type": "float", "label": "Shift Y", "min_value": -200.0, "max_value": 200.0, "default_value": 0.0},
            {"name": "pre_blur", "type": "int", "label": "Pre-Blur", "min_value": 1, "max_value": 21, "default_value": 3},
        ]
        super().__init__("CUDA Edge Slicer", parent_node_editor, params)

class CUDAChannelDriftNode(GlitchEffectNode):
    """GPU-accelerated independent RGB channel transformation."""
    def __init__(self, parent_node_editor):
        params = [
            {"name": "r_shift_x", "type": "float", "label": "R Shift X", "min_value": -100.0, "max_value": 100.0, "default_value": -10.0},
            {"name": "r_shift_y", "type": "float", "label": "R Shift Y", "min_value": -100.0, "max_value": 100.0, "default_value": -5.0},
            {"name": "g_shift_x", "type": "float", "label": "G Shift X", "min_value": -100.0, "max_value": 100.0, "default_value": 0.0},
            {"name": "g_shift_y", "type": "float", "label": "G Shift Y", "min_value": -100.0, "max_value": 100.0, "default_value": 0.0},
            {"name": "b_shift_x", "type": "float", "label": "B Shift X", "min_value": -100.0, "max_value": 100.0, "default_value": 10.0},
            {"name": "b_shift_y", "type": "float", "label": "B Shift Y", "min_value": -100.0, "max_value": 100.0, "default_value": 5.0},
            {"name": "scale_var", "type": "float", "label": "Scale Variance", "min_value": 0.0, "max_value": 0.5, "default_value": 0.02},
            {"name": "rot_var", "type": "float", "label": "Rot Variance", "min_value": 0.0, "max_value": 45.0, "default_value": 0.5},
            {"name": "drift_speed", "type": "float", "label": "Drift Speed", "min_value": 0.0, "max_value": 5.0, "default_value": 1.0},
        ]
        super().__init__("CUDA Channel Drift", parent_node_editor, params)

class CUDADigitalMeltNode(GlitchEffectNode):
    """GPU-accelerated directional pixel melting."""
    def __init__(self, parent_node_editor):
        params = [
            {"name": "threshold", "type": "float", "label": "Luma Threshold", "min_value": 0.0, "max_value": 1.0, "default_value": 0.6},
            {"name": "melt_length", "type": "int", "label": "Melt Length", "min_value": 1, "max_value": 100, "default_value": 20},
            {"name": "direction", "type": "combo", "label": "Direction", "items": ["down", "up", "left", "right"], "default_value": "down"},
            {"name": "melt_mix", "type": "float", "label": "Melt Mix", "min_value": 0.0, "max_value": 1.0, "default_value": 0.8},
        ]
        super().__init__("CUDA Digital Melt", parent_node_editor, params)
