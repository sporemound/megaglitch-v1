#!/usr/bin/env python3
"""
Quantum-Gestalt Glitch Engine
==============================
Generates glitch art GIFs with parameters controlled by cognitive state predictions.

Each cognitive state has unique glitch characteristics:
- Delta (Relaxed): Subtle, flowing glitches
- Theta (Meditative): Wave-like, smooth distortions
- Alpha (Calm): Balanced, moderate effects
- Beta (Active): Sharp, aggressive glitches
- Gamma (Hyper-focus): Extreme, chaotic distortions

Integrated into MegaGlitch-Node for real-time cognitive glitch effects.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import io
import os
import sys
import gc  # Garbage collection for memory management
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing as mp

# Suppress output from worker processes to prevent console spam
if mp.current_process().name != 'MainProcess':
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


# Worker function for CPU multiprocessing (must be at module level for pickling)
def _generate_frame_cpu_worker(img, profile, intensity, progress, active_effects, frame_idx):
    """
    Worker function for CPU-based frame generation in separate processes.
    Must be at module level to be picklable for ProcessPoolExecutor.
    """
    try:
        # Apply glitch effects directly without creating full engine instance
        glitched = _apply_glitch_effects_standalone(img, profile, intensity, progress, active_effects)
        return glitched
    except Exception as e:
        print(f"Error in worker for frame {frame_idx}: {e}")
        import traceback
        traceback.print_exc()
        # Return original image on error
        return img.copy()


def _apply_glitch_effects_standalone(img, profile, intensity, progress, active_effects):
    """
    Standalone function to apply glitch effects (for multiprocessing).
    Simplified version that doesn't require QuantumGestaltEngine instance.
    """
    from PIL import Image, ImageFilter, ImageEnhance
    import numpy as np
    
    glitched = img.copy()
    
    try:
        # Convert to numpy for processing
        img_array = np.array(glitched)
        
        for effect in active_effects:
            # Simple glitch effects that work without engine instance
            if effect == 'pixel_shift':
                shift_amount = int(profile.get('displacement', 5) * intensity)
                if shift_amount > 0:
                    img_array = np.roll(img_array, shift_amount, axis=1)
            
            elif effect == 'color_drift':
                color_shift = profile.get('color_shift', (5, 10, 15))
                for i, shift in enumerate(color_shift):
                    if i < img_array.shape[2]:
                        shift_px = int(shift * intensity)
                        img_array[:, :, i] = np.roll(img_array[:, :, i], shift_px, axis=0)
            
            elif effect == 'wave':
                amplitude = int(10 * intensity)
                frequency = 0.05 + progress * 0.1
                for y in range(img_array.shape[0]):
                    offset = int(amplitude * np.sin(y * frequency))
                    img_array[y] = np.roll(img_array[y], offset, axis=0)
            
            elif effect == 'glitch_blocks':
                block_size = int(20 * intensity)
                num_blocks = int(10 * intensity)
                for _ in range(num_blocks):
                    x = np.random.randint(0, img_array.shape[1] - block_size)
                    y = np.random.randint(0, img_array.shape[0] - block_size)
                    img_array[y:y+block_size, x:x+block_size] = np.random.randint(0, 255, (block_size, block_size, 3))
        
        glitched = Image.fromarray(img_array.astype(np.uint8))
        
        # Apply blur if specified
        blur_amount = profile.get('blur', 0)
        if blur_amount > 0:
            glitched = glitched.filter(ImageFilter.GaussianBlur(radius=blur_amount * intensity))
        
    except Exception as e:
        print(f"Error applying standalone glitch effects: {e}")
        glitched = img.copy()
    
    return glitched


class QuantumGestaltEngine:
    """Generate glitch effects controlled by cognitive state predictions."""
    
    # Standard output settings
    OUTPUT_SIZE = 1920  # 1920x1920 pixels (Full HD quality)
    FPS = 55
    DURATION_SECONDS = 30
    TOTAL_FRAMES = FPS * DURATION_SECONDS  # 1650 frames
    
    # Cognitive state glitch profiles
    GLITCH_PROFILES = {
        0: {  # Delta (Relaxed)
            'name': 'Delta Wave',
            'base_intensity': 0.3,
            'effects': ['wave', 'soft_shift', 'color_drift'],
            'color_shift': (5, 10, 15),
            'displacement': 3,
            'blur': 1.5,
            'dynamics': {
                'intensity_variation': 0.2,  # ±20%
                'effect_cycling': True,
                'phase_shifts': 3  # Number of distinct phases
            }
        },
        1: {  # Theta (Meditative)
            'name': 'Theta Flow',
            'base_intensity': 0.4,
            'effects': ['ripple', 'smooth_wave', 'gentle_scan'],
            'color_shift': (10, 5, 20),
            'displacement': 5,
            'blur': 1.0,
            'dynamics': {
                'intensity_variation': 0.25,
                'effect_cycling': True,
                'phase_shifts': 4
            }
        },
        2: {  # Alpha (Calm)
            'name': 'Alpha Balance',
            'base_intensity': 0.5,
            'effects': ['pixel_sort', 'rgb_shift', 'scan_lines'],
            'color_shift': (15, 15, 15),
            'displacement': 8,
            'blur': 0.5,
            'dynamics': {
                'intensity_variation': 0.3,
                'effect_cycling': True,
                'phase_shifts': 4
            }
        },
        3: {  # Beta (Active)
            'name': 'Beta Pulse',
            'base_intensity': 0.7,
            'effects': ['harsh_shift', 'channel_split', 'sharp_glitch'],
            'color_shift': (30, 20, 40),
            'displacement': 15,
            'blur': 0.2,
            'dynamics': {
                'intensity_variation': 0.4,
                'effect_cycling': True,
                'phase_shifts': 5
            }
        },
        4: {  # Gamma (Hyper-focus)
            'name': 'Gamma Chaos',
            'base_intensity': 0.9,
            'effects': ['extreme_glitch', 'data_mosh', 'corruption'],
            'color_shift': (50, 40, 60),
            'displacement': 25,
            'blur': 0.1,
            'dynamics': {
                'intensity_variation': 0.5,
                'effect_cycling': True,
                'phase_shifts': 6
            }
        }
    }
    
    def __init__(self, use_cuda=True, num_workers=None, verbose=True):
        """
        Initialize glitch engine.
        
        Args:
            use_cuda (bool): Use CUDA/GPU if available
            num_workers (int): Number of parallel workers (None = auto-detect)
            verbose (bool): Print initialization messages (False for worker processes)
        """
        self.current_profile = None
        
        # Setup CUDA
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        # Setup parallel processing
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)
        self.verbose = verbose
        
        # Only print from main process to avoid spam
        if verbose and mp.current_process().name == 'MainProcess':
            print(f"Quantum-Gestalt Engine initialized:")
            print(f"  Device: {self.device}")
            print(f"  CUDA available: {torch.cuda.is_available()}")
            if self.use_cuda:
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Parallel workers: {self.num_workers}")
    
    def apply_cognitive_glitch(self, img, cognitive_state, intensity=1.0, progress=0.0):
        """
        Apply a single-frame cognitive glitch effect.
        This is for real-time use in MegaGlitch-Node.
        
        Args:
            img: PIL Image or numpy array
            cognitive_state: int (0-4) or dict with probabilities
            intensity: float (0.0-1.0) intensity multiplier
            progress: float (0.0-1.0) animation progress for dynamic effects
            
        Returns:
            PIL Image with glitch applied
        """
        # Convert numpy to PIL if needed
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        
        # Handle different cognitive state formats
        if isinstance(cognitive_state, int):
            # Single state - use that profile
            profile = self.GLITCH_PROFILES[cognitive_state]
            confidence = 1.0
        elif isinstance(cognitive_state, dict):
            # Probability distribution - blend profiles
            probabilities = np.array([cognitive_state.get(i, 0.0) for i in range(5)])
            probabilities = probabilities / probabilities.sum()  # Normalize
            profile = self._blend_profiles(probabilities)
            confidence = probabilities.max()
        else:
            # Assume it's an array of probabilities
            probabilities = np.array(cognitive_state)
            probabilities = probabilities / probabilities.sum()
            profile = self._blend_profiles(probabilities)
            confidence = probabilities.max()
        
        # Calculate dynamic intensity
        dynamic_intensity = self._calculate_dynamic_intensity(profile, confidence, progress)
        dynamic_intensity *= intensity  # Apply user intensity multiplier
        
        # Select effects for current phase
        active_effects = self._select_effects_for_phase(profile, progress)
        
        # Apply glitch effects
        glitched = self._apply_dynamic_glitch_effects(
            img, profile, dynamic_intensity, progress, active_effects
        )
        
        return glitched
    
    def generate_glitch_gif(self, image_path, predicted_class, probabilities, output_path='glitch_output.gif'):
        """
        Generate a glitch GIF controlled by cognitive state prediction.
        
        ALL 5 class probabilities influence the glitch parameters, creating unique
        hybrid styles based on the full probability distribution.
        
        Standard format: 1920x1920, 55 FPS, 30 seconds (1650 frames)
        
        Args:
            image_path (str): Path to input image
            predicted_class (int): Predicted cognitive class (0-4)
            probabilities (np.ndarray): Full probability distribution (all 5 classes)
            output_path (str): Output GIF path
            
        Returns:
            str: Path to generated GIF
        """
        # Create blended profile from ALL 5 states weighted by probabilities
        blended_profile = self._blend_profiles(probabilities)
        self.current_profile = blended_profile
        
        dominant_profile = self.GLITCH_PROFILES[predicted_class]
        
        if self.verbose:
            print(f"Generating hybrid glitch GIF...")
            print(f"Dominant: {dominant_profile['name']} ({float(probabilities[predicted_class]):.1%})")
            print(f"Blend: " + ", ".join([f"{self.GLITCH_PROFILES[i]['name'].split()[0]}: {float(probabilities[i]):.1%}" 
                                           for i in range(5) if float(probabilities[i]) > 0.05]))
            print(f"Format: {self.OUTPUT_SIZE}x{self.OUTPUT_SIZE}, {self.FPS} FPS, {self.DURATION_SECONDS}s")
            print(f"Total frames: {self.TOTAL_FRAMES}")
        
        # Load and prepare image
        img = Image.open(image_path).convert('RGB')
        img = self._prepare_image(img)
        
        # Generate frames with dynamic effects
        frames = []
        frame_duration = int(1000 / self.FPS)  # ms per frame
        
        if self.verbose:
            print("  Generating frames...")
            # Use ProcessPoolExecutor for CPU (true parallelism) or ThreadPoolExecutor for GPU
            executor_class = ThreadPoolExecutor if self.use_cuda else ProcessPoolExecutor
            print(f"  Using {'GPU (CUDA)' if self.use_cuda else 'CPU'} with {self.num_workers} parallel workers")
            print(f"  Executor: {executor_class.__name__}")
        else:
            executor_class = ThreadPoolExecutor if self.use_cuda else ProcessPoolExecutor
        
        # Convert image to tensor for GPU processing
        if self.use_cuda:
            img_tensor = self._pil_to_tensor(img).to(self.device)
        
        try:
            # Generate frames in parallel with progress bar
            from tqdm import tqdm
            
            # Prepare all frame parameters
            frame_params = []
            for i in range(self.TOTAL_FRAMES):
                progress = i / (self.TOTAL_FRAMES - 1) if self.TOTAL_FRAMES > 1 else 0.0
                
                # Calculate parameters
                intensity = self._calculate_dynamic_intensity(
                    blended_profile, probabilities[predicted_class], progress
                )
                active_effects = self._select_effects_for_phase(
                    blended_profile, progress
                )
                
                frame_params.append({
                    'index': i,
                    'img': img_tensor if self.use_cuda else img,
                    'profile': blended_profile,
                    'intensity': intensity,
                    'progress': progress,
                    'effects': active_effects
                })
            
            # Process frames in parallel
            frames = [None] * self.TOTAL_FRAMES
            
            with executor_class(max_workers=self.num_workers) as executor:
                if self.use_cuda:
                    # GPU processing with threads
                    futures = {
                        executor.submit(
                            self._generate_frame_gpu,
                            params['img'], params['profile'], params['intensity'],
                            params['progress'], params['effects'], params['index']
                        ): params['index']
                        for params in frame_params
                    }
                else:
                    # CPU processing with processes - use map for better memory management
                    futures = {
                        executor.submit(
                            _generate_frame_cpu_worker,
                            params['img'], params['profile'], params['intensity'],
                            params['progress'], params['effects'], params['index']
                        ): params['index']
                        for params in frame_params
                    }
                
                # Collect results with progress bar
                from concurrent.futures import as_completed
                
                # Progress bar only if verbose
                if self.verbose:
                    iterator = tqdm(as_completed(futures), total=self.TOTAL_FRAMES, 
                                   desc="Processing frames", unit="frame")
                else:
                    iterator = as_completed(futures)
                
                for future in iterator:
                    try:
                        frame_idx = futures[future]
                        frames[frame_idx] = future.result()
                    except Exception as e:
                        if self.verbose:
                            print(f"  Error processing frame {futures[future]}: {e}")
                        raise
                
        except Exception as e:
            if self.verbose:
                print(f"  Error generating frames: {e}")
            raise
        
        # Clean up GPU memory
        if self.use_cuda:
            del img_tensor
            torch.cuda.empty_cache()
        
        if self.verbose:
            print(f"  ✓ All {len(frames)} frames generated successfully")
            print("  Saving GIF (this may take a moment)...")
        
        # Save as high-quality GIF
        try:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=frame_duration,
                loop=0,  # 0 = loop forever (infinite loop)
                optimize=False,
                quality=100,  # Maximum quality
                subsampling=0  # Best chroma subsampling
            )
        except Exception as e:
            if self.verbose:
                print(f"  Error saving GIF: {e}")
            raise
        
        # Calculate file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        if self.verbose:
            print(f"✓ Glitch GIF saved: {output_path}")
            print(f"  Size: {file_size:.2f} MB")
            print(f"  Format: {self.OUTPUT_SIZE}x{self.OUTPUT_SIZE}, {self.FPS} FPS, {self.DURATION_SECONDS}s")
        
        return output_path
    
    def _prepare_image(self, img):
        """Prepare image to standard size format."""
        # Calculate padding to maintain aspect ratio
        width, height = img.size
        
        if width == height == self.OUTPUT_SIZE:
            return img
        
        # Determine scaling
        scale = self.OUTPUT_SIZE / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize with highest quality
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create padded canvas
        canvas = Image.new('RGB', (self.OUTPUT_SIZE, self.OUTPUT_SIZE), (0, 0, 0))
        
        # Center the image
        x_offset = (self.OUTPUT_SIZE - new_width) // 2
        y_offset = (self.OUTPUT_SIZE - new_height) // 2
        canvas.paste(img, (x_offset, y_offset))
        
        return canvas
    
    def _blend_profiles(self, probabilities):
        """
        Blend all 5 cognitive profiles weighted by their probabilities.
        Creates unique hybrid glitch styles based on full prediction distribution.
        """
        # Start with empty blended profile
        blended = {
            'name': 'Hybrid',
            'base_intensity': 0.0,
            'effects': [],
            'color_shift': [0, 0, 0],
            'displacement': 0,
            'blur': 0.0,
            'dynamics': {
                'intensity_variation': 0.0,
                'effect_cycling': True,
                'phase_shifts': 0
            }
        }
        
        # Weighted sum of all parameters
        for class_idx in range(5):
            weight = float(probabilities[class_idx])
            profile = self.GLITCH_PROFILES[class_idx]
            
            # Blend numeric parameters
            blended['base_intensity'] += profile['base_intensity'] * weight
            blended['displacement'] += profile['displacement'] * weight
            blended['blur'] += profile['blur'] * weight
            blended['dynamics']['intensity_variation'] += profile['dynamics']['intensity_variation'] * weight
            blended['dynamics']['phase_shifts'] += profile['dynamics']['phase_shifts'] * weight
            
            # Blend color shifts
            for i in range(3):
                blended['color_shift'][i] += profile['color_shift'][i] * weight
        
        # Convert color_shift back to tuple
        blended['color_shift'] = tuple(int(x) for x in blended['color_shift'])
        
        # Round phase shifts to integer
        blended['dynamics']['phase_shifts'] = max(1, int(round(blended['dynamics']['phase_shifts'])))
        
        # Collect all effects weighted by probability (include effect if prob > 5%)
        effect_pool = []
        for class_idx in range(5):
            prob_value = float(probabilities[class_idx])
            if prob_value > 0.05:  # Only include if >5% probability
                profile = self.GLITCH_PROFILES[class_idx]
                effect_pool.extend(profile['effects'])
        
        # Remove duplicates while preserving order
        seen = set()
        blended['effects'] = [x for x in effect_pool if not (x in seen or seen.add(x))]
        
        # If we have too many effects, prioritize by dominant classes
        if len(blended['effects']) > 6:
            # Keep effects from top 3 classes
            probs_array = np.array(probabilities)
            top_3_indices = np.argsort(probs_array)[-3:][::-1]
            priority_effects = []
            for idx in top_3_indices:
                priority_effects.extend(self.GLITCH_PROFILES[int(idx)]['effects'])
            # Remove duplicates
            seen = set()
            blended['effects'] = [x for x in priority_effects if not (x in seen or seen.add(x))]
        
        return blended
    
    def _pil_to_tensor(self, img):
        """Convert PIL image to PyTorch tensor."""
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW
        return tensor
    
    def _tensor_to_pil(self, tensor):
        """Convert PyTorch tensor to PIL image."""
        if tensor.is_cuda:
            tensor = tensor.cpu()
        arr = tensor.permute(1, 2, 0).numpy()  # CHW -> HWC
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    
    def _generate_frame_gpu(self, img_tensor, profile, intensity, progress, active_effects, frame_idx):
        """Generate a single frame using GPU acceleration."""
        try:
            # Convert tensor back to PIL for effects (most effects use PIL)
            img_pil = self._tensor_to_pil(img_tensor)
            
            # Apply effects (TODO: could be optimized further with pure PyTorch ops)
            glitched = self._apply_dynamic_glitch_effects(
                img_pil, profile, intensity, progress, active_effects
            )
            
            return glitched
            
        except Exception as e:
            print(f"  Warning: Frame {frame_idx} failed on GPU, falling back to CPU: {e}")
            # Fallback to CPU
            img_pil = self._tensor_to_pil(img_tensor)
            return self._generate_frame_cpu(img_pil, profile, intensity, progress, active_effects, frame_idx)
    
    def _generate_frame_cpu(self, img, profile, intensity, progress, active_effects, frame_idx):
        """Generate a single frame using CPU."""
        try:
            glitched = self._apply_dynamic_glitch_effects(
                img, profile, intensity, progress, active_effects
            )
            return glitched
        except Exception as e:
            print(f"  Error in frame {frame_idx}: {e}")
            # Return original image on error
            return img.copy()
    
    def _calculate_dynamic_intensity(self, profile, confidence, progress):
        """Calculate intensity that varies dynamically over time."""
        base = profile['base_intensity']
        variation = profile['dynamics']['intensity_variation']
        phase_shifts = profile['dynamics']['phase_shifts']
        
        # Multi-wave intensity variation
        wave1 = np.sin(progress * 2 * np.pi * phase_shifts)
        wave2 = np.sin(progress * 4 * np.pi * phase_shifts)
        wave3 = np.cos(progress * 3 * np.pi * phase_shifts)
        
        # Combine waves
        combined_wave = (wave1 + wave2 * 0.5 + wave3 * 0.3) / 1.8
        
        # Apply variation
        intensity = base + (combined_wave * variation)
        
        # Clamp and apply confidence
        intensity = np.clip(intensity, 0.1, 1.0)
        intensity *= (0.7 + 0.3 * confidence)
        
        return intensity
    
    def _select_effects_for_phase(self, profile, progress):
        """Select which effects to apply based on current phase."""
        if not profile['dynamics']['effect_cycling']:
            return profile['effects']
        
        all_effects = profile['effects']
        num_effects = len(all_effects)
        phase_shifts = profile['dynamics']['phase_shifts']
        
        # Determine current phase (0 to phase_shifts-1)
        current_phase = int(progress * phase_shifts) % phase_shifts
        
        # Rotate effect selection based on phase
        if num_effects <= 2:
            return all_effects
        
        # Select subset of effects for variety
        start_idx = current_phase % num_effects
        num_active = max(2, num_effects - 1)
        
        active = []
        for i in range(num_active):
            idx = (start_idx + i) % num_effects
            active.append(all_effects[idx])
        
        return active
    
    def _apply_dynamic_glitch_effects(self, img, profile, intensity, progress, active_effects):
        """Apply glitch effects with dynamic parameters."""
        glitched = img.copy()
        
        # Apply each active effect
        for effect in active_effects:
            if effect == 'wave':
                glitched = self._wave_distortion(glitched, intensity, progress)
            elif effect == 'soft_shift':
                glitched = self._soft_pixel_shift(glitched, profile['displacement'], intensity)
            elif effect == 'color_drift':
                glitched = self._color_drift(glitched, profile['color_shift'], intensity)
            elif effect == 'ripple':
                glitched = self._ripple_effect(glitched, intensity, progress)
            elif effect == 'smooth_wave':
                glitched = self._smooth_wave(glitched, intensity, progress)
            elif effect == 'gentle_scan':
                glitched = self._scan_lines(glitched, intensity, gentle=True)
            elif effect == 'pixel_sort':
                glitched = self._pixel_sort(glitched, intensity)
            elif effect == 'rgb_shift':
                glitched = self._rgb_shift(glitched, profile['displacement'], intensity)
            elif effect == 'scan_lines':
                glitched = self._scan_lines(glitched, intensity, gentle=False)
            elif effect == 'harsh_shift':
                glitched = self._harsh_shift(glitched, profile['displacement'], intensity)
            elif effect == 'channel_split':
                glitched = self._channel_split(glitched, profile['displacement'], intensity)
            elif effect == 'sharp_glitch':
                glitched = self._sharp_glitch(glitched, intensity)
            elif effect == 'extreme_glitch':
                glitched = self._extreme_glitch(glitched, intensity, progress)
            elif effect == 'data_mosh':
                glitched = self._data_mosh(glitched, intensity)
            elif effect == 'corruption':
                glitched = self._corruption(glitched, intensity, progress)
        
        # Apply blur if specified
        if profile['blur'] > 0 and intensity > 0.3:
            glitched = glitched.filter(ImageFilter.GaussianBlur(profile['blur']))
        
        return glitched
    
    # ===== GLITCH EFFECTS =====
    
    def _wave_distortion(self, img, intensity, progress):
        """Gentle wave distortion."""
        arr = np.array(img)
        height, width = arr.shape[:2]
        
        wave_img = Image.new('RGB', img.size)
        wave_arr = np.array(wave_img)
        
        for y in range(height):
            shift = int(intensity * 10 * np.sin(2 * np.pi * (y / 50 + progress)))
            if 0 <= shift < width:
                wave_arr[y] = np.roll(arr[y], shift, axis=0)
        
        return Image.fromarray(wave_arr)
    
    def _soft_pixel_shift(self, img, displacement, intensity):
        """Soft horizontal pixel shifting."""
        arr = np.array(img)
        shift = int(displacement * intensity)
        
        # Split into bands and shift
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        r = np.roll(r, shift, axis=1)
        b = np.roll(b, -shift, axis=1)
        
        return Image.fromarray(np.stack([r, g, b], axis=2))
    
    def _color_drift(self, img, color_shift, intensity):
        """Drift color channels."""
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.0 + intensity * 0.5)
        
        arr = np.array(img).astype(np.float32)
        arr[:, :, 0] += color_shift[0] * intensity
        arr[:, :, 1] += color_shift[1] * intensity
        arr[:, :, 2] += color_shift[2] * intensity
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        
        return Image.fromarray(arr)
    
    def _ripple_effect(self, img, intensity, progress):
        """Ripple distortion."""
        arr = np.array(img)
        height, width = arr.shape[:2]
        
        ripple_img = Image.new('RGB', img.size)
        ripple_arr = np.array(ripple_img)
        
        for y in range(height):
            for x in range(width):
                offset_x = int(intensity * 5 * np.sin(2 * np.pi * (x / 30 + progress)))
                offset_y = int(intensity * 5 * np.cos(2 * np.pi * (y / 30 + progress)))
                
                new_x = (x + offset_x) % width
                new_y = (y + offset_y) % height
                
                ripple_arr[y, x] = arr[new_y, new_x]
        
        return Image.fromarray(ripple_arr)
    
    def _smooth_wave(self, img, intensity, progress):
        """Smooth vertical wave."""
        arr = np.array(img)
        height, width = arr.shape[:2]
        
        wave_img = Image.new('RGB', img.size)
        wave_arr = np.array(wave_img)
        
        for x in range(width):
            shift = int(intensity * 8 * np.sin(2 * np.pi * (x / 40 + progress * 2)))
            wave_arr[:, x] = np.roll(arr[:, x], shift, axis=0)
        
        return Image.fromarray(wave_arr)
    
    def _scan_lines(self, img, intensity, gentle=False):
        """Add scan line effect."""
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        line_spacing = 4 if gentle else 2
        line_alpha = int(50 * intensity) if gentle else int(100 * intensity)
        
        for y in range(0, height, line_spacing):
            draw.line([(0, y), (width, y)], fill=(0, 0, 0, line_alpha), width=1)
        
        return img
    
    def _pixel_sort(self, img, intensity):
        """Pixel sorting glitch."""
        arr = np.array(img)
        height, width = arr.shape[:2]
        
        # Sort random rows
        num_rows = int(height * intensity * 0.3)
        rows = np.random.choice(height, num_rows, replace=False)
        
        for row in rows:
            # Sort pixels by brightness
            brightness = np.mean(arr[row], axis=1)
            sorted_indices = np.argsort(brightness)
            arr[row] = arr[row][sorted_indices]
        
        return Image.fromarray(arr)
    
    def _rgb_shift(self, img, displacement, intensity):
        """RGB channel shift."""
        arr = np.array(img)
        shift = int(displacement * intensity)
        
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        r = np.roll(r, shift, axis=1)
        g = np.roll(g, -shift // 2, axis=1)
        b = np.roll(b, -shift, axis=1)
        
        return Image.fromarray(np.stack([r, g, b], axis=2))
    
    def _harsh_shift(self, img, displacement, intensity):
        """Harsh pixel displacement."""
        arr = np.array(img)
        height, width = arr.shape[:2]
        
        # Random horizontal shifts in chunks
        chunk_height = max(2, int(20 * (1 - intensity)))
        for y in range(0, height, chunk_height):
            shift = np.random.randint(-displacement, displacement)
            arr[y:y+chunk_height] = np.roll(arr[y:y+chunk_height], shift, axis=1)
        
        return Image.fromarray(arr)
    
    def _channel_split(self, img, displacement, intensity):
        """Aggressive channel splitting."""
        arr = np.array(img)
        shift = int(displacement * intensity * 1.5)
        
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        
        # Different shifts and directions
        r = np.roll(np.roll(r, shift, axis=0), shift, axis=1)
        g = np.roll(g, -shift // 2, axis=1)
        b = np.roll(np.roll(b, -shift, axis=0), -shift, axis=1)
        
        return Image.fromarray(np.stack([r, g, b], axis=2))
    
    def _sharp_glitch(self, img, intensity):
        """Sharp block glitches."""
        arr = np.array(img)
        height, width = arr.shape[:2]
        
        # Random rectangular glitches
        num_glitches = int(20 * intensity)
        
        for _ in range(num_glitches):
            x = np.random.randint(0, max(1, width - 50))
            y = np.random.randint(0, max(1, height - 50))
            w = np.random.randint(10, min(50, width - x))
            h = np.random.randint(10, min(50, height - y))
            
            # Copy or invert block
            if np.random.random() > 0.5:
                arr[y:y+h, x:x+w] = 255 - arr[y:y+h, x:x+w]
            else:
                shift = np.random.randint(-100, 100)
                arr[y:y+h, x:x+w] = np.roll(arr[y:y+h, x:x+w], shift, axis=0)
        
        return Image.fromarray(arr)
    
    def _extreme_glitch(self, img, intensity, progress):
        """Extreme chaotic glitching."""
        arr = np.array(img).astype(np.float32)
        height, width = arr.shape[:2]
        
        # Heavy noise
        noise = np.random.randint(-100, 100, arr.shape) * intensity
        arr = arr + noise
        
        # Random channel swaps
        if np.random.random() > 0.5:
            arr = arr[:, :, [2, 0, 1]]  # BGR
        
        # Extreme shifts
        shift = int(50 * intensity * np.sin(progress * 4 * np.pi))
        arr = np.roll(arr, shift, axis=np.random.randint(0, 2))
        
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    
    def _data_mosh(self, img, intensity):
        """Data moshing effect."""
        arr = np.array(img)
        height, width = arr.shape[:2]
        
        # Corrupt random sections
        num_corruptions = int(30 * intensity)
        
        for _ in range(num_corruptions):
            y = np.random.randint(0, max(1, height - 20))
            source_y = np.random.randint(0, max(1, height - 20))
            h = np.random.randint(5, min(20, height - y))
            
            # Copy row from random location
            arr[y:y+h] = arr[source_y:source_y+h]
        
        return Image.fromarray(arr)
    
    def _corruption(self, img, intensity, progress):
        """Digital corruption artifacts."""
        arr = np.array(img)
        
        # Random bit flips
        mask = np.random.random(arr.shape) < (intensity * 0.05)
        arr[mask] = np.random.randint(0, 256, np.sum(mask))
        
        # Horizontal tear lines
        if intensity > 0.5:
            num_tears = int(10 * intensity)
            for _ in range(num_tears):
                y = np.random.randint(0, arr.shape[0])
                arr[y] = np.roll(arr[y], np.random.randint(-50, 50), axis=0)
        
        return Image.fromarray(arr)


