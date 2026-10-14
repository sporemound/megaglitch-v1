import cv2, numpy as np
import gpu_ops as gpo
gpo.USE_CUDA = True

from cuda_effects import apply_cuda_edge_slicer, apply_cuda_channel_drift, apply_cuda_digital_melt

img = np.zeros((100, 100, 3), dtype=np.uint8)
p = {'edge_thresh': 0.2, 'shift_x': 10, 'shift_y': 0, 'pre_blur': 3, 'threshold': 0.5, 'melt_len': 10, 'direction': 'down', 'melt_mix': 1.0}

try:
    print('Slicer:', type(apply_cuda_edge_slicer(img, p, 1.0)))
except Exception as e:
    print('Slicer error:', e)

try:
    print('Drift:', type(apply_cuda_channel_drift(img, p, 1.0)))
except Exception as e:
    print('Drift error:', e)

try:
    print('Melt:', type(apply_cuda_digital_melt(img, p, 1.0)))
except Exception as e:
    print('Melt error:', e)
