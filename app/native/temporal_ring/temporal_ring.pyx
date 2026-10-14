# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False

from libc.stdint cimport uint8_t
from libc.string cimport memcpy, memset
from libc.math cimport atan2, cos, sin, sqrt
from libcpp.vector cimport vector
cimport numpy as cnp
import numpy as np

cnp.import_array()


def amplify_prediction_errors_uniform(object current, object reference,
                                      double intensity, int error_threshold,
                                      object output=None):
    """Exact RGB8 uniform prediction-error kernel without temporary frame arrays."""
    cdef cnp.ndarray current_array = np.asarray(current)
    cdef cnp.ndarray reference_array = np.asarray(reference)
    if current_array.dtype != np.uint8 or reference_array.dtype != np.uint8:
        raise TypeError("current and reference frames must be uint8")
    if current_array.ndim != 3 or reference_array.ndim != 3:
        raise TypeError("current and reference frames must be three-dimensional")
    if (
        current_array.shape[0] != reference_array.shape[0]
        or current_array.shape[1] != reference_array.shape[1]
        or current_array.shape[2] != reference_array.shape[2]
    ):
        raise ValueError("current and reference frame shapes must match")
    if not current_array.flags.c_contiguous or not reference_array.flags.c_contiguous:
        raise ValueError("current and reference frames must be C-contiguous")
    cdef cnp.ndarray output_array
    if output is None:
        output_array = np.empty_like(current_array)
    else:
        output_array = np.asarray(output)
        if (
            output_array.dtype != np.uint8
            or output_array.ndim != 3
            or output_array.shape[0] != current_array.shape[0]
            or output_array.shape[1] != current_array.shape[1]
            or output_array.shape[2] != current_array.shape[2]
            or not output_array.flags.c_contiguous
        ):
            raise ValueError("output must be a matching contiguous uint8 frame")
    cdef Py_ssize_t length = current_array.size
    cdef const uint8_t* current_ptr = <const uint8_t*>current_array.data
    cdef const uint8_t* reference_ptr = <const uint8_t*>reference_array.data
    cdef uint8_t* output_ptr = <uint8_t*>output_array.data
    cdef double factor = 1.0 + intensity * 3.0
    cdef double residual
    cdef double value
    cdef Py_ssize_t index
    with nogil:
        for index in range(length):
            residual = <double>current_ptr[index] - <double>reference_ptr[index]
            if residual > error_threshold or residual < -error_threshold:
                value = <double>reference_ptr[index] + residual * factor
                if value < 0.0:
                    value = 0.0
                elif value > 255.0:
                    value = 255.0
                output_ptr[index] = <uint8_t>value
            else:
                output_ptr[index] = current_ptr[index]
    return output_array


def process_pframe_spiral_uniform(object current, object reference,
                                  double motion_intensity, int block_size,
                                  object random_values,
                                  double prediction_intensity, int error_threshold,
                                  object output=None):
    """Fused RGB8 spiral-motion and uniform prediction-error segment."""
    cdef cnp.ndarray current_array = np.asarray(current)
    cdef cnp.ndarray reference_array = np.asarray(reference)
    cdef cnp.ndarray random_array = np.asarray(random_values, dtype=np.float64)
    if current_array.dtype != np.uint8 or reference_array.dtype != np.uint8:
        raise TypeError("current and reference frames must be uint8")
    if current_array.ndim != 3 or reference_array.ndim != 3:
        raise TypeError("current and reference frames must be three-dimensional")
    if (
        current_array.shape[0] != reference_array.shape[0]
        or current_array.shape[1] != reference_array.shape[1]
        or current_array.shape[2] != reference_array.shape[2]
        or current_array.shape[2] != 3
    ):
        raise ValueError("current and reference must be matching RGB frames")
    if not current_array.flags.c_contiguous or not reference_array.flags.c_contiguous:
        raise ValueError("current and reference frames must be C-contiguous")
    if random_array.ndim != 1 or not random_array.flags.c_contiguous:
        raise ValueError("random_values must be a contiguous one-dimensional array")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    cdef Py_ssize_t height = current_array.shape[0]
    cdef Py_ssize_t width = current_array.shape[1]
    cdef Py_ssize_t channels = current_array.shape[2]
    cdef Py_ssize_t blocks_x = 0 if width <= block_size else (width - 1) // block_size
    cdef Py_ssize_t blocks_y = 0 if height <= block_size else (height - 1) // block_size
    cdef Py_ssize_t block_count = blocks_x * blocks_y
    if random_array.size != block_count:
        raise ValueError(f"expected {block_count} random values, got {random_array.size}")

    cdef cnp.ndarray output_array
    if output is None:
        output_array = np.empty_like(current_array)
    else:
        output_array = np.asarray(output)
        if (
            output_array.dtype != np.uint8
            or output_array.ndim != 3
            or output_array.shape[0] != height
            or output_array.shape[1] != width
            or output_array.shape[2] != channels
            or not output_array.flags.c_contiguous
        ):
            raise ValueError("output must be a matching contiguous uint8 frame")

    cdef const uint8_t* current_ptr = <const uint8_t*>current_array.data
    cdef const uint8_t* reference_ptr = <const uint8_t*>reference_array.data
    cdef uint8_t* output_ptr = <uint8_t*>output_array.data
    cdef const double* random_ptr = <const double*>random_array.data
    cdef Py_ssize_t frame_bytes = current_array.size
    cdef Py_ssize_t row_bytes = width * channels
    cdef Py_ssize_t block_row_bytes = block_size * channels
    cdef Py_ssize_t x, y, row, source_x, source_y, block_index = 0
    cdef Py_ssize_t source_offset, destination_offset, index
    cdef double center_x = <double>(width // 2)
    cdef double center_y = <double>(height // 2)
    cdef double block_center_x, block_center_y, distance, angle, spiral_angle
    cdef double displacement = <double>(<int>(block_size * motion_intensity * 2.0)) * 0.5
    cdef int dx, dy
    cdef double residual, value
    cdef double prediction_factor = 1.0 + prediction_intensity * 3.0

    with nogil:
        if output_ptr != current_ptr:
            memcpy(output_ptr, current_ptr, <size_t>frame_bytes)
        y = 0
        while y < height - block_size:
            x = 0
            while x < width - block_size:
                if random_ptr[block_index] < motion_intensity:
                    block_center_x = <double>(x + block_size // 2)
                    block_center_y = <double>(y + block_size // 2)
                    distance = sqrt(
                        (block_center_x - center_x) * (block_center_x - center_x)
                        + (block_center_y - center_y) * (block_center_y - center_y)
                    )
                    angle = atan2(block_center_y - center_y, block_center_x - center_x)
                    spiral_angle = angle + distance * 0.01 * motion_intensity * 10.0
                    dx = <int>(displacement * cos(spiral_angle))
                    dy = <int>(displacement * sin(spiral_angle))
                    source_x = x + dx
                    source_y = y + dy
                    if source_x < 0:
                        source_x = 0
                    elif source_x > width - block_size:
                        source_x = width - block_size
                    if source_y < 0:
                        source_y = 0
                    elif source_y > height - block_size:
                        source_y = height - block_size
                    for row in range(block_size):
                        source_offset = (source_y + row) * row_bytes + source_x * channels
                        destination_offset = (y + row) * row_bytes + x * channels
                        memcpy(output_ptr + destination_offset,
                               reference_ptr + source_offset,
                               <size_t>block_row_bytes)
                block_index += 1
                x += block_size
            y += block_size

        for index in range(frame_bytes):
            residual = <double>output_ptr[index] - <double>reference_ptr[index]
            if residual > error_threshold or residual < -error_threshold:
                value = <double>reference_ptr[index] + residual * prediction_factor
                if value < 0.0:
                    value = 0.0
                elif value > 255.0:
                    value = 255.0
                output_ptr[index] = <uint8_t>value
    return output_array


cdef class NativeTemporalRingCore:
    cdef vector[uint8_t] storage
    cdef Py_ssize_t width
    cdef Py_ssize_t height
    cdef Py_ssize_t channels
    cdef Py_ssize_t capacity
    cdef Py_ssize_t frame_bytes
    cdef Py_ssize_t count
    cdef Py_ssize_t write_index
    cdef Py_ssize_t delay
    cdef double mix
    cdef bint frozen
    cdef bint configured
    cdef unsigned long long writes
    cdef unsigned long long reads
    cdef unsigned long long mixes
    cdef unsigned long long frozen_writes
    cdef unsigned long long bytes_copied

    def __cinit__(self):
        self.configured = False
        self.mix = 0.5

    def configure(self, int width, int height, int channels, int capacity):
        if width <= 0 or height <= 0 or channels <= 0 or capacity <= 0:
            raise ValueError("width, height, channels, and capacity must be positive")
        cdef object total_value = width
        total_value *= height
        total_value *= channels
        total_value *= capacity
        cdef size_t maximum_size = <size_t>-1
        if total_value > maximum_size:
            raise OverflowError("requested temporal allocation exceeds addressable size")
        cdef size_t total = total_value
        self.storage = vector[uint8_t](total, 0)
        self.width = width
        self.height = height
        self.channels = channels
        self.capacity = capacity
        self.frame_bytes = width * height * channels
        self.count = 0
        self.write_index = 0
        self.delay = 0
        self.frozen = False
        self.configured = True
        self.writes = 0
        self.reads = 0
        self.mixes = 0
        self.frozen_writes = 0
        self.bytes_copied = 0

    cdef void _require_configured(self) except *:
        if not self.configured:
            raise RuntimeError("temporal ring is not configured")

    cdef Py_ssize_t _slot_for_delay(self, Py_ssize_t delay) except -1:
        self._require_configured()
        if delay < 0 or delay >= self.count:
            raise ValueError(f"delay {delay} is outside written history [0, {self.count - 1}]")
        return (self.write_index + self.capacity - 1 - delay) % self.capacity

    cdef cnp.ndarray _frame_array(self, object frame):
        cdef cnp.ndarray result = np.asarray(frame)
        if result.dtype != np.uint8 or result.ndim != 3:
            raise TypeError("frame must be a three-dimensional uint8 array")
        if result.shape[0] != self.height or result.shape[1] != self.width or result.shape[2] != self.channels:
            raise ValueError(
                f"expected frame shape {(self.height, self.width, self.channels)}, got "
                f"{(result.shape[0], result.shape[1], result.shape[2])}"
            )
        if not result.flags.c_contiguous:
            raise ValueError("frame must be C-contiguous")
        return result

    def write_frame(self, object frame):
        self._require_configured()
        if self.frozen:
            self.frozen_writes += 1
            return False
        cdef cnp.ndarray array = self._frame_array(frame)
        cdef Py_ssize_t offset = self.write_index * self.frame_bytes
        cdef uint8_t* destination = &self.storage[0] + offset
        cdef const uint8_t* source = <const uint8_t*>array.data
        cdef size_t length = <size_t>self.frame_bytes
        with nogil:
            memcpy(destination, source, length)
        self.write_index = (self.write_index + 1) % self.capacity
        if self.count < self.capacity:
            self.count += 1
        self.writes += 1
        self.bytes_copied += self.frame_bytes
        return True

    def read_frame(self, delay=None):
        cdef Py_ssize_t requested = self.delay if delay is None else int(delay)
        cdef Py_ssize_t slot = self._slot_for_delay(requested)
        cdef cnp.ndarray output = np.empty((self.height, self.width, self.channels), dtype=np.uint8)
        cdef const uint8_t* source = &self.storage[0] + slot * self.frame_bytes
        cdef uint8_t* destination = <uint8_t*>output.data
        cdef size_t length = <size_t>self.frame_bytes
        with nogil:
            memcpy(destination, source, length)
        self.reads += 1
        self.bytes_copied += self.frame_bytes
        return output

    def process_temporal_mix(self, object current, delay=None, mix=None, object output=None):
        cdef cnp.ndarray current_array = self._frame_array(current)
        cdef Py_ssize_t requested = self.delay if delay is None else int(delay)
        cdef Py_ssize_t slot = self._slot_for_delay(requested)
        cdef float amount = <float>(self.mix if mix is None else float(mix))
        if amount < 0.0 or amount > 1.0:
            raise ValueError("mix must be between 0 and 1")
        cdef cnp.ndarray output_array
        if output is None:
            output_array = np.empty((self.height, self.width, self.channels), dtype=np.uint8)
        else:
            output_array = self._frame_array(output)
        cdef const uint8_t* current_ptr = <const uint8_t*>current_array.data
        cdef const uint8_t* history_ptr = &self.storage[0] + slot * self.frame_bytes
        cdef uint8_t* output_ptr = <uint8_t*>output_array.data
        cdef Py_ssize_t index
        cdef float inverse = 1.0 - amount
        cdef float value
        with nogil:
            for index in range(self.frame_bytes):
                value = current_ptr[index] * inverse + history_ptr[index] * amount
                output_ptr[index] = <uint8_t>(value + 0.5)
        self.mixes += 1
        self.bytes_copied += self.frame_bytes
        return output_array

    def set_delay(self, int delay):
        if delay < 0:
            raise ValueError("delay must be non-negative")
        self.delay = delay

    def set_mix(self, double mix):
        if mix < 0.0 or mix > 1.0:
            raise ValueError("mix must be between 0 and 1")
        self.mix = mix

    def freeze(self, enabled=True):
        self._require_configured()
        self.frozen = bool(enabled)

    def clear(self):
        self._require_configured()
        cdef size_t length = self.storage.size()
        cdef uint8_t* destination = &self.storage[0]
        with nogil:
            memset(destination, 0, length)
        self.count = 0
        self.write_index = 0

    def get_statistics(self):
        return {
            "backend": "native_cpp",
            "configured": bool(self.configured),
            "width": self.width if self.configured else 0,
            "height": self.height if self.configured else 0,
            "channels": self.channels if self.configured else 0,
            "capacity": self.capacity if self.configured else 0,
            "count": self.count if self.configured else 0,
            "write_index": self.write_index if self.configured else 0,
            "allocated_bytes": self.storage.size(),
            "writes": self.writes,
            "reads": self.reads,
            "mixes": self.mixes,
            "frozen_writes": self.frozen_writes,
            "bytes_copied": self.bytes_copied,
            "frozen": bool(self.frozen),
        }

    def shutdown(self):
        vector[uint8_t]().swap(self.storage)
        self.configured = False
        self.count = 0
        self.write_index = 0
        self.frozen = False
