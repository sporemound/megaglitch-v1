"""
MegaGlitch Performance Profiler
Lightweight profiling for hot-path optimization
"""
import time
from collections import defaultdict, deque
from typing import Dict, Deque
import logging


class PerformanceProfiler:
    """Lightweight profiler for tracking per-stage timings."""
    
    def __init__(self, window_size: int = 120):
        self.window_size = window_size
        self.stage_times: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window_size))
        self.frame_times: Deque[float] = deque(maxlen=window_size)
        self.frame_count = 0
        self.last_report_frame = 0
        self.report_interval = window_size
        
    def time_stage(self, stage_name: str):
        """Context manager for timing a stage."""
        return _StageTimer(self, stage_name)
    
    def record_frame_time(self, frame_time: float):
        """Record total frame processing time."""
        self.frame_times.append(frame_time)
        self.frame_count += 1
        
        # Auto-report every N frames
        if self.frame_count - self.last_report_frame >= self.report_interval:
            self.print_report()
            self.last_report_frame = self.frame_count
    
    def record_stage(self, stage_name: str, elapsed: float):
        """Record elapsed time for a stage."""
        self.stage_times[stage_name].append(elapsed)
    
    def get_avg(self, stage_name: str) -> float:
        """Get average time for a stage in ms."""
        times = self.stage_times.get(stage_name)
        if not times:
            return 0.0
        return (sum(times) / len(times)) * 1000.0  # Convert to ms
    
    def get_max(self, stage_name: str) -> float:
        """Get max time for a stage in ms."""
        times = self.stage_times.get(stage_name)
        if not times:
            return 0.0
        return max(times) * 1000.0
    
    def get_frame_avg(self) -> float:
        """Get average frame time in ms."""
        if not self.frame_times:
            return 0.0
        return (sum(self.frame_times) / len(self.frame_times)) * 1000.0
    
    def get_frame_max(self) -> float:
        """Get max frame time in ms."""
        if not self.frame_times:
            return 0.0
        return max(self.frame_times) * 1000.0
    
    def get_fps(self) -> float:
        """Get average FPS."""
        avg_frame = self.get_frame_avg()
        if avg_frame == 0:
            return 0.0
        return 1000.0 / avg_frame
    
    def print_report(self):
        """Print performance report."""
        if not self.frame_times:
            return
        
        logging.info("=" * 80)
        logging.info(f"PERFORMANCE REPORT (last {len(self.frame_times)} frames)")
        logging.info("=" * 80)
        
        # Frame stats
        avg_frame = self.get_frame_avg()
        max_frame = self.get_frame_max()
        fps = self.get_fps()
        
        logging.info(f"Frame Time:  avg={avg_frame:.2f}ms  max={max_frame:.2f}ms")
        logging.info(f"FPS:         {fps:.1f}")
        
        # Stage breakdown
        if self.stage_times:
            logging.info("\nStage Breakdown:")
            total_stage_time = sum(self.get_avg(stage) for stage in self.stage_times.keys())
            
            # Sort by average time
            sorted_stages = sorted(
                self.stage_times.keys(),
                key=lambda s: self.get_avg(s),
                reverse=True
            )
            
            for stage in sorted_stages:
                avg = self.get_avg(stage)
                max_time = self.get_max(stage)
                pct = (avg / avg_frame * 100.0) if avg_frame > 0 else 0
                logging.info(f"  {stage:30s} avg={avg:6.2f}ms  max={max_time:6.2f}ms  ({pct:4.1f}%)")
            
            logging.info(f"\nTotal tracked: {total_stage_time:.2f}ms ({total_stage_time/avg_frame*100:.1f}%)")
        
        logging.info("=" * 80)
    
    def reset(self):
        """Clear all timing data."""
        self.stage_times.clear()
        self.frame_times.clear()
        self.frame_count = 0
        self.last_report_frame = 0


class _StageTimer:
    """Context manager for timing stages."""
    
    def __init__(self, profiler: PerformanceProfiler, stage_name: str):
        self.profiler = profiler
        self.stage_name = stage_name
        self.start_time = 0.0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        self.profiler.record_stage(self.stage_name, elapsed)
        return False


# Global profiler instance
_global_profiler = PerformanceProfiler(window_size=120)


def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    return _global_profiler






