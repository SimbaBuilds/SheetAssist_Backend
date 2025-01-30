import psutil
import logging
from functools import wraps
from fastapi import Request
from typing import Callable
import time

logger = logging.getLogger(__name__)

class MemoryTracker:
    def __init__(self):
        self.peak_memory = 0
        self.start_memory = 0
        self.process = psutil.Process()
    
    def update_peak(self, current):
        self.peak_memory = max(self.peak_memory, current)
        return self.peak_memory
    
    def get_memory(self):
        """Get current memory usage in MB"""
        current = self.process.memory_info().rss / 1024 / 1024
        return current, self.update_peak(current)

class MemoryProfilerMiddleware:
    def __init__(self, app):
        self.app = app
        self.trackers = {}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        tracker = MemoryTracker()
        path = scope.get("path", "unknown")
        self.trackers[path] = tracker
        
        # Record start memory
        start_memory, _ = tracker.get_memory()
        start_time = time.time()
        
        # Execute request
        await self.app(scope, receive, send)
        
        # Record end memory
        end_memory, peak_memory = tracker.get_memory()
        end_time = time.time()
        
        # Calculate memory differences and duration
        memory_diff = end_memory - start_memory
        duration = end_time - start_time
        
        # Log memory usage with peak
        logger.info(
            f"Memory Profile - Endpoint: {path}\n"
            f"Start Memory: {start_memory:.2f}MB\n"
            f"End Memory: {end_memory:.2f}MB\n"
            f"Peak Memory: {peak_memory:.2f}MB\n"
            f"Memory Change: {memory_diff:.2f}MB\n"
            f"Duration: {duration:.2f}s"
        )
        
        # Cleanup
        del self.trackers[path]

def profile_memory(func: Callable):
    """Decorator to profile memory usage of specific functions"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        tracker = MemoryTracker()
        start_memory, _ = tracker.get_memory()
        start_time = time.time()
        
        result = await func(*args, **kwargs)
        
        end_memory, peak_memory = tracker.get_memory()
        end_time = time.time()
        
        memory_diff = end_memory - start_memory
        duration = end_time - start_time
        
        logger.info(
            f"Memory Profile - Function: {func.__name__}\n"
            f"Start Memory: {start_memory:.2f}MB\n"
            f"End Memory: {end_memory:.2f}MB\n"
            f"Peak Memory: {peak_memory:.2f}MB\n"
            f"Memory Change: {memory_diff:.2f}MB\n"
            f"Duration: {duration:.2f}s"
        )
        
        return result
    
    return wrapper 