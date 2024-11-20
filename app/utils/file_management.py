import os
import shutil
from pathlib import Path
import tempfile
import atexit
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TempFileManager:
    def __init__(self, base_dir: str = None, max_age_hours: int = 24):
        """Initialize the temporary file manager
        
        Args:
            base_dir: Base directory for temp files. If None, uses system temp directory
            max_age_hours: Maximum age of temp files before cleanup (in hours)
        """
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            system_temp = Path(tempfile.gettempdir())
            self.base_dir = system_temp / "data_processor_temp"
        
        self.max_age = timedelta(hours=max_age_hours)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Register cleanup on program exit
        atexit.register(self.cleanup_old_files)
        
        self._pending_cleanup = set()
    
    def get_temp_dir(self) -> Path:
        """Create a new temporary directory for the current session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.base_dir / f"session_{timestamp}"
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir
    
    def cleanup_old_files(self):
        """Remove temporary files/directories older than max_age"""
        try:
            current_time = datetime.now()
            
            for item in self.base_dir.glob("session_*"):
                try:
                    # Get creation time from directory name
                    timestamp_str = item.name.replace("session_", "")
                    created_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    
                    if current_time - created_time > self.max_age:
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                        logger.info(f"Cleaned up old temp files: {item}")
                except Exception as e:
                    logger.error(f"Error cleaning up {item}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def save_temp_file(self, file_data, filename: str, session_dir: Path = None) -> Path:
        """Save a temporary file and return its path
        
        Args:
            file_data: File data to save
            filename: Name for the saved file
            session_dir: Optional specific session directory to use
            
        Returns:
            Path: Path to the saved temporary file
        """
        if session_dir is None:
            session_dir = self.get_temp_dir()
            
        file_path = session_dir / filename
        
        # Handle different types of file data
        if hasattr(file_data, 'read'):
            # For file-like objects
            with open(file_path, 'wb') as f:
                f.write(file_data.read())
        elif isinstance(file_data, (str, bytes)):
            # For string or bytes content
            mode = 'wb' if isinstance(file_data, bytes) else 'w'
            with open(file_path, mode) as f:
                f.write(file_data)
        else:
            raise ValueError(f"Unsupported file data type: {type(file_data)}")
            
        return file_path
    
    def mark_for_cleanup(self, *paths: Path) -> None:
        """Mark files or directories for cleanup after they're no longer needed
        
        Args:
            *paths: Paths to mark for cleanup
        """
        for path in paths:
            self._pending_cleanup.add(Path(path))
    
    def cleanup_marked(self) -> None:
        """Clean up all marked files and directories"""
        while self._pending_cleanup:
            path = self._pending_cleanup.pop()
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
                logger.info(f"Cleaned up marked path: {path}")
            except Exception as e:
                logger.error(f"Error cleaning up marked path {path}: {str(e)}")

# Global instance
temp_file_manager = TempFileManager() 