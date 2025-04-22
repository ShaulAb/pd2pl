"""Logging configuration for the pandas to polars translator."""
import sys
from pathlib import Path
from loguru import logger

def setup_logging(log_file: Path = None):
    """Configure logging for the application.
    
    Args:
        log_file: Optional path to log file. If not provided, logs will only go to stderr.
    """
    # Remove default handler
    logger.remove()
    
    # Add stderr handler with custom format
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        level="INFO"
    )
    
    # Add file handler if log_file is provided
    if log_file is not None:
        logger.add(
            log_file,
            rotation="10 MB",  # Rotate when file reaches 10MB
            retention="1 week",  # Keep logs for 1 week
            compression="zip",   # Compress rotated logs
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                   "{name}:{function}:{line} - {message}",
            level="DEBUG"       # More detailed logging to file
        ) 