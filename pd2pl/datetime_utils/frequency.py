"""Frequency parsing and mapping utilities for pandas to polars translation."""

import re
from typing import Tuple

def parse_frequency(freq: str) -> Tuple[str, int]:
    """
    Parse a pandas frequency string into its base frequency and multiplier.
    
    Args:
        freq: A pandas frequency string (e.g., 'D', '2D', 'W', '3W', 'ME', 'QE')
        
    Returns:
        A tuple of (base_frequency, multiplier)
        
    Raises:
        ValueError: If the frequency string is invalid
    """
    if not freq:
        raise ValueError("Frequency string cannot be empty")
        
    # Match pattern: optional number followed by one or more letters
    match = re.match(r'^(\d*)([A-Za-z]+)$', freq)
    if not match:
        raise ValueError(f"Invalid frequency string: {freq}")
        
    multiplier_str, base_freq = match.groups()
    
    # Handle multiplier validation
    if not multiplier_str:
        multiplier = 1
    else:
        multiplier = int(multiplier_str)
        if multiplier <= 0:
            raise ValueError(f"Frequency multiplier must be positive, got {multiplier}")
    
    # Verify that base_freq is a recognized frequency
    valid_freqs = {
        'D', 'W', 'M', 'ME', 'MS', 'Q', 'QE', 'QS', 'Y', 'YE', 'YS', 'H', 'T', 'S', 'L', 'U', 'N'
    }
    if base_freq.upper() not in valid_freqs:
        raise ValueError(f"Unrecognized frequency: {base_freq}")
        
    return base_freq.upper(), multiplier

def map_pandas_freq_to_polars_interval(freq: str) -> str:
    """
    Map a pandas frequency string to a polars interval string.
    
    Args:
        freq: A pandas frequency string (e.g., 'D', '2D', 'W', '3W', 'ME', 'QE')
        
    Returns:
        A polars interval string (e.g., '1d', '2d', '1w', '3w')
        
    Raises:
        ValueError: If the frequency string is invalid
    """
    base_freq, multiplier = parse_frequency(freq)
    
    # Map pandas frequency to polars interval
    freq_map = {
        'D': 'd',
        'W': 'w',
        'M': 'mo',
        'ME': 'mo',  # Month End
        'MS': 'mo',  # Month Start
        'Q': 'q',
        'QE': 'q',   # Quarter End
        'QS': 'q',   # Quarter Start
        'Y': 'y',
        'YE': 'y',   # Year End
        'YS': 'y',   # Year Start
        'H': 'h',
        'T': 'm',    # pandas uses 'T' for minutes
        'S': 's',
        'L': 'ms',   # pandas uses 'L' for milliseconds
        'U': 'us',   # pandas uses 'U' for microseconds
        'N': 'ns',   # pandas uses 'N' for nanoseconds
    }
    
    if base_freq not in freq_map:
        raise ValueError(f"Unsupported frequency: {base_freq}")
        
    return f"{multiplier}{freq_map[base_freq]}" 