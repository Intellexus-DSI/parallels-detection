"""Utility functions for handling detect_and_convert submodule imports."""

import sys
from pathlib import Path


def setup_converter_path():
    """
    Add detect_and_convert submodule to Python path if it exists.
    
    This allows imports like 'from conversion import Converter' to work
    when detect_and_convert is included as a Git submodule.
    
    Returns:
        bool: True if path was found and added, False otherwise
    """
    # Get the project root (assumes this file is in segmentation/segmenter/)
    project_root = Path(__file__).parent.parent.parent
    
    # Check for submodule in project root
    submodule_path = project_root / "detect_and_convert"
    
    if submodule_path.exists() and submodule_path.is_dir():
        submodule_str = str(submodule_path)
        if submodule_str not in sys.path:
            sys.path.insert(0, submodule_str)
        return True
    
    return False


def get_converter():
    """
    Attempt to import and return Converter class.
    
    Returns:
        Converter class if available, None otherwise
    """
    # Setup path first
    setup_converter_path()
    
    try:
        from conversion import Converter
        return Converter
    except ImportError:
        return None
