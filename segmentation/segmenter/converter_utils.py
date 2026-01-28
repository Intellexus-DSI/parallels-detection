"""Utility functions for handling detect_and_convert submodule imports."""

import subprocess
import sys
from pathlib import Path


def run_setup_script():
    """Run setup_submodule.py to initialize the submodule."""
    project_root = Path(__file__).parent.parent.parent
    setup_script = project_root / "setup_submodule.py"
    
    if not setup_script.exists():
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(setup_script)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        return result.returncode == 0
    except Exception:
        return False


def setup_converter_path():
    """
    Add detect_and_convert submodule to Python path if it exists.
    
    This allows imports like 'from conversion import Converter' to work
    when detect_and_convert is included as a Git submodule.
    
    If submodule is not found, automatically runs setup_submodule.py.
    
    Returns:
        bool: True if path was found and added, False otherwise
    """
    # Get the project root (assumes this file is in segmentation/segmenter/)
    project_root = Path(__file__).parent.parent.parent
    
    # Check for submodule in project root
    submodule_path = project_root / "detect_and_convert"
    
    # Check if submodule exists and has content
    if submodule_path.exists() and submodule_path.is_dir():
        # Check if it has actual content (not just empty directory)
        try:
            if any(submodule_path.iterdir()):
                submodule_str = str(submodule_path)
                if submodule_str not in sys.path:
                    sys.path.insert(0, submodule_str)
                return True
        except Exception:
            pass
    
    # Submodule not found or empty - try to set it up automatically
    print("detect_and_convert submodule not found. Running setup script...")
    if run_setup_script():
        # Try again after setup
        if submodule_path.exists() and submodule_path.is_dir():
            try:
                if any(submodule_path.iterdir()):
                    submodule_str = str(submodule_path)
                    if submodule_str not in sys.path:
                        sys.path.insert(0, submodule_str)
                    return True
            except Exception:
                pass
    
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
