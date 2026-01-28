"""Utility functions for handling detect_and_convert submodule imports."""

import subprocess
import sys
from pathlib import Path


def run_setup_script():
    """Run setup_submodule.py to initialize the submodule."""
    project_root = Path(__file__).parent.parent.parent
    setup_script = project_root / "setup_submodule.py"
    
    if not setup_script.exists():
        print(f"[ERROR] Setup script not found at: {setup_script}")
        return False
    
    try:
        print(f"[INFO] Running setup script: {setup_script}")
        result = subprocess.run(
            [sys.executable, str(setup_script)],
            cwd=project_root,
            capture_output=False,  # Show output in real-time
            text=True,
            timeout=600  # 10 minute timeout for network operations
        )
        if result.returncode == 0:
            print("[OK] Setup script completed successfully")
            return True
        else:
            print(f"[WARNING] Setup script exited with code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print("[ERROR] Setup script timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"[ERROR] Setup script failed with exception: {e}")
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
    print("\n" + "="*60)
    print("detect_and_convert submodule not found.")
    print("Attempting automatic setup...")
    print("="*60)
    
    if run_setup_script():
        # Try again after setup
        if submodule_path.exists() and submodule_path.is_dir():
            try:
                if any(submodule_path.iterdir()):
                    submodule_str = str(submodule_path)
                    if submodule_str not in sys.path:
                        sys.path.insert(0, submodule_str)
                    print(f"[OK] Submodule path added: {submodule_str}")
                    return True
            except Exception as e:
                print(f"[WARNING] Error checking submodule after setup: {e}")
    
    print("[ERROR] Automatic setup failed or submodule still not found")
    print("\nPlease try manually:")
    print("  python setup_submodule.py")
    print("Or:")
    print("  git clone https://github.com/Intellexus-DSI/detect_and_convert detect_and_convert")
    print("  cd detect_and_convert && pip install -e .")
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
