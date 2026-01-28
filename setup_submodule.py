#!/usr/bin/env python
"""Setup script to initialize detect_and_convert submodule."""

import subprocess
import sys
from pathlib import Path


def main():
    """Initialize and install the detect_and_convert submodule."""
    project_root = Path(__file__).parent
    submodule_path = project_root / "detect_and_convert"
    gitmodules_path = project_root / ".gitmodules"
    
    # Repository URL - fallback if git submodule fails
    REPO_URL = "https://github.com/Intellexus-DSI/detect_and_convert.git"
    
    print("Setting up detect_and_convert submodule...")
    
    # Check if submodule directory already exists and has content
    if submodule_path.exists() and submodule_path.is_dir():
        try:
            if any(submodule_path.iterdir()):
                print("[OK] Submodule directory already exists with content.")
                # Still proceed to installation check
        except Exception:
            pass
    
    # If submodule doesn't exist or is empty, try to set it up
    if not submodule_path.exists() or not any(submodule_path.iterdir()):
        print("Submodule directory not found or empty. Initializing...")
        
        success = False
        
        # Method 1: Try git submodule (if .gitmodules exists)
        if gitmodules_path.exists():
            print("Attempting to initialize via git submodule...")
            result = subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=project_root,
                capture_output=True,
                text=True
            )
            
            if submodule_path.exists() and any(submodule_path.iterdir()):
                print("[OK] Submodule initialized via git submodule!")
                success = True
            elif result.returncode != 0:
                print(f"[INFO] git submodule failed: {result.stderr.strip() if result.stderr else result.stdout.strip()}")
        
        # Method 2: Fallback - clone directly if git submodule failed
        if not success:
            print("Attempting to clone repository directly...")
            try:
                # Remove empty directory if it exists
                if submodule_path.exists():
                    import shutil
                    shutil.rmtree(submodule_path)
                
                result = subprocess.run(
                    ["git", "clone", REPO_URL, str(submodule_path)],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0 and submodule_path.exists() and any(submodule_path.iterdir()):
                    print("[OK] Repository cloned successfully!")
                    success = True
                else:
                    error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
                    print(f"[ERROR] Clone failed: {error_msg}")
            except subprocess.TimeoutExpired:
                print("[ERROR] Clone operation timed out after 5 minutes")
            except Exception as e:
                print(f"[ERROR] Clone failed with exception: {e}")
        
        if not success:
            print("\n[ERROR] Failed to initialize detect_and_convert submodule")
            print("Please try manually:")
            print(f"  git clone {REPO_URL} detect_and_convert")
            sys.exit(1)
    
    # Install detect_and_convert
    if submodule_path.exists() and any(submodule_path.iterdir()):
        setup_py = submodule_path / "setup.py"
        pyproject_toml = submodule_path / "pyproject.toml"
        
        # Check for either setup.py or pyproject.toml
        if setup_py.exists() or pyproject_toml.exists():
            print("Installing detect_and_convert package...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", str(submodule_path)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
                print(f"[WARNING] Installation had issues: {error_msg}")
                print("[INFO] Continuing anyway - package may still work if already installed")
            else:
                print("[OK] detect_and_convert installed successfully!")
            print("\nSetup complete! The converter is ready to use.")
        else:
            print(f"[WARNING] No setup.py or pyproject.toml found in {submodule_path}")
            print("[INFO] The submodule may not require installation, or structure may be different")
            print("[INFO] Continuing anyway - will attempt to import directly")
    else:
        print("[ERROR] detect_and_convert submodule directory still not found after initialization")
        sys.exit(1)


if __name__ == "__main__":
    main()
