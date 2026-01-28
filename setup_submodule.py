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
    
    print("Setting up detect_and_convert submodule...")
    
    # Check if .gitmodules exists
    if not gitmodules_path.exists():
        print("Error: .gitmodules file not found")
        print("The submodule configuration is missing.")
        sys.exit(1)
    
    # Check if submodule directory exists and has content
    if not submodule_path.exists() or not any(submodule_path.iterdir()):
        print("Submodule directory not found or empty. Initializing...")
        
        # Check if .gitmodules is uncommitted (first-time setup)
        check_result = subprocess.run(
            ["git", "status", "--porcelain", ".gitmodules"],
            capture_output=True,
            text=True
        )
        if check_result.returncode == 0 and check_result.stdout.strip():
            print("\n[INFO] .gitmodules file exists but is not committed yet.")
            print("For first-time setup, you need to commit .gitmodules first:")
            print("  git add .gitmodules")
            print("  git commit -m 'Add detect_and_convert submodule'")
            print("  python setup_submodule.py")
            print("\nAlternatively, you can manually add the submodule:")
            print("  git submodule add https://github.com/Intellexus-DSI/detect_and_convert detect_and_convert")
            sys.exit(1)
        
        # Try to initialize the submodule
        result = subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            capture_output=True,
            text=True
        )
        
        # Check again if directory was created
        if not submodule_path.exists() or not any(submodule_path.iterdir()):
            error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
            if error_msg:
                print(f"Error: {error_msg}")
            else:
                print("Warning: git submodule command succeeded but directory was not created.")
            print("\nTroubleshooting:")
            print("1. Make sure .gitmodules is committed: git add .gitmodules && git commit")
            print("2. Or manually add: git submodule add https://github.com/Intellexus-DSI/detect_and_convert detect_and_convert")
            sys.exit(1)
        
        print("[OK] Submodule initialized successfully!")
    
    # Install detect_and_convert
    if submodule_path.exists():
        setup_py = submodule_path / "setup.py"
        if setup_py.exists():
            print("Installing detect_and_convert package...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", str(submodule_path)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"Error installing detect_and_convert: {result.stderr}", file=sys.stderr)
                sys.exit(1)
            print("[OK] detect_and_convert installed successfully!")
            print("\nSetup complete! The converter is ready to use.")
        else:
            print("Warning: detect_and_convert setup.py not found")
            print(f"Checked path: {setup_py}")
            sys.exit(1)
    else:
        print("Error: detect_and_convert submodule directory still not found after initialization")
        print("Make sure .gitmodules is committed and try again")
        sys.exit(1)


if __name__ == "__main__":
    main()
