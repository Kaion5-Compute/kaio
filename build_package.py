#!/usr/bin/env python3
"""Build script for the kaio package."""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(f"✓ {description} completed successfully")
    return True

def main():
    """Build the package."""
    # Ensure we're in the right directory
    os.chdir(Path(__file__).parent)
    
    # Clean previous builds
    print("Cleaning previous builds...")
    for dir_name in ["build", "dist", "kaio.egg-info"]:
        if os.path.exists(dir_name):
            import shutil
            shutil.rmtree(dir_name)
            print(f"✓ Removed {dir_name}")
    
    # Install build dependencies
    if not run_command(f"{sys.executable} -m pip install --upgrade build twine", "Installing build tools"):
        return False
    
    # Build the package
    if not run_command(f"{sys.executable} -m build", "Building package"):
        return False
    
    # Check the package
    if not run_command(f"{sys.executable} -m twine check dist/*", "Checking package"):
        return False
    
    print("\n✓ Package built successfully!")
    print("Files created:")
    dist_path = Path("dist")
    if dist_path.exists():
        for file in dist_path.iterdir():
            print(f"  - {file}")
    
    print("\nTo upload to PyPI:")
    print("  python -m twine upload dist/*")
    print("\nTo test locally:")
    print("  pip install dist/kaio-0.1.0-py3-none-any.whl")

if __name__ == "__main__":
    main()