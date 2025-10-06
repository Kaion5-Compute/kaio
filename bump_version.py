#!/usr/bin/env python3
import re
import sys
import toml
from pathlib import Path

def bump_version(version_type):
    """Bump version in setup.py, __init__.py, and pyproject.toml"""
    
    # Read current version from setup.py
    setup_py = Path("setup.py")
    if not setup_py.exists():
        print("setup.py not found")
        return
    
    content = setup_py.read_text()
    version_match = re.search(r'version="(\d+)\.(\d+)\.(\d+)"', content)
    if not version_match:
        print("Could not find version in setup.py")
        return
    
    major, minor, patch = map(int, version_match.groups())
    
    # Bump version
    if version_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif version_type == "minor":
        minor += 1
        patch = 0
    elif version_type == "patch":
        patch += 1
    else:
        print("Invalid version type. Use: major, minor, or patch")
        return
    
    new_version = f"{major}.{minor}.{patch}"
    print(f"Bumping version to {new_version}")
    
    # Update setup.py
    new_content = re.sub(
        r'version="\d+\.\d+\.\d+"',
        f'version="{new_version}"',
        content
    )
    setup_py.write_text(new_content)
    print(f"✓ Updated setup.py")
    
    # Update __init__.py
    init_py = Path("__init__.py")
    if init_py.exists():
        init_content = init_py.read_text()
        new_init_content = re.sub(
            r'__version__ = "\d+\.\d+\.\d+"',
            f'__version__ = "{new_version}"',
            init_content
        )
        init_py.write_text(new_init_content)
        print(f"✓ Updated __init__.py")
    
    # Update pyproject.toml if it exists
    pyproject_toml = Path("pyproject.toml")
    if pyproject_toml.exists():
        try:
            pyproject_data = toml.load(pyproject_toml)
            if "project" in pyproject_data and "version" in pyproject_data["project"]:
                pyproject_data["project"]["version"] = new_version
                with open(pyproject_toml, "w") as f:
                    toml.dump(pyproject_data, f)
                print(f"✓ Updated pyproject.toml")
        except Exception as e:
            print(f"Warning: Could not update pyproject.toml: {e}")
    
    print(f"Updated version to {new_version}")
    return new_version

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bump_version.py [major|minor|patch]")
        sys.exit(1)
    
    version_type = sys.argv[1]
    new_version = bump_version(version_type)
    
    if new_version:
        print(f"\nNext steps:")
        print(f"1. git add .")
        print(f"2. git commit -m 'Bump version to {new_version}'")
        print(f"3. git tag v{new_version}")
        print(f"4. git push origin main --tags")