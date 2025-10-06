# PyPI Deployment Guide for Kaio Package

## Package Structure

Your repository is now structured as a proper Python package:

```
kaio/
├── __init__.py          # Package initialization, exports Client
├── client.py            # Main Client class
├── image_resolver.py    # Image resolution utilities
├── setup.py            # Package setup configuration
├── pyproject.toml      # Modern Python packaging config
├── README.md           # Package documentation
├── LICENSE             # MIT license
├── MANIFEST.in         # Files to include in distribution
├── build_package.py    # Build automation script
└── test_import.py      # Import test script
```

## Installation and Usage

Once published to PyPI, users can install and use your package like this:

```bash
pip install kaio
```

```python
from kaio import Client

client = Client("https://api.kaion5.com")
client.login("your-api-key")

result = client.submit_job(
    directory="./my_code",
    job_name="training-job",
    instance_type="ml.g4dn.xlarge",
    entrypoint="train.py"
)
```

## Building the Package

1. **Install build tools:**
   ```bash
   pip install --upgrade build twine
   ```

2. **Build the package:**
   ```bash
   python -m build
   ```

3. **Check the package:**
   ```bash
   python -m twine check dist/*
   ```

## Publishing to PyPI

### Test PyPI (Recommended first)

1. **Create account at https://test.pypi.org**

2. **Upload to Test PyPI:**
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```

3. **Test installation:**
   ```bash
   pip install --index-url https://test.pypi.org/simple/ kaio
   ```

### Production PyPI

1. **Create account at https://pypi.org**

2. **Upload to PyPI:**
   ```bash
   python -m twine upload dist/*
   ```

3. **Test installation:**
   ```bash
   pip install kaio
   ```

## Automated Build Script

Use the provided build script:

```bash
python build_package.py
```

This script will:
- Clean previous builds
- Install build dependencies
- Build the package
- Check the package
- Show next steps

## Version Management

To release a new version:

1. Update version in `setup.py` and `pyproject.toml`
2. Update version in `__init__.py`
3. Update `CHANGELOG.md` (if you create one)
4. Build and upload new version

## Package Features

- ✅ Proper package structure with `__init__.py`
- ✅ Clean import: `from kaio import Client`
- ✅ Modern packaging with `pyproject.toml`
- ✅ Comprehensive documentation
- ✅ MIT license
- ✅ Dependency management
- ✅ Build automation
- ✅ Import testing

## Next Steps

1. Test the package locally
2. Upload to Test PyPI first
3. Test installation from Test PyPI
4. Upload to production PyPI
5. Update documentation with installation instructions