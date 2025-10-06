# Kaio Python Client

A Python client for the Kaio multi-tenant machine learning platform that enables developers to run SageMaker jobs through simple APIs with automatic image resolution and secure file uploads.

## Installation

```bash
pip install kaio
```

## Quick Start

```python
from kaio import Client

# Initialize client
client = Client("https://api.kaion5.com")

# Login with your API key
client.login("your-api-key")

# Submit a job
result = client.submit_job(
    directory="./my_code",
    job_name="training-job",
    instance_type="ml.g4dn.xlarge",
    entrypoint="train.py"
)
```

## Features

- **Automatic Image Resolution**: Detects your local ML framework and selects appropriate Docker images
- **Secure File Uploads**: Handles code packaging and S3 uploads automatically
- **Job Management**: Submit, monitor, and download results from SageMaker jobs
- **Multi-Framework Support**: Works with PyTorch, TensorFlow, and Scikit-learn
- **GPU/CPU Instance Matching**: Automatically selects GPU or CPU optimized containers
- **Automatic Dependencies**: Adds required packages (nbconvert, psutil, GPUtil) to requirements.txt
- **JWT Token Management**: Handles authentication token refresh automatically

## API Reference

### Client

#### `Client(api_base, verbose=False)`

Initialize the Kaio client.

**Parameters:**
- `api_base` (str): Base URL of the Kaio API endpoint
- `verbose` (bool): Enable verbose logging for debugging. Defaults to False.

**Example:**
```python
client = Client("https://api.kaion5.com", verbose=True)
```

#### `login(api_key)`

Authenticate with API key and obtain JWT token.

**Parameters:**
- `api_key` (str): Your Kaio platform API key

**Returns:**
- `Client`: Self for method chaining

**Raises:**
- `requests.HTTPError`: If authentication fails

#### `submit_job(**kwargs)`

Submit a SageMaker job with automatic image resolution.

**Parameters:**
- `directory` (str): Path to code directory. Defaults to current directory.
- `job_name` (str): Unique name for the job. Defaults to "job".
- `instance_type` (str): SageMaker instance type. Defaults to "ml.m5.large".
- `instance_count` (int): Number of instances. Defaults to 1.
- `volume_size_gb` (int): EBS volume size in GB. Defaults to 5.
- `entrypoint` (str): Main script to execute (.py or .ipynb). Defaults to "train.py".
- `input_data` (str, optional): S3 URI for input data (not implemented yet).
- `framework` (str, optional): ML framework ("pytorch", "tensorflow", "sklearn").
- `framework_version` (str, optional): Framework version.

**Returns:**
- `dict`: Job submission result with status, job_name, and entrypoint

**Raises:**
- `requests.HTTPError`: If API calls fail
- `FileNotFoundError`: If directory or entrypoint doesn't exist
- `ValueError`: If code package exceeds volume capacity

#### `get_job(job_id)`

Get job status and details.

**Parameters:**
- `job_id` (str): Job identifier

**Returns:**
- `dict`: Job details including status, logs, and output URLs

**Raises:**
- `requests.HTTPError`: If job not found or API error

#### `download_output(job_id, output_dir=".")`

Download completed job output files.

**Parameters:**
- `job_id` (str): Job identifier
- `output_dir` (str): Local directory to save output. Defaults to current directory.

**Returns:**
- `Path`: Path to downloaded output tar.gz file

**Raises:**
- `RuntimeError`: If job is not completed
- `requests.HTTPError`: If download fails

## Supported Instance Types

### CPU Instances
- `ml.m5.large`, `ml.m5.xlarge`, `ml.m5.2xlarge`, `ml.m5.4xlarge`
- `ml.c5.large`, `ml.c5.xlarge`, `ml.c5.2xlarge`, `ml.c5.4xlarge`

### GPU Instances
- `ml.g4dn.xlarge`, `ml.g4dn.2xlarge`, `ml.g4dn.4xlarge`, `ml.g4dn.8xlarge`
- `ml.p3.2xlarge`, `ml.p3.8xlarge`, `ml.p3.16xlarge`
- `ml.g5.xlarge`, `ml.g5.2xlarge`, `ml.g5.4xlarge`, `ml.g5.8xlarge`

## Framework Auto-Detection

The SDK automatically detects your local ML framework and selects appropriate Docker images:

- **PyTorch**: Detects version and selects matching SageMaker PyTorch container
- **TensorFlow**: Detects version and selects matching SageMaker TensorFlow container
- **Scikit-learn**: Falls back to scikit-learn container for general ML workloads

## Code Requirements

### File Size Limits
Code packages must not exceed half your volume size:
- 5GB volume → 2.5GB max code package
- 10GB volume → 5GB max code package

### Automatic Dependencies
The SDK automatically adds these packages to your requirements.txt:
- `nbconvert` - For Jupyter notebook execution
- `psutil` - For system monitoring
- `GPUtil` - For GPU monitoring

## Examples

### Basic PyTorch Training

```python
from kaio import Client

client = Client("https://api.kaion5.com")
client.login("your-api-key")

result = client.submit_job(
    directory="./pytorch_model",
    job_name="pytorch-training",
    instance_type="ml.g4dn.xlarge",
    entrypoint="train.py",
    volume_size_gb=10
)
```

### TensorFlow with Custom Framework Version

```python
result = client.submit_job(
    directory="./tensorflow_model",
    job_name="tf-experiment",
    instance_type="ml.p3.2xlarge",
    framework="tensorflow",
    framework_version="2.13.0",
    entrypoint="model.py"
)
```

### Jupyter Notebook Execution

```python
result = client.submit_job(
    directory="./notebooks",
    job_name="data-analysis",
    instance_type="ml.m5.xlarge",
    entrypoint="analysis.ipynb"
)
```

## Error Handling

```python
import requests

try:
    result = client.submit_job(
        directory="./code",
        job_name="my-job",
        instance_type="ml.g4dn.xlarge"
    )
except requests.HTTPError as e:
    if e.response.status_code == 403:
        print("Access denied - check your API key")
    else:
        print(f"API error: {e}")
except ValueError as e:
    print(f"Configuration error: {e}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
```

## License

MIT License