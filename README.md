# Kaio Python Client

A Python client for the Kaio multi-tenant machine learning platform that enables developers to run SageMaker Processing jobs through simple APIs with automatic base docker image resolution, secure file uploads, and multi-instance parallel processing capabilities.

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

- **SageMaker Processing Jobs**: Runs data processing and ML workloads on AWS SageMaker
- **Multi-Instance Processing**: Parallel processing across multiple instances for large datasets
- **Automatic Image Resolution**: Detects your local ML framework and selects appropriate Docker images
- **Secure File Uploads**: Handles code packaging and S3 uploads automatically
- **Job Management**: Submit, monitor, and download results from SageMaker jobs
- **Web Dashboard**: Access your jobs dashboard at https://www.kaion5.com/home/kaio-platform.html to monitor jobs and download outputs
- **Multi-Framework Support**: Works with PyTorch, TensorFlow, and Scikit-learn
- **GPU/CPU Instance Matching**: Automatically selects GPU or CPU optimized containers
- **Automatic Dependencies**: Adds required packages (nbconvert, psutil, GPUtil) to requirements.txt
- **JWT Token Management**: Handles authentication token refresh automatically
- **Environment Variable Access**: Provides SM_CURRENT_HOST and SM_HOSTS for custom parallelization logic

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

Submit a SageMaker Processing job with automatic image resolution.

**Parameters:**
- `directory` (str): Path to code directory containing your code and data. All files in this directory will be packaged and uploaded to SageMaker. Defaults to current directory.
- `job_name` (str): Unique name for the job. Defaults to "job".
- `instance_type` (str): SageMaker instance type. Defaults to "ml.m5.large".
- `instance_count` (int): Number of instances for parallel processing. Defaults to 1.
  - **Multi-Instance Processing**: When `instance_count > 1`, SageMaker launches multiple instances to process large datasets in parallel
  - **Environment Variables**: Each instance receives `SM_CURRENT_HOST` (current instance identifier) and `SM_HOSTS` (comma-separated list of all instance identifiers)
  - **No Inter-Instance Networking**: Instances run independently without network communication between them
  - **Custom Parallelization Logic**: Use `SM_CURRENT_HOST` and `SM_HOSTS` to implement data partitioning and parallel processing
- `volume_size_gb` (int): EBS volume size in GB. Defaults to 5. Maximum 50GB.
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

#### `get_jobs()`

Get all jobs for the authenticated user.

**Returns:**
- `dict`: List of jobs with count

**Raises:**
- `requests.HTTPError`: If API error

**Example:**
```python
jobs = client.get_jobs()
print(f"Total jobs: {jobs['count']}")
```

#### `get_job(unique_job_name)`

Get specific job status and details.

**Parameters:**
- `unique_job_name` (str): Unique job name returned from submit_job

**Returns:**
- `dict`: Job details including status and timestamp

**Raises:**
- `requests.HTTPError`: If job not found or API error

**Example:**
```python
job = client.get_job("unique-job-name-123")
print(f"Status: {job['status']}")
```

#### `download_output(unique_job_name, output_dir=".")`

Download completed job output files.

**Parameters:**
- `unique_job_name` (str): Unique job name returned from submit_job
- `output_dir` (str): Local directory to save output. Defaults to current directory.

**Returns:**
- `list`: List of downloaded file paths

**Raises:**
- `RuntimeError`: If job is not completed
- `requests.HTTPError`: If download fails

**Example:**
```python
files = client.download_output("unique-job-name-123", "./results")
for file_path in files:
    print(f"Downloaded: {file_path}")
```

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

### Data and Code Structure
**Important**: All data needed for your job must be included in the directory you submit with `client.submit_job()`. The entire directory is packaged and uploaded to SageMaker.

**Security Notice**: Jobs run securely in Kaion5 Compute's AWS account. Do not upload sensitive or confidential data. All uploaded data and job outputs are automatically deleted after 7 days. There will be future development and Kaio abilities for even more secure Bring-Your-Own-Account workflows, where you'd have access to a the Kaio dashbaord for jobs submitted though Kaio in your own account. 

**Telemetry Data**: By using this client, you acknowledge and agree that Kaion5 Compute will retain job telemetry data (job names, instance types, job status, runtime, compute metrics, storage configurations) even after job cleanup. This data helps optimize the platform and provision future instances based on usage patterns. After 7 days, logs, code, and data are permanently deleted and not stored.

### File Size Limits
Code packages must not exceed half your volume size:
- 5GB volume → 2.5GB max code package
- 10GB volume → 5GB max code package
- **Maximum storage per job**: 50GB

### Multi-Instance Code Structure
For multi-instance processing jobs, your code should:

```python
import os

def main():
    # Get SageMaker environment variables
    current_host = os.environ.get('SM_CURRENT_HOST', 'algo-1')
    all_hosts = os.environ.get('SM_HOSTS', 'algo-1').split(',')
    
    # Determine this instance's role
    host_rank = all_hosts.index(current_host)
    total_hosts = len(all_hosts)
    
    print(f"Running on {current_host} (rank {host_rank}/{total_hosts})")
    
    # Implement data partitioning logic
    process_data_partition(host_rank, total_hosts)

if __name__ == "__main__":
    main()
```

### Requirements File
**Important**: Include a `requirements.txt` file in your code directory listing all Python packages your job needs. This file will be automatically installed within the job environment.

### Automatic Dependencies
The SDK automatically adds these packages to your requirements.txt:
- `nbconvert` - For Jupyter notebook execution
- `psutil` - For system monitoring
- `GPUtil` - For GPU monitoring

## Multi-Instance Processing

### Current Implementation: SageMaker Processing Jobs

The Kaio client currently implements **SageMaker Processing Jobs**, which are designed for data processing, feature engineering, model evaluation, and **model training** tasks. When using `instance_count > 1`, you can parallelize processing of large datasets across multiple instances.

**Note**: Processing jobs can absolutely be used for training models - simply include your training code as the entrypoint. The distinction from Training jobs is primarily about distributed training capabilities and job orchestration features.

### SageMaker Environment Variables

Each instance in a multi-instance processing job receives these environment variables:

- **`SM_CURRENT_HOST`**: Identifier for the current instance (e.g., "algo-1", "algo-2", "algo-3")
- **`SM_HOSTS`**: Comma-separated list of all instance identifiers (e.g., "algo-1,algo-2,algo-3")

### Multi-Instance Processing Example

```python
import os

# Get instance information
current_host = os.environ.get('SM_CURRENT_HOST', 'algo-1')
all_hosts = os.environ.get('SM_HOSTS', 'algo-1').split(',')
host_rank = all_hosts.index(current_host)
total_hosts = len(all_hosts)

print(f"Instance {host_rank + 1} of {total_hosts} (Host: {current_host})")

# Partition data based on host rank
data_partition_size = len(dataset) // total_hosts
start_idx = host_rank * data_partition_size
end_idx = start_idx + data_partition_size if host_rank < total_hosts - 1 else len(dataset)

# Process assigned data partition
my_data = dataset[start_idx:end_idx]
process_data_partition(my_data)
```

### Important Limitations

- **No Inter-Instance Communication**: Instances cannot communicate with each other during processing
- **No Distributed Training**: Current implementation does not support distributed ML training
- **Independent Processing**: Each instance works on its assigned data partition independently

### Future: Distributed Training Support

Distributed training capabilities will be available in future updates when SageMaker Training Jobs are implemented, which will include:
- Multi-GPU distributed training
- Parameter server configurations
- Gradient synchronization between instances
- Framework-specific distributed training (PyTorch DDP, TensorFlow MultiWorkerMirroredStrategy)

## Examples

### Single Instance Model Training

```python
from kaio import Client

client = Client("https://api.kaion5.com")
client.login("your-api-key")

# Training a model using Processing jobs
result = client.submit_job(
    directory="./pytorch_training",  # Contains train.py, data/, and requirements.txt
    job_name="pytorch-model-training",
    instance_type="ml.g4dn.xlarge",
    entrypoint="train.py",  # Your training script
    volume_size_gb=10
)
```

### Multi-Instance Data Processing

```python
# Submit job with 4 instances for parallel processing with SM_HOSTS and your custom parallelization logic
result = client.submit_job(
    directory="./image_classification_model",  # Contains processing code and data
    job_name="multi-instance-image-processing",
    instance_type="ml.m5.2xlarge",
    instance_count=4,  # 4 instances for parallel processing
    entrypoint="process_images.py",  # Your parallel processing script using SM_HOSTS
    volume_size_gb=50
)
```

### Framework-Specific Processing

```python
result = client.submit_job(
    directory="./tensorflow_processing",
    job_name="tf-data-prep",
    instance_type="ml.p3.2xlarge",
    framework="tensorflow",
    framework_version="2.13.0",
    entrypoint="preprocess.py"
)
```

### Jupyter Notebook Processing

```python
result = client.submit_job(
    directory="./notebooks",
    job_name="data-analysis",
    instance_type="ml.m5.xlarge",
    entrypoint="analysis.ipynb"
)
```

## Job Management

### Getting All Jobs

```python
# Get list of all your jobs
jobs = client.get_jobs()
print(f"Total jobs: {jobs['count']}")
for job in jobs['jobs']:
    print(f"Job: {job['job_name']} - Status: {job['status']}")
```

### Getting Job Status

```python
# Get specific job details (use unique_job_name from submit_job response)
job_details = client.get_job("unique-job-name-123")
print(f"Status: {job_details['status']}")
print(f"Started: {job_details['start_time']}")
if job_details['status'] == 'Completed':
    print(f"Completed: {job_details['end_time']}")
```

### Downloading Job Outputs

```python
# Download all output files from a completed job
downloaded_files = client.download_output("unique-job-name-123", "./results")
for file_path in downloaded_files:
    print(f"Downloaded: {file_path}")
```

### Complete Job Workflow

```python
# Submit job
result = client.submit_job(
    directory="./my_training",
    job_name="model-training",
    instance_type="ml.g4dn.xlarge"
)

unique_job_name = result['unique_job_name']
print(f"Job submitted: {unique_job_name}")

# Monitor job status
import time
while True:
    job = client.get_job(unique_job_name)
    status = job['status']
    print(f"Job status: {status}")
    
    if status in ['Completed', 'Failed', 'Stopped']:
        break
    
    time.sleep(30)  # Check every 30 seconds

# Download results if completed
if status == 'Completed':
    files = client.download_output(unique_job_name, "./results")
    print(f"Downloaded {len(files)} files")
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

## Job Dashboard

**Access your jobs at**: https://www.kaion5.com/home/kaio-platform.html

After signing in to the Kaio Platform, you can:
- Monitor job status and progress
- View job logs and details
- Download job outputs and results
- Manage your job history

## License

MIT License