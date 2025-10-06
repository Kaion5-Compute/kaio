import os, time, json, requests, tarfile, tempfile
from pathlib import Path
from .image_resolver import resolve_image, get_env_info


class Client:
    """Kaio Platform Client for SageMaker job submission and management.

    A Python client for the Kaio multi-tenant machine learning platform that enables
    developers to run SageMaker jobs through simple APIs with automatic image resolution
    and secure file uploads.

    Args:
        api_base (str): Base URL of the Kaio API endpoint
        verbose (bool): Enable verbose logging for debugging. Defaults to False.

    Example:
        >>> client = Client("https://api.kaion5.com")
        >>> client.login("your-api-key")
        >>> result = client.submit_job(
        ...     directory="./my_code",
        ...     job_name="training-job",
        ...     instance_type="ml.g4dn.xlarge",
        ...     entrypoint="train.py"
        ... )
    """

    def __init__(self, api_base, verbose=False):
        self.api_base = api_base
        self.jwt = None
        self.jwt_exp = 0
        self.api_key = None
        self.verbose = verbose

    def login(self, api_key):
        """Authenticate with API key and obtain JWT token.

        Args:
            api_key (str): Your Kaio platform API key

        Returns:
            Client: Self for method chaining

        Raises:
            requests.HTTPError: If authentication fails
        """
        resp = requests.post(f"{self.api_base}/login", json={"api_key": api_key})
        resp.raise_for_status()
        data = resp.json()
        self.jwt = data["access_token"]
        self.jwt_exp = data["exp"]
        self.api_key = api_key
        return self

    def _auth_headers(self):
        if time.time() > self.jwt_exp - 60:
            # refresh token automatically
            self.login(self.api_key)
        return {"Authorization": f"Bearer {self.jwt}"}

    def _ensure_nbconvert_in_requirements(self, directory: str):
        """Ensure requirements.txt exists and includes nbconvert"""
        req_path = Path(directory) / "requirements.txt"
        if req_path.exists():
            with open(req_path, "r+", encoding="utf-8") as f:
                lines = f.read().splitlines()
                if not any("nbconvert" in line for line in lines):
                    lines.append("nbconvert")
                if not any("psutil" in line for line in lines):
                    lines.append("psutil")
                if not any("GPUtil" in line for line in lines):
                    lines.append("GPUtil")
                    f.seek(0)
                    f.write("\n".join(lines) + "\n")
                    f.truncate()
        else:
            with open(req_path, "w", encoding="utf-8") as f:
                f.write("nbconvert\npsutil\nGPUtil\n")
        if self.verbose:
            print(f"✓ requirements.txt updated at {req_path}")

    def _tar_directory(self, directory: str, entrypoint: str) -> str:
        """Tar up code + requirements.txt (run_this_file.py now created server-side)"""
        self._ensure_nbconvert_in_requirements(directory)

        tmp_fd, tmp_tar = tempfile.mkstemp(suffix=".tar.gz")
        os.close(tmp_fd)
        with tarfile.open(tmp_tar, "w:gz") as tf:
            for root, _, files in os.walk(directory):
                for f in files:
                    path = Path(root) / f
                    arcname = str(path.relative_to(directory))
                    tf.add(path, arcname)
        return tmp_tar

    def submit_job(
        self,
        directory=".",
        job_name="job",
        instance_type="ml.m5.large",
        instance_count=1,
        volume_size_gb=5,
        entrypoint="train.py",
        input_data=None,
        framework=None,
        framework_version=None,
    ):
        """Submit a SageMaker Processing job with automatic image resolution.

        Packages your code directory, uploads it to S3, and triggers SageMaker Processing job execution.
        Automatically detects your local ML framework and selects appropriate Docker images.
        
        This implementation uses SageMaker Processing Jobs, which are designed for data processing,
        feature engineering, model evaluation, and model training tasks. Multi-instance processing 
        enables parallel processing of large datasets across multiple instances.
        
        Note: Processing jobs can be used for model training by simply providing your training
        code as the entrypoint. All data needed for the job must be included in the directory.
        
        Security: Jobs run in Kaion5 Compute's AWS account. Do not upload sensitive data.
        All data and outputs are deleted after 7 days. Job telemetry data (job names,
        instance types, status, runtime, compute metrics, storage configs) is retained
        for platform optimization. Logs, code, and data are permanently deleted after 7 days.

        Args:
            directory (str): Path to code directory containing your code and data. All files in this
                directory will be packaged and uploaded to SageMaker. Defaults to current directory.
            job_name (str): Unique name for the job. Defaults to "job".
            instance_type (str): SageMaker instance type. Defaults to "ml.m5.large".
            instance_count (int): Number of instances for parallel processing. Defaults to 1.
                When instance_count > 1:
                - SageMaker launches multiple instances to process data in parallel
                - Each instance receives SM_CURRENT_HOST (e.g., "algo-1", "algo-2") and 
                  SM_HOSTS (comma-separated list: "algo-1,algo-2,algo-3") environment variables
                - No networking is configured between instances - they run independently
                - Use SM_CURRENT_HOST and SM_HOSTS to implement custom data partitioning logic
                - Ideal for embarrassingly parallel workloads like data preprocessing
                Note: Distributed training is NOT supported in current Processing job implementation.
                Future SageMaker Training job support will enable distributed ML training.
            volume_size_gb (int): EBS volume size in GB. Defaults to 5. Maximum 50GB.
            entrypoint (str): Main script to execute (.py or .ipynb). Defaults to "train.py".
            input_data (str, optional): S3 URI for input data. Not implemented yet.
            framework (str, optional): ML framework ("pytorch", "tensorflow", "sklearn").
                Auto-detected from environment if not specified.
            framework_version (str, optional): Framework version. Auto-detected if not specified.

        Returns:
            dict: Job submission result with status, job_name, and entrypoint

        Raises:
            requests.HTTPError: If API calls fail
            FileNotFoundError: If directory or entrypoint doesn't exist
            ValueError: If code package exceeds volume capacity

        Examples:
            Single instance processing:
            >>> result = client.submit_job(
            ...     directory="./processing_code",
            ...     job_name="data-preprocessing",
            ...     instance_type="ml.m5.xlarge",
            ...     entrypoint="preprocess.py"
            ... )
            
            Multi-instance parallel processing:
            >>> result = client.submit_job(
            ...     directory="./parallel_processing",
            ...     job_name="large-dataset-processing",
            ...     instance_type="ml.m5.2xlarge",
            ...     instance_count=4,  # 4 instances for parallel processing
            ...     entrypoint="parallel_process.py"
            ... )
        """
        # Resolve image URI using advanced resolution logic
        image_uri = resolve_image(instance_type, framework, framework_version)
        if self.verbose:
            print(f"Resolved image: {image_uri}")

        # 1. Ask server for a presigned URL
        job_spec = {
            "instance_type": instance_type,
            "instance_count": instance_count,
            "volume_size_gb": volume_size_gb,
            "entrypoint": entrypoint,
            "image_uri": image_uri,
        }

        resp = requests.post(
            f"{self.api_base}/jobs/presign",
            headers=self._auth_headers(),
            json={"job_name": job_name, "job_spec": job_spec},
        )
        resp.raise_for_status()
        presign = resp.json()
        upload_url = presign["upload_url"]
        s3_uri = presign["s3_uri"]

        # 2. Tar & upload code
        tar_path = self._tar_directory(directory, entrypoint)

        # Check file size against volume capacity
        tar_size_mb = os.path.getsize(tar_path) / (1024 * 1024)
        max_allowed_mb = (volume_size_gb * 1024) / 2  # Half of volume size

        if tar_size_mb > max_allowed_mb:
            os.unlink(tar_path)  # Clean up temp file
            raise ValueError(
                f"Code package size ({tar_size_mb:.1f} MB) exceeds half the volume size "
                f"({max_allowed_mb:.1f} MB). Increase volume_size_gb to at least "
                f"{int((tar_size_mb * 2) / 1024) + 1} GB as we unpack the data on the instance."
            )

        try:
            with open(tar_path, "rb") as f:
                file_data = f.read()
                if self.verbose:
                    print(
                        f"Uploading {len(file_data)} bytes ({tar_size_mb:.1f} MB) to S3..."
                    )

            r = requests.put(upload_url, data=file_data)
            if self.verbose:
                print(f"S3 response status: {r.status_code}")
            if r.status_code != 200:
                if self.verbose:
                    print(f"S3 error response: {r.text}")
            r.raise_for_status()
            if self.verbose:
                print("✓ S3 upload successful")
        except Exception as e:
            if self.verbose:
                print(f"S3 upload failed: {e}")
            raise
        finally:
            try:
                os.unlink(tar_path)
            except OSError:
                pass

        # 3. Return upload success
        return {
            "status": "uploaded",
            "job_name": job_name,
            "entrypoint": entrypoint,
        }

    def get_job(self, job_id):
        """Get job status and details.

        Args:
            job_id (str): Job identifier

        Returns:
            dict: Job details including status, logs, and output URLs

        Raises:
            requests.HTTPError: If job not found or API error
        """
        resp = requests.get(
            f"{self.api_base}/jobs/{job_id}", headers=self._auth_headers()
        )
        resp.raise_for_status()
        return resp.json()

    def download_output(self, job_id, output_dir="."):
        """Download completed job output files.

        Args:
            job_id (str): Job identifier
            output_dir (str): Local directory to save output. Defaults to current directory.

        Returns:
            Path: Path to downloaded output tar.gz file

        Raises:
            RuntimeError: If job is not completed
            requests.HTTPError: If download fails
        """
        job = self.get_job(job_id)
        if job["status"] != "Completed":
            raise RuntimeError(f"Job not completed yet: {job['status']}")
        output_url = job["output_url"]
        resp = requests.get(output_url)
        resp.raise_for_status()
        out_path = Path(output_dir) / f"{job_id}_output.tar.gz"
        with open(out_path, "wb") as f:
            f.write(resp.content)
        return out_path
