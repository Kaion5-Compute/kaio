import sys
import importlib
import subprocess
import re
from packaging import version


def get_env_info():
    """
    Resolve local environment details:
      - Python version
      - PyTorch version
      - TensorFlow version
      - Scikit-learn version
      - GPU availability and count

    Returns a dictionary with detected versions (or None if not installed).
    """
    info = {}

    # Python version
    py_major, py_minor = sys.version_info[:2]
    info["python_version"] = f"py{py_major}{py_minor}"

    # Try PyTorch
    torch = None
    try:
        torch = importlib.import_module("torch")
        info["pytorch_version"] = torch.__version__.split("+")[0]
    except ImportError:
        info["pytorch_version"] = None

    # Try TensorFlow
    tf = None
    try:
        tf = importlib.import_module("tensorflow")
        info["tensorflow_version"] = tf.__version__
    except ImportError:
        info["tensorflow_version"] = None

    # Try Scikit-learn
    try:
        sklearn = importlib.import_module("sklearn")
        info["sklearn_version"] = sklearn.__version__
    except ImportError:
        info["sklearn_version"] = None

    # GPU check
    gpu_available = False
    gpu_count = 0

    if torch is not None:
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
    elif tf is not None:
        try:
            gpus = tf.config.list_physical_devices("GPU")
            gpu_available = len(gpus) > 0
            gpu_count = len(gpus)
        except Exception:
            gpu_available = False
            gpu_count = 0
    else:
        # fallback: check nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if result.returncode == 0 and "GPU" in result.stdout:
                gpu_available = True
                gpu_count = len(result.stdout.strip().splitlines())
        except FileNotFoundError:
            gpu_available = False

    info["gpu_available"] = gpu_available
    info["gpu_count"] = gpu_count

    return info


# GPU-capable SageMaker instances
GPU_INSTANCES = {
    "ml.g4dn.xlarge",
    "ml.g4dn.2xlarge",
    "ml.g4dn.4xlarge",
    "ml.g4dn.8xlarge",
    "ml.g4dn.12xlarge",
    "ml.g4dn.16xlarge",
    "ml.p3.2xlarge",
    "ml.p3.8xlarge",
    "ml.p3.16xlarge",
    "ml.p4d.24xlarge",
    "ml.g5.xlarge",
    "ml.g5.2xlarge",
    "ml.g5.4xlarge",
    "ml.g5.8xlarge",
    "ml.g5.12xlarge",
    "ml.g5.16xlarge",
    "ml.g5.24xlarge",
    "ml.g5.48xlarge",
}


def is_gpu_instance(instance_type: str) -> bool:
    return instance_type in GPU_INSTANCES


def parse_uri(uri: str):
    """
    Extract framework, version, py_version, device (cpu/gpu) from a URI.
    Example: '.../tensorflow-training:2.13.0-gpu-py310'
    """
    m = re.search(r":([\d\.\-]+)-(gpu|cpu)-(py\d+|py3)", uri)
    if not m:
        return None
    fw_version, device, py_version = m.groups()
    if "pytorch" in uri:
        framework = "pytorch"
    elif "tensorflow" in uri:
        framework = "tensorflow"
    elif "scikit-learn" in uri:
        framework = "sklearn"
    else:
        framework = "unknown"
    return {
        "uri": uri,
        "framework": framework,
        "version": fw_version,
        "py_version": py_version,
        "device": device,
    }


def resolve_container(
    all_uris, framework: str, req_version: str, req_py: str, instance_type: str
):
    gpu = is_gpu_instance(instance_type)

    # Parse all URIs
    parsed = [p for u in all_uris if (p := parse_uri(u))]

    # Filter by framework
    candidates = [p for p in parsed if p["framework"] == framework]

    # Filter GPU/CPU
    device = "gpu" if gpu else "cpu"
    candidates = [p for p in candidates if p["device"] == device]

    if not candidates:
        return {"choices": [], "selection": None}

    # Step 1: try exact version & py_version match
    exact = [
        p
        for p in candidates
        if p["version"] == req_version and p["py_version"] == req_py
    ]
    if exact:
        return {"choices": exact, "selection": exact[0]}

    # Step 2: relax python version match but keep framework version exact
    py_relaxed = [p for p in candidates if p["version"] == req_version]
    if py_relaxed:
        # pick closest py_version <= requested
        py_sorted = sorted(py_relaxed, key=lambda p: p["py_version"], reverse=True)
        selection = max(
            (p for p in py_sorted if p["py_version"] <= req_py),
            default=py_sorted[0],
            key=lambda p: p["py_version"],
        )
        return {"choices": py_relaxed, "selection": selection}

    # Step 3: relax framework version (pick closest lower or equal)
    fw_sorted = sorted(candidates, key=lambda p: version.parse(p["version"]))
    lower_versions = [
        p
        for p in fw_sorted
        if version.parse(p["version"]) <= version.parse(req_version)
    ]
    if lower_versions:
        best = max(lower_versions, key=lambda p: version.parse(p["version"]))
        return {"choices": lower_versions, "selection": best}

    # Step 4: fallback to lowest available
    return {
        "choices": candidates,
        "selection": min(candidates, key=lambda p: version.parse(p["version"])),
    }


def smart_resolve(all_uris, reqs, instance_type: str):
    """
    Resolve URIs for multiple frameworks, prioritizing PyTorch > TensorFlow > Sklearn.
    reqs = {
        "pytorch_version": "2.2.0",
        "tensorflow_version": "2.13.0",
        "sklearn_version": "1.2",
        "python_version": "py310"
    }
    """
    results = {}

    for fw in ["pytorch", "tensorflow", "sklearn"]:
        fw_version = reqs.get(f"{fw}_version")
        if fw_version:
            res = resolve_container(
                all_uris, fw, fw_version, reqs["python_version"], instance_type
            )
            results[fw] = res

    # Select preferred (pytorch > tensorflow > sklearn)
    selection = None
    for fw in ["pytorch", "tensorflow", "sklearn"]:
        if fw in results and results[fw]["selection"]:
            selection = results[fw]["selection"]
            break

    return {"all_results": results, "final_selection": selection}


# Available container URIs for af-south-1
ALL_URIS = [
    "313743910680.dkr.ecr.af-south-1.amazonaws.com/sagemaker-pytorch:0.4.0-gpu-py3",
    "313743910680.dkr.ecr.af-south-1.amazonaws.com/sagemaker-pytorch:0.4.0-cpu-py3",
    "313743910680.dkr.ecr.af-south-1.amazonaws.com/sagemaker-pytorch:1.0.0-gpu-py3",
    "313743910680.dkr.ecr.af-south-1.amazonaws.com/sagemaker-pytorch:1.0.0-cpu-py3",
    "313743910680.dkr.ecr.af-south-1.amazonaws.com/sagemaker-pytorch:1.1.0-gpu-py3",
    "313743910680.dkr.ecr.af-south-1.amazonaws.com/sagemaker-pytorch:1.1.0-cpu-py3",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.2.0-gpu-py3",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.2.0-cpu-py3",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.3.1-gpu-py3",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.3.1-cpu-py3",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.4.0-gpu-py3",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.4.0-cpu-py3",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.4.0-gpu-py36",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.4.0-cpu-py36",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.5.0-gpu-py3",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.5.0-cpu-py3",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.5.0-gpu-py36",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.5.0-cpu-py36",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.6.0-gpu-py3",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.6.0-cpu-py3",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.6.0-gpu-py36",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.6.0-cpu-py36",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.7.1-gpu-py3",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.7.1-cpu-py3",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.7.1-gpu-py36",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.7.1-cpu-py36",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.8.1-gpu-py3",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.8.1-cpu-py3",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.8.1-gpu-py36",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.8.1-cpu-py36",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.9.1-gpu-py38",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.9.1-cpu-py38",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.10.2-gpu-py38",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.10.2-cpu-py38",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.11.0-gpu-py38",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.11.0-cpu-py38",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.12.1-gpu-py38",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.12.1-cpu-py38",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.13.1-gpu-py39",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.13.1-cpu-py39",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:2.0.1-gpu-py310",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:2.0.1-cpu-py310",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:2.1.0-cpu-py310",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:2.2.0-gpu-py310",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:2.2.0-cpu-py310",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:2.3.0-gpu-py311",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:2.3.0-cpu-py311",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:2.4.0-gpu-py311",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:2.4.0-cpu-py311",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:2.5.1-gpu-py311",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:2.5.1-cpu-py311",
    "313743910680.dkr.ecr.af-south-1.amazonaws.com/sagemaker-tensorflow-scriptmode:1.11.0-gpu-py3",
    "313743910680.dkr.ecr.af-south-1.amazonaws.com/sagemaker-tensorflow-scriptmode:1.11.0-cpu-py3",
    "313743910680.dkr.ecr.af-south-1.amazonaws.com/sagemaker-tensorflow-scriptmode:1.12.0-gpu-py3",
    "313743910680.dkr.ecr.af-south-1.amazonaws.com/sagemaker-tensorflow-scriptmode:1.12.0-cpu-py3",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:1.14.0-gpu-py3",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:1.14.0-cpu-py3",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:1.15.5-gpu-py3",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:1.15.5-cpu-py3",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:1.15.5-gpu-py36",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:1.15.5-cpu-py36",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:1.15.5-gpu-py37",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:1.15.5-cpu-py37",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.0.4-gpu-py3",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.0.4-cpu-py3",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.0.4-gpu-py36",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.0.4-cpu-py36",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.1.3-gpu-py3",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.1.3-cpu-py3",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.1.3-gpu-py36",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.1.3-cpu-py36",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.2.2-gpu-py37",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.2.2-cpu-py37",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.3.2-gpu-py37",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.3.2-cpu-py37",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.4.3-gpu-py37",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.4.3-cpu-py37",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.5.1-gpu-py37",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.5.1-cpu-py37",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.6.3-gpu-py38",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.6.3-cpu-py38",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.7.1-gpu-py38",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.7.1-cpu-py38",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.8.0-gpu-py39",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.8.0-cpu-py39",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.9.2-gpu-py39",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.9.2-cpu-py39",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.10.1-gpu-py39",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.10.1-cpu-py39",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.11.0-gpu-py39",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.11.0-cpu-py39",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.12.0-gpu-py310",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.12.0-cpu-py310",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.13.0-gpu-py310",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.13.0-cpu-py310",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.14.1-gpu-py310",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.14.1-cpu-py310",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.16.2-gpu-py310",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.16.2-cpu-py310",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.18.0-gpu-py310",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.18.0-cpu-py310",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.19.0-gpu-py312",
    "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.19.0-cpu-py312",
    "510948584623.dkr.ecr.af-south-1.amazonaws.com/sagemaker-scikit-learn:0.20.0-cpu-py3",
    "510948584623.dkr.ecr.af-south-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
    "510948584623.dkr.ecr.af-south-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3",
    "510948584623.dkr.ecr.af-south-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
]


def resolve_image(instance_type, framework=None, version=None):
    """Resolve SageMaker image URI using smart resolution logic"""
    
    env_info = get_env_info()
    
    # If framework is specified, prioritize it
    if framework:
        # Create requirements dict with specified framework
        reqs = {"python_version": env_info["python_version"]}
        
        if version:
            # Use specified version
            reqs[f"{framework}_version"] = version
        else:
            # Use detected version or latest available
            detected_version = env_info.get(f"{framework}_version")
            if detected_version:
                reqs[f"{framework}_version"] = detected_version
            else:
                # Find latest available version for this framework
                parsed = [p for u in ALL_URIS if (p := parse_uri(u)) and p["framework"] == framework]
                if parsed:
                    latest = max(parsed, key=lambda p: version.parse(p["version"]))
                    reqs[f"{framework}_version"] = latest["version"]
        
        result = smart_resolve(ALL_URIS, reqs, instance_type)
        if result["final_selection"]:
            return result["final_selection"]["uri"]
    
    # Auto-detect from environment
    result = smart_resolve(ALL_URIS, env_info, instance_type)
    
    if result["final_selection"]:
        return result["final_selection"]["uri"]
    
    # Fallback to defaults
    is_gpu = is_gpu_instance(instance_type)
    if is_gpu:
        return "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:2.3.0-gpu-py311"
    else:
        return "510948584623.dkr.ecr.af-south-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
