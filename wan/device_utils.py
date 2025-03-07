import torch

def get_device():
    """Get the best available device for the current system."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_device_memory():
    """Get available device memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1048576
    elif torch.backends.mps.is_available():
        # MPS doesn't expose memory info, use a conservative estimate
        return 4 * 1024  # 4GB default
    return 2 * 1024  # 2GB for CPU

def current_device():
    """Get current device index, returns -1 for CPU/MPS."""
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return -1

def is_cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()

def is_mps_available():
    """Check if MPS (Metal) is available."""
    return torch.backends.mps.is_available()

def empty_cache():
    """Clear device memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # MPS and CPU don't need explicit cache clearing 