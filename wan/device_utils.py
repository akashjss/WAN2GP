import torch

def get_device():
    """Get the best available device for the current system."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

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