# Create a new file to patch mmgp
def patch_mmgp():
    """Patch mmgp to work with MPS devices"""
    import mmgp.offload as offload
    import torch
    
    original_init = offload.offload.__init__
    
    def new_init(self):
        if torch.cuda.is_available():
            self.device_mem_capacity = torch.cuda.get_device_properties(0).total_memory
        elif torch.backends.mps.is_available():
            # MPS doesn't expose memory info, use a conservative estimate
            self.device_mem_capacity = 4 * 1024 * 1024 * 1024  # 4GB default
        else:
            self.device_mem_capacity = 2 * 1024 * 1024 * 1024  # 2GB for CPU
        
        # Rest of original init
        self.verbose = 0
        self.profile_type = None
        self.compile = False
        self.quantize_transformer = False
        self.attention_mode = None
        self.preload_n_first_mb = 0
    
    offload.offload.__init__ = new_init 