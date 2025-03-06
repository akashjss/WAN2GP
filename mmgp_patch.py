def patch_mmgp():
    """Patch mmgp to work with MPS devices"""
    import mmgp.offload as offload
    import torch
    
    original_init = offload.offload.__init__
    
    def new_init(self):
        # Device setup
        if torch.cuda.is_available():
            self.device_mem_capacity = torch.cuda.get_device_properties(0).total_memory
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device_mem_capacity = 4 * 1024 * 1024 * 1024  # 4GB default
            self.device = 'mps'
        else:
            self.device_mem_capacity = 2 * 1024 * 1024 * 1024  # 2GB for CPU
            self.device = 'cpu'
        
        # Basic configuration (needed by profile())
        self.verbose = 0
        self.profile_type = None
        self.compile = False
        self.quantize_transformer = False
        self.attention_mode = None
        self.preload_n_first_mb = 0
        self.initialized = False
        self.memory_buffer = 0
        
        # Model and module tracking (needed by all())
        self.models = {}
        self.model_devices = {}
        self.model_sizes = {}
        self.model_blocks = {}
        self.model_config = {}
        self.module_names = {}
        self.module_paths = {}
        
        # Block tracking (needed by add_module_to_blocks())
        self.loaded_blocks = {}
        self.block_sizes = {}
        self.block_devices = {}
        self.blocks_of_modules = {}
        self.modules_of_blocks = {}
        self.block_dependencies = {}
        self.blocks_of_modules_sizes = {}
        self.blocks_of_modules_devices = {}
        self.blocks_of_modules_status = {}
        self.blocks_of_modules_deps = {}
        
        # Parameter tracking (needed by final error)
        self.parameters_ref = {}  # Track parameter references
        self.tied_parameters = {}  # Track tied parameters
        self.param_to_name = {}   # Map parameters to names
        self.name_to_param = {}   # Map names to parameters
        self.param_device = {}    # Track parameter devices
        self.param_dtype = {}     # Track parameter data types
        
        # LoRA support (needed for module processing)
        self.lora_parents = {}
        self.lora_children = {}
        self.lora_names = {}
        self.lora_ranks = {}
        
        # Status tracking
        self.block_status = {}
        self.current_blocks = set()
        self.block_queue = []
        
        # Memory management
        self.memory_used = 0
        self.peak_memory = 0
        
        # Performance tracking
        self.call_counter = 0
        self.profiling_enabled = False
    
    offload.offload.__init__ = new_init 