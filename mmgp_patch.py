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
        
        # Basic configuration
        self.verbose = 0
        self.profile_type = None
        self.compile = False
        self.quantize_transformer = False
        self.attention_mode = None
        self.preload_n_first_mb = 0
        self.initialized = False
        
        # Block tracking and dependencies
        self.loaded_blocks = {}
        self.block_sizes = {}
        self.block_devices = {}
        self.blocks_of_modules = {}
        self.modules_of_blocks = {}
        self.block_dependencies = {}
        self.blocks_of_modules_sizes = {}
        self.prev_blocks_names = {}  # Add this for block tracking
        self.next_blocks_names = {}  # Add this for block tracking
        self.entry_names = set()     # Add this for entry tracking
        
        # Model tracking
        self.models = {}
        self.model_devices = {}
        self.model_sizes = {}
        self.model_blocks = {}
        self.model_config = {}
        
        # Module tracking
        self.module_names = {}
        self.module_paths = {}
        self.module_types = {}
        self.module_shapes = {}
        self.module_hooks = {}
        
        # Parameter tracking
        self.parameters_ref = {}
        self.param_to_name = {}
        self.name_to_param = {}
        self.param_device = {}
        self.param_dtype = {}
        
        # LoRA support (required by mmgp)
        self.lora_parents = {}
        self.lora_children = {}
        self.lora_names = {}
        self.lora_ranks = {}
        self.lora_weights = {}
        self.lora_modules = {}
        
        # Memory management
        self.memory_used = 0
        self.peak_memory = 0
        self.memory_buffer = 0
        
        # Block status
        self.block_status = {}
        self.current_blocks = set()
        self.block_queue = []
        
        # Additional tracking
        self.call_stats = {}
        self.profiling_enabled = False
        self.call_counter = 0
        
        # Entry tracking
        self.entry_points = set()
        self.entry_dependencies = {}
        self.entry_blocks = {}
        self.entry_modules = {}
        
        # Block graph
        self.block_graph = {}
        self.block_weights = {}
        self.block_scores = {}
        
        # Cache
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_entries = {}
        
        # Preloading configuration
        self.preloaded_blocks_per_model = {}  # Track preloaded blocks for each model
        self.preload_budget = {}  # Budget for preloading per model
        self.preload_strategy = 'default'  # Strategy for preloading
        self.preload_scores = {}  # Scores for block preloading
        self.preload_order = {}  # Order of block preloading
        self.preload_status = {}  # Status of preloaded blocks
    
    offload.offload.__init__ = new_init 