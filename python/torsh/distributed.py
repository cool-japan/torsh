"""Distributed training utilities."""

try:
    from . import torsh_python as _C
except ImportError:
    import torsh_python as _C

# Import from Rust module
from torsh_python.distributed import (
    ProcessGroup,
    DistributedDataParallel,
    init_process_group,
    destroy_process_group,
    get_rank,
    get_world_size,
    is_initialized,
    is_available,
    barrier,
    all_reduce,
    all_gather,
    broadcast,
    reduce,
    scatter,
    gather,
)

# Alias for common usage
DDP = DistributedDataParallel

__all__ = [
    'ProcessGroup', 'DistributedDataParallel', 'DDP',
    'init_process_group', 'destroy_process_group',
    'get_rank', 'get_world_size', 'is_initialized', 'is_available',
    'barrier', 'all_reduce', 'all_gather', 'broadcast', 'reduce', 'scatter', 'gather',
]