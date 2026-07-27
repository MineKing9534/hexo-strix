from .model import (
    AxisGineCompatNet,
    AxisGineConfig,
    AxisModelOutput,
    PersistentRayAxisNet,
    PersistentRayConfig,
    axis_gine_compat_4x128,
    persistent_ray_4x128,
)
from .ops import (
    AXIS_RAY_PAIRS,
    NUM_RAYS,
    PACKED_RAY_RADIUS,
    RAY_DIRS,
    gather_dense_actions,
    unpack_ray_bits,
)

__all__ = [
    "AxisGineCompatNet",
    "AxisGineConfig",
    "AxisModelOutput",
    "PersistentRayAxisNet",
    "PersistentRayConfig",
    "axis_gine_compat_4x128",
    "persistent_ray_4x128",
    "AXIS_RAY_PAIRS",
    "NUM_RAYS",
    "PACKED_RAY_RADIUS",
    "RAY_DIRS",
    "gather_dense_actions",
    "unpack_ray_bits",
]
from .checkpoint import (
    ConversionReport,
    convert_strix_axis_state_dict,
    extract_state_dict,
    load_strix_axis_checkpoint,
)
from .wire import RasterTensorBatch, decode_hxr1

__all__ += [
    "ConversionReport",
    "convert_strix_axis_state_dict",
    "extract_state_dict",
    "load_strix_axis_checkpoint",
    "RasterTensorBatch",
    "decode_hxr1",
]
from .klent import (
    KlentPolicy,
    actor_aware_lambda_returns,
    klent_policy_from_dense,
    sample_segmented_gumbel,
)

__all__ += [
    "KlentPolicy",
    "actor_aware_lambda_returns",
    "klent_policy_from_dense",
    "sample_segmented_gumbel",
]
