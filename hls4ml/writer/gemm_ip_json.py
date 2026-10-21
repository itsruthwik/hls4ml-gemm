import json

import numpy as np


def _numpy_scalar_json_default(value):
    """Encode NumPy scalars while retaining JSONEncoder's normal strictness.

    Manifest array data is converted to ordinary lists by the writers; accepting
    arbitrary NumPy arrays here would conceal an unnormalized manifest field.
    """
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def write_gemm_config_json(path, manifest):
    """Write a GEMM-IP manifest, accepting NumPy scalar metadata from frontends."""
    with open(path, 'w') as manifest_file:
        json.dump(manifest, manifest_file, indent=4, default=_numpy_scalar_json_default)
