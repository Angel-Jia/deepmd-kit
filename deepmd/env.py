import os
import numpy as np
import torch


# FLOAT_PREC
dp_float_prec = os.environ.get("DP_INTERFACE_PREC", "high").lower()
if dp_float_prec in ("high", ""):
    # default is high
    GLOBAL_TF_FLOAT_PRECISION = torch.float64
    GLOBAL_NP_FLOAT_PRECISION = np.float64
    GLOBAL_ENER_FLOAT_PRECISION = np.float64
    global_float_prec = "double"
elif dp_float_prec == "low":
    GLOBAL_TF_FLOAT_PRECISION = torch.float32
    GLOBAL_NP_FLOAT_PRECISION = np.float32
    GLOBAL_ENER_FLOAT_PRECISION = np.float64
    global_float_prec = "float"
else:
    raise RuntimeError(
        "Unsupported float precision option: %s. Supported: high,"
        "low. Please set precision with environmental variable "
        "DP_INTERFACE_PREC." % dp_float_prec
    )
