import torch
import torch.nn as nn
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from deepmd.env import GLOBAL_TF_FLOAT_PRECISION, GLOBAL_NP_FLOAT_PRECISION

if TYPE_CHECKING:
    _DICT_VAL = TypeVar("_DICT_VAL")
    _OBJ = TypeVar("_OBJ")
    try:
        from typing import Literal  # python >3.6
    except ImportError:
        from typing_extensions import Literal  # type: ignore
    _ACTIVATION = Literal["relu", "relu6", "softplus", "sigmoid", "tanh", "gelu", "gelu_tf"]
    _PRECISION = Literal["default", "float16", "float32", "float64"]

ACTIVATION_FN_DICT = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "softplus": nn.Softplus,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh
}

# define constants
PRECISION_DICT = {
    "default": GLOBAL_TF_FLOAT_PRECISION,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def get_activation_func(
    activation_fn: "_ACTIVATION",
) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation_fn not in ACTIVATION_FN_DICT:
        raise RuntimeError(f"{activation_fn} is not a valid activation function")
    return ACTIVATION_FN_DICT[activation_fn]


def get_precision(precision: "_PRECISION") -> Any:
    if precision not in PRECISION_DICT:
        raise RuntimeError(f"{precision} is not a valid precision")
    return PRECISION_DICT[precision]
