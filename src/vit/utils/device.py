"""Pick the torch device: the GPU only when this PyTorch build can run on it."""

import re
from functools import cache

import torch

from vit.utils.logging import get_logger

logger = get_logger(__name__)


def _capability_of(arch: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"(?:sm|compute)_(\d+)[a-z]?", arch)
    if match is None:
        return None
    number = int(match.group(1))
    return number // 10, number % 10


def cuda_build_supports(capability: tuple[int, int], arch_list: list[str]) -> bool:
    """Whether a GPU of `capability` can run kernels from a build for `arch_list`.

    A binary for sm_XY runs on GPUs of the same major version and minor >= Y;
    PTX (compute_XY) is compiled on the fly for any GPU >= XY.
    """
    for arch in arch_list:
        built_for = _capability_of(arch)
        if built_for is None:
            continue
        if arch.startswith("sm_"):
            if built_for[0] == capability[0] and built_for[1] <= capability[1]:
                return True
        elif built_for <= capability:
            return True
    return False


@cache
def select_device() -> torch.device:
    """CUDA if a GPU is present and supported by this PyTorch build, else CPU."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    capability = torch.cuda.get_device_capability(0)
    if not cuda_build_supports(capability, torch.cuda.get_arch_list()):
        logger.warning(
            "GPU %s (sm_%d%d) is not supported by this PyTorch build; using the CPU",
            torch.cuda.get_device_name(0),
            *capability,
        )
        return torch.device("cpu")
    return torch.device("cuda")


def device_name(device: torch.device) -> str:
    """Human-readable name for reports: the GPU model or "cpu"."""
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
