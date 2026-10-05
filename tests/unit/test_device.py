import torch

from vit.utils.device import cuda_build_supports, select_device

BUILD = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]


def test_supported_gpus_match_a_binary_of_the_same_major_version():
    assert cuda_build_supports((8, 9), BUILD)  # RTX 40xx runs the sm_86 binary
    assert cuda_build_supports((7, 5), BUILD)
    assert cuda_build_supports((9, 0), ["sm_90a"])


def test_old_or_unknown_gpus_are_not_supported():
    assert not cuda_build_supports((5, 0), BUILD)  # GeForce 940MX
    assert not cuda_build_supports((7, 0), BUILD)
    assert not cuda_build_supports((8, 0), [])


def test_ptx_runs_on_newer_gpus():
    assert cuda_build_supports((8, 9), ["compute_80"])
    assert not cuda_build_supports((7, 5), ["compute_80"])


def test_unsupported_gpu_falls_back_to_the_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _=0: (5, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _=0: "Old GPU")
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: BUILD)
    select_device.cache_clear()
    try:
        assert select_device() == torch.device("cpu")
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _=0: (8, 9))
        select_device.cache_clear()
        assert select_device() == torch.device("cuda")
    finally:
        select_device.cache_clear()
