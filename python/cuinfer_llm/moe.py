# MIT License
#
# Copyright (c) 2026 CUI Xin (崔 欣)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Fused grouped expert routing on CUDA tensors."""

import ctypes
import functools
import math
import os
from pathlib import Path

import torch


_DTYPES = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}


@functools.lru_cache(maxsize=1)
def _library() -> ctypes.CDLL:
    override = os.environ.get("CUINFER_MOE_LIBRARY")
    if override:
        path = Path(override)
    elif os.environ.get("TEST_SRCDIR") and os.environ.get("TEST_WORKSPACE"):
        path = (
            Path(os.environ["TEST_SRCDIR"])
            / os.environ["TEST_WORKSPACE"]
            / "calculator/moe/libmoe.so"
        )
    else:
        path = Path(__file__).resolve().parents[2] / "bazel-bin/calculator/moe/libmoe.so"
    lib = ctypes.CDLL(str(path))
    pointer = ctypes.c_void_p
    integer = ctypes.c_int
    lib.cuinfer_moe_fused_routing.argtypes = (
        [pointer] * 5 + [integer] * 5 + [ctypes.c_double] + [integer] * 3 + [pointer]
    )
    lib.cuinfer_moe_fused_routing.restype = integer
    lib.cuinfer_moe_cuda_error_string.argtypes = [integer]
    lib.cuinfer_moe_cuda_error_string.restype = ctypes.c_char_p
    return lib


def fused_topk_deepseek(
    scores: torch.Tensor,
    bias: torch.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    routed_scaling_factor: float,
    topk_values: torch.Tensor,
    topk_indices: torch.Tensor,
    routing_replay_out: torch.Tensor | None = None,
    *,
    use_ptx: bool = False,
) -> None:
    """Write normalized routing weights and expert IDs into CUDA output buffers."""
    if not isinstance(scores, torch.Tensor) or scores.ndim != 2:
        raise ValueError("scores must be a 2D tensor")
    if scores.device.type != "cuda" or scores.dtype not in _DTYPES:
        raise ValueError("scores must be a CUDA tensor with float32, float16 or bfloat16 dtype")
    tokens, experts = scores.shape
    for name, value in (("n_group", n_group), ("topk_group", topk_group), ("topk", topk)):
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")
    if (
        tokens > 2**31 - 1
        or not 2 <= experts <= 384
        or not 1 <= n_group <= 32
        or experts % n_group
        or experts // n_group < 2
        or not 1 <= topk_group <= n_group
        or not 1 <= topk <= 8
        or topk > experts // n_group * topk_group
    ):
        raise ValueError("unsupported routing dimensions or group configuration")
    if not math.isfinite(routed_scaling_factor):
        raise ValueError("routed_scaling_factor must be finite")
    if not isinstance(use_ptx, bool):
        raise TypeError("use_ptx must be a bool")

    tensors = [
        ("scores", scores, scores.dtype, (tokens, experts)),
        ("bias", bias, None, (experts,)),
        ("topk_values", topk_values, scores.dtype, (tokens, topk)),
        ("topk_indices", topk_indices, torch.int32, (tokens, topk)),
    ]
    for name, tensor, dtype, shape in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device != scores.device or (dtype is not None and tensor.dtype != dtype):
            raise ValueError(f"{name} has an invalid device or dtype")
        if name == "bias" and tensor.dtype not in _DTYPES:
            raise ValueError("bias must have float32, float16 or bfloat16 dtype")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.requires_grad:
            raise ValueError("fused routing does not support autograd")

    if routing_replay_out is not None:
        replay = routing_replay_out
        if not isinstance(replay, torch.Tensor):
            raise TypeError("routing_replay_out must be a torch.Tensor")
        if (
            replay.device != scores.device
            or replay.dtype != torch.int16
            or replay.ndim != 2
            or replay.shape[0] < tokens
            or replay.shape[1] != topk
        ):
            raise ValueError("routing_replay_out must be int16 [at least tokens, topk] on the same device")
        if not replay.is_contiguous():
            raise ValueError("routing_replay_out must be contiguous")

    with torch.cuda.device(scores.device):
        stream = torch.cuda.current_stream(scores.device)
        status = _library().cuinfer_moe_fused_routing(
            scores.data_ptr(), bias.data_ptr(), topk_values.data_ptr(),
            topk_indices.data_ptr(),
            routing_replay_out.data_ptr() if routing_replay_out is not None else None,
            tokens, experts, n_group, topk_group, topk, routed_scaling_factor,
            _DTYPES[scores.dtype], _DTYPES[bias.dtype], int(use_ptx), stream.cuda_stream,
        )
        if status:
            message = _library().cuinfer_moe_cuda_error_string(status).decode()
            raise RuntimeError(f"MoE CUDA launch failed: {message}")
        for _, tensor, _, _ in tensors:
            tensor.record_stream(stream)
        if routing_replay_out is not None:
            routing_replay_out.record_stream(stream)