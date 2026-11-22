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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""CUDA mHC operators for contiguous BF16 residual streams."""

import ctypes
import functools
import math
import os
from pathlib import Path

import torch


@functools.lru_cache(maxsize=1)
def _library() -> ctypes.CDLL:
    override = os.environ.get("CUINFER_MHC_LIBRARY")
    if override:
        path = Path(override)
    elif os.environ.get("TEST_SRCDIR") and os.environ.get("TEST_WORKSPACE"):
        path = (
            Path(os.environ["TEST_SRCDIR"])
            / os.environ["TEST_WORKSPACE"]
            / "calculator/mhc/libmhc.so"
        )
    else:
        path = Path(__file__).resolve().parents[2] / "bazel-bin/calculator/mhc/libmhc.so"
    lib = ctypes.CDLL(str(path))
    pointer = ctypes.c_void_p
    integer = ctypes.c_int
    real = ctypes.c_float
    lib.cuinfer_mhc_post.argtypes = [pointer] * 5 + [integer, integer, pointer]
    lib.cuinfer_mhc_post.restype = integer
    lib.cuinfer_mhc_pre.argtypes = (
        [pointer] * 8 + [integer] * 6 + [real] * 4 + [integer, pointer]
    )
    lib.cuinfer_mhc_pre.restype = integer
    lib.cuinfer_cuda_error_string.argtypes = [integer]
    lib.cuinfer_cuda_error_string.restype = ctypes.c_char_p
    return lib


def _check_result(status: int) -> None:
    if status:
        message = _library().cuinfer_cuda_error_string(status).decode()
        raise RuntimeError(f"mHC CUDA launch failed: {message}")


def _tensor(tensor: torch.Tensor, dtype: torch.dtype, device: torch.device) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("mHC inputs must be torch.Tensor objects")
    if tensor.device != device or tensor.device.type != "cuda" or tensor.dtype != dtype:
        raise ValueError(
            f"expected a {dtype} tensor on {device}, got {tensor.dtype} on {tensor.device}"
        )
    if tensor.requires_grad:
        raise ValueError("mHC operators do not support autograd")


def _residual_shape(residual: torch.Tensor) -> tuple[tuple[int, ...], int, int]:
    if not isinstance(residual, torch.Tensor):
        raise TypeError("residual must be a torch.Tensor")
    _tensor(residual, torch.bfloat16, residual.device)
    if residual.ndim < 3 or residual.shape[-2] != 4 or residual.shape[-1] <= 0:
        raise ValueError("residual must have shape [..., 4, H] with H > 0")
    outer = tuple(residual.shape[:-2])
    hidden = residual.shape[-1]
    tokens = math.prod(outer)
    if tokens > 2**31 - 1 or hidden > (2**31 - 1) // 4:
        raise ValueError("mHC dimensions exceed the supported integer range")
    return outer, hidden, tokens


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """Apply the mHC post-map, returning a BF16 tensor shaped like residual."""
    outer, hidden, tokens = _residual_shape(residual)
    for tensor, dtype in (
        (x, torch.bfloat16),
        (post_layer_mix, torch.float32),
        (comb_res_mix, torch.float32),
    ):
        _tensor(tensor, dtype, residual.device)
    if tuple(x.shape) != outer + (hidden,):
        raise ValueError("x shape must match residual's outer and hidden dimensions")
    if tuple(post_layer_mix.shape) not in (outer + (4,), outer + (4, 1)):
        raise ValueError("post_layer_mix must have shape [..., 4] or [..., 4, 1]")
    if tuple(comb_res_mix.shape) != outer + (4, 4):
        raise ValueError("comb_res_mix must have shape [..., 4, 4]")
    with torch.cuda.device(residual.device):
        x, residual, post_layer_mix, comb_res_mix = (
            t.contiguous() for t in (x, residual, post_layer_mix, comb_res_mix)
        )
        output = torch.empty_like(residual)
        stream = torch.cuda.current_stream(residual.device)
        status = _library().cuinfer_mhc_post(
            x.data_ptr(), residual.data_ptr(), post_layer_mix.data_ptr(),
            comb_res_mix.data_ptr(), output.data_ptr(), tokens, hidden,
            stream.cuda_stream,
        )
        _check_result(status)
        for tensor in (x, residual, post_layer_mix, comb_res_mix):
            tensor.record_stream(stream)
    return output


def _pre(
    dot_mix: torch.Tensor,
    square_sum: torch.Tensor | None,
    residual: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    k: int,
    splits: int,
    rms_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_multiplier: float,
    repeat: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outer, hidden, tokens = _residual_shape(residual)
    for tensor in (dot_mix, scale, base):
        _tensor(tensor, torch.float32, residual.device)
    if square_sum is not None:
        _tensor(square_sum, torch.float32, residual.device)
    if not isinstance(splits, int) or splits not in (1, 2, 4, 8, 16):
        raise ValueError("num_splits must be one of 1, 2, 4, 8, 16")
    if not isinstance(k, int) or not 0 < k <= 2**31 - 1:
        raise ValueError("k must be a positive 32-bit integer")
    if not isinstance(repeat, int) or not 0 < repeat <= 2**31 - 1:
        raise ValueError("sinkhorn_repeat must be a positive 32-bit integer")
    if not isinstance(block_size, int) or not 0 <= block_size <= 2**31 - 1:
        raise ValueError("block_size must be a nonnegative 32-bit integer")
    if any(not math.isfinite(v) or v <= 0 for v in (rms_eps, pre_eps, sinkhorn_eps)):
        raise ValueError("epsilon values must be finite and positive")
    if not math.isfinite(post_multiplier):
        raise ValueError("post multiplier must be finite")
    if tuple(scale.shape) != (3,) or tuple(base.shape) != (24,):
        raise ValueError("mhc_scale and mhc_base must have shapes [3] and [24]")
    expected_dot = outer + (24,)
    if square_sum is None:
        if splits != 1 or tuple(dot_mix.shape) not in (expected_dot, (1,) + expected_dot):
            raise ValueError("prenorm dot_mix must have shape [..., 24] or [1, ..., 24]")
    else:
        prefix = () if splits == 1 else (splits,)
        if tuple(dot_mix.shape) != prefix + expected_dot or tuple(square_sum.shape) != prefix + outer:
            raise ValueError("dot_mix or square_sum shape does not match num_splits and residual")

    with torch.cuda.device(residual.device):
        dot_mix, residual, scale, base = (
            t.contiguous() for t in (dot_mix, residual, scale, base)
        )
        if square_sum is not None:
            square_sum = square_sum.contiguous()
        post = torch.empty(outer + (4, 1), dtype=torch.float32, device=residual.device)
        comb = torch.empty(outer + (4, 4), dtype=torch.float32, device=residual.device)
        layer = torch.empty(outer + (hidden,), dtype=torch.bfloat16, device=residual.device)
        stream = torch.cuda.current_stream(residual.device)
        status = _library().cuinfer_mhc_pre(
            dot_mix.data_ptr(), square_sum.data_ptr() if square_sum is not None else None,
            residual.data_ptr(), scale.data_ptr(), base.data_ptr(), post.data_ptr(),
            comb.data_ptr(), layer.data_ptr(), tokens, hidden, k, splits, repeat, block_size,
            rms_eps, pre_eps, sinkhorn_eps, post_multiplier, int(square_sum is None),
            stream.cuda_stream,
        )
        _check_result(status)
        for tensor in (dot_mix, residual, scale, base):
            tensor.record_stream(stream)
        if square_sum is not None:
            square_sum.record_stream(stream)
    return post, comb, layer


def mhc_pre_big_fuse(
    dot_mix: torch.Tensor,
    sqrsum: torch.Tensor,
    residual: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    k: int,
    rms_eps: float = 1e-6,
    mhc_pre_eps: float = 1e-6,
    mhc_sinkhorn_eps: float = 1e-6,
    mhc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
    num_splits: int = 1,
    block_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute mHC pre-map from external projection and square sums."""
    if not isinstance(sqrsum, torch.Tensor):
        raise TypeError("sqrsum must be a torch.Tensor")
    return _pre(dot_mix, sqrsum, residual, mhc_scale, mhc_base, k, num_splits,
                rms_eps, mhc_pre_eps, mhc_sinkhorn_eps, mhc_post_mult_value,
                sinkhorn_repeat, block_size)


def mhc_pre_big_fuse_with_prenorm(
    dot_mix: torch.Tensor,
    residual: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float = 1e-6,
    mhc_pre_eps: float = 1e-6,
    mhc_sinkhorn_eps: float = 1e-6,
    mhc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
    block_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute mHC pre-map and the residual square sum on the GPU."""
    _, hidden, _ = _residual_shape(residual)
    return _pre(dot_mix, None, residual, mhc_scale, mhc_base, 4 * hidden, 1,
                rms_eps, mhc_pre_eps, mhc_sinkhorn_eps, mhc_post_mult_value,
                sinkhorn_repeat, block_size)