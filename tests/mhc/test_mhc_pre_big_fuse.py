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

import unittest

import torch

from cuinfer_llm import mhc
from mhc_test_utils import assert_close


def reference_pre(dot, square, residual, scale, base, k, splits,
                  rms_eps=1e-6, pre_eps=1e-6, sinkhorn_eps=1e-6,
                  post_multiplier=1.0, repeat=20):
    if splits != 1:
        dot = dot.sum(0)
        square = square.sum(0)
    normalized = dot * torch.rsqrt(square[..., None] / k + rms_eps)
    pre = torch.sigmoid(normalized[..., :4] * scale[0] + base[:4]) + pre_eps
    post = post_multiplier * torch.sigmoid(normalized[..., 4:8] * scale[1] + base[4:8])[..., None]
    matrix = (normalized[..., 8:] * scale[2] + base[8:]).reshape(*residual.shape[:-2], 4, 4)
    matrix = torch.softmax(matrix, dim=-1) + sinkhorn_eps
    matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + sinkhorn_eps)
    for _ in range(repeat - 1):
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + sinkhorn_eps)
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + sinkhorn_eps)
    layer = (pre[..., :, None] * residual.float()).sum(dim=-2).bfloat16()
    return post, matrix, layer


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class MhcPreTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability()[0] < 8:
            self.skipTest("BF16 requires SM80 or later")
        torch.manual_seed(42)

    def test_pre_with_external_sums(self):
        for outer, hidden in [((2,), 64), ((1,), 4096), ((2, 3), 128)]:
            for splits in (1, 2, 4, 8, 16):
                with self.subTest(outer=outer, hidden=hidden, splits=splits):
                    residual = (torch.randn(*outer, 4, hidden, device="cuda") * 0.01).bfloat16()
                    scale = torch.randn(3, device="cuda") * 0.1
                    base = torch.randn(24, device="cuda") * 0.1
                    prefix = () if splits == 1 else (splits,)
                    dot = torch.randn(*prefix, *outer, 24, device="cuda") * 0.01
                    square = torch.rand(*prefix, *outer, device="cuda") * (4 * hidden / splits)
                    actual = mhc.mhc_pre_big_fuse(dot, square, residual, scale, base,
                                                  4 * hidden, num_splits=splits)
                    expected = reference_pre(dot, square, residual, scale, base, 4 * hidden, splits)
                    self.check_pre(actual, expected, outer, hidden)

    def test_pre_with_internal_sums(self):
        for outer, hidden, leading in [((2,), 64, False), ((1,), 4096, True),
                                       ((4,), 7168, False)]:
            with self.subTest(outer=outer, hidden=hidden):
                residual = (torch.randn(*outer, 4, hidden, device="cuda") * 0.01).bfloat16()
                scale = torch.randn(3, device="cuda") * 0.1
                base = torch.randn(24, device="cuda") * 0.1
                dot = torch.randn(*outer, 24, device="cuda") * 0.01
                actual = mhc.mhc_pre_big_fuse_with_prenorm(
                    dot.unsqueeze(0) if leading else dot, residual, scale, base
                )
                square = residual.float().square().sum(dim=(-2, -1))
                expected = reference_pre(dot, square, residual, scale, base, 4 * hidden, 1)
                self.check_pre(actual, expected, outer, hidden)

    def check_pre(self, actual, expected, outer, hidden):
        for got, ref, shape, dtype, tol in zip(
            actual, expected, [outer + (4, 1), outer + (4, 4), outer + (hidden,)],
            [torch.float32, torch.float32, torch.bfloat16], [2e-3, 2e-3, 1e-2],
        ):
            self.assertEqual(got.shape, shape)
            self.assertEqual(got.dtype, dtype)
            assert_close(self, got, ref, tol, 0.006)

    def test_validation(self):
        residual = torch.randn(2, 4, 64, device="cuda", dtype=torch.bfloat16)
        dot = torch.zeros(2, 24, device="cuda")
        square = torch.ones(2, device="cuda")
        with self.assertRaises(ValueError):
            mhc.mhc_pre_big_fuse(dot, square, residual, torch.ones(3, device="cuda"),
                                 torch.zeros(24, device="cuda"), 256, num_splits=3)
        with self.assertRaises(ValueError):
            mhc.mhc_pre_big_fuse(dot, square, residual, torch.ones(3, device="cuda"),
                                 torch.zeros(24, device="cuda"), 256, rms_eps=float("inf"))

    def test_nondefault_stream(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            residual = torch.ones(2, 4, 64, dtype=torch.bfloat16, device="cuda")
            dot = torch.zeros(2, 24, device="cuda")
            scale = torch.ones(3, device="cuda")
            base = torch.zeros(24, device="cuda")
            pre = mhc.mhc_pre_big_fuse_with_prenorm(dot, residual, scale, base)
        stream.synchronize()
        torch.testing.assert_close(pre[0], torch.full_like(pre[0], 0.5))
        torch.testing.assert_close(pre[2], torch.full_like(pre[2], 2.0))

    def test_custom_parameters_and_empty_batch(self):
        residual = (torch.randn(2, 4, 65, device="cuda") * 0.1).bfloat16()
        dot = torch.randn(2, 24, device="cuda")
        square = torch.rand(2, device="cuda") + 0.1
        scale = torch.tensor([0.3, 0.6, 2.0], device="cuda")
        base = torch.linspace(-1, 1, 24, device="cuda")
        expected = reference_pre(dot, square, residual, scale, base, 123, 1,
                                 0.02, 0.03, 0.04, 1.7, 3)
        for block_size in (0, 128, 256, 512):
            with self.subTest(block_size=block_size):
                actual = mhc.mhc_pre_big_fuse(
                    dot, square, residual, scale, base, 123,
                    rms_eps=0.02, mhc_pre_eps=0.03, mhc_sinkhorn_eps=0.04,
                    mhc_post_mult_value=1.7, sinkhorn_repeat=3, block_size=block_size,
                )
                self.check_pre(actual, expected, (2,), 65)
                prenorm = mhc.mhc_pre_big_fuse_with_prenorm(
                    dot, residual, scale, base, rms_eps=0.02, mhc_pre_eps=0.03,
                    mhc_sinkhorn_eps=0.04, mhc_post_mult_value=1.7,
                    sinkhorn_repeat=3, block_size=block_size,
                )
                prenorm_square = residual.float().square().sum(dim=(-2, -1))
                prenorm_ref = reference_pre(dot, prenorm_square, residual, scale, base,
                                            260, 1, 0.02, 0.03, 0.04, 1.7, 3)
                self.check_pre(prenorm, prenorm_ref, (2,), 65)
        single_iteration = mhc.mhc_pre_big_fuse(
            dot, square, residual, scale, base, 123, sinkhorn_repeat=1
        )
        self.check_pre(single_iteration,
                       reference_pre(dot, square, residual, scale, base, 123, 1, repeat=1),
                       (2,), 65)
        empty = residual[:0]
        result = mhc.mhc_pre_big_fuse(
            dot[:0], square[:0], empty, scale, base, 123)
        self.assertEqual([x.shape for x in result],
                         [torch.Size((0, 4, 1)), torch.Size((0, 4, 4)),
                          torch.Size((0, 65))])
        with self.assertRaises(TypeError):
            mhc.mhc_pre_big_fuse(dot, None, residual, scale, base, 123)


if __name__ == "__main__":
    unittest.main()
