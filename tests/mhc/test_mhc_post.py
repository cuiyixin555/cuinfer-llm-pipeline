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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class MhcPostTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability()[0] < 8:
            self.skipTest("BF16 requires SM80 or later")
        torch.manual_seed(42)

    def test_post(self):
        cases = [
            ((1,), 64, False), ((2, 3), 127, True),
            ((1,), 1536, False), ((1,), 2048, True),
            ((1,), 4096, True), ((1,), 7168, False),
            ((1,), 8192, True), ((1,), 16392, False),
        ]
        for outer, hidden, trailing_one in cases:
            with self.subTest(outer=outer, hidden=hidden):
                x = torch.randn(*outer, hidden, device="cuda", dtype=torch.bfloat16)
                residual = torch.randn(*outer, 4, hidden, device="cuda", dtype=torch.bfloat16)
                post = torch.randn(*outer, 4, device="cuda")
                comb = torch.randn(*outer, 4, 4, device="cuda")
                supplied = post[..., None] if trailing_one else post
                actual = mhc.mhc_post(x, residual, supplied, comb)
                mixed = torch.matmul(comb.transpose(-1, -2), residual.float())
                expected = (mixed + post[..., :, None] * x.float()[..., None, :]).bfloat16()
                self.assertEqual(actual.shape, residual.shape)
                self.assertEqual(actual.dtype, torch.bfloat16)
                assert_close(self, actual, expected, 1e-2, 0.005)

    def test_validation_and_noncontiguous_inputs(self):
        residual = torch.randn(2, 4, 64, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)[:, ::2]
        post = torch.randn(2, 4, device="cuda")
        comb = torch.randn(2, 4, 4, device="cuda")
        actual = mhc.mhc_post(x, residual, post, comb)
        expected = (torch.matmul(comb.transpose(-1, -2), residual.float())
                    + post[..., None] * x.float()[:, None, :]).bfloat16()
        assert_close(self, actual, expected, 1e-2, 0.005)
        with self.assertRaises(ValueError):
            mhc.mhc_post(x.float(), residual, post, comb)
        with self.assertRaises(ValueError):
            mhc.mhc_post(x, residual, post, comb[:, :, :3])

    def test_nondefault_stream(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x = torch.ones(2, 64, dtype=torch.bfloat16, device="cuda")
            residual = torch.ones(2, 4, 64, dtype=torch.bfloat16, device="cuda")
            post = torch.ones(2, 4, device="cuda")
            comb = torch.zeros(2, 4, 4, device="cuda")
            output = mhc.mhc_post(x, residual, post, comb)
        stream.synchronize()
        torch.testing.assert_close(output, residual)

    def test_empty_batch(self):
        residual = torch.empty(0, 4, 65, device="cuda", dtype=torch.bfloat16)
        output = mhc.mhc_post(
            torch.empty(0, 65, device="cuda", dtype=torch.bfloat16),
            residual, torch.empty(0, 4, device="cuda"),
            torch.empty(0, 4, 4, device="cuda"),
        )
        self.assertEqual(output.shape, residual.shape)


if __name__ == "__main__":
    unittest.main()
