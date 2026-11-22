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

import itertools
import unittest

import torch

from cuinfer_llm import moe


def reference(scores, bias, groups, selected_groups, top_k, scale):
    tokens, experts = scores.shape
    per_group = experts // groups
    activated = scores.float().sigmoid()
    ranked = activated + bias.float()
    grouped = ranked.reshape(tokens, groups, per_group)
    top_two = torch.sort(grouped, dim=-1, descending=True, stable=True).values[..., :2]
    group_order = torch.argsort(top_two.sum(dim=-1), dim=-1, descending=True, stable=True)
    group_mask = torch.zeros((tokens, groups), dtype=torch.bool, device=scores.device)
    group_mask.scatter_(1, group_order[:, :selected_groups], True)
    masked = ranked.masked_fill(~group_mask.repeat_interleave(per_group, dim=1), -torch.inf)
    indices = torch.argsort(masked, dim=-1, descending=True, stable=True)[:, :top_k]
    selected = activated.gather(1, indices)
    weights = (selected.double() * scale / (selected.sum(dim=1, keepdim=True).double() + 1e-20))
    return weights.to(scores.dtype), indices.to(torch.int32)


def run_routing(scores, bias, groups, selected_groups, top_k, scale, use_ptx, replay=True):
    tokens = scores.shape[0]
    weights = torch.empty((tokens, top_k), device=scores.device, dtype=scores.dtype)
    indices = torch.empty((tokens, top_k), device=scores.device, dtype=torch.int32)
    recorded = (
        torch.full((tokens + 2, top_k), -1, device=scores.device, dtype=torch.int16)
        if replay else None
    )
    moe.fused_topk_deepseek(
        scores, bias, groups, selected_groups, top_k, scale, weights, indices,
        recorded, use_ptx=use_ptx,
    )
    return weights, indices, recorded


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FusedRoutingTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability()[0] < 8:
            self.skipTest("BF16 requires SM80 or later")
        torch.manual_seed(47)

    def test_reference_for_dtypes_and_configurations(self):
        configs = [
            (7, 256, 8, 4, 8),
            (7, 128, 4, 2, 4),
            (7, 96, 3, 2, 5),
            (3, 384, 1, 1, 1),
            (3, 128, 1, 1, 8),
            (3, 64, 32, 3, 5),
        ]
        dtypes = (torch.float32, torch.float16, torch.bfloat16)
        for config, score_type, bias_type, use_ptx in itertools.product(
            configs, dtypes, dtypes, (False, True)
        ):
            with self.subTest(config=config, score_type=score_type, bias_type=bias_type, ptx=use_ptx):
                tokens, experts, groups, selected_groups, top_k = config
                scores = torch.randn(tokens, experts, device="cuda").to(score_type)
                bias = (torch.randperm(experts, device="cuda").float() / 8).to(bias_type)
                weights, ids, replay = run_routing(
                    scores, bias, groups, selected_groups, top_k, 2.5, use_ptx
                )
                expected_weights, expected_ids = reference(
                    scores, bias, groups, selected_groups, top_k, 2.5
                )
                torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
                torch.testing.assert_close(weights, expected_weights, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(replay[:tokens].int(), ids, rtol=0, atol=0)
                self.assertTrue(torch.all(replay[tokens:] == -1).item())

    def test_negative_scores_ties_and_optional_replay(self):
        for use_ptx in (False, True):
            scores = torch.zeros((1, 8), device="cuda")
            bias = torch.tensor([2.5, 1.5, 0.5, -1.5, 0.0, -0.1, -0.2, -0.3], device="cuda")
            weights, ids, replay = run_routing(scores, bias, 2, 1, 4, 2.0, use_ptx, False)
            self.assertIsNone(replay)
            self.assertEqual(ids.tolist(), [[0, 1, 2, 3]])
            torch.testing.assert_close(weights, torch.full_like(weights, 0.5))

            tie_scores = torch.zeros((1, 16), device="cuda")
            tie_bias = torch.zeros(16, device="cuda")
            _, tied_ids, _ = run_routing(tie_scores, tie_bias, 4, 2, 8, 1.0, use_ptx)
            self.assertEqual(tied_ids.tolist(), [list(range(8))])

            for dtype, logit in itertools.product(
                (torch.float32, torch.float16, torch.bfloat16), (-20.0, -100.0)
            ):
                negative = torch.full((3, 256), logit, dtype=dtype, device="cuda")
                ordered_bias = -torch.arange(256, device="cuda").float() / 8
                weights, ids, _ = run_routing(negative, ordered_bias, 8, 4, 8, 2.5, use_ptx)
                self.assertEqual(ids[0].tolist(), list(range(8)))
                expected = 0.3125 if logit == -20.0 else 0.0
                torch.testing.assert_close(weights, torch.full_like(weights, expected), rtol=0, atol=0)

    def test_empty_batch_and_grid_stride(self):
        for use_ptx in (False, True):
            bias = torch.tensor([0.0, 0.5], device="cuda")
            empty = torch.empty((0, 2), device="cuda")
            weights, ids, replay = run_routing(empty, bias, 1, 1, 1, 1.0, use_ptx)
            self.assertEqual(tuple(weights.shape), (0, 1))
            self.assertEqual(tuple(ids.shape), (0, 1))
            self.assertTrue(torch.all(replay == -1).item())
            self.assertEqual(torch.ones(1, device="cuda").item(), 1)

            scores = torch.zeros((65536, 2), device="cuda")
            scores[-1] = torch.tensor([10.0, -10.0], device="cuda")
            weights = torch.full((65536, 1), -1.0, device="cuda")
            ids = torch.full((65536, 1), -1, device="cuda", dtype=torch.int32)
            moe.fused_topk_deepseek(scores, bias, 1, 1, 1, 1.0, weights, ids, use_ptx=use_ptx)
            self.assertTrue(torch.all(ids[:-1] == 1).item())
            self.assertEqual(ids[-1].item(), 0)
            torch.testing.assert_close(weights, torch.ones_like(weights), rtol=0, atol=0)

    def test_nondefault_stream_and_cuda_graph(self):
        for use_ptx in (False, True):
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                scores = torch.randn((4, 32), device="cuda")
                bias = torch.arange(32, device="cuda").float()
                weights, ids, replay = run_routing(scores, bias, 4, 2, 4, 2.0, use_ptx)
            stream.synchronize()
            expected_weights, expected_ids = reference(scores, bias, 4, 2, 4, 2.0)
            torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
            torch.testing.assert_close(weights, expected_weights, rtol=1e-5, atol=1e-6)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                moe.fused_topk_deepseek(scores, bias, 4, 2, 4, 2.0, weights, ids, replay, use_ptx=use_ptx)
            scores.fill_(-20.0)
            graph.replay()
            torch.cuda.synchronize()
            expected_weights, expected_ids = reference(scores, bias, 4, 2, 4, 2.0)
            torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
            torch.testing.assert_close(weights, expected_weights, rtol=1e-5, atol=1e-6)

    def test_validation(self):
        scores = torch.zeros((2, 8), device="cuda")
        bias = torch.zeros(8, device="cuda")
        weights = torch.empty((2, 2), device="cuda")
        ids = torch.empty((2, 2), device="cuda", dtype=torch.int32)
        replay = torch.empty((2, 2), device="cuda", dtype=torch.int16)
        args = [scores, bias, 2, 1, 2, 1.0, weights, ids, replay]
        for index in (0, 1, 6, 7, 8):
            original = args[index]
            args[index] = torch.empty(
                (*original.shape[:-1], original.shape[-1] * 2),
                device="cuda", dtype=original.dtype,
            )[..., ::2]
            with self.subTest(strided=index), self.assertRaisesRegex(ValueError, "contiguous"):
                moe.fused_topk_deepseek(*args)
            args[index] = original
        for index, value in ((2, 0), (3, 3), (4, 9), (5, float("inf"))):
            original = args[index]
            args[index] = value
            with self.subTest(index=index), self.assertRaises(ValueError):
                moe.fused_topk_deepseek(*args)
            args[index] = original
        with self.assertRaises(ValueError):
            moe.fused_topk_deepseek(scores, bias.cpu(), 2, 1, 2, 1.0, weights, ids)
        with self.assertRaises(ValueError):
            moe.fused_topk_deepseek(scores, bias[:-1], 2, 1, 2, 1.0, weights, ids)
        with self.assertRaises(ValueError):
            moe.fused_topk_deepseek(scores, bias, 2, 1, 2, 1.0, weights, ids.long())
        for invalid_replay in (replay[:1], replay[:, :1], replay.int()):
            with self.assertRaises(ValueError):
                moe.fused_topk_deepseek(
                    scores, bias, 2, 1, 2, 1.0, weights, ids, invalid_replay
                )
        with self.assertRaises(ValueError):
            moe.fused_topk_deepseek(scores.requires_grad_(), bias, 2, 1, 2, 1.0, weights, ids)


if __name__ == "__main__":
    unittest.main()