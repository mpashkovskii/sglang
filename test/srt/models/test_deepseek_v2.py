import unittest
from unittest.mock import MagicMock, patch

import sglang.srt.models.deepseek_v2 as deepseek_v2
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE


class MockConfig:
    def __init__(self):
        self.n_routed_experts = 8
        self.n_shared_experts = 2
        self.num_experts_per_tok = 2
        self.n_group = 4
        self.topk_group = 1
        self.moe_intermediate_size = 4096
        self.hidden_size = 4096
        self.hidden_act = "silu"
        self.norm_topk_prob = True
        self.routed_scaling_factor = 1.0
        self.topk_method = "noaux_tc"


@patch("sglang.srt.models.deepseek_v2.get_bool_env_var", return_value=True)
@patch("sglang.srt.distributed.get_tensor_model_parallel_world_size", return_value=1)
@patch("sglang.srt.distributed.get_tensor_model_parallel_rank", return_value=0)
@patch(
    "sglang.srt.distributed.parallel_state.get_tp_group",
    return_value=MagicMock(world_size=1),
)
class TestDeepseekV2MoE(unittest.TestCase):
    def setUp(self):
        deepseek_v2.AiterTopKRoutingBuffersInstance = None
        self.config = MockConfig()

    def test_fp8moe_blockscale_buffers_registration(self, *_):
        deepseek_v2._use_aiter = True
        moe0 = deepseek_v2.DeepseekV2MoE(self.config, prefix="moe", layer_id="layer0")

        # Check if the experts is a FusedMoE instance
        self.assertIsInstance(moe0.experts, FusedMoE)

        # Verify the buffers are registered in the experts
        self.assertTrue(hasattr(moe0.experts, "total_topk_ids"))
        self.assertTrue(hasattr(moe0.experts, "non_shared_topk_ids"))
        self.assertTrue(hasattr(moe0.experts, "total_topk_weights"))
        self.assertTrue(hasattr(moe0.experts, "non_shared_topk_weights"))

        # Verify the buffers have the expected shape
        self.assertEqual(
            moe0.experts.total_topk_ids.shape[1],
            self.config.num_experts_per_tok + self.config.n_shared_experts,
        )

        # Verify singleton is working
        self.assertIsNotNone(deepseek_v2.AiterTopKRoutingBuffersInstance)

        # Create another MoE instance should use the same buffers
        moe1 = deepseek_v2.DeepseekV2MoE(self.config, prefix="moe", layer_id="layer1")
        self.assertIs(moe0.experts.total_topk_ids, moe1.experts.total_topk_ids)
        self.assertIs(
            moe0.experts.non_shared_topk_ids, moe1.experts.non_shared_topk_ids
        )
        self.assertIs(moe0.experts.total_topk_weights, moe1.experts.total_topk_weights)
        self.assertIs(
            moe0.experts.non_shared_topk_weights, moe1.experts.non_shared_topk_weights
        )

    def test_no_fp8moe_blockscale_buffers_registration(self, *_):
        deepseek_v2._use_aiter = False
        moe0 = deepseek_v2.DeepseekV2MoE(self.config, prefix="moe", layer_id="layer2")

        # Check if the experts is a FusedMoE instance
        self.assertIsInstance(moe0.experts, FusedMoE)

        # Verify the buffers are not registered in the experts
        self.assertFalse(hasattr(moe0.experts, "total_topk_ids"))
        self.assertFalse(hasattr(moe0.experts, "non_shared_topk_ids"))
        self.assertFalse(hasattr(moe0.experts, "total_topk_weights"))
        self.assertFalse(hasattr(moe0.experts, "non_shared_topk_weights"))

        # Verify singleton is missing
        self.assertIsNone(deepseek_v2.AiterTopKRoutingBuffersInstance)


if __name__ == "__main__":
    unittest.main()
