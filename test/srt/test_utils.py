import unittest
import torch.nn as nn

from sglang.srt.utils import AiterTopKRoutingBuffers


class TestAiterTopKRoutingBuffers(unittest.TestCase):
    def test_initialization(self):
        param_list = [
            # top_k, n_routed_experts, n_shared_experts, mandatory_shared_expert_ids
            (1, 4, 1, [4]),
            (2, 6, 2, [6, 7]),
            (3, 8, 3, [8, 9, 10]),
        ]
        for top_k, n_routed_experts, n_shared_experts, mandatory_shared_expert_ids in param_list:
            with self.subTest(top_k=top_k, n_routed_experts=n_routed_experts, n_shared_experts=n_shared_experts, mandatory_shared_expert_ids=mandatory_shared_expert_ids):

                buffers = AiterTopKRoutingBuffers(
                    top_k=top_k,
                    n_routed_experts=n_routed_experts,
                    n_shared_experts=n_shared_experts
                )
                
                # Test IDs
                self.assertEqual(
                    buffers.total_topk_ids.shape,
                    (AiterTopKRoutingBuffers.MAX_NUM_TOKENS, top_k + n_shared_experts)
                )
                self.assertEqual(buffers.non_shared_topk_ids.shape, (AiterTopKRoutingBuffers.MAX_NUM_TOKENS, top_k))
                
                first_token_expert_ids = buffers.total_topk_ids[0].tolist()
                first_token_non_shared_experts_ids = first_token_expert_ids[:-top_k]
                self.assertEqual(first_token_non_shared_experts_ids, [0] * top_k, "Non shared expert ids should be empty")

                first_token_shared_experts_ids = first_token_expert_ids[-top_k:]
                self.assertEqual(first_token_shared_experts_ids, mandatory_shared_expert_ids, "Each token has to be also routed to all shared experts")

                # Test weights
                self.assertEqual(
                    buffers.total_topk_weights.shape,
                    (AiterTopKRoutingBuffers.MAX_NUM_TOKENS, top_k + n_shared_experts)
                )
                
                self.assertEqual(buffers.non_shared_topk_weights.shape, (AiterTopKRoutingBuffers.MAX_NUM_TOKENS, top_k))

                first_token_shared_weights = buffers.total_topk_weights[0].tolist()[-top_k:]
                self.assertTrue(
                    all(weight == AiterTopKRoutingBuffers.SHARED_EXPERTS_SCORE for weight in first_token_shared_weights),
                    "Shared experts weights has to be initiated with SHARED_EXPERTS_SCORE"
                )

    def test_register_in_layer(self):
        top_k = 2
        n_routed_experts = 8
        n_shared_experts = 2
        buffers = AiterTopKRoutingBuffers(
            top_k=top_k,
            n_routed_experts=n_routed_experts,
            n_shared_experts=n_shared_experts
        )
        
        layer = nn.Module()
        
        # Register buffers in the layer
        buffers.register_in_layer(layer)
        
        # Check that buffers are registered correctly
        self.assertTrue(hasattr(layer, "total_topk_ids"))
        self.assertTrue(hasattr(layer, "non_shared_topk_ids"))
        self.assertTrue(hasattr(layer, "total_topk_weights"))
        self.assertTrue(hasattr(layer, "non_shared_topk_weights"))

        # Check that the buffers are registered as non-persistent buffers
        model_dict = layer.state_dict()
        self.assertNotIn("total_topk_ids", model_dict)
        self.assertNotIn("non_shared_topk_ids", model_dict)
        self.assertNotIn("total_topk_weights", model_dict)
        self.assertNotIn("non_shared_topk_weights", model_dict)


if __name__ == "__main__":
    unittest.main()
