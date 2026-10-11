import torch
from transformers import EsmConfig, EsmForMaskedLM

from cla.train.lora import ARMS, HPARAMS, build_arm, trainable_counts

TINY = EsmConfig(
    vocab_size=33, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
    intermediate_size=64, pad_token_id=1, mask_token_id=32, position_embedding_type="rotary",
)  # fmt: skip


def test_every_arm_trains_the_same_number_of_parameters_and_padding_is_inert():
    counts = {}
    for arm in ARMS:
        torch.manual_seed(0)
        bound = build_arm(EsmForMaskedLM(TINY), arm, HPARAMS)
        counts[arm] = trainable_counts(bound)
        has_embedding = ARMS[arm]["chain_embedding"]
        assert (counts[arm]["padding"] == 0) == has_embedding
        assert (counts[arm]["role_embedding"] > 0) == has_embedding
        if bound.padding is not None:
            ids = torch.tensor([[0, 4, 5, 2]])
            out = bound(ids, torch.ones_like(ids), chain_ids=torch.tensor([[0, 1, 2, 0]]))
            out.logits.sum().backward()
            assert bound.padding.weight.grad is None
    assert len({c["total"] for c in counts.values()}) == 1
    assert len({c["lora"] for c in counts.values()}) == 1
