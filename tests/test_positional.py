import pytest
import torch
from transformers import AutoTokenizer, EsmConfig, EsmForMaskedLM

from cla.layout.concat import render_naive
from cla.layout.geometry import gap_at_chain_breaks
from cla.layout.spec import Chain, ChainSpec, Role
from cla.model.positional import RESET_BASE, ChainAwarePositions

SPEC = ChainSpec(
    chains=(
        Chain(Role.TCRA, "CAVRDSNYQLIW"),
        Chain(Role.TCRB, "CASSLAPGATNEKLFF"),
        Chain(Role.PEPTIDE, "GILGFVFTL"),
        Chain(Role.MHC_HEAVY, "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQ"),
        Chain(Role.B2M, "IQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSF"),
    ),
    example_id="ex0",
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")


def _module(**flags):
    base = {"reset": False, "chain_embedding": False, "offset": None}
    return ChainAwarePositions(16, **{**base, **flags})


def test_all_off_is_plain_arange_even_with_separators(tokenizer):
    r = render_naive(SPEC, tokenizer, separator_tokens=2, symmetric_ends=True)
    out = _module().apply(r)
    assert out.position_ids is None and r.position_ids is None


def test_reset_gives_every_chain_the_same_indices_under_any_order(tokenizer):
    # Holds for the position indices of each chain. It does not extend to hidden states:
    # concatenation order still decides which chains a residue attends to and in what
    # sequence, so full embedding invariance under P1 is not a property of this module.
    reset = _module(reset=True)
    seen: dict[str, list[torch.Tensor]] = {}
    for order in [(0, 1, 2, 3, 4), (4, 3, 2, 1, 0), (2, 0, 4, 1, 3)]:
        r = render_naive(SPEC.reordered(order), tokenizer, separator_tokens=1, symmetric_ends=True)
        pos = reset.position_ids(r)
        for role, sl in r.chain_slices.items():
            seen.setdefault(role, []).append(pos[sl])
    for role, runs in seen.items():
        expected = RESET_BASE + torch.arange(len(runs[0]))
        assert all(torch.equal(p, expected) for p in runs), role

    # With an offset the within-chain indices stay order-invariant while the absolute
    # index of a chain shifts with its rank.
    both = _module(reset=True, offset=200)
    a = render_naive(SPEC, tokenizer)
    b = render_naive(SPEC.reordered((4, 3, 2, 1, 0)), tokenizer)
    for r in (a, b):
        pos = both.position_ids(r)
        for sl in r.chain_slices.values():
            assert torch.equal(pos[sl] - pos[sl.start], torch.arange(sl.stop - sl.start))
    pa, pb = (
        both.position_ids(a)[a.chain_slices["TCRA"]],
        both.position_ids(b)[b.chain_slices["TCRA"]],
    )
    assert pa[0] == RESET_BASE and pb[0] == RESET_BASE + 4 * 200


def test_each_flag_changes_output_alone_and_only_its_own_output(tokenizer):
    r = render_naive(SPEC, tokenizer, separator_tokens=1, symmetric_ends=True)
    base = r.position_ids

    reset = _module(reset=True).position_ids(r)
    offset = _module(offset=50).position_ids(r)
    assert not torch.equal(reset, torch.arange(len(reset)))
    assert not torch.equal(offset, torch.arange(len(offset)))
    assert not torch.equal(reset, offset)
    assert base is None
    # An offset alone is exactly the Phase 1 G1 operator, so audit numbers transfer.
    assert torch.equal(offset, gap_at_chain_breaks(r, 50).position_ids)

    config = EsmConfig(
        vocab_size=33, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, position_embedding_type="rotary", pad_token_id=1, mask_token_id=32,
    )  # fmt: skip
    model = EsmForMaskedLM(config).eval()
    emb = _module(chain_embedding=True)
    torch.nn.init.normal_(emb.role_embedding.weight)
    ids = r.input_ids[None]
    with torch.no_grad():
        plain = model.esm.embeddings(input_ids=ids)
        emb.attach(model)
        unbound = model.esm.embeddings(input_ids=ids)
        with emb.bound(emb.chain_ids(r)[None]):
            bound = model.esm.embeddings(input_ids=ids)
        emb.detach()
        detached = model.esm.embeddings(input_ids=ids)
    assert torch.equal(unbound, plain) and torch.equal(detached, plain)
    assert not torch.equal(bound, plain)
    assert emb.position_ids(r) is None


def test_learned_offsets_are_trainable_and_start_at_the_constant(tokenizer):
    r = render_naive(SPEC, tokenizer)
    m = _module(offset=100, learned_offset=True)
    pos = m.position_ids(r)
    assert torch.equal(pos.long(), gap_at_chain_breaks(r, 100).position_ids)
    pos.sum().backward()
    assert m.pair_offsets.grad.abs().sum() > 0


def test_chains_with_clipped_slices_are_skipped(tokenizer):
    spec = ChainSpec(
        chains=(Chain(Role.PEPTIDE, "GILGFVFTL"), Chain(Role.TCRB, "CASSLAPG")), example_id="x"
    )
    r = render_naive(spec, tokenizer, max_length=12, strict=False)
    pos = _module(reset=True, offset=10).position_ids(r)
    assert pos.shape == r.input_ids.shape
