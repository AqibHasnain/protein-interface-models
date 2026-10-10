import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

from cla.layout.concat import render_naive
from cla.layout.geometry import GEOMETRY_OPERATORS
from cla.layout.perturb import (
    CONTROLS,
    OPERATORS,
    OperatorNotApplicable,
    insert_neutral_spacer,
    mutate_context_residues,
    permute_chain_order,
    resample_chain_length,
    restore_native_flank,
    vary_separator_count,
)
from cla.layout.spec import Chain, ChainSpec, Role

SPEC = ChainSpec(
    chains=(
        Chain(Role.TCRA, "CAVRDSNYQLIW"),
        Chain(Role.TCRB, "CASSLAPGATNEKLFF"),
        Chain(Role.PEPTIDE, "GILGFVFTL"),
        Chain(Role.MHC_HEAVY, "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQ"),
        Chain(
            Role.B2M,
            "IQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSF",
        ),
    ),
    example_id="ex0",
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")


@pytest.mark.parametrize("op", [permute_chain_order, vary_separator_count, insert_neutral_spacer])
def test_null_operators_preserve_identity(op):
    for seed in range(20):
        out, _ = op(SPEC, np.random.default_rng(seed))
        assert out.identity == SPEC.identity


def test_permutation_is_never_identity_order():
    for seed in range(50):
        out, _ = permute_chain_order(SPEC, np.random.default_rng(seed))
        assert out.chains != SPEC.chains


def test_positive_control_changes_identity():
    pool = ["CASSYQGAEQFF", "CASSLAPGATNEKLFF", "CASRDRVNTEAFF"]
    out, _ = resample_chain_length(SPEC, np.random.default_rng(0), pool=pool)
    assert out.identity != SPEC.identity
    assert len(out.by_role(Role.TCRB)) != len(SPEC.by_role(Role.TCRB))
    with pytest.raises(ValueError):
        resample_chain_length(SPEC, np.random.default_rng(0))


def test_native_flank_not_applicable_without_flanks():
    with pytest.raises(OperatorNotApplicable):
        restore_native_flank(SPEC, np.random.default_rng(0))


def test_registry_keys():
    assert sorted(OPERATORS) == ["P1", "P2", "P3", "P4", "P5"]
    assert sorted(CONTROLS) == ["C1", "C2"]
    assert sorted(GEOMETRY_OPERATORS) == ["G1", "G2"]


@pytest.mark.parametrize("baseline", [0, 1])
def test_separator_draw_never_returns_baseline(baseline):
    draws = {
        vary_separator_count(SPEC, np.random.default_rng(s), baseline=baseline)[1][
            "separator_tokens"
        ]
        for s in range(100)
    }
    assert baseline not in draws and len(draws) > 1


@pytest.mark.parametrize("target", [Role.PEPTIDE, Role.TCRB, Role.MHC_HEAVY])
@pytest.mark.parametrize("n", [1, 5])
def test_context_mutation_changes_exactly_n_residues_outside_target(target, n):
    out, _ = mutate_context_residues(
        SPEC, np.random.default_rng(0), n_mutations=n, target_role=target
    )
    assert out.identity != SPEC.identity
    diffs = [
        (a.role, x != y)
        for a, b in zip(SPEC.chains, out.chains, strict=True)
        for x, y in zip(a.sequence, b.sequence, strict=True)
    ]
    assert sum(changed for _, changed in diffs) == n
    assert not any(changed for role, changed in diffs if role is target)


@pytest.mark.parametrize("sep", [0, 1, 5, 25])
@pytest.mark.parametrize("spacer", [0, 10])
def test_chain_slices_recover_chain_sequences(tokenizer, sep, spacer):
    r = render_naive(SPEC, tokenizer, separator_tokens=sep, spacer_before_peptide=spacer)
    for chain in SPEC.chains:
        ids = r.input_ids[r.chain_slices[chain.role.value]]
        decoded = tokenizer.decode(ids, skip_special_tokens=False).replace(" ", "")
        assert decoded == chain.sequence
    assert r.input_ids[0] == tokenizer.cls_token_id
    assert r.input_ids[-1] == tokenizer.eos_token_id
    assert bool(r.attention_mask.all())


def test_overrides_render_and_strict_truncation(tokenizer):
    spec, spacer = insert_neutral_spacer(SPEC, np.random.default_rng(0))
    base = render_naive(SPEC, tokenizer, separator_tokens=1)
    spaced = render_naive(spec, tokenizer, separator_tokens=1, **spacer)
    assert len(spaced.input_ids) == len(base.input_ids) + spacer["spacer_before_peptide"]
    with pytest.raises(ValueError):
        render_naive(SPEC, tokenizer, max_length=50)
    clipped = render_naive(SPEC, tokenizer, max_length=50, strict=False)
    assert len(clipped.input_ids) == 50


# Nullness guarantee for geometry operators: the token tensors are equal, so no content
# differs, and only position_ids moves. This subsumes any multiset-of-chains check.
@pytest.mark.parametrize("op", sorted(GEOMETRY_OPERATORS))
@pytest.mark.parametrize("gap", [1, 5, 25, 200])
@pytest.mark.parametrize("sep", [0, 3])
def test_geometry_operators_change_only_position_ids(tokenizer, op, gap, sep):
    original = render_naive(SPEC, tokenizer, separator_tokens=sep)
    perturbed = GEOMETRY_OPERATORS[op](original, gap)
    assert torch.equal(original.input_ids, perturbed.input_ids)
    assert torch.equal(original.attention_mask, perturbed.attention_mask)
    assert original.chain_slices == perturbed.chain_slices
    assert perturbed.position_ids is not None
    assert not torch.equal(perturbed.position_ids, torch.arange(len(original.input_ids)))
    # Residue offsets within a chain are preserved.
    for sl in perturbed.chain_slices.values():
        assert bool((perturbed.position_ids[sl].diff() == 1).all())


@pytest.mark.parametrize("sep", [0, 2])
def test_render_gap_matches_geometry_operator_and_zero_gap_is_arange(tokenizer, sep):
    plain = render_naive(SPEC, tokenizer, separator_tokens=sep)
    assert plain.position_ids is None
    gapped = render_naive(SPEC, tokenizer, separator_tokens=sep, chain_break_gap=7)
    assert torch.equal(gapped.position_ids, GEOMETRY_OPERATORS["G1"](plain, 7).position_ids)


def test_model_treats_absent_position_ids_as_arange(tokenizer):
    # position_ids=None on a layout is only safe if the model's own default is arange.
    model = EsmForMaskedLM.from_pretrained("facebook/esm2_t12_35M_UR50D").eval()
    r = render_naive(SPEC, tokenizer)
    ids, mask = r.input_ids[None], r.attention_mask[None]
    with torch.inference_mode():
        default = model(input_ids=ids, attention_mask=mask).logits
        explicit = model(
            input_ids=ids, attention_mask=mask, position_ids=torch.arange(ids.shape[1])[None]
        ).logits
    assert torch.equal(default, explicit)


def test_symmetric_ends_separates_chains_from_special_tokens(tokenizer):
    r = render_naive(SPEC, tokenizer, separator_tokens=2, symmetric_ends=True)
    sl = list(r.chain_slices.values())
    assert sl[0].start == 3 and r.input_ids.shape[0] - sl[-1].stop == 3
    assert bool((r.input_ids[1:3] == tokenizer.pad_token_id).all())
