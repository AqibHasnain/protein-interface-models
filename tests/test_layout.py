import numpy as np
import pytest
from transformers import AutoTokenizer

from cla.layout.concat import render_naive
from cla.layout.perturb import (
    OPERATORS,
    OperatorNotApplicable,
    insert_neutral_spacer,
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
