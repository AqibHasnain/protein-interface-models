from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from cla.model.backbone import load_esm, resolve_device
from cla.model.scoring import score_many, span_pll, windowed_span_pll


@dataclass
class FakeRendered:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    chain_slices: dict[str, slice]
    position_ids: torch.Tensor | None = None
    chain_ids: torch.Tensor | None = None


VOCAB = 6
TOK = SimpleNamespace(mask_token_id=5, pad_token_id=1)


class FixedLogits(nn.Module):
    """Ignores its input and returns the same logits for every batch row."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = logits
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        return SimpleNamespace(logits=self.logits[None].expand(input_ids.shape[0], -1, -1))


class CopyInput(nn.Module):
    """Predicts, with high confidence, whatever token it sees at each position."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        return SimpleNamespace(logits=20.0 * torch.nn.functional.one_hot(input_ids, VOCAB).float())


def _rendered(ids, span):
    ids = torch.tensor(ids)
    return FakeRendered(ids, torch.ones_like(ids), {"PEPTIDE": span})


def test_span_pll_matches_hand_computed_mean():
    logits = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 9.0],
        ]
    )
    ids = [0, 2, 3, 4, 0]
    rendered = _rendered(ids, slice(1, 4))
    expected = np.mean(
        [
            logits[i, t] - np.log(np.exp(logits[i].numpy()).sum())
            for i, t in zip((1, 2, 3), ids[1:4], strict=True)
        ]
    )
    got = span_pll(FixedLogits(logits), TOK, rendered, {"PEPTIDE"})
    assert got == pytest.approx(float(expected), abs=1e-6)


def test_windowed_span_pll_averages_window_means_and_anchors_the_tail():
    # 5 span residues with window=stride=2 give windows [0,1], [2,3], [3,4]: the last is
    # anchored to the span end, so residue 3 is scored twice and the mean is over windows.
    logits = torch.arange(35, dtype=torch.float).reshape(7, 5) ** 1.3 % 4
    logits = torch.cat([logits, torch.zeros(7, 1)], dim=1)
    ids = [0, 2, 3, 4, 2, 3, 0]
    lp = torch.log_softmax(logits, dim=-1)
    per_res = [float(lp[i, t]) for i, t in zip(range(1, 6), ids[1:6], strict=True)]
    expected = np.mean([np.mean(per_res[0:2]), np.mean(per_res[2:4]), np.mean(per_res[3:5])])
    rendered = _rendered(ids, slice(1, 6))
    got = windowed_span_pll(FixedLogits(logits), TOK, rendered, {"PEPTIDE"}, window=2, stride=2)
    assert got == pytest.approx(float(expected), abs=1e-6)
    # A window covering the whole span reduces to span_pll.
    whole = windowed_span_pll(FixedLogits(logits), TOK, rendered, {"PEPTIDE"}, window=9)
    assert whole == pytest.approx(span_pll(FixedLogits(logits), TOK, rendered, {"PEPTIDE"}))


def test_span_pll_applies_the_mask():
    # A copy-the-input model scores ~0 nats on any unmasked token, so a score near -20 means
    # the span was replaced by the mask id.
    model = CopyInput()
    rendered = _rendered([0, 2, 3, 4, 0], slice(1, 4))
    masked = span_pll(model, TOK, rendered, {"PEPTIDE"})
    assert masked < -19.0


@pytest.fixture(scope="module")
def esm35():
    return load_esm("35M", device=resolve_device("cpu"))


def test_span_pll_real_model_is_deterministic_and_batch_consistent(esm35):
    model, tok = esm35
    ids = torch.tensor(tok("MKTAYIAKQRQISFVKSHFSRQ")["input_ids"])
    rendered = FakeRendered(ids, torch.ones_like(ids), {"PEPTIDE": slice(1, len(ids) - 1)})
    a = span_pll(model, tok, rendered, {"PEPTIDE"})
    b = span_pll(model, tok, rendered, {"PEPTIDE"})
    assert a == b
    assert np.isfinite(a) and a < 0
    assert score_many(model, tok, [rendered, rendered], {"PEPTIDE"})[0] == pytest.approx(
        a, abs=1e-4
    )
