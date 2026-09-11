import numpy as np
import pytest
import torch

from gemflair import verify_rope


def test_blocks_pack_contiguously_and_drop_the_remainder():
    out = verify_rope.blocks([[1, 2, 3], [4, 5, 6, 7]], seq_len=3)
    assert out.tolist() == [[1, 2, 3], [4, 5, 6]]


def test_blocks_do_not_rebase_across_subjects():
    out = verify_rope.blocks([[0, 300], [0, 300]], seq_len=2)
    assert out.tolist() == [[0, 300], [0, 300]]


def test_blocks_returns_empty_when_shorter_than_seq_len():
    assert verify_rope.blocks([[1, 2]], seq_len=5).shape == (0, 5)


def test_blocks_do_not_rebase_when_subject_boundary_falls_mid_block():
    rows = [[0, 300, 600, 900], [0, 50]]
    out = verify_rope.blocks(rows, seq_len=3)
    assert out.tolist() == [[0, 300, 600], [900, 0, 50]]


def test_eval_loss_raises_on_empty_blocks():
    tok = np.empty((0, 3), dtype=np.int64)
    ela = np.empty((0, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="tuning split"):
        verify_rope.eval_loss(None, tok, ela, 300, "cpu")


class _StubModel:
    def __init__(self):
        self.received_position_ids = []

    def __call__(self, input_ids, position_ids, labels):
        self.received_position_ids.append(position_ids)
        class _Out:
            loss = torch.tensor(0.0)
        return _Out()


def test_eval_loss_passes_time_based_position_ids():
    tok = np.array([[1, 2, 3], [4, 5, 6]])
    ela = np.array([[0, 300, 600], [900, 1200, 1500]], dtype=np.float64)
    model = _StubModel()
    verify_rope.eval_loss(model, tok, ela, 300, "cpu", batch_size=4)
    pos = model.received_position_ids[0]
    expected = (ela / 300 + np.arange(3)).astype(np.int64)
    assert pos.tolist() == expected.tolist()


def test_eval_loss_passes_none_position_ids_without_rope():
    tok = np.array([[1, 2, 3], [4, 5, 6]])
    ela = np.array([[0, 300, 600], [900, 1200, 1500]], dtype=np.float64)
    model = _StubModel()
    verify_rope.eval_loss(model, tok, ela, None, "cpu", batch_size=4)
    assert model.received_position_ids[0] is None
