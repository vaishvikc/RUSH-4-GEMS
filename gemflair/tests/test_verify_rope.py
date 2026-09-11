import numpy as np

from gemflair import verify_rope


def test_blocks_pack_contiguously_and_drop_the_remainder():
    out = verify_rope.blocks([[1, 2, 3], [4, 5, 6, 7]], seq_len=3)
    assert out.tolist() == [[1, 2, 3], [4, 5, 6]]


def test_blocks_do_not_rebase_across_subjects():
    out = verify_rope.blocks([[0, 300], [0, 300]], seq_len=2)
    assert out.tolist() == [[0, 300], [0, 300]]


def test_blocks_returns_empty_when_shorter_than_seq_len():
    assert verify_rope.blocks([[1, 2]], seq_len=5).shape == (0, 5)
