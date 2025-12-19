import tor
import pytest

def test_tensor_slicing_all():
    x = tor.tensor([[1,2,3],[4,5,6], [7,8,9]])
    y = x[:, :]
    assert y.tolist() == [[1,2,3],[4,5,6], [7,8,9]]
    assert y.storage == x.storage

def test_tensor_slicing_row():
    x = tor.tensor([[1,2,3],[4,5,6], [7,8,9]])
    y = x[1, :]
    assert y.tolist() == [4,5,6]
    assert y.storage == x.storage

def test_tensor_slicing_column():
    x = tor.tensor([[1,2,3],[4,5,6], [7,8,9]])
    y = x[:, 1]
    assert y.tolist() == [2,5,8]
    assert y.storage == x.storage

def test_tensor_slicing():
    x = tor.tensor([[1,2,3],[4,5,6], [7,8,9]])
    y = x[1:, 1:]
    assert y.tolist() == [[5,6], [8,9]]
    assert y.storage == x.storage

def test_tensor_slicing_negative():
    x = tor.tensor([[1,2,3],[4,5,6], [7,8,9]])
    y = x[-2:, -2:]
    assert y.tolist() == [[5,6], [8,9]]
    assert y.storage == x.storage

def test_tensor_slicing_error():
    x = tor.tensor([[1,2,3],[4,5,6], [7,8,9]])
    with pytest.raises(IndexError):
        x[1, 1, 1]
