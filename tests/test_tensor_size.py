import tor
import pytest


def test_tensor_size():
    x = tor.tensor([[1,2,3],[4,5,6]])
    assert x.size() == (2, 3)
    assert x.size(0) == 2
    assert x.size(1) == 3
    
def test_tensor_size_error():
    x = tor.tensor([[1,2,3],[4,5,6]])
    with pytest.raises(IndexError):
        x.size(2)

def test_tensor_size_scalar():
    x = tor.tensor(1)
    assert x.size() == ()
    with pytest.raises(IndexError):
        x.size(0)