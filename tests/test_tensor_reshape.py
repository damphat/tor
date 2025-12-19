import tor
import pytest

def test_reshape_2d():
    x = tor.tensor([[1, 2], [3, 4]])
    y = x.reshape((4,))
    assert y.dtype == int
    assert y.shape == (4,)
    assert y.strides == (1,)
    assert y.storage == [1, 2, 3, 4]
    # giữ nguyên storage
    assert y.storage == x.storage

def test_reshape_scalar_to_scalar():
    x = tor.tensor(1)
    y = x.reshape(())
    assert y.dtype == int
    assert y.shape == ()
    assert y.strides == ()
    assert y.storage == x.storage
    
def test_reshape_scalar_to_1d():
    x = tor.tensor(1)
    y = x.reshape((1,))
    assert y.dtype == int
    assert y.shape == (1,)
    assert y.strides == (1,)
    assert y.storage == x.storage
    
def test_reshape_error():
    x = tor.tensor([1,2,3])
    with pytest.raises(ValueError):
        y = x.reshape((2,2))
    