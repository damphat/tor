import tor

def test_tensor_scalar():
    x = tor.tensor(1)
    assert x is not None
    assert x.dtype == int
    assert x.shape == ()
    assert x.strides == ()
    assert x.storage == [1]

def test_tensor_1d():
    x = tor.tensor([1, 2, 3])
    assert x is not None
    assert x.dtype == int
    assert x.shape == (3,)
    assert x.strides == (1,)
    assert x.storage == [1, 2, 3]

def test_tensor_2d():
    x = tor.tensor([[1, 2], [3, 4]])
    assert x is not None
    assert x.dtype == int
    assert x.shape == (2, 2)
    assert x.strides == (2, 1)
    assert x.storage == [1, 2, 3, 4]

def test_tensor_3d():
    x = tor.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert x is not None
    assert x.dtype == int
    assert x.shape == (2, 2, 2)
    assert x.strides == (4, 2, 1)
    assert x.storage == [1, 2, 3, 4, 5, 6, 7, 8]

def test_tensor_type_inference():
    # nếu có 1 float thì sẽ coi là float
    x = tor.tensor([1, 2, 3.0])
    assert x.dtype == float
    # nếu không có float thì sẽ coi là int
    y = tor.tensor([1, 2, 3])
    assert y.dtype == int
