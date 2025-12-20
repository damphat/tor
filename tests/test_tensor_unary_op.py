import tor

def test_unary_op():
    x = tor.tensor([1, 2, 3])
    y = x.unary_op(lambda x: x * 2)
    assert y.tolist() == [2, 4, 6]
    assert y.dtype == x.dtype
    assert y.shape == x.shape
    assert y.storage != x.storage

    y = x.unary_op(lambda x: x * 0.5)
    assert y.tolist() == [0.5, 1.0, 1.5]
    assert y.dtype == float
    assert y.shape == x.shape
    assert y.storage != x.storage

    x = tor.tensor(5)

    y = x.unary_op(lambda x: x * 2)
    assert y.tolist() == 10
    assert y.dtype == x.dtype
    assert y.shape == x.shape
    assert y.storage != x.storage