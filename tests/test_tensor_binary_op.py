import tor

def test_binary_op():
    x = tor.tensor([1, 2, 3])
    y = tor.tensor([4, 5, 6])
    z = x.binary_op(y, lambda a, b: a + b)
    assert z.tolist() == [5, 7, 9]
    assert z.dtype == x.dtype
    assert z.shape == x.shape
    assert z.storage != x.storage

def test_binary_op_broadcast():
    x = tor.tensor([1, 2, 3])
    y = tor.tensor(4)
    z = x.binary_op(y, lambda a, b: a + b)
    assert z.tolist() == [5, 6, 7]
    
    y = tor.tensor([4])
    z = x.binary_op(y, lambda a, b: a + b)
    assert z.tolist() == [5, 6, 7]

    y = tor.tensor([[4]])
    z = x.binary_op(y, lambda a, b: a + b)
    assert z.tolist() == [[5, 6, 7]]

    y = tor.tensor([[4],[5]])
    z = x.binary_op(y, lambda a, b: a + b)
    assert z.tolist() == [[5, 6, 7], [6, 7, 8]]
    
def test_binary_op_type():
    x = tor.tensor([1, 2, 3])
    y = tor.tensor([2])
    z = x.binary_op(y, lambda a, b: a / b)
    assert z.dtype == float
    assert z.shape == x.shape
    assert z.storage != x.storage
    assert z.tolist() == [0.5, 1.0, 1.5]