import tor

def test_tensor_tolist():
    x = tor.tensor([1, 2, 3])
    assert x.tolist() == [1, 2, 3]
    
    y = tor.tensor([[1, 2], [3, 4]])
    assert y.tolist() == [[1, 2], [3, 4]]
    
def test_tensor_tolist_scalar():
    x = tor.tensor(1)
    assert x.tolist() == 1
