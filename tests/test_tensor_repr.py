from tor.tensor import tensor

def test_repr_1d():
    t = tensor([1, 2, 3])
    expected = "Tensor([1, 2, 3], dtype=int)"
    assert repr(t) == expected

def test_repr_2d():
    t = tensor([[1, 2], [3, 4]])
    expected = (
        "Tensor([[1, 2],\n"
        "        [3, 4]], dtype=int)"
    )
    assert repr(t) == expected

def test_repr_3d():
    t = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    expected = (
        "Tensor([[[1, 2],\n"
        "         [3, 4]],\n"
        "\n"
        "        [[5, 6],\n"
        "         [7, 8]]], dtype=int)"
    )
    assert repr(t) == expected

def test_repr_scalar():
    t = tensor(5)
    expected = "Tensor(5, dtype=int)"
    assert repr(t) == expected

def test_repr_float_no_zeros():
    t = tensor([1.0, 2.0, float('nan')])
    expected = "Tensor([1., 2., nan], dtype=float)"
    assert repr(t) == expected

def test_repr_float_fixed_precision_5digits_no_exponent_of_zero():
    t = tensor([1.5, 1.0, float('nan')])
    expected = "Tensor([1.5000, 1.0000,    nan], dtype=float)"
    assert repr(t) == expected

def test_repr_float_scientific_notation():
    t = tensor([1e-10, 0, float('nan')])
    expected = "Tensor([1.0000e-10, 0.0000e+00,        nan], dtype=float)"
    assert repr(t) == expected

def test_repr_empty():
    t = tensor([])
    expected = "Tensor([], dtype=float)"
    assert repr(t) == expected

