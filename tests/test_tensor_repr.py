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

def test_repr_float():
    t = tensor([1.0, 2.0])
    expected = "Tensor([1.0, 2.0], dtype=float)"
    assert repr(t) == expected

def test_repr_empty():
    t = tensor([])
    expected = "Tensor([], dtype=float)"
    assert repr(t) == expected

if __name__ == "__main__":
    # Chạy các test thủ công nếu không dùng pytest
    test_repr_1d()
    test_repr_2d()
    test_repr_3d()
    test_repr_scalar()
    test_repr_float()
    test_repr_empty()
    print("All repr tests passed!")
