import tor
import pytest

def test_simple_indexing():
    x = tor.tensor([[1,2],[3,4]])
    assert x.tolist() == [[1, 2], [3, 4]]

    x0 = x[0]
    assert x0.tolist() == [1,2]

    x1 = x[1]
    assert x1.tolist() == [3,4]

    x11 = x1[1]
    assert x11.tolist() == 4

    # x[1,1] = 5
    x[1,1] = 5

    # bởi vì x1 và x11 chỉ vào x nên khi thay đổi giá trị của x thì x1 và x11 cũng thay đổi
    assert x.tolist() == [[1, 2], [3, 5]]
    assert x1.tolist() == [3,5] 
    assert x11.tolist() == 5

def test_simple_indexing_negative():
    x = tor.tensor([[1,2],[3,4]])
    assert x[-1].tolist() == [3,4]
    assert x[-1,-1].tolist() == 4
    
def test_indexing_error():
    x = tor.tensor([[1,2],[3,4]])
    with pytest.raises(IndexError):
        x[2]
    with pytest.raises(IndexError):
        x[1,2]
    with pytest.raises(IndexError):
        x[1,1,1]