import pytest
import tor

@pytest.fixture
def sample_tensor():
    return tor.tensor([1, 2, 3])
