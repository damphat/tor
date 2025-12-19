from typing import List, Tuple, Any, Type, Union


def _compute_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    if not shape:
        return ()
    strides_list: List[int] = []
    s = 1
    for d in reversed(shape):
        strides_list.append(s)
        s *= d
    return tuple(reversed(strides_list))


class Tensor:
    def __init__(
        self,
        storage: List[Any],
        shape: Tuple[int, ...],
        dtype: Type,
        strides: Tuple[int, ...],
    ):
        self.storage: List[Any] = storage
        self.shape: Tuple[int, ...] = shape
        self.dtype: Type = dtype
        self.strides: Tuple[int, ...] = strides

    def reshape(self, shape: Tuple[int, ...]) -> "Tensor":
        size = 1
        for d in shape:
            size *= d
        
        if not shape and len(self.storage) == 1: # Scalar case
            pass
        elif size != len(self.storage):
            raise ValueError(f"Cannot reshape tensor of size {len(self.storage)} into shape {shape}")

        return Tensor(
            storage=self.storage,
            shape=shape,
            dtype=self.dtype,
            strides=_compute_strides(shape),
        )


def tensor(data: Any) -> Tensor:
    def get_shape(lst: Any) -> List[int]:
        if not isinstance(lst, list):
            return []
        if not lst:
            return [0]
        return [len(lst)] + get_shape(lst[0])

    def flatten(lst: Any) -> List[Any]:
        if not isinstance(lst, list):
            return [lst]
        res: List[Any] = []
        for i in lst:
            res.extend(flatten(i))
        return res

    shape = tuple(get_shape(data))
    storage = flatten(data)

    if not storage:
        dtype = float
    elif any(isinstance(x, float) for x in storage):
        dtype = float
    else:
        dtype = int

    storage = [dtype(x) for x in storage]
    strides = _compute_strides(shape)

    return Tensor(storage, shape, dtype, strides)

