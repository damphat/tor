from typing import List, Tuple, Any, Type, Union, Optional


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
        storage_offset: int = 0,
    ):
        self.storage: List[Any] = storage
        self.shape: Tuple[int, ...] = shape
        self.dtype: Type = dtype
        self.strides: Tuple[int, ...] = strides
        self.storage_offset: int = storage_offset

    def __repr__(self) -> str:
        return f"Tensor({self.tolist()}, dtype={self.dtype.__name__})"

    def tolist(self) -> Any:
        def recursive_nest(offset: int, shape_idx: int) -> Any:
            if shape_idx == len(self.shape):
                return self.storage[offset]
            
            dim_size = self.shape[shape_idx]
            stride = self.strides[shape_idx]
            
            if shape_idx == len(self.shape) - 1:
                return self.storage[offset : offset + dim_size]
            
            return [
                recursive_nest(offset + i * stride, shape_idx + 1)
                for i in range(dim_size)
            ]

        if not self.shape:
            return self.storage[self.storage_offset] if self.storage else []
        
        return recursive_nest(self.storage_offset, 0)

    def size(self, dim: Optional[int] = None) -> Union[Tuple[int, ...], int]:
        if dim is None:
            return self.shape
        if dim < 0 or dim >= len(self.shape):
            raise IndexError(f"Dimension out of range (expected to be in range [0, {len(self.shape)-1}], but got {dim})")
        return self.shape[dim]

    def reshape(self, shape: Tuple[int, ...]) -> "Tensor":
        size = 1
        for d in shape:
            size *= d
        
        if not shape and len(self.storage) == 1:
            pass
        elif size != len(self.storage):
            raise ValueError(f"Cannot reshape tensor of size {len(self.storage)} into shape {shape}")

        return Tensor(
            storage=self.storage,
            shape=shape,
            dtype=self.dtype,
            strides=_compute_strides(shape),
            storage_offset=self.storage_offset,
        )

    def _resolve_indices(self, indices: Any) -> Tuple[int, Tuple[int, ...]]:
        if not isinstance(indices, tuple):
            indices = (indices,)
        
        if len(indices) > len(self.shape):
            raise IndexError(f"Too many indices for tensor of dimension {len(self.shape)}")
        
        offset = self.storage_offset
        resolved_indices = list(indices)
        for i, idx in enumerate(indices):
            if idx < 0:
                idx += self.shape[i]
                resolved_indices[i] = idx
                
            if idx < 0 or idx >= self.shape[i]:
                original_idx = indices[i]
                raise IndexError(f"Index {original_idx} is out of bounds for dimension {i} with size {self.shape[i]}")
            offset += idx * self.strides[i]
        
        return offset, tuple(resolved_indices)

    def __getitem__(self, indices: Any) -> "Tensor":
        offset, idx_tuple = self._resolve_indices(indices)
        
        # New shape and strides are the remaining dimensions
        new_shape = self.shape[len(idx_tuple):]
        new_strides = self.strides[len(idx_tuple):]
        
        return Tensor(
            storage=self.storage,
            shape=new_shape,
            dtype=self.dtype,
            strides=new_strides,
            storage_offset=offset
        )

    def __setitem__(self, indices: Any, value: Any):
        offset, idx_tuple = self._resolve_indices(indices)
        
        if len(idx_tuple) != len(self.shape):
            raise ValueError("Only scalar assignment is supported via indexing currently")
            
        self.storage[offset] = self.dtype(value)


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

