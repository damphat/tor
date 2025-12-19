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
            raise IndexError("Dimension out of range")
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

    def __getitem__(self, indices: Any) -> "Tensor":
        if not isinstance(indices, tuple):
            indices = (indices,)
        
        if len(indices) > len(self.shape):
            raise IndexError(f"Too many indices for tensor of dimension {len(self.shape)}")

        new_shape = []
        new_strides = []
        new_offset = self.storage_offset
        
        for i, idx in enumerate(indices):
            dim_size = self.shape[i]
            stride = self.strides[i]
            
            if isinstance(idx, int):
                # Integer indexing reduces dimension
                if idx < 0:
                    idx += dim_size
                if idx < 0 or idx >= dim_size:
                    raise IndexError(f"Index {idx} is out of bounds for dimension {i} with size {dim_size}")
                new_offset += idx * stride
            elif isinstance(idx, slice):
                # Slicing keeps dimension but changes size/stride
                start, stop, step = idx.indices(dim_size)
                new_offset += start * stride
                
                # Calculate the length of the slice
                if (step > 0 and start >= stop) or (step < 0 and start <= stop):
                    slice_len = 0
                else:
                    slice_len = (stop - start + (step - (1 if step > 0 else -1))) // step
                
                new_shape.append(slice_len)
                new_strides.append(step * stride)
            elif idx is Ellipsis:
                # Basic Ellipsis support could be added but skipping for now or handling as all
                raise NotImplementedError("Ellipsis not yet supported")
            else:
                raise TypeError(f"Invalid index type: {type(idx)}")
        
        # Append remaining dimensions that weren't indexed
        new_shape.extend(self.shape[len(indices):])
        new_strides.extend(self.strides[len(indices):])
        
        return Tensor(
            storage=self.storage,
            shape=tuple(new_shape),
            dtype=self.dtype,
            strides=tuple(new_strides),
            storage_offset=new_offset
        )

    def __setitem__(self, indices: Any, value: Any):
        target = self[indices]
        if target.shape != ():
            raise ValueError("Only scalar assignment is supported via indexing currently")
        self.storage[target.storage_offset] = self.dtype(value)


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

