import math
import itertools
from typing import List, Tuple, Any, Type, Union, Optional, Callable
 
 
Scalar = Union[int, float]


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
        storage: List[Scalar],
        shape: Tuple[int, ...],
        dtype: Type[Scalar],
        strides: Tuple[int, ...],
        storage_offset: int = 0,
    ):
        self.storage: List[Scalar] = storage
        self.shape: Tuple[int, ...] = shape
        self.dtype: Type[Scalar] = dtype
        self.strides: Tuple[int, ...] = strides
        self.storage_offset: int = storage_offset

    def __repr__(self) -> str:
        prefix = "Tensor("
        suffix = f", dtype={self.dtype.__name__})"
        
        data_list = self.tolist()
        
        if self.dtype == float:
            # Analyze all values for formatting
            all_values = self._flatten_list(data_list) if self.shape else [data_list]
            
            use_sci = False
            is_all_int = True
            max_val = 0.0
            min_pos = float('inf')
            non_special_values = []
            
            for v in all_values:
                if v is None: continue
                if not math.isfinite(v):
                    continue
                non_special_values.append(v)
                abs_v = abs(v)
                if abs_v > max_val: max_val = abs_v
                if 0 < abs_v < min_pos: min_pos = abs_v
                if v != int(v):
                    is_all_int = False
            
            if max_val >= 1e4 or (min_pos < 1e-4 and min_pos != float('inf')):
                use_sci = True
            
            if use_sci:
                base_formatter = lambda x: f"{x:.4e}"
            elif is_all_int:
                base_formatter = lambda x: f"{x:.0f}."
            else:
                base_formatter = lambda x: f"{x:.4f}"

            # Calculate width for special values alignment
            if non_special_values:
                example_formatted = base_formatter(non_special_values[0])
                width = len(example_formatted)
            else:
                width = 3

            def formatter(x: Any) -> str:
                if not isinstance(x, (int, float)):
                    return str(x)
                if math.isnan(x):
                    return "nan".rjust(width)
                if math.isinf(x):
                    s = "inf" if x > 0 else "-inf"
                    return s.rjust(width)
                return base_formatter(x)
        else:
            formatter = str

        if not self.shape:
            # Handle 0-dim tensor (scalar)
            return f"{prefix}{formatter(data_list)}{suffix}"

        formatted_data = self._format_data(data_list, indent=len(prefix), formatter=formatter)
        return f"{prefix}{formatted_data}{suffix}"

    def _flatten_list(self, lst: Any) -> List[Scalar]:
        if not isinstance(lst, list):
            return [lst]
        res: List[Scalar] = []
        for i in lst:
            res.extend(self._flatten_list(i))
        return res

    def _format_data(self, data: Any, indent: int, formatter: Any) -> str:
        if not isinstance(data, list):
            return formatter(data)
        
        if len(data) == 0:
            return "[]"
        
        # If it's a 1D list (contains non-lists)
        if not isinstance(data[0], list):
            return "[" + ", ".join(map(formatter, data)) + "]"
        
        # Higher dimensions
        depth = 0
        curr = data
        while isinstance(curr, list) and curr:
            depth += 1
            curr = curr[0]
        
        # Separator includes newlines and spaces for alignment
        # depth-1 newlines: 1 for 2D, 2 for 3D, etc.
        sep = "," + "\n" * (depth - 1) + " " * (indent + 1)
        
        parts: List[str] = []
        for item in data:
            parts.append(self._format_data(item, indent + 1, formatter))
            
        return "[" + sep.join(parts) + "]"


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

    def unary_op(self, op: Callable[[Scalar], Scalar]) -> "Tensor":
        elements = self._flatten_list(self.tolist())
        applied_values = [op(x) for x in elements]

        if not applied_values:
            new_dtype = self.dtype
        elif any(isinstance(x, float) for x in applied_values):
            new_dtype = float
        else:
            new_dtype = int

        new_storage = [new_dtype(x) for x in applied_values]
        return Tensor(
            storage=new_storage,
            shape=self.shape,
            dtype=new_dtype,
            strides=_compute_strides(self.shape),
            storage_offset=0,
        )

    def binary_op(self, other: Union["Tensor", Any], op: Callable[[Scalar, Scalar], Scalar]) -> "Tensor":
        if not isinstance(other, Tensor):
            other = tensor(other)

        shape1 = self.shape
        shape2 = other.shape

        ndim1 = len(shape1)
        ndim2 = len(shape2)
        ndim = max(ndim1, ndim2)

        padded_shape1 = (1,) * (ndim - ndim1) + shape1
        padded_shape2 = (1,) * (ndim - ndim2) + shape2

        result_shape_list = []
        for s1, s2 in zip(padded_shape1, padded_shape2):
            if s1 == s2:
                result_shape_list.append(s1)
            elif s1 == 1:
                result_shape_list.append(s2)
            elif s2 == 1:
                result_shape_list.append(s1)
            else:
                raise ValueError(
                    f"Shapes {shape1} and {shape2} are not compatible for broadcasting"
                )

        result_shape = tuple(result_shape_list)
        applied_values = []

        for res_idx in itertools.product(*(range(s) for s in result_shape)):
            off1 = self.storage_offset
            for i in range(ndim1):
                res_dim_idx = res_idx[ndim - ndim1 + i]
                if shape1[i] > 1:
                    off1 += res_dim_idx * self.strides[i]

            off2 = other.storage_offset
            for i in range(ndim2):
                res_dim_idx = res_idx[ndim - ndim2 + i]
                if shape2[i] > 1:
                    off2 += res_dim_idx * other.strides[i]

            applied_values.append(op(self.storage[off1], other.storage[off2]))

        if not applied_values:
            # Fallback for empty tensors or other edge cases
            new_dtype = self.dtype
        elif any(isinstance(v, float) for v in applied_values):
            new_dtype = float
        else:
            new_dtype = int

        new_storage = [new_dtype(v) for v in applied_values]

        return Tensor(
            storage=new_storage,
            shape=result_shape,
            dtype=new_dtype,
            strides=_compute_strides(result_shape),
            storage_offset=0,
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

