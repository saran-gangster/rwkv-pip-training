import os
import torch
import numpy as np
import struct
from functools import lru_cache
from itertools import accumulate

def print_rank_0(*message):
    """Print messages only if the process is rank 0."""
    pass

def _warmup_mmap_file(path):
    """Read file to warm up mmap."""
    pass

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: float,
    7: np.double,
    8: np.uint16,
}

def code(dtype):
    """Return the code corresponding to the given dtype."""
    for k, v in dtypes.items():
        if v == dtype:
            return k
    raise ValueError(f"Unsupported dtype: {dtype}")

def index_file_path(prefix_path):
    """Return the index file path for a given prefix path."""
    return f"{prefix_path}.idx"

def data_file_path(prefix_path):
    """Return the data file path for a given prefix path."""
    return f"{prefix_path}.bin"

class MMapIndexedDataset(torch.utils.data.Dataset):
    class Index:
        _HDR_MAGIC = b"MMIDIDX\x00\x00"

        @classmethod
        def writer(cls, path, dtype):
            """Context manager to write an index file."""
            class _Writer:
                def __enter__(self):
                    self._file = open(path, "wb")
                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack("<Q", 1))
                    self._file.write(struct.pack("<B", code(dtype)))
                    return self

                @staticmethod
                def _get_pointers(sizes):
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []
                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size
                    return pointers

                def write(self, sizes, doc_idx):
                    pointers = self._get_pointers(sizes)
                    self._file.write(struct.pack("<Q", len(sizes)))
                    self._file.write(struct.pack("<Q", len(doc_idx)))
                    self._file.write(np.array(sizes, dtype=np.int32).tobytes(order="C"))
                    self._file.write(np.array(pointers, dtype=np.int64).tobytes(order="C"))
                    self._file.write(np.array(doc_idx, dtype=np.int64).tobytes(order="C"))

                def __exit__(self, exc_type, exc_value, traceback):
                    self._file.close()

            return _Writer()

        def __init__(self, path, skip_warmup=False):
            with open(path, "rb") as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    "Index file doesn't match expected format. "
                    "Make sure that --dataset-impl is configured properly."
                )
                version = struct.unpack("<Q", stream.read(8))
                assert (1,) == version

                dtype_code = struct.unpack("<B", stream.read(1))[0]
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack("<Q", stream.read(8))[0]
                self._doc_count = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()

            if not skip_warmup:
                print_rank_0("    warming up index mmap file...")
                _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            print_rank_0("    reading sizes...")
            self._sizes = np.frombuffer(
                self._bin_buffer, dtype=np.int32, count=self._len, offset=offset
            )
            print_rank_0("    reading pointers...")
            self._pointers = np.frombuffer(
                self._bin_buffer, dtype=np.int64, count=self._len, offset=offset + self._sizes.nbytes
            )
            print_rank_0("    reading document index...")
            self._doc_idx = np.frombuffer(
                self._bin_buffer, dtype=np.int64, count=self._doc_count, offset=offset + self._sizes.nbytes + self._pointers.nbytes
            )

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, skip_warmup=False):
        super().__init__()
        self._do_init(path, skip_warmup)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path, skip_warmup):
        self._path = path
        self._index = self.Index(index_file_path(self._path), skip_warmup)
        if not skip_warmup:
            print_rank_0("    warming up data mmap file...")
            _warmup_mmap_file(data_file_path(self._path))
        print_rank_0("    creating numpy buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode="r", order="C")
        print_rank_0("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            ptr, size = self._index[idx]
            return np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=total_size, offset=ptr)
            return np.split(np_array, offsets[:-1])

    def get(self, idx, offset=0, length=None):
        """Retrieve a single item with optional offset and length."""
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * self._index.dtype().itemsize
        return np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=length, offset=ptr)

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index.doc_idx

    def set_doc_idx(self, doc_idx):
        self._index._doc_idx = doc_idx

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        """Check if both index and data files exist for the given path."""
        return os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))
