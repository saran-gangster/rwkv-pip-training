import json
import math
import random
from typing import Any, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from .binidx import MMapIndexedDataset
from .utils import MaybeIsPrime


class MyDataset(Dataset):
    def __init__(self, args: Any):
        """
        Initialize the dataset based on the provided arguments.
        
        Args:
            args: Arguments containing dataset configuration.
        """
        self.args = args
        self.data = None
        self.vocab_size = None
        self.data_size = None
        self.data_pile = None
        self.data_pile_size = 0

        if args.data_type == "binidx":
            self._load_binidx_data()
        elif args.data_type == "numpy":
            self._load_numpy_data()
        elif args.data_type == "uint16":
            self._load_uint16_data()
        else:
            self._load_text_data()

    def _load_binidx_data(self):
        """Load binary indexed data."""
        self.vocab_size = self.args.vocab_size
        rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")

        if self.args.my_pile_version == 1:
            self.data = MMapIndexedDataset(self.args.data_file)
            self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
            rank_zero_info(f"Data has {self.data_size} tokens.")
        elif self.args.my_pile_version == 2:
            self._load_pile_version_2_data()
        
        if self.args.my_qa_mask > 0:
            self._load_pile_data()
        
        if self.args.my_pile_stage > 0:
            self._validate_pile_stage()
    
    def _load_pile_version_2_data(self):
        """Load data for pile version 2."""
        with open(self.args.data_file, "r", encoding='utf-8') as file:
            data_list = [line.strip().split(' ') for line in file.read().strip().split('\n')]
        
        self.data = []
        self.data_size = int(data_list[-1][-1])
        rank_zero_info(f"Data has {self.data_size} chunks.")
        
        for d in data_list:
            dataset = MMapIndexedDataset(d[0])
            data_size = len(dataset._bin_buffer) // dataset._index._dtype_size
            assert (data_size - self.args.ctx_len) == int(d[1])
            self.data.append([int(d[-1]), int(d[1]), dataset])

    def _load_pile_data(self):
        """Load pile data for QA mask."""
        self.data_pile = MMapIndexedDataset('/fsx/pile_deduped/pile_0.87_deduped_text_document')
        self.data_pile_size = len(self.data_pile._bin_buffer) // self.data_pile._index._dtype_size

    def _validate_pile_stage(self):
        """Validate and set up pile stage configurations."""
        self.samples_per_epoch = self.args.epoch_steps * self.args.real_bsz
        assert self.samples_per_epoch == 40320
        rank_zero_info(f"########## Pile 20b-tokenized stage {self.args.my_pile_stage} ##########")
        
        dataset_slot = self.data_size // self.args.ctx_len
        if self.args.my_pile_stage != 4:
            assert MaybeIsPrime(self.args.magic_prime)
            assert self.args.magic_prime % 3 == 2
            assert 0.9 < self.args.magic_prime / dataset_slot <= 1

    def _load_numpy_data(self):
        """Load numpy data."""
        self.data = np.load(self.args.data_file).astype("int")
        self.vocab_size = self.args.vocab_size
        rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")
        self.data_size = len(self.data)
        rank_zero_info(f"Data has {self.data_size} tokens.")

    def _load_uint16_data(self):
        """Load uint16 data."""
        self.data = np.fromfile(self.args.data_file, dtype=np.uint16).astype("int32").reshape(-1, self.args.my_sample_len)
        self.vocab_size = self.args.vocab_size
        rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")
        self.data_size = self.data.shape[0]
        rank_zero_info(f"Data has {self.data_size} samples.")

    def _load_text_data(self):
        """Load text data."""
        if self.args.data_type == "dummy":
            rank_zero_info("Building dummy data...")
            self.data = "".join(f".{i % 10000}+{(i * i) % 10000}={(i + (i * i) % 10000)}." for i in range(100000))
        else:
            with open(self.args.data_file, "r", encoding=self.args.data_type) as file:
                self.data = file.read()

        rank_zero_info("Building token list...")
        unique = sorted(set(self.data))
        self.vocab_size = len(unique)

        vocab = {i: u for i, u in enumerate(unique)}
        with open(f"{self.args.proj_dir}/vocab.json", "w", encoding="utf-8") as vocab_file:
            json.dump(vocab, vocab_file, ensure_ascii=False)
        
        self.data_size = len(self.data)
        rank_zero_info(f"Data has {self.data_size} tokens, {self.vocab_size} vocab size.")
        self.stoi = {ch: i for i, ch in enumerate(unique)}
        self.itos = {i: ch for i, ch in enumerate(unique)}

    def __len__(self) -> int:
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Retrieve a single data sample.
        
        Args:
            idx: Index of the data sample.
        
        Returns:
            A tuple of input and target tensors, and optionally the QA mask tensor.
        """
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size

        if args.data_type == "uint16":
            return self._get_uint16_sample()
        
        ctx_len = args.ctx_len
        req_len = ctx_len + 1
        magic_prime = args.magic_prime

        if args.my_pile_stage > 0:
            i = self._get_pile_sample_index(epoch, idx, rank, world_size, req_len, ctx_len, magic_prime)
        else:
            i = random.randint(0, self.data_size - req_len)
        
        dix = self._get_data_slice(i, req_len)
        
        if args.my_qa_mask == 1:
            z = self._get_qa_mask(dix, ctx_len)
            return dix[:-1], dix[1:], z

        return dix[:-1], dix[1:]

    def _get_uint16_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve a sample for uint16 data type."""
        i = random.randint(0, self.data_size - 1)
        dix = self.data[i]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

    def _get_pile_sample_index(self, epoch: int, idx: int, rank: int, world_size: int, req_len: int, ctx_len: int, magic_prime: int) -> int:
        """Calculate the sample index for pile data."""
        ii = 1 + epoch * self.samples_per_epoch + (idx * world_size) + rank

        if self.args.my_qa_mask > 0:
            ii_orig = ii
            if ii % 2 == 0:
                ii = -1
                data = self.data_pile
            else:
                ii = ii // 2
        else:
            data = self.data

        if data == self.data_pile:
            return random.randint(0, self.data_pile_size - req_len)

        if self.args.my_pile_stage == 4 or ii < self.args.my_random_steps:
            if self.args.my_pile_version == 1:
                return random.randint(0, self.data_size - req_len)
            return random.randint(0, self.data_size)
        
        ii -= self.args.my_random_steps
        factor = (math.sqrt(5) - 1) / 2
        factor = int(magic_prime * factor)
        i = ((factor * ii * ii * ii) % magic_prime) * ctx_len
        i += self.args.my_pile_shift
        return i

    def _get_data_slice(self, i: int, req_len: int) -> torch.Tensor:
        """Retrieve a slice of data."""
        if self.args.data_type == "binidx":
            if self.args.my_pile_version == 1:
                dix = self.data.get(idx=0, offset=i, length=req_len).astype(int)
            else:
                for cutoff, chunk_count, dataset in self.data:
                    if i < cutoff:
                        offset = i - (cutoff - chunk_count if chunk_count else 0)
                        dix = dataset.get(idx=0, offset=offset, length=req_len).astype(int)
                        break
        elif self.args.data_type == "numpy":
            dix = self.data[i:i + req_len]
        else:
            dix = torch.tensor([self.stoi[s] for s in self.data[i:i + req_len]], dtype=torch.long)
        return dix

    def _get_qa_mask(self, dix: torch.Tensor, ctx_len: int) -> torch.Tensor:
        """Generate the QA mask."""
        if self.data == self.data_pile:
            return torch.ones(ctx_len, dtype=torch.bfloat16)
        
        z = torch.zeros(ctx_len, dtype=torch.bfloat16)
        z_sum = 0
        is_good = False
        
        for i in range(3, ctx_len):
            if dix[i] == 27 and dix[i-1] == 34 and dix[i-2] == 187 and dix[i-3] == 187:
                is_good = True
            if dix[i] == 0:
                is_good = False
            if is_good:
                z[i] = 1
                z_sum += 1

        if z_sum == 0:
            z = torch.ones(ctx_len, dtype=torch.bfloat16)
            i = random.randint(0, self.data_pile_size - (ctx_len + 1))
            dix = self.data_pile.get(idx=0, offset=i, length=ctx_len + 1).astype(int)
        
        return z