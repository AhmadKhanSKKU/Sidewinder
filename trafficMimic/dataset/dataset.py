# NetMasquerade-main/src/trafficMimic/dataset/dataset.py

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import read_flow_pkl
from dataset.vocab import *


class FlowDataset(Dataset):
    def __init__(self, args, timevocab, sizevocab, train=False) -> None:
        super(Dataset, self).__init__()
        self.seq_len = args.model.seq_len
        if train:
            self.data = list(read_flow_pkl(args.trainer.train_data_pth))
        else:
            self.data = list(read_flow_pkl(args.trainer.test_data_pth))

        self.timevocab = timevocab
        self.sizevocab = sizevocab

        self.ipd_list = [flow[0] for flow in self.data]
        self.size_list = [flow[1] for flow in self.data]

    def __len__(self):
        return len(self.ipd_list)

    def __getitem__(self, item):
        flow_ipd = self._float_to_tokens(self.timevocab, self.ipd_list[item])
        flow_size = self._float_to_tokens(self.sizevocab, self.size_list[item])

        flow_ipd = flow_ipd[: self.seq_len]
        flow_size = flow_size[: self.seq_len]
        real_length = len(flow_ipd)

        # Extract the pure, unsquashed physical floats
        raw_ipd = list(self.ipd_list[item][: self.seq_len])
        raw_size = list(self.size_list[item][: self.seq_len])

        # Labels match the exact unmasked token sequences
        flow_ipd_label = flow_ipd.copy()
        flow_size_label = flow_size.copy()

        padding = [
            self.timevocab.pad_index for _ in range(self.seq_len - len(flow_ipd))
        ]
        raw_padding = [0.0 for _ in range(self.seq_len - len(raw_ipd))]

        flow_ipd.extend(padding)
        flow_size.extend(padding)
        flow_ipd_label.extend(padding)
        flow_size_label.extend(padding)

        raw_ipd.extend(raw_padding)
        raw_size.extend(raw_padding)

        output = {
            "flow_ipd": flow_ipd,
            "flow_size": flow_size,
            "flow_ipd_label": flow_ipd_label,
            "flow_size_label": flow_size_label,
            "real_length": real_length,
            "raw_ipd": raw_ipd,
            "raw_size": raw_size,
        }

        return {key: torch.tensor(value) for key, value in output.items()}

    def _float_to_tokens(self, vocab, float_list):
        """Bypasses the broken hash() function and strictly finds the closest mathematical token bucket."""
        if not hasattr(vocab, "np_data"):
            values = []
            max_idx = len(vocab) if hasattr(vocab, "__len__") else vocab.max_size
            for i in range(vocab.special_size, max_idx):
                try:
                    if hasattr(vocab, "itos"):
                        val = vocab.itos(i) if callable(vocab.itos) else vocab.itos[i]
                    else:
                        val = vocab.data[i - vocab.special_size]
                    values.append(float(val))
                except Exception:
                    values.append(0.0)
            vocab.np_data = np.array(values)

        arr = np.array(float_list, dtype=np.float32)
        diffs = np.abs(arr[:, None] - vocab.np_data[None, :])
        closest_indices = np.argmin(diffs, axis=1)

        token_ids = closest_indices + vocab.special_size
        return token_ids.tolist()


if __name__ == "__main__":
    pass
