# NetMasquerade-main/src/trafficMimic/dataset/dataset.py

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import random
import torch
from torch.utils.data import Dataset, DataLoader
from utils import feature_extract, read_flow_pkl
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

        # self.ipd_list, self.size_list = list(self.data[0]), list(self.data[1])
        self.ipd_list = [flow[0] for flow in self.data]
        self.size_list = [flow[1] for flow in self.data]

    def __len__(
        self,
    ):
        return len(self.ipd_list)

    def __getitem__(self, item):
        flow_ipd = self._float_to_tokens(self.timevocab, self.ipd_list[item])
        flow_size = self._float_to_tokens(self.sizevocab, self.size_list[item])
        
        flow_ipd = flow_ipd[:self.seq_len]
        flow_size = flow_size[:self.seq_len]
        real_length = len(flow_ipd)

        # [NEW] Extract the pure, unsquashed physical floats
        raw_ipd = self.ipd_list[item][:self.seq_len]
        raw_size = self.size_list[item][:self.seq_len]

        flow_ipd_random, flow_ipd_label = self.random_word(flow_ipd.copy(), self.timevocab)
        flow_size_random, flow_size_label = self.random_word(flow_size.copy(), self.sizevocab)

        padding = [self.timevocab.pad_index for _ in range(self.seq_len - len(flow_ipd))]
        raw_padding = [0.0 for _ in range(self.seq_len - len(raw_ipd))]
        
        flow_ipd.extend(padding)
        flow_size.extend(padding)
        flow_ipd_label.extend(padding)
        flow_size_label.extend(padding)
        
        # [NEW] Pad the raw floats with 0.0
        raw_ipd.extend(raw_padding)
        raw_size.extend(raw_padding)

        output = {
            "flow_ipd": flow_ipd,             
            "flow_size": flow_size,           
            "flow_ipd_label": flow_ipd_label,
            "flow_size_label": flow_size_label,
            "real_length": real_length,
            "raw_ipd": raw_ipd,             # <--- Passed to Environment
            "raw_size": raw_size,           # <--- Passed to Environment
        }

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, tokens, vocab):
        output_label = []
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    # tokens[i] = random.randrange(len(vocab))
                    tokens[i] = random.randint(5, vocab.__len__() - 1)

                # 10% ratndomly change token to current token
                else:
                    tokens[i] = vocab.stoi(token)

                output_label.append(token)

            else:
                tokens[i] = token
                output_label.append(0)
        return tokens, output_label

    def _float_to_tokens(self, vocab, float_list):
        """Bypasses the broken hash() function and strictly finds the closest mathematical token bucket."""
        if not hasattr(vocab, "np_data"):
            # Extract the actual bucket values into a numpy array once for speed
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

        # Use fast numpy broadcasting to find the nearest neighbor for every packet
        arr = np.array(float_list, dtype=np.float32)
        diffs = np.abs(arr[:, None] - vocab.np_data[None, :])
        closest_indices = np.argmin(diffs, axis=1)

        # Add special_size (e.g., 5) back to get the true Token ID
        token_ids = closest_indices + vocab.special_size
        return token_ids.tolist()


if __name__ == "__main__":
    pass
