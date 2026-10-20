# NetMasquerade-main/src/advGenerate/rl_dataset.py

# This file is a simulation version of NetMasquerade.

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from trafficMimic.utils import *
from torch.utils.data import Dataset
from trafficMimic.dataset.vocab import *
from copy import deepcopy
from typing import List, Tuple, Dict

class RLFullFlowDataset(Dataset):
    def __init__(
        self,
        args: argparse.Namespace,
        timevocab: TimeVocab,
        sizevocab: SizeVocab,
        mode: str,
    ) -> None:
        super(Dataset, self).__init__()
        self.seq_len = args.model.state_dim

        # Bypass feature extraction and load the raw arrays directly
        if mode == "train":
            self.data: List[Tuple[torch.Tensor, torch.Tensor]] = list(
                read_flow_pkl(args.trainer.train_data_pth)
            )
        else:
            self.data: List[Tuple[torch.Tensor, torch.Tensor]] = list(
                read_flow_pkl(args.trainer.test_data_pth)
            )

        self.timevocab = timevocab
        self.sizevocab = sizevocab

        self.ipd_list = list(self.data[0])
        self.size_list = list(self.data[1])

        # Provide dummy metadata to satisfy the environment's return expectations
        num_flows = len(self.ipd_list)
        self.src = ["192.168.1.1"] * num_flows
        self.dst = ["192.168.1.2"] * num_flows
        self.src_port = [12345] * num_flows
        self.dst_port = [80] * num_flows
        self.flow_list = [None] * num_flows

    def __len__(self):
        return len(self.ipd_list)

    def __getitem__(
        self, item: int
    ) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, List[int]], str, str, int, int, object
    ]:
        flow_ipd: list[float] = self.timevocab.to_seq(self.ipd_list[item], None)
        flow_size: list[int] = self.sizevocab.to_seq(self.size_list[item], None)

        flow_ipd = [self.timevocab.sos_index] + flow_ipd + [self.timevocab.eos_index]
        flow_size = [self.sizevocab.sos_index] + flow_size + [self.sizevocab.eos_index]

        real_length = len(flow_size)

        flow_ipd = flow_ipd[: self.seq_len]
        flow_size = flow_size[: self.seq_len]

        padding = [
            self.timevocab.pad_index for _ in range(self.seq_len - len(flow_ipd))
        ]
        flow_ipd.extend(padding), flow_size.extend(padding)

        output = {
            "flow_ipd": flow_ipd,
            "flow_size": flow_size,
            "real_length": real_length,
        }

        ori_output = {
            "ipd": self.ipd_list[item][: self.seq_len - 1],
            "size": self.size_list[item][: self.seq_len - 1],
        }
        return (
            {key: torch.tensor(value) for key, value in output.items()},
            ori_output,
            self.src[item],
            self.dst[item],
            self.src_port[item],
            self.dst_port[item],
            self.flow_list[item],
        )

    def collate_fn(self, batch):
        return batch


class RLFullFlowNoDiffDataset(Dataset):
    def __init__(
        self,
        args: argparse.Namespace,
        timevocab: TimeVocab,
        sizevocab: SizeVocab,
        mode: str,
    ) -> None:
        super(Dataset, self).__init__()
        self.seq_len = args.model.state_dim

        # Bypass feature extraction
        if mode == "train":
            self.data = list(read_flow_pkl(args.trainer.train_data_pth))
        else:
            self.data = list(read_flow_pkl(args.trainer.test_data_pth))

        self.timevocab = timevocab
        self.sizevocab = sizevocab

        self.ipd_list = list(self.data[0])
        self.size_list = list(self.data[1])

        # Provide dummy metadata
        num_flows = len(self.ipd_list)
        self.src = ["192.168.1.1"] * num_flows
        self.dst = ["192.168.1.2"] * num_flows
        self.src_port = [12345] * num_flows
        self.dst_port = [80] * num_flows
        self.flow_list = [None] * num_flows

    def __len__(self):
        return len(self.ipd_list)

    def __getitem__(self, item):
        # We ALREADY have IPDs, so we skip the timestamp subtraction math!
        stp = deepcopy(self.ipd_list[item])

        flow_ipd = self.timevocab.to_seq(stp, None)
        flow_size = self.sizevocab.to_seq(self.size_list[item], None)

        flow_ipd = [self.timevocab.sos_index] + flow_ipd + [self.timevocab.eos_index]
        flow_size = [self.sizevocab.sos_index] + flow_size + [self.sizevocab.eos_index]

        real_length = len(flow_size)

        flow_ipd = flow_ipd[: self.seq_len]
        flow_size = flow_size[: self.seq_len]

        padding = [
            self.timevocab.pad_index for _ in range(self.seq_len - len(flow_ipd))
        ]
        flow_ipd.extend(padding), flow_size.extend(padding)

        output = {
            "flow_ipd": flow_ipd,
            "flow_size": flow_size,
            "real_length": real_length,
        }

        ori_output = {
            "ipd": stp[: self.seq_len - 1],
            "size": self.size_list[item][: self.seq_len - 1],
        }
        return (
            {key: torch.tensor(value) for key, value in output.items()},
            ori_output,
            self.src[item],
            self.dst[item],
            self.src_port[item],
            self.dst_port[item],
            self.flow_list[item],
        )

    def collate_fn(self, batch):
        return batch


class RLFlowDataset(Dataset):
    def __init__(
        self,
        args: argparse.Namespace,
        timevocab: TimeVocab,
        sizevocab: SizeVocab,
        mode: str,
    ) -> None:
        super(Dataset, self).__init__()
        self.seq_len = args.model.state_dim
        if mode == "train":
            self.data: List[Tuple[torch.Tensor, torch.Tensor]] = [
                read_flow_pkl(args.trainer.train_data_pth)
            ]
        else:
            self.data: List[Tuple[torch.Tensor, torch.Tensor]] = list(
                read_flow_pkl(args.trainer.test_data_pth)
            )
        self.timevocab = timevocab
        self.sizevocab = sizevocab

        self.ipd_list, self.size_list = list(self.data[0]), list(self.data[1])

    def __len__(
        self,
    ):
        return len(self.ipd_list)

    def __getitem__(self, item):
        flow_ipd = self.timevocab.to_seq(self.ipd_list[item], None)
        flow_size = self.sizevocab.to_seq(self.size_list[item], None)

        flow_ipd = [self.timevocab.sos_index] + flow_ipd + [self.timevocab.eos_index]
        flow_size = [self.sizevocab.sos_index] + flow_size + [self.sizevocab.eos_index]

        real_length = len(flow_size)

        flow_ipd = flow_ipd[: self.seq_len]
        flow_size = flow_size[: self.seq_len]

        padding = [
            self.timevocab.pad_index for _ in range(self.seq_len - len(flow_ipd))
        ]
        flow_ipd.extend(padding), flow_size.extend(padding)

        output = {
            "flow_ipd": flow_ipd,
            "flow_size": flow_size,
            "real_length": real_length,
        }

        return {key: torch.tensor(value) for key, value in output.items()}


if __name__ == "__main__":
    pass
