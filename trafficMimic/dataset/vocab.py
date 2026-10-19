# # NetMasquerade-main/src/trafficMimic/dataset/vocab.py

# import os
# import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import torch
# import tqdm
# import pickle
# import numpy as np
# from utils import recursive_namespace, feature_extract, read_yaml, read_flow_pkl
# from typing import List, Tuple, Dict, Union


# class Vocab(object):
#     def __init__(self, data: List[List[float]], max_size: int):
#         """
#         no use of '[cls]', '[sep]' in this project
#         """
#         self.special_list = ["[pad]", "[unk]", "[cls]", "[sep]", "[mask]"]
#         self.special_size = len(self.special_list)
#         self.reverse_list = {token: i for i, token in enumerate(self.special_list)}
#         self.pad_index = 0
#         self.unk_index = 1
#         self.eos_index = 2
#         self.sos_index = 3
#         self.mask_index = 4
#         self.data = data
#         self.max_size = max_size + self.special_size

#     def __len__(self):
#         return self.max_size + 1

#     # def to_seq(self, sentence, seq_len, with_eos=False, with_sos=False) -> list:
#     #     pass

#     # def from_seq(self, seq, with_pad=False):
#     #     pass

#     # def itos(self, index):
#     #     pass

#     # def stoi(self, word):
#     #     pass

#     @staticmethod
#     def load_vocab(vocab_path: str) -> "Vocab":
#         with open(vocab_path, "rb") as f:
#             return pickle.load(f)

#     def save_vocab(self, vocab_path: str):
#         original_data = self.data
#         self.data = None
#         with open(vocab_path, "wb") as f:
#             pickle.dump(self, f)
#         self.data = original_data


# class TimeVocab(Vocab):
#     def __init__(
#         self,
#         data: List[List[float]],
#         max_size: int = 50,
#         upper_bound: float = 300,
#         lower_bound: Union[None, float] = None,
#     ):
#         super(TimeVocab, self).__init__(data, max_size)
#         if upper_bound:
#             self.upper_bound = upper_bound
#         else:
#             self.upper_bound = max(max(i[1:]) for i in self.data)

#         if lower_bound:
#             self.lower_bound = lower_bound
#         else:
#             self.lower_bound = min(min(i[1:]) for i in self.data)
#         self.bins = self.binning()

#     def binning(self) -> List[float]:
#         min_val = np.log10(self.lower_bound + 1e-9)
#         max_val = np.log10(self.upper_bound + 1e-9)
#         bin_width = (max_val - min_val) / (self.max_size - self.special_size)
#         bins: List[float] = np.arange(min_val, max_val, bin_width)

#         return bins

#     def itos(self, index: int):
#         if index < self.special_size:
#             return self.special_list[index]
#         if index >= self.max_size:
#             return self.unk_index
#         if index == self.max_size - 1:
#             return self.bins[-1]
#         index -= self.special_size
#         return np.power(10, (self.bins[index + 1] + self.bins[index]) / 2.0)

#     """KITSUNE"""
#     # def stoi(self, word):
#     #     if word in self.reverse_list:
#     #         return self.reverse_list[word]
#     #     if word + 1e-9 >= self.upper_bound:
#     #         return self.unk_index
#     #     if word + 1e-9 < self.bins[0]:
#     #         return self.unk_index
#     #     return np.digitize(np.log10(word + 1e-9), self.bins) + self.special_size - 1

#     """ML"""

#     def stoi(self, word):
#         try:
#             val = float(word)
#         except (ValueError, TypeError):
#             val = 0.0

#         # 2. ENFORCE POSITIVE BOUNDS
#         # Ensure the value is strictly non-negative before applying log transformations.
#         # If your bins expect a minimum scale, clip the lower bound tightly.
#         val_clean = np.clip(val, a_min=0.0, a_max=None)

#         # 3. Compute log10 safely with a robust epsilon offset
#         log_val = np.log10(val_clean + 1e-7)

#         return np.digitize(log_val, self.bins) + self.special_size - 1

#     def to_seq(
#         self,
#         sentence: List[float],
#         seq_len: int,
#         with_eos: bool = False,
#         with_sos: bool = False,
#         with_ori_len: bool = False,
#     ) -> list:
#         seq = [self.stoi(word) for word in sentence]

#         if with_eos:
#             seq += [self.eos_index]
#         if with_sos:
#             seq = [self.sos_index] + seq

#         origin_seq_len = len(seq)

#         if seq_len is None:
#             pass
#         elif len(seq) <= seq_len:
#             seq += [self.pad_index for _ in range(seq_len - len(seq))]
#         else:
#             seq = seq[:seq_len]

#         return (seq, origin_seq_len) if with_ori_len else seq

#     def from_seq(self, seq, with_pad=False):
#         sentence = [self.itos(index) for index in seq]
#         cnt = len(sentence)
#         if not with_pad:
#             while cnt > 0 and sentence[cnt - 1] == 0:
#                 cnt -= 1
#         return sentence[:cnt]


# class SizeVocab(Vocab):
#     def __init__(self, data, ratio=1):
#         self.ratio = ratio
#         max_size = 1600 // ratio
#         super(SizeVocab, self).__init__(data, max_size)

#     def hash(self, word):
#         return int(word) // self.ratio

#     def rehash(self, index):
#         return int(index * self.ratio)

#     def itos(self, index):
#         if index < self.special_size:
#             return self.special_list[index]
#         if index > self.max_size:
#             return self.special_list[self.unk_index]
#         return self.rehash(index - self.special_size)

#     def stoi(self, word):
#         if word in self.reverse_list:
#             return self.reverse_list[word]
#         if self.hash(word) + self.special_size >= self.max_size:
#             return self.unk_index
#         return self.hash(word) + self.special_size

#     def to_seq(
#         self, sentence, seq_len, with_eos=False, with_sos=False, with_ori_len=False
#     ) -> list:
#         seq = [self.stoi(word) for word in sentence]

#         if with_eos:
#             seq += [self.eos_index]
#         if with_sos:
#             seq = [self.sos_index] + seq

#         origin_seq_len = len(seq)

#         if seq_len is None:
#             pass
#         elif len(seq) <= seq_len:
#             seq += [self.pad_index for _ in range(seq_len - len(seq))]
#         else:
#             seq = seq[:seq_len]

#         return (seq, origin_seq_len) if with_ori_len else seq

#     def from_seq(self, seq, with_pad=False):
#         sentence = [self.itos(index) for index in seq]
#         cnt = len(sentence)
#         if not with_pad:
#             while cnt > 0 and sentence[cnt - 1] == 0:
#                 cnt -= 1
#         return sentence[:cnt]


# if __name__ == "__main__":
#     pass


# NetMasquerade-main/src/trafficMimic/dataset/vocab.py

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import numpy as np
from typing import List, Optional, Union

class Vocab(object):
    def __init__(self, data: List[List[float]], max_size: int):
        """
        Base Vocabulary class handling special tokens and serialization.
        """
        self.special_list = ["[pad]", "[unk]", "[cls]", "[sep]", "[mask]"]
        self.special_size = len(self.special_list)
        self.reverse_list = {token: i for i, token in enumerate(self.special_list)}
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        self.data = data
        self.max_size = max_size + self.special_size

    def __len__(self):
        return self.max_size + 1

    @staticmethod
    def load_vocab(vocab_path: str) -> "Vocab":
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path: str):
        original_data = self.data
        self.data = None
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)
        self.data = original_data


class TimeVocab(Vocab):
    def __init__(
        self,
        data: List[List[float]],
        max_size: int = 50,
        upper_bound: float = 300.0,
        lower_bound: Union[None, float] = None,
    ):
        super(TimeVocab, self).__init__(data, max_size)

        # Determine bounds safely
        if upper_bound:
            self.upper_bound = upper_bound
        else:
            self.upper_bound = (
                max(max(i[1:]) for i in self.data) if self.data else 300.0
            )

        if lower_bound is not None:
            self.lower_bound = lower_bound
        else:
            self.lower_bound = min(min(i[1:]) for i in self.data) if self.data else 0.0

        self.bins = self.binning()

    def binning(self) -> List[float]:
        min_val = np.log10(self.lower_bound + 1e-9)
        max_val = np.log10(self.upper_bound + 1e-9)
        bin_width = (max_val - min_val) / (self.max_size - self.special_size)
        return np.arange(min_val, max_val, bin_width).tolist()

    def itos(self, index: int) -> Union[str, float]:
        """Returns standard token representations (can include text tags)."""
        if index < self.special_size:
            return self.special_list[index]
        if index >= self.max_size:
            return self.special_list[self.unk_index]
        if index == self.max_size - 1:
            return float(np.power(10, self.bins[-1]))

        bin_idx = index - self.special_size
        bin_idx = max(0, min(bin_idx, len(self.bins) - 2))
        return float(np.power(10, (self.bins[bin_idx + 1] + self.bins[bin_idx]) / 2.0))

    def stoi(self, word: Union[str, float, int]) -> int:
        """Encodes strings or numerical floating-point attributes safely into token IDs."""
        if isinstance(word, str) and word in self.reverse_list:
            return self.reverse_list[word]

        try:
            val = float(word)
        except (ValueError, TypeError):
            return self.unk_index

        val_clean = np.clip(val, a_min=self.lower_bound, a_max=self.upper_bound)
        log_val = np.log10(val_clean + 1e-9)

        # Safeguard digitized index from overlapping with special tokens
        idx = int(np.digitize(log_val, self.bins))
        idx = np.clip(idx, 1, len(self.bins))
        return idx + self.special_size - 1

    def to_seq(
        self,
        sentence: List[float],
        seq_len: int,
        with_eos: bool = False,
        with_sos: bool = False,
        with_ori_len: bool = False,
    ) -> list[float]:
        seq = [self.stoi(word) for word in sentence]
        if with_eos:
            seq += [self.eos_index]
        if with_sos:
            seq = [self.sos_index] + seq
        origin_seq_len = len(seq)

        if seq_len is not None:
            if len(seq) <= seq_len:
                seq += [self.pad_index for _ in range(seq_len - len(seq))]
            else:
                seq = seq[:seq_len]
        return (seq, origin_seq_len) if with_ori_len else seq

    def from_seq_to_values(self, seq: List[int]) -> List[float]:
        """Decodes token sequences into pure numeric floats for ML evaluators, using 0.0 as baseline fallback."""
        raw_values = []
        for idx in seq:
            if idx < self.special_size:
                raw_values.append(0.0)
            elif idx >= self.max_size:
                raw_values.append(float(self.upper_bound))
            elif idx == self.max_size - 1:
                raw_values.append(float(np.power(10, self.bins[-1])))
            else:
                bin_idx = idx - self.special_size
                bin_idx = max(0, min(bin_idx, len(self.bins) - 2))
                raw_values.append(
                    float(
                        np.power(
                            10, (self.bins[bin_idx + 1] + self.bins[bin_idx]) / 2.0
                        )
                    )
                )
        return raw_values

    # Inside your Vocab / TimeVocab / SizeVocab classes in vocab.py
    def decode_tokens(self, token_indices: list[int]) -> list[float]:
        """Converts a sequence of vocabulary indices back into real physical floats/ints."""
        decoded_values = []
        for idx in token_indices:
            if idx < self.special_size:
                # Handle special tokens ([pad], [mask], [unk]) safely using normal defaults
                decoded_values.append(0.0)
            else:
                # Safely extract the original value from the dictionary or array
                try:
                    # Adjust depending on how your specific vocabulary stores the inverse mapping
                    if hasattr(self, "itos"):
                        val = (
                            self.itos[idx]
                            if isinstance(self.itos, list)
                            else self.itos(idx)
                        )
                    else:
                        val = self.data[idx - self.special_size]
                    decoded_values.append(float(val))
                except Exception:
                    decoded_values.append(0.0)
        return decoded_values


class SizeVocab(Vocab):
    def __init__(self, data, ratio=1):
        self.ratio = ratio
        max_size = 1600 // ratio
        super(SizeVocab, self).__init__(data, max_size)

    def hash(self, word):
        try:
            val = int(float(word))
        except (ValueError, TypeError):
            val = 0
        return val // self.ratio

    def rehash(self, index):
        return int(index * self.ratio)

    def itos(self, index: int) -> Union[str, int]:
        if index < self.special_size:
            return self.special_list[index]
        if index >= self.max_size:
            return self.special_list[self.unk_index]
        return self.rehash(index - self.special_size)

    def stoi(self, word: Union[str, float, int]) -> int:
        if isinstance(word, str) and word in self.reverse_list:
            return self.reverse_list[word]

        h_val = self.hash(word)
        token_idx = h_val + self.special_size
        if token_idx >= self.max_size:
            return self.max_size - 1
        return max(self.special_size, token_idx)

    def to_seq(
        self,
        sentence: list[Union[str, float, int]],
        seq_len: int,
        with_eos: bool = False,
        with_sos: bool = False,
        with_ori_len: bool = False,
    ) -> list[int]:
        seq = [self.stoi(word) for word in sentence]
        if with_eos:
            seq += [self.eos_index]
        if with_sos:
            seq = [self.sos_index] + seq
        origin_seq_len = len(seq)

        if seq_len is not None:
            if len(seq) <= seq_len:
                seq += [self.pad_index for _ in range(seq_len - len(seq))]
            else:
                seq = seq[:seq_len]
        return (seq, origin_seq_len) if with_ori_len else seq

    def from_seq_to_values(self, seq: List[int]) -> List[int]:
        """Decodes size tokens into pure numerical integers for ML evaluators, using 0 as fallback."""
        raw_values = []
        for idx in seq:
            if idx < self.special_size:
                raw_values.append(0)
            elif idx >= self.max_size:
                raw_values.append(1600)
            else:
                raw_values.append(self.rehash(idx - self.special_size))
        return raw_values

    # Inside your Vocab / TimeVocab / SizeVocab classes in vocab.py
    def decode_tokens(self, token_indices: list[int]) -> list[float]:
        """Converts a sequence of vocabulary indices back into real physical floats/ints."""
        decoded_values = []
        for idx in token_indices:
            if idx < self.special_size:
                # Handle special tokens ([pad], [mask], [unk]) safely using normal defaults
                decoded_values.append(0.0)
            else:
                # Safely extract the original value from the dictionary or array
                try:
                    # Adjust depending on how your specific vocabulary stores the inverse mapping
                    if hasattr(self, "itos"):
                        val = (
                            self.itos[idx]
                            if isinstance(self.itos, list)
                            else self.itos(idx)
                        )
                    else:
                        val = self.data[idx - self.special_size]
                    decoded_values.append(float(val))
                except Exception:
                    decoded_values.append(0.0)
        return decoded_values
