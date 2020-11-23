# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import utils.aes as AES


logger = logging.getLogger(__name__)


class RawAudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sample_rate,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        min_length=0,
        pad=False,
        normalize=True,
        aes=False,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.sizes = []
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.min_length = min_length
        self.pad = pad
        self.shuffle = shuffle
        self.normalize = normalize
        self.aes = aes
        if aes:
            self.key = [143, 194, 34, 208, 145, 203, 230, 143, 177, 246, 97, 206, 145, 92, 255, 84]
            self.iv = [103, 35, 148, 239, 76, 213, 47, 118, 255, 222, 123, 176, 106, 134, 98, 92]
            self.moo = AES.AESModeOfOperation()

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    def postprocess(self, feats, curr_sample_rate):
        if feats.dim() == 2:
            feats = feats.mean(-1)

        if curr_sample_rate != self.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        assert feats.dim() == 1, feats.dim()

        if self.normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        return feats

    def crop_to_max_size(self, wav1, wav2, target_size):
        size = len(wav1)
        diff = size - target_size
        if diff <= 0:
            return wav1, wav2

        start = np.random.randint(0, diff + 1)
        end = size - diff + start
        return (wav1[start:end], wav2[start:end])

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        aes_samples = [s for s in samples if s["aes_source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        aes_sources = [s["aes_source"] for s in aes_samples]
        sizes = [len(s) for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)
            #make it devisible by 20
            # [(512,10,5)] + [(512, 3, 2)]*5  div 160 or aes
            # [(512,10,5)] + [(512, 3, 2)]*4  div 80
            # [(512,10,5)] + [(512, 3, 2)]*3  div 40
            # [(512,10,5)] + [(512, 3, 2)]*2  div 20
            # [(512,10,5)]  div 10
            target_size = target_size - (target_size %160)

        collated_sources = sources[0].new_zeros(len(sources), target_size)
        aes_collated = sources[0].new_zeros(len(sources), target_size)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i], aes_collated[i] = self.crop_to_max_size(source, aes_sources[i], target_size)


        input = {"source": collated_sources, "aes":aes_collated}
        if self.pad:
            input["padding_mask"] = padding_mask
        return {"id": torch.LongTensor([s["id"] for s in samples]), "net_input": input}

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]

# loading data from regular and aes files
class AesFileAudioDataset(RawAudioDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        min_length=0,
        pad=False,
        normalize=True,
        aes=False
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            min_length=min_length,
            pad=pad,
            normalize=normalize,
            aes=aes,
        )

        self.fnames = []

        skipped = 0
        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for line in f:
                items = line.strip().split("\t")
                assert len(items) == 2, line
                sz = int(items[1])
                if min_length is not None and sz < min_length:
                    skipped += 1
                    continue
                self.fnames.append(items[0])
                self.sizes.append(sz)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

    def __getitem__(self, index):
        import soundfile as sf

        fname = os.path.join(self.root_dir, self.fnames[index])
        wav, curr_sample_rate = sf.read(fname)
        aes_fname = fname.replace(".flac", "-aes.flac")
        aes, curr_sample_rate = sf.read(aes_fname)
        # from -0.5 - 0.5 tp -2.0 - 2.0
        wav *=4
        aes *=4
        feats = torch.from_numpy(wav).float()
        aes_feats = torch.from_numpy(aes).float()
        feats = self.postprocess(feats, curr_sample_rate)
        aes_feats = self.postprocess(aes_feats, curr_sample_rate)
        return {"id": index, "source": feats, "aes_source": aes_feats}
