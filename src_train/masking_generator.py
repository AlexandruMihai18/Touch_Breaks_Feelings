# --------------------------------------------------------
# Images Speak in Images: A Generalist Painter for In-Context Visual Learning (https://arxiv.org/abs/2212.02499)
# Github source: https://github.com/baaivision/Painter
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Xinlong Wang, Wen Wang
# Based on MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------'

import math
import random
from typing import Optional, Tuple, Union

import numpy as np


class MaskingGenerator:
    """
    MaskingGenerator produces random masks for masked image modeling pretraining.

    Args:
        input_size: int or (int, int) tuple for height and width of the mask. If int, the mask is square.
            Used to compute the number of patches in the mask, which is num_patches = height * width.
        num_masking_patches: total number of patches to mask in the output mask.
        min_num_patches: minimum number of patches to mask in a single contiguous region. Default 4.
        max_num_patches: maximum number of patches to mask in a single contiguous region.
            Default is num_masking_patches (i.e. no limit).
        min_aspect: minimum aspect ratio (h/w) of a single masked region. Default 0.3.
        max_aspect: maximum aspect ratio (h/w) of a single masked region. Default is 1/min_aspect.
    """

    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]],
        num_masking_patches: int,
        min_num_patches: int = 4,
        max_num_patches: Optional[int] = None,
        min_aspect: float = 0.3,
        max_aspect: Optional[float] = None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = (
            num_masking_patches if max_num_patches is None else max_num_patches
        )

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask: np.ndarray, max_mask_patches: int) -> int:
        n_new_masked_patches = 0
        # Try up to 10 times to find a valid region to mask
        for attempt in range(10):

            # Sample a target ratio and aspect ratio
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w >= self.width or h >= self.height:
                # Region is too large to fit in the mask, try again
                continue

            top = random.randint(0, self.height - h)
            left = random.randint(0, self.width - w)

            # Count how many patches in this region are already masked
            num_masked = mask[top : top + h, left : left + w].sum()

            if num_masked == h * w:
                # Skip if all the patches have been masked
                continue
            if h * w - num_masked > max_mask_patches:
                # Skip if masking this region would exceed the max number
                # of masking patches
                continue

            # TODO: vectorize this operation
            # Set the patches in this region to masked (1)
            # and count how many new patches we masked
            for i in range(top, top + h):
                for j in range(left, left + w):
                    if mask[i, j] == 0:
                        mask[i, j] = 1
                        n_new_masked_patches += 1

            if n_new_masked_patches > 0:
                break
        return n_new_masked_patches

    def __call__(self) -> np.ndarray:
        # Initialize mask array and count of masked patches
        mask = np.zeros(shape=self.get_shape(), dtype=np.int32)
        mask_count = 0

        # Iteratively add masked regions until we reach the desired number
        # of masking patches
        while mask_count < self.num_masking_patches:
            # Compute max number of patches available to mask in this
            # iteration
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            # Mask a new region and get the number of newly masked patches
            n_new_masked_patches = self._mask(mask, max_mask_patches)

            if n_new_masked_patches == 0:
                # If we failed to find a valid region to mask after 10 attempts,
                # break to avoid an infinite loop
                break
            else:
                # Add the number of newly masked patches to the total count
                mask_count += n_new_masked_patches

        # maintain a fix number (self.num_masking_patches)
        if mask_count > self.num_masking_patches:
            # If number of masked patches exceeds the desired number,
            # randomly unmask some patches
            n_unmask_patches = mask_count - self.num_masking_patches
            mask_x, mask_y = mask.nonzero()
            to_vis = np.random.choice(mask_x.shape[0], n_unmask_patches, replace=False)
            mask[mask_x[to_vis], mask_y[to_vis]] = 0

        elif mask_count < self.num_masking_patches:
            # We failed to mask enough patches,
            # so randomly mask some additional patches
            n_missing_patches = self.num_masking_patches - mask_count
            mask_x, mask_y = (mask == 0).nonzero()
            to_mask = np.random.choice(
                mask_x.shape[0], n_missing_patches, replace=False
            )
            mask[mask_x[to_mask], mask_y[to_mask]] = 1

        # Double check that masked number is exactly num_masking_patches
        assert (
            mask.sum() == self.num_masking_patches
        ), f"mask: {mask}, mask count {mask.sum()}"

        return mask
