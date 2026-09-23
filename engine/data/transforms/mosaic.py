"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import random
from copy import deepcopy

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from PIL import Image

from ...core import register
from .._misc import convert_to_tv_tensor


@register()
class Mosaic(T.Transform):
    """
    Applies Mosaic augmentation to a batch of images. Combines four randomly selected images
    into a single composite image with randomized transformations.
    """

    def __init__(self, output_size=320, max_size=None, rotation_range=0, translation_range=(0.1, 0.1),
                 scaling_range=(0.5, 1.5), probability=1.0, fill_value=114, use_cache=True, max_cached_images=50,
                 random_pop=True) -> None:
        """
        Args:
            output_size (int): Target size for resizing individual images.
            rotation_range (float): Range of rotation in degrees for affine transformation.
            translation_range (tuple): Range of translation for affine transformation.
            scaling_range (tuple): Range of scaling factors for affine transformation.
            probability (float): Probability of applying the Mosaic augmentation.
            fill_value (int): Fill value for padding or affine transformations.
            use_cache (bool): Whether to use cache. Defaults to True.
            max_cached_images (int): The maximum length of the cache.
            random_pop (bool): Whether to randomly pop a result from the cache.
        """
        super().__init__()
        self.resize = T.Resize(size=output_size, max_size=max_size)
        self.probability = probability
        self.affine_transform = T.RandomAffine(degrees=rotation_range, translate=translation_range,
                                               scale=scaling_range, fill=fill_value)
        self.use_cache = use_cache
        self.mosaic_cache = []
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop

    def state_dict(self):
        return {
            'use_cache': self.use_cache,
            'mosaic_cache': deepcopy(self.mosaic_cache),
            'max_cached_images': self.max_cached_images,
            'random_pop': self.random_pop,
        }

    def load_state_dict(self, state):
        if not state:
            return

        self.use_cache = state.get('use_cache', self.use_cache)
        self.max_cached_images = state.get('max_cached_images', self.max_cached_images)
        self.random_pop = state.get('random_pop', self.random_pop)
        self.mosaic_cache = deepcopy(state.get('mosaic_cache', []))

    def load_samples_from_dataset(self, image, target, dataset):
        """Loads and resizes a set of images and their corresponding targets."""
        # Append the main image
        get_size_func = F.get_size if hasattr(F, "get_size") else F.get_spatial_size  # torchvision >=0.17 is get_size
        image, target = self.resize(image, target)
        resized_images, resized_targets = [image], [target]
        max_height, max_width = get_size_func(resized_images[0])

        # randomly select 3 images
        sample_indices = random.choices(range(len(dataset)), k=3)
        for idx in sample_indices:
            # image, target = dataset.load_item(idx)
            image, target = self.resize(dataset.load_item(idx))
            height, width = get_size_func(image)
            max_height, max_width = max(max_height, height), max(max_width, width)
            resized_images.append(image)
            resized_targets.append(target)

        return resized_images, resized_targets, max_height, max_width

    def load_samples_from_cache(self, image, target, cache):
        image, target = self.resize(image, target)
        cache.append(dict(img=image, labels=target))

        if len(cache) > self.max_cached_images:
            if self.random_pop:
                index = random.randint(0, len(cache) - 2)  # do not remove last image
            else:
                index = 0
            cache.pop(index)
        sample_indices = random.choices(range(len(cache)), k=3)
        mosaic_samples = [dict(img=cache[idx]["img"].copy(), labels=self._clone(cache[idx]["labels"])) for idx in
                          sample_indices]  # sample 3 images
        mosaic_samples = [dict(img=image.copy(), labels=self._clone(target))] + mosaic_samples

        get_size_func = F.get_size if hasattr(F, "get_size") else F.get_spatial_size
        sizes = [get_size_func(mosaic_samples[idx]["img"]) for idx in range(4)]
        max_height = max(size[0] for size in sizes)
        max_width = max(size[1] for size in sizes)

        return mosaic_samples, max_height, max_width

    def create_mosaic_from_cache(self, mosaic_samples, max_height, max_width):
        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]
        merged_image = Image.new(mode=mosaic_samples[0]["img"].mode, size=(max_width * 2, max_height * 2), color=0)
        offsets = torch.tensor([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]).repeat(1, 2)

        mosaic_target = []
        all_masks = []
        has_masks = 'masks' in mosaic_samples[0]["labels"]

        for i, sample in enumerate(mosaic_samples):
            img = sample["img"]
            target = sample["labels"]

            merged_image.paste(img, placement_offsets[i])
            target['boxes'] = target['boxes'] + offsets[i]

            if has_masks:
                # Paste each tile's masks onto a full-size canvas so they stay aligned
                # with the merged image (tile masks have per-tile sizes and offsets).
                curr_m = target['masks']
                full_size_m = torch.zeros((curr_m.shape[0], max_height * 2, max_width * 2),
                                          dtype=curr_m.dtype, device=curr_m.device)
                # Index (not unpack): PIL paste() extends a 2-element list box to 4 in place.
                x_off, y_off = placement_offsets[i][0], placement_offsets[i][1]
                h_c, w_c = curr_m.shape[1], curr_m.shape[2]
                full_size_m[:, y_off: y_off + h_c, x_off: x_off + w_c] = curr_m
                all_masks.append(full_size_m)
                # Remove tile-size masks to avoid the cat size mismatch below.
                del target['masks']

            mosaic_target.append(target)

        merged_target = {}
        for key in mosaic_target[0]:
            merged_target[key] = torch.cat([target[key] for target in mosaic_target])

        if has_masks:
            merged_target['masks'] = torch.cat(all_masks, dim=0)

        return merged_image, merged_target

    def create_mosaic_from_dataset(self, images, targets, max_height, max_width):
        """Creates a mosaic image by combining multiple images."""
        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]
        merged_image = Image.new(mode=images[0].mode, size=(max_width * 2, max_height * 2), color=0)
        for i, img in enumerate(images):
            merged_image.paste(img, placement_offsets[i])

        """Merges targets into a single target dictionary for the mosaic."""
        offsets = torch.tensor([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]).repeat(1, 2)
        merged_target = {}
        for key in targets[0]:
            if key == 'boxes':
                values = [target[key] + offsets[i] for i, target in enumerate(targets)]
            elif key == 'masks':
                # Paste tile masks onto full-size canvases so they stay aligned with the
                # merged image (same quadrant layout as the image paste above).
                values = []
                for i, target in enumerate(targets):
                    curr_m = target[key]
                    full_size_m = torch.zeros((curr_m.shape[0], max_height * 2, max_width * 2),
                                              dtype=curr_m.dtype, device=curr_m.device)
                    # Index (not unpack): PIL paste() extends a 2-element list box to 4 in place.
                    x_off, y_off = placement_offsets[i][0], placement_offsets[i][1]
                    h_c, w_c = curr_m.shape[1], curr_m.shape[2]
                    full_size_m[:, y_off: y_off + h_c, x_off: x_off + w_c] = curr_m
                    values.append(full_size_m)
            else:
                values = [target[key] for target in targets]

            merged_target[key] = torch.cat(values, dim=0) if isinstance(values[0], torch.Tensor) else values

        return merged_image, merged_target

    @staticmethod
    def _clone(tensor_dict):
        return {key: value.clone() for (key, value) in tensor_dict.items()}

    def forward(self, *inputs):
        """
        Args:
            inputs (tuple): Input tuple containing (image, target, dataset).

        Returns:
            tuple: Augmented (image, target, dataset).
        """
        if len(inputs) == 1:
            inputs = inputs[0]
        image, target, dataset = inputs

        # Skip mosaic augmentation with probability 1 - self.probability
        if self.probability < 1.0 and random.random() > self.probability:
            return image, target, dataset

        # Prepare mosaic components
        if self.use_cache:
            mosaic_samples, max_height, max_width = self.load_samples_from_cache(image, target, self.mosaic_cache)
            mosaic_image, mosaic_target = self.create_mosaic_from_cache(mosaic_samples, max_height, max_width)
        else:
            resized_images, resized_targets, max_height, max_width = self.load_samples_from_dataset(image, target,dataset)
            mosaic_image, mosaic_target = self.create_mosaic_from_dataset(resized_images, resized_targets, max_height, max_width)

        # Clamp boxes and convert target formats
        if 'boxes' in mosaic_target:
            mosaic_target['boxes'] = convert_to_tv_tensor(mosaic_target['boxes'], 'boxes', box_format='xyxy',
                                                          spatial_size=mosaic_image.size[::-1])
        if 'masks' in mosaic_target:
            mosaic_target['masks'] = convert_to_tv_tensor(mosaic_target['masks'], 'masks')

        # Apply affine transformations
        mosaic_image, mosaic_target = self.affine_transform(mosaic_image, mosaic_target)

        return mosaic_image, mosaic_target, dataset
