"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
LMDB-backed COCO detection dataset.

This keeps the large annotation table out of Python worker heaps.  The dataset
only keeps image ids in memory and reads per-image metadata/annotations from
LMDB on demand.
"""

import json
import os
import pickle
from importlib import import_module
from pathlib import Path

import torch
from PIL import Image

from ...core import register
from .._misc import convert_to_tv_tensor
from ._dataset import DetDataset
from .coco_dataset import ConvertCocoPolysToMask, mscoco_category2label

Image.MAX_IMAGE_PIXELS = None

__all__ = ["CocoLmdbDetection"]


def _require_module(name, install_hint):
    try:
        return import_module(name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Missing dependency `{name}`. Install it with `{install_hint}`.") from exc


def _image_id_to_key(image_id):
    return int(image_id).to_bytes(8, byteorder="big", signed=False)


@register()
class CocoLmdbDetection(DetDataset):
    __inject__ = ["transforms"]
    __share__ = ["remap_mscoco_category"]

    def __init__(
        self,
        img_folder,
        index_file,
        lmdb_path,
        transforms,
        categories_file=None,
        return_masks=False,
        remap_mscoco_category=False,
        max_readers=2048,
    ):
        self.img_folder = str(img_folder)
        self.index_file = str(index_file)
        self.lmdb_path = str(lmdb_path)
        self.categories_file = str(categories_file) if categories_file else None
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category
        self.max_readers = max_readers

        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self._env = None
        self._env_pid = None
        self._msgpack = None

        self.ids, self._legacy_records = self._load_index(self.index_file)
        self._record_by_id = None
        self._categories = self._load_categories(self.categories_file)

    @staticmethod
    def _load_index(index_file):
        with open(index_file, "rb") as f:
            index = pickle.load(f)

        if isinstance(index, dict):
            fmt = index.get("format", "")
            if "ids" in index:
                return index["ids"], None
            if "records" in index:
                records = list(index["records"])
                return [record[0] for record in records], records
            raise ValueError(f"Unsupported LMDB index format `{fmt}` in {index_file}")

        records = list(index)
        if not records:
            return [], records

        first = records[0]
        if isinstance(first, (tuple, list)):
            return [record[0] for record in records], records
        return records, None

    @staticmethod
    def _load_categories(categories_file):
        if not categories_file:
            return []

        path = Path(categories_file)
        if not path.exists():
            return []

        with path.open() as f:
            categories = json.load(f)
        if isinstance(categories, dict):
            categories = categories.get("categories", [])
        return categories

    def _init_lmdb(self):
        pid = os.getpid()
        if self._env is not None and self._env_pid == pid:
            return

        lmdb = _require_module("lmdb", "pip install lmdb")
        self._msgpack = _require_module("msgpack", "pip install msgpack")
        self._env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=self.max_readers,
        )
        self._env_pid = pid

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        state["_env_pid"] = None
        state["_msgpack"] = None
        return state

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        return img, target

    def load_item(self, idx):
        image_id = int(self.ids[idx])
        record = self._get_record(image_id)
        image = self._load_image(record["file_name"])
        target = {"image_id": image_id, "annotations": record["annotations"]}

        if self.remap_mscoco_category:
            image, target = self.prepare(image, target, category2label=mscoco_category2label)
        else:
            image, target = self.prepare(image, target)

        target["idx"] = torch.tensor([idx])

        if "boxes" in target:
            target["boxes"] = convert_to_tv_tensor(target["boxes"], key="boxes", spatial_size=image.size[::-1])
        if "masks" in target:
            target["masks"] = convert_to_tv_tensor(target["masks"], key="masks")

        return image, target

    def _get_record(self, image_id):
        self._init_lmdb()
        with self._env.begin(write=False) as txn:
            blob = txn.get(_image_id_to_key(image_id))

        if blob is None:
            raise KeyError(f"image_id={image_id} is missing from {self.lmdb_path}")

        record = self._msgpack.unpackb(blob, raw=False)
        if isinstance(record, list):
            record = self._legacy_record(image_id, record)
        return record

    def _legacy_record(self, image_id, annotations):
        if self._record_by_id is None:
            if self._legacy_records is None:
                raise RuntimeError(
                    "LMDB contains legacy annotation-only values, but the index "
                    "does not contain file_name/height/width records."
                )
            self._record_by_id = {
                int(image_id): (file_name, height, width)
                for image_id, file_name, height, width in self._legacy_records
            }

        file_name, height, width = self._record_by_id[int(image_id)]
        return {
            "file_name": file_name,
            "height": height,
            "width": width,
            "annotations": annotations,
        }

    def _load_image(self, file_name):
        path = Path(self.img_folder) / file_name
        return Image.open(path).convert("RGB")

    def extra_repr(self):
        s = f" img_folder: {self.img_folder}\n"
        s += f" index_file: {self.index_file}\n lmdb_path: {self.lmdb_path}\n"
        s += f" return_masks: {self.return_masks}\n"
        if self._transforms is not None:
            s += f" transforms:\n   {repr(self._transforms)}"
        return s

    @property
    def categories(self):
        return self._categories

    @property
    def category2name(self):
        return {cat["id"]: cat["name"] for cat in self.categories}

    @property
    def category2label(self):
        return {cat["id"]: i for i, cat in enumerate(self.categories)}

    @property
    def label2category(self):
        return {i: cat["id"] for i, cat in enumerate(self.categories)}
