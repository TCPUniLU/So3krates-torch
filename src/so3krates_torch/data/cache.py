import hashlib
import json
import logging
import os
import tempfile
from typing import Optional

import torch

from .atomic_data import AtomicData

logger = logging.getLogger(__name__)


class DiskCache:
    """Disk cache for AtomicData objects.

    Caches graph-constructed AtomicData to avoid recomputing neighbor
    lists on subsequent epochs. Each item is stored as a .pt file.
    Cache is invalidated when cutoff parameters or source file change.
    """

    def __init__(
        self,
        cache_dir: str,
        cutoff: float,
        cutoff_lr: Optional[float] = None,
        source_file: Optional[str] = None,
    ):
        self.cache_dir = cache_dir
        self.cutoff = cutoff
        self.cutoff_lr = cutoff_lr
        self.source_file = source_file
        os.makedirs(cache_dir, exist_ok=True)
        if not self._validate_cache():
            self.clear()
            self._write_metadata()

    def _compute_metadata(self) -> dict:
        meta = {
            "cutoff": self.cutoff,
            "cutoff_lr": self.cutoff_lr,
        }
        if self.source_file and os.path.isfile(self.source_file):
            stat = os.stat(self.source_file)
            meta["file_size"] = stat.st_size
            meta["file_mtime"] = stat.st_mtime
            with open(self.source_file, "rb") as f:
                chunk = f.read(1024 * 1024)  # first 1 MB
            meta["file_hash"] = hashlib.sha256(chunk).hexdigest()
        return meta

    def _metadata_path(self) -> str:
        return os.path.join(self.cache_dir, "metadata.json")

    def _validate_cache(self) -> bool:
        meta_path = self._metadata_path()
        if not os.path.isfile(meta_path):
            return False
        try:
            with open(meta_path, "r") as f:
                stored = json.load(f)
            current = self._compute_metadata()
            return stored == current
        except (json.JSONDecodeError, OSError):
            return False

    def _write_metadata(self) -> None:
        meta = self._compute_metadata()
        meta_path = self._metadata_path()
        with open(meta_path, "w") as f:
            json.dump(meta, f)

    def _item_path(self, idx: int) -> str:
        return os.path.join(self.cache_dir, f"{idx:08d}.pt")

    def load(self, idx: int) -> Optional[AtomicData]:
        path = self._item_path(idx)
        if os.path.isfile(path):
            try:
                return torch.load(path, weights_only=False)
            except Exception:
                logger.warning(f"Failed to load cache item {idx}")
                return None
        return None

    def save(self, idx: int, data: AtomicData) -> None:
        path = self._item_path(idx)
        # Atomic write: save to temp file then rename
        fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, suffix=".pt.tmp")
        try:
            os.close(fd)
            torch.save(data, tmp_path)
            os.rename(tmp_path, path)
        except Exception:
            logger.warning(f"Failed to save cache item {idx}")
            if os.path.isfile(tmp_path):
                os.unlink(tmp_path)

    def clear(self) -> None:
        if not os.path.isdir(self.cache_dir):
            return
        logger.info(f"Clearing cache at {self.cache_dir}")
        for fname in os.listdir(self.cache_dir):
            fpath = os.path.join(self.cache_dir, fname)
            if os.path.isfile(fpath):
                os.unlink(fpath)
