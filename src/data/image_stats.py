"""Per-image statistics for EDA: size, colour mode, integrity and duplicate detection."""
import hashlib
import io

import imagehash
import numpy as np
import pandas as pd
from datasets import Image as ImageFeature
from PIL import Image
from tqdm import tqdm


def collect_image_stats(split, split_name: str) -> pd.DataFrame:
    """One row per image: file size and hash, width/height/mode, perceptual hash, and whether it decodes."""
    # decode=False gives the raw file bytes, so a broken file raises in our try/except
    # instead of crashing inside the datasets library.
    raw = split.select_columns(["image", "label", "file_name"]).cast_column("image", ImageFeature(decode=False))
    rows = []
    for index, record in enumerate(tqdm(raw, desc=f"scanning {split_name} images")):
        data = record["image"]["bytes"]
        if data is None:
            with open(record["image"]["path"], "rb") as f:
                data = f.read()
        row = {
            "split": split_name,
            "index": index,
            "file_name": record["file_name"],
            "label": record["label"],
            "file_kb": len(data) / 1024,
            "md5": hashlib.md5(data).hexdigest(),
        }
        try:
            with Image.open(io.BytesIO(data)) as img:
                img.verify()  # checks the file structure without decoding pixels
            with Image.open(io.BytesIO(data)) as img:
                img.load()  # full decode: catches truncated pixel data
                row.update(
                    width=img.width,
                    height=img.height,
                    mode=img.mode,
                    format=img.format,
                    phash=str(imagehash.phash(img)),
                    dhash=str(imagehash.dhash(img)),
                    corrupted=False,
                )
        except Exception as err:
            row.update(corrupted=True, error=repr(err))
        rows.append(row)

    stats = pd.DataFrame(rows)
    stats["aspect_ratio"] = stats["width"] / stats["height"]
    return stats


def _hash_array(hex_hashes: pd.Series) -> np.ndarray:
    return np.array([int(h, 16) for h in hex_hashes], dtype=np.uint64)


def near_duplicate_pairs(stats: pd.DataFrame, max_phash: int = 8, max_dhash: int = 10) -> pd.DataFrame:
    """Pairs of images that are the same photo (possibly re-encoded, recoloured or re-cropped).

    Both 64-bit hashes must be close: pHash alone also matches different birds on plain
    sky/snow backgrounds, and dHash (based on edges) rejects those. Thresholds were picked
    by looking at the candidate pairs. All splits are compared together, so a pair with
    split_a != split_b is train/test leakage.
    """
    valid = stats.dropna(subset=["phash"]).reset_index(drop=True)
    phashes, dhashes = _hash_array(valid["phash"]), _hash_array(valid["dhash"])

    pairs = []
    for i in range(len(valid) - 1):
        phash_dist = np.bitwise_count(phashes[i] ^ phashes[i + 1:])
        dhash_dist = np.bitwise_count(dhashes[i] ^ dhashes[i + 1:])
        for offset in np.flatnonzero((phash_dist <= max_phash) & (dhash_dist <= max_dhash)):
            a, b = valid.loc[i], valid.loc[i + 1 + offset]
            pairs.append({
                "split_a": a["split"], "index_a": a["index"], "file_a": a["file_name"], "label_a": a["label"],
                "split_b": b["split"], "index_b": b["index"], "file_b": b["file_name"], "label_b": b["label"],
                "phash_distance": int(phash_dist[offset]),
                "dhash_distance": int(dhash_dist[offset]),
                "identical_file": a["md5"] == b["md5"],
            })

    columns = ["split_a", "index_a", "file_a", "label_a", "split_b", "index_b", "file_b", "label_b",
               "phash_distance", "dhash_distance", "identical_file"]
    pairs = pd.DataFrame(pairs, columns=columns)
    pairs["kind"] = pairs["split_a"] + "-" + pairs["split_b"]
    pairs["same_label"] = pairs["label_a"] == pairs["label_b"]
    return pairs.sort_values(["kind", "phash_distance"], ignore_index=True)
