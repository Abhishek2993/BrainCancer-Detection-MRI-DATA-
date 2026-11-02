"""Data audit and leakage-aware train/val/test split (the Phase 0 notebook logic as a library).

Exact duplicates (MD5) and near-duplicates (64-bit dHash within ``max_bits`` across all 8
flips/rotations) are merged into groups. Groups that span two classes are dropped as label
conflicts. The remaining groups are split, stratified by class, so no group crosses splits.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from .data import IMG_EXTS, dihedral_hashes, popcount64


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[rj] = ri


def audit(dataset_path) -> pd.DataFrame:
    """One row per readable image: path, class, label, mode, size, md5 and dihedral hashes."""
    dataset_path = Path(dataset_path)
    classes = sorted(d.name for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith("."))
    records = []
    for label, cls in enumerate(classes):
        for f in sorted((dataset_path / cls).iterdir()):
            if not f.name.lower().endswith(IMG_EXTS):
                continue
            raw = f.read_bytes()
            try:
                with Image.open(io.BytesIO(raw)) as im:
                    mode, size = im.mode, im.size
                    hashes = dihedral_hashes(im.convert("L"))
            except Exception:
                continue
            records.append(dict(path=str(f), cls=cls, label=label, mode=mode, width=size[0], height=size[1],
                                md5=hashlib.md5(raw).hexdigest(), hashes=hashes))
    return pd.DataFrame(records)


def group_duplicates(df: pd.DataFrame, max_bits: int = 4, chunk: int = 64) -> tuple[pd.DataFrame, dict]:
    """Adds a ``group`` column. Returns (df, stats)."""
    n = len(df)
    uf = _UnionFind(n)
    for _, idx in df.groupby("md5").indices.items():
        for j in idx[1:]:
            uf.union(idx[0], j)
    orig = np.array([h[0] for h in df["hashes"]], dtype=np.uint64)
    allh = np.array(df["hashes"].tolist(), dtype=np.uint64)
    pairs = set()
    for start in range(0, n, chunk):
        dist = popcount64(np.bitwise_xor(orig[start:start + chunk, None, None], allh[None])).min(axis=2)
        for r, j in zip(*np.nonzero(dist <= max_bits)):
            i, j = start + int(r), int(j)
            if i != j:
                uf.union(i, j)
                pairs.add((min(i, j), max(i, j)))
    df = df.copy()
    df["group"] = [uf.find(i) for i in range(n)]
    n_classes = df.groupby("group")["label"].nunique()
    conflicts = set(n_classes[n_classes > 1].index)
    stats = {"images": n, "exact_duplicates": int(df.duplicated("md5").sum()), "near_duplicate_pairs": len(pairs),
             "groups": int(df["group"].nunique()), "conflict_groups": len(conflicts),
             "conflict_images": int(df["group"].isin(conflicts).sum())}
    df["conflict"] = df["group"].isin(conflicts)
    return df, stats


def grouped_split(df: pd.DataFrame, val_frac=0.15, test_frac=0.15, seed=42) -> pd.DataFrame:
    """Drops label-conflict groups and assigns whole groups to train/val/test, stratified by class."""
    clean = df[~df["conflict"]].copy()
    groups = clean.groupby("group")["label"].first()
    g_trainval, g_test = train_test_split(groups.index, test_size=test_frac, stratify=groups.values, random_state=seed)
    g_train, g_val = train_test_split(g_trainval, test_size=val_frac / (1 - test_frac),
                                      stratify=groups.loc[g_trainval].values, random_state=seed)
    split_of = {**{g: "train" for g in g_train}, **{g: "val" for g in g_val}, **{g: "test" for g in g_test}}
    clean["split"] = clean["group"].map(split_of)
    assert clean.groupby("group")["split"].nunique().max() == 1
    return clean


def make_split(dataset_path, out_csv, max_bits=4, val_frac=0.15, test_frac=0.15, seed=42):
    """Audit + group + split, written in the same format as the Phase 0 notebook's split.csv."""
    dataset_path = Path(dataset_path)
    df, stats = group_duplicates(audit(dataset_path), max_bits)
    clean = grouped_split(df, val_frac, test_frac, seed)
    out = clean[["path", "cls", "label", "group", "split"]].copy()
    out["path"] = [str(Path(p).relative_to(dataset_path)) for p in out["path"]]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    stats["split_counts"] = out["split"].value_counts().to_dict()
    return out, stats
