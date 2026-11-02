import zipfile

import h5py
import numpy as np
from PIL import Image, ImageOps

from bcdc import figshare
from bcdc.data import cross_near_duplicates, dihedral_hashes, query_hash


def write_fake_mat(path, label, pid, image, mask):
    """Mimics MATLAB v7.3: HDF5 with a 'cjdata' group, arrays stored transposed, PID as uint16 codes."""
    with h5py.File(path, "w") as f:
        g = f.create_group("cjdata")
        g["label"] = np.array([[float(label)]])
        g["PID"] = np.array([[ord(c)] for c in pid], dtype=np.uint16).T
        g["image"] = image.T.astype(np.int16)
        g["tumorMask"] = mask.T.astype(np.uint8)
        g["tumorBorder"] = np.zeros((1, 4))


def test_read_slice_orientation_and_load(tmp_path):
    img = np.zeros((40, 60), np.int16)
    img[5:10, 40:50] = 900                         # bright, off-centre, non-square image
    mask = img > 0
    inner = tmp_path / "inner"
    inner.mkdir()
    write_fake_mat(inner / "1.mat", 2, "MR040", img, mask)
    write_fake_mat(inner / "2.mat", 3, "MR041", img, mask)
    with h5py.File(inner / "cvind.mat", "w") as f:
        f["cvind"] = np.arange(3)
    # nested zips, as on Figshare
    with zipfile.ZipFile(tmp_path / "part1.zip", "w") as z:
        for p in inner.iterdir():
            z.write(p, p.name)
    with zipfile.ZipFile(tmp_path / "outer.zip", "w") as z:
        z.write(tmp_path / "part1.zip", "part1.zip")
    root = tmp_path / "dl"
    root.mkdir()
    (root / "outer.zip").write_bytes((tmp_path / "outer.zip").read_bytes())
    figshare._extract_nested(root)

    s = figshare.read_slice(next(root.rglob("1.mat")))
    assert s["label"] == "glioma" and s["pid"] == "MR040"
    assert s["image"].shape == (40, 60) and s["mask"][7, 45] and not s["mask"][45 % 40, 7]

    meta, images, masks = figshare.load(root, size=32)
    assert list(meta["label"]) == ["glioma", "pituitary"]          # cvind.mat skipped
    assert images.shape == (2, 32, 32) and images.dtype == np.uint8 and masks.dtype == bool
    assert masks[0].any()


def test_cross_dataset_duplicates_catch_flips():
    rng = np.random.default_rng(0)
    a = Image.fromarray(rng.integers(0, 255, (80, 80), dtype=np.uint8))
    b = Image.fromarray(rng.integers(0, 255, (80, 80), dtype=np.uint8))
    ref = np.array([dihedral_hashes(a)], dtype=np.uint64)
    q = [query_hash(ImageOps.mirror(a)), query_hash(b)]
    hit = cross_near_duplicates(q, ref, max_bits=4)
    assert hit.tolist() == [True, False]


def test_thumbnail_match_finds_rotated_copy():
    from bcdc.data import best_thumbnail_match, thumbnail, thumbnail_variants
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[:120, :120]
    refs = []
    for cy, cx in [(30, 40), (80, 90), (60, 20)]:
        a = 200 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 300) + rng.normal(0, 5, (120, 120))
        refs.append(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)))
    ref_v = np.stack([thumbnail_variants(r) for r in refs])
    rotated_copy = refs[1].transpose(Image.Transpose.ROTATE_90).resize((300, 300))   # rotated + rescaled
    best, which = best_thumbnail_match(np.stack([thumbnail(rotated_copy)]), ref_v)
    assert which[0] == 1 and best[0] > 0.99
