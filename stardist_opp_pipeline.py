"""
CZI / TIF -> StarDist OPP -> BiofilmQ Label Mask Pipeline
==========================================================
Batch-processes .czi or .tif microscopy files through StarDist OPP segmentation
and exports label masks formatted for import into BiofilmQ.

NEW IN THIS VERSION: automatic z-anisotropy correction.
StarDist3D bakes a fixed z:xy anisotropy ratio into its ray geometry at
training time (see config.json -> "anisotropy"). If the true voxel spacing
of an acquisition doesn't match that ratio, segmentations come out blocky
and under-resolved in z (boxy XZ/YZ views, poorly separated touching cells).
This version reads the true voxel spacing from the CZI/TIF metadata,
resamples the z axis to match what the model was trained on, segments,
then resamples the label mask back down to the original z-slice count
before saving -- so outputs still line up with your raw data in BiofilmQ.

Requirements (install before running):
    pip install aicsimageio[czi] merge-stardist-masks stardist tensorflow-cpu tifffile numpy scikit-image

Usage:
    python stardist_opp_pipeline.py
    (The script will prompt you for all required paths and settings)
"""

import math
import sys
from pathlib import Path

import numpy as np
import tifffile

# ── CZI reading ───────────────────────────────────────────────────────────────
try:
    from aicsimageio import AICSImage
except ImportError:
    print("ERROR: aicsimageio is not installed. Run: pip install aicsimageio[czi]")
    sys.exit(1)

# ── StarDist ──────────────────────────────────────────────────────────────────
try:
    from stardist.models import StarDist3D
    from csbdeep.utils import normalize
except ImportError:
    print("ERROR: stardist is not installed. Run: pip install stardist tensorflow")
    sys.exit(1)

# ── StarDist OPP (merge-stardist-masks) ──────────────────────────────────────
try:
    from merge_stardist_masks.naive_fusion import naive_fusion
except ImportError:
    print("ERROR: merge-stardist-masks is not installed. Run: pip install merge-stardist-masks")
    sys.exit(1)

# ── Resampling (for anisotropy correction) ───────────────────────────────────
try:
    from skimage.transform import resize
except ImportError:
    print("ERROR: scikit-image is not installed. Run: pip install scikit-image")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers - image loading
# ─────────────────────────────────────────────────────────────────────────────

def load_czi(path: Path, channel: int = 0) -> np.ndarray:
    """Load a single channel from a .czi file and return a ZYX numpy array."""
    img = AICSImage(str(path))
    # dims: TCZYX  — grab first time-point, specified channel
    data = img.get_image_data("ZYX", T=0, C=channel)
    return data.astype(np.float32)


def load_tif(path: Path, channel: int = 0) -> np.ndarray:
    """
    Load a .tif file and return a ZYX numpy array.

    Handles a range of common TIF layouts:
      - ZYX  (3-D stack, single channel)        — used as-is
      - CZYX (multi-channel 3-D stack)           — channel axis sliced
      - TZYX (time-series 3-D stack)             — first time-point taken
      - YX   (single 2-D slice)                  — expanded to 1×Y×X
      - TCZYX / other 5-D layouts                — T=0, specified channel
    """
    raw = tifffile.imread(str(path))
    ndim = raw.ndim

    if ndim == 2:
        # Single 2-D slice → treat as a 1-slice Z-stack
        data = raw[np.newaxis, ...]          # 1 × Y × X

    elif ndim == 3:
        # Most common TIF from microscopes: ZYX stack, single channel
        data = raw                           # Z × Y × X

    elif ndim == 4:
        # Could be CZYX or TZYX — assume CZYX (channel-first) and slice channel
        if raw.shape[0] <= 4:               # heuristic: small first dim → channel
            data = raw[channel]              # Z × Y × X
        else:                               # large first dim → time
            data = raw[0]                   # first time-point, Z × Y × X

    elif ndim == 5:
        # TCZYX — take first time-point and specified channel
        data = raw[0, channel]               # Z × Y × X

    else:
        raise ValueError(
            f"Unexpected TIF dimensionality: {raw.shape}. "
            "Expected 2-D, 3-D (ZYX), 4-D (CZYX/TZYX), or 5-D (TCZYX)."
        )

    return data.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers - voxel size / anisotropy correction  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def get_czi_voxel_size(path: Path) -> tuple | None:
    """
    Return (z_um, y_um, x_um) physical voxel size for a CZI file, or None
    if the metadata doesn't contain physical pixel sizes.
    """
    try:
        img = AICSImage(str(path))
        pz = img.physical_pixel_sizes
        if pz.Z is None or pz.X is None:
            return None
        y_um = pz.Y if pz.Y is not None else pz.X
        return (float(pz.Z), float(y_um), float(pz.X))
    except Exception:
        return None


def get_tif_voxel_size(path: Path) -> tuple | None:
    """
    Best-effort extraction of (z_um, y_um, x_um) physical voxel size from a
    TIF file's ImageJ metadata (z spacing) and TIFF resolution tags (x/y).
    Returns None if either piece can't be determined -- TIFs frequently
    lack this metadata, in which case set MANUAL_Z_VOXEL_SIZE /
    MANUAL_XY_VOXEL_SIZE below instead.

    IMPORTANT: for ImageJ-written TIFFs, XResolution/YResolution are
    pixels-per-unit where "unit" is whatever ImageJ recorded in its own
    metadata (imagej_metadata['unit'], e.g. 'micron') -- NOT necessarily
    inches/cm per the standard TIFF ResolutionUnit tag. Treating an
    ImageJ file's resolution tags as inch-based (the previous bug here)
    silently produces voxel sizes off by orders of magnitude. We check
    imagej_metadata['unit'] first and only fall back to the standard
    inch/cm TIFF convention for non-ImageJ files.
    """
    try:
        with tifffile.TiffFile(str(path)) as tif:
            z_um = None
            xy_um = None

            ij_meta = tif.imagej_metadata or {}
            unit = str(ij_meta.get("unit", "")).strip().lower()
            micron_aliases = ("um", "micron", "microns", "\u00b5m", "micrometer", "micrometre")

            # Z spacing: ImageJ hyperstack metadata
            if "spacing" in ij_meta:
                z_um = float(ij_meta["spacing"])
                if unit == "nm":
                    z_um /= 1000.0
                elif unit == "mm":
                    z_um *= 1000.0
                elif unit not in micron_aliases and unit not in ("",):
                    # Unrecognized unit — don't guess
                    z_um = None

            # X/Y spacing
            page = tif.pages[0]
            x_res_tag = page.tags.get("XResolution")
            res_unit_tag = page.tags.get("ResolutionUnit")

            def _tag_to_float(tag):
                if tag is None:
                    return None
                val = tag.value
                if isinstance(val, tuple) and len(val) == 2 and val[1] != 0:
                    return val[0] / val[1]
                return float(val)

            x_val = _tag_to_float(x_res_tag)

            if x_val:
                if unit in micron_aliases:
                    # ImageJ convention: XResolution is pixels-per-unit,
                    # and that unit is directly imagej_metadata['unit'].
                    px_per_um = x_val
                elif unit == "nm":
                    px_per_um = x_val * 1000.0   # pixels per nm -> pixels per um
                elif unit == "mm":
                    px_per_um = x_val / 1000.0   # pixels per mm -> pixels per um
                elif unit in ("", "pixel", "pixels", "inch"):
                    # Standard (non-ImageJ) TIFF: use the ResolutionUnit tag.
                    # 2 = inch (default), 3 = centimeter
                    unit_code = res_unit_tag.value if res_unit_tag is not None else 2
                    if unit_code == 3:
                        px_per_um = x_val / 10000.0   # pixels per cm -> pixels per um
                    else:
                        px_per_um = x_val / 25400.0   # pixels per inch -> pixels per um
                else:
                    px_per_um = None

                if px_per_um and px_per_um > 0:
                    xy_um = 1.0 / px_per_um

            if z_um is None or xy_um is None:
                return None
            return (z_um, xy_um, xy_um)
    except Exception:
        return None


def compute_resample_factor(
    z_voxel_um: float,
    xy_voxel_um: float,
    model_anisotropy_z: float,
    threshold: float = 0.05,
    min_sane_factor: float = 0.1,
    max_sane_factor: float = 10.0,
) -> float | None:
    """
    Compare the data's true z:xy anisotropy to what the StarDist model was
    trained on, and return the factor by which the z axis should be
    resampled (via interpolation) so the two match.

    factor > 1  -> upsample z (add slices, data is coarser in z than training)
    factor < 1  -> downsample z (data is finer in z than training)
    Returns None if the mismatch is within `threshold` (no correction needed).

    SAFETY CHECK: real microscopy anisotropy mismatches are typically well
    under 10x in either direction. A factor outside [min_sane_factor,
    max_sane_factor] almost always means the voxel size metadata was
    misread (e.g. a bad unit-conversion bug) rather than a genuine
    acquisition difference. In that case we refuse to resample -- silently
    collapsing or exploding a z-stack based on bad metadata is far worse
    than skipping the correction for that file.
    """
    if z_voxel_um <= 0 or xy_voxel_um <= 0 or model_anisotropy_z <= 0:
        return None
    actual_anisotropy = z_voxel_um / xy_voxel_um
    factor = actual_anisotropy / model_anisotropy_z
    if abs(factor - 1.0) < threshold:
        return None
    if not (min_sane_factor <= factor <= max_sane_factor):
        print(f"   WARNING: computed resample factor ({factor:.6f}) is outside the "
              f"plausible range [{min_sane_factor}, {max_sane_factor}] — this almost "
              f"always means the voxel size metadata was misread, not a real "
              f"anisotropy mismatch. Skipping resampling for this file rather than "
              f"risk corrupting the z-stack. Check the printed voxel size above, "
              f"or set MANUAL_Z_VOXEL_SIZE / MANUAL_XY_VOXEL_SIZE.")
        return None
    return factor


def resample_z(img: np.ndarray, factor: float) -> np.ndarray:
    """Resample a ZYX image along z only, by `factor` (via linear interpolation)."""
    orig_z = img.shape[0]
    new_z = max(1, round(orig_z * factor))
    resized = resize(
        img,
        (new_z,) + img.shape[1:],
        order=1,                       # linear — appropriate for intensity data
        mode="reflect",
        preserve_range=True,
        anti_aliasing=(factor < 1),    # only needed when downsampling
    )
    return resized.astype(np.float32)


def resample_labels_to_shape(labels: np.ndarray, target_z: int) -> np.ndarray:
    """Resample a ZYX label mask back to `target_z` slices using nearest-neighbor
    interpolation (order=0), so label IDs are never blended/invented."""
    if labels.shape[0] == target_z:
        return labels
    resized = resize(
        labels,
        (target_z,) + labels.shape[1:],
        order=0,                       # nearest-neighbor — required for label masks
        mode="edge",
        preserve_range=True,
        anti_aliasing=False,
    )
    return resized.astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers - segmentation
# ─────────────────────────────────────────────────────────────────────────────

def pad_to_even(img: np.ndarray, grid: tuple) -> tuple[np.ndarray, tuple]:
    """
    Pad a ZYX image so every dimension is divisible by the corresponding grid
    factor. StarDist downsamples by `grid` internally; if the image size is not
    a multiple of grid the upsampled output is 1 px larger than the input,
    causing a shape mismatch between the raw image and the label mask.

    Returns (padded_image, original_shape) so the caller can crop back later.
    """
    original_shape = img.shape
    pad_width = []
    for size, g in zip(img.shape, grid):
        remainder = size % g
        pad = (g - remainder) % g   # 0 if already divisible
        pad_width.append((0, pad))
    if any(p[1] > 0 for p in pad_width):
        img = np.pad(img, pad_width, mode="reflect")
    return img, original_shape


def crop_to_shape(arr: np.ndarray, shape: tuple) -> np.ndarray:
    """Crop array back to `shape` along every axis (removes padding)."""
    slices = tuple(slice(0, s) for s in shape)
    return arr[slices]


def compute_tiles(img_shape: tuple, base_tiles: tuple) -> tuple:
    """
    Adjust the tile count so that each tile is at least 16 voxels in every
    dimension. This prevents excessively small tiles when images are cropped
    to a variety of sizes, while still capping memory use for large images.

    img_shape : (Z, Y, X)
    base_tiles : (tz, ty, tx) from the user / default config
    returns   : (tz, ty, tx) safe tile counts
    """
    MIN_TILE_SIZE = 16
    safe = []
    for size, tiles in zip(img_shape, base_tiles):
        max_tiles = max(1, size // MIN_TILE_SIZE)
        safe.append(min(tiles, max_tiles))
    return tuple(safe)


def normalize_image(img: np.ndarray, pmin: float = 1.0, pmax: float = 99.8) -> np.ndarray:
    """Percentile-based normalisation as recommended by StarDist."""
    return normalize(img, pmin, pmax, axis=None)


def run_stardist_opp(
    model: StarDist3D,
    img_norm: np.ndarray,
    prob_thresh: float = None,
    nms_thresh: float = None,
    max_full_overlaps: int = 2,
    n_tiles: tuple = (4, 4, 4),
    small_volume_voxel_threshold: int = 2_000_000,
    use_opp_fusion: bool = True,
) -> np.ndarray:
    """
    Run StarDist prediction followed by the OPP (Optimised Post-Processing)
    mask-merging step from merge-stardist-masks.

    If prob_thresh or nms_thresh are None, the model's own optimised thresholds
    from thresholds.json are used automatically.

    n_tiles splits the volume into a grid of overlapping tiles so the full
    image never needs to fit in RAM at once. Tile counts are clamped per-image
    to avoid tiles smaller than 16 voxels (important for variable-size crops).
    If a tile size still causes OOM, the fallback automatically doubles tile count.

    SIZE-AWARE TILING: tiling exists purely to keep large volumes within
    memory; on a small crop the tile seams can cut straight through cells,
    causing them to be truncated, suppressed, or dropped near tile
    boundaries. If the (padded) volume has fewer than
    `small_volume_voxel_threshold` voxels, tiling is skipped entirely
    (n_tiles=(1,1,1)) regardless of the configured `n_tiles`, since such a
    volume comfortably fits in memory as a single tile.

    Returns a 3-D integer label array (ZYX).
    """
    # Resolve thresholds — fall back to model's own optimised values if not set
    if prob_thresh is None:
        prob_thresh = model.thresholds.prob
    if nms_thresh is None:
        nms_thresh = model.thresholds.nms
    print(f"         Using prob_thresh={prob_thresh:.4f}, nms_thresh={nms_thresh:.4f}")

    # Grid from model config — needed by naive_fusion to map predictions back
    # to image coordinates (this model uses grid=(1,2,2))
    grid = tuple(model.config.grid)

    # Pad to grid-even dimensions to prevent the label mask growing 1 px
    # larger than the raw image when Y or X is odd (grid=(1,2,2) issue).
    original_shape = img_norm.shape
    img_norm, _ = pad_to_even(img_norm, grid)
    if img_norm.shape != original_shape:
        print(f"         Padded {original_shape} → {img_norm.shape} to match grid {grid}")

    # Skip tiling entirely for small volumes -- tile seams can cut through
    # cells, truncating or dropping them near boundaries. Only fall back to
    # the configured n_tiles (clamped to image size) for volumes large
    # enough that a single tile risks running out of memory.
    total_voxels = int(np.prod(img_norm.shape))
    if total_voxels <= small_volume_voxel_threshold:
        safe_base = (1, 1, 1)
        print(f"         Volume has {total_voxels:,} voxels (<= {small_volume_voxel_threshold:,} "
              f"threshold) — skipping tiling (n_tiles=(1,1,1)) to avoid seams cutting through cells.")
    else:
        safe_base = compute_tiles(img_norm.shape, n_tiles)

    # Tiling strategies to try in order if OOM occurs
    tile_schedule = [
        safe_base,
        compute_tiles(img_norm.shape, (safe_base[0] * 2, safe_base[1] * 2, safe_base[2] * 2)),
        compute_tiles(img_norm.shape, (safe_base[0] * 4, safe_base[1] * 4, safe_base[2] * 4)),
    ]

    for tiles in tile_schedule:
        try:
            print(f"         Predicting with n_tiles={tiles} ...")

            # 1. Raw StarDist prediction with return_predict=True so we get the
            #    raw distance and probability arrays needed by the OPP fusion step.
            (labels_raw, polys), (probs_raw, dists_raw) = model.predict_instances(
                img_norm,
                prob_thresh=prob_thresh,
                nms_thresh=nms_thresh,
                n_tiles=tiles,
                return_predict=True,
            )
            break  # success — exit the retry loop

        except Exception as e:
            if "OOM" in str(e) or "out of memory" in str(e).lower() or "RESOURCE_EXHAUSTED" in str(e):
                print(f"         OOM with n_tiles={tiles} — retrying with more tiles...")
            else:
                raise  # non-OOM error — re-raise immediately
    else:
        raise RuntimeError("OOM persisted across all tile strategies — image may be too large for available RAM.")

    # 2. StarDist OPP fusion step (v0.2.x functional API)
    #    naive_fusion() merges overlapping StarDist masks whose centres fall
    #    inside a neighbouring mask — intended to handle non-star-convex
    #    objects like elongated or curved bacteria, where one physical cell
    #    gets split into multiple star-convex proposals. On straighter cells
    #    it can instead wrongly merge separate touching cells (it can't tell
    #    "one bent cell, two proposals" from "two straight cells touching"),
    #    so this is a per-dataset choice, not a universal improvement.
    if use_opp_fusion:
        print("         Running OPP fusion...")
        labels_final = naive_fusion(
            dists=dists_raw,
            probs=probs_raw,
            rays=model.rays,
            prob_thresh=prob_thresh,
            grid=grid,
            max_full_overlaps=max_full_overlaps,
        )
    else:
        print("         OPP fusion disabled (use_opp_fusion=False) — using raw StarDist labels.")
        labels_final = labels_raw

    # Crop back to the original (pre-padding) shape
    labels_final = crop_to_shape(labels_final, original_shape)

    return labels_final.astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers - large-volume blockwise processing
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY THIS EXISTS: run_stardist_opp() calls predict_instances(..., return_predict=True)
# to get the raw dists/probs arrays that OPP fusion needs. This forces StarDist
# to materialize ONE DENSE ARRAY holding all n_rays distances for EVERY voxel in
# the whole (padded) volume -- regardless of n_tiles, which only controls the
# internal prediction computation, not the size of this final returned array.
# For small crops this is trivial (megabytes). For a full ~1163x1163xZ frame it
# can be 10-20+ GiB, which reliably OOMs on a normal workstation even though the
# tiled prediction step itself completes fine.
#
# THE FIX: split the volume into overlapping XY blocks small enough that each
# block's dense array fits comfortably in memory, run full detection+OPP on
# each block independently (reusing run_stardist_opp), then keep only the
# detections whose centroid falls inside that block's non-overlapping "core"
# region and stitch them into one seamless full-size output. This mirrors how
# StarDist's own internal tiling stitches label images -- we just have to do
# it ourselves at this level because return_predict=True bypasses it.
# Overlap must exceed the largest expected cell diameter so no real cell's
# centroid can be ambiguous about which block's core it belongs to.

def estimate_dense_array_gb(shape_zyx: tuple, grid: tuple, n_rays: int) -> float:
    """Estimate the size (in GiB) of the dense dists array predict_instances
    would need to return for a volume of this shape -- the thing that OOMs on
    large frames. Used to decide whether blockwise processing is needed."""
    z, y, x = shape_zyx
    gz, gy, gx = grid
    elems = math.ceil(z / gz) * math.ceil(y / gy) * math.ceil(x / gx) * n_rays
    return elems * 4 / (1024 ** 3)  # float32 = 4 bytes


def compute_block_core_size(
    z_size: int,
    grid: tuple,
    n_rays: int,
    max_dense_array_gb: float,
    overlap_px: int,
    min_core_size: int = 96,
) -> int:
    """
    Given the volume's z-extent (after any anisotropy resampling), work out
    how large an XY block (core region, excluding overlap halo) can be while
    keeping that block's own dense dists array under `max_dense_array_gb`.
    Assumes square blocks and grid[1] == grid[2] (true for the (1,2,2) grids
    used throughout this pipeline).
    """
    gz, gy, gx = grid
    budget_elems = max_dense_array_gb * (1024 ** 3) / 4  # float32 = 4 bytes
    z_downsampled = math.ceil(z_size / gz)
    # block_xy solves: (block_xy/gy) * (block_xy/gx) * z_downsampled * n_rays <= budget_elems
    block_xy = gy * math.sqrt(budget_elems / (z_downsampled * n_rays))
    core_size = int(block_xy) - 2 * overlap_px
    return max(min_core_size, core_size)


def run_stardist_opp_large(
    model: StarDist3D,
    img_norm: np.ndarray,
    prob_thresh: float = None,
    nms_thresh: float = None,
    max_full_overlaps: int = 2,
    n_tiles: tuple = (4, 4, 4),
    small_volume_voxel_threshold: int = 2_000_000,
    use_opp_fusion: bool = True,
    max_dense_array_gb: float = 3.0,
    overlap_px: int = 64,
) -> np.ndarray:
    """
    Segment a large ZYX volume by splitting it into overlapping XY blocks so
    the dense per-voxel distance array required by predict_instances never
    exceeds `max_dense_array_gb` for any single block. Each block is run
    through the normal run_stardist_opp() pipeline (thresholds, OPP fusion,
    small-volume tiling logic all still apply per-block), then only
    detections whose centroid falls inside that block's own non-overlapping
    "core" region are kept and written into the full-size output -- so cells
    are never double-counted or cut at a block seam, provided `overlap_px`
    exceeds the largest real cell diameter in the image (default 64 px is
    generous for typical bacterial cells at ~0.09 um/px).

    Returns a full-size ZYX int32 label array, same shape as img_norm.
    """
    grid = tuple(model.config.grid)
    n_rays = model.config.n_rays
    Z, Y, X = img_norm.shape

    core_size = compute_block_core_size(
        Z, grid, n_rays, max_dense_array_gb, overlap_px
    )
    n_blocks_y = math.ceil(Y / core_size)
    n_blocks_x = math.ceil(X / core_size)
    print(f"         Large volume: splitting into {n_blocks_y}x{n_blocks_x} blocks "
          f"(core size {core_size}px, overlap {overlap_px}px) to stay under "
          f"{max_dense_array_gb:.1f} GiB per block.")

    output = np.zeros((Z, Y, X), dtype=np.int32)
    next_label_id = 0
    block_num = 0
    total_blocks = n_blocks_y * n_blocks_x

    for y0 in range(0, Y, core_size):
        y1 = min(y0 + core_size, Y)
        for x0 in range(0, X, core_size):
            x1 = min(x0 + core_size, X)
            block_num += 1

            # Extract block with overlap halo, clipped to image bounds
            by0, by1 = max(0, y0 - overlap_px), min(Y, y1 + overlap_px)
            bx0, bx1 = max(0, x0 - overlap_px), min(X, x1 + overlap_px)
            block = img_norm[:, by0:by1, bx0:bx1]

            print(f"         Block {block_num}/{total_blocks}: "
                  f"y[{y0}:{y1}] x[{x0}:{x1}] (extracted with halo: "
                  f"y[{by0}:{by1}] x[{bx0}:{bx1}], shape {block.shape})")

            block_labels = run_stardist_opp(
                model, block,
                prob_thresh=prob_thresh,
                nms_thresh=nms_thresh,
                max_full_overlaps=max_full_overlaps,
                n_tiles=n_tiles,
                small_volume_voxel_threshold=small_volume_voxel_threshold,
                use_opp_fusion=use_opp_fusion,
            )

            # Core boundaries in this block's own local coordinate frame
            core_y0_local, core_y1_local = y0 - by0, y1 - by0
            core_x0_local, core_x1_local = x0 - bx0, x1 - bx0

            kept_this_block = 0
            for lbl_id in np.unique(block_labels):
                if lbl_id == 0:
                    continue
                mask = block_labels == lbl_id
                ys, xs = np.where(mask.any(axis=0))  # collapse z for centroid in y,x
                cy, cx = ys.mean(), xs.mean()
                if (core_y0_local <= cy < core_y1_local) and (core_x0_local <= cx < core_x1_local):
                    next_label_id += 1
                    kept_this_block += 1
                    # Write this cell's full mask (not clipped to core) into
                    # the global output at this block's offset, so cells that
                    # extend into the halo are not truncated.
                    global_slice = output[:, by0:by1, bx0:bx1]
                    global_slice[mask] = next_label_id

            print(f"         Block {block_num}/{total_blocks}: kept {kept_this_block} "
                  f"cells (centroid inside core region)")

    print(f"         Large-volume stitching complete: {next_label_id} total cells")
    return output


def save_tiff(data: np.ndarray, out_path: Path) -> None:
    """Save a numpy array as a uint16 ImageJ-compatible TIFF stack."""
    tifffile.imwrite(
        str(out_path),
        data.astype(np.uint16),
        imagej=True,          # embeds metadata BiofilmQ/Fiji can read
        photometric="minisblack",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_batch(
    input_dir: Path,
    output_dir: Path,
    model: StarDist3D,
    file_type: str,               # "czi" or "tif"
    segmentation_channels: list,
    label_channel_name: str,
    prob_thresh: float,
    nms_thresh: float,
    max_full_overlaps: int,
    norm_pmin: float,
    norm_pmax: float,
    n_tiles: tuple,
    resample_threshold: float,
    small_volume_voxel_threshold: int,
    use_opp_fusion: bool,
    manual_z_voxel_um: float,
    manual_xy_voxel_um: float,
    max_dense_array_gb: float = 3.0,
    block_overlap_px: int = 64,
) -> None:

    # Collect files of the chosen type
    glob_pattern = "*.czi" if file_type == "czi" else "*.tif"
    input_files = sorted(input_dir.glob(glob_pattern))

    # Also catch .tiff extension
    if file_type == "tif":
        input_files = sorted(
            list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
        )

    if not input_files:
        print(f"No .{file_type} files found in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(input_files)} .{file_type} file(s) → saving to subfolders in {output_dir}\n")

    # Model's trained z:xy anisotropy ratio — this is the target we resample to.
    model_anisotropy = tuple(model.config.anisotropy) if model.config.anisotropy else (1.0, 1.0, 1.0)
    model_anisotropy_z = model_anisotropy[0]
    print(f"Model trained anisotropy (z,y,x): {model_anisotropy}\n")

    for i, file_path in enumerate(input_files, 1):
        stem = file_path.stem
        print(f"[{i}/{len(input_files)}] Processing: {file_path.name}")

        # ── Determine true voxel size for anisotropy correction ────────────
        if file_type == "czi":
            voxel_size = get_czi_voxel_size(file_path)
        else:
            voxel_size = get_tif_voxel_size(file_path)

        if voxel_size is None:
            # Fall back to manually configured values if metadata is missing
            if manual_z_voxel_um is not None and manual_xy_voxel_um is not None:
                voxel_size = (manual_z_voxel_um, manual_xy_voxel_um, manual_xy_voxel_um)
                print("   (voxel size not found in metadata — using MANUAL_Z/XY_VOXEL_SIZE config)")
            else:
                print("   WARNING: could not determine voxel size from metadata, and no "
                      "MANUAL_Z_VOXEL_SIZE/MANUAL_XY_VOXEL_SIZE fallback is set. "
                      "Skipping anisotropy correction for this file — segmentation "
                      "quality may be reduced if z-spacing doesn't match the model's "
                      "training anisotropy.")

        resample_factor = None
        if voxel_size is not None:
            z_um, y_um, x_um = voxel_size
            xy_um = (y_um + x_um) / 2.0
            actual_anisotropy = z_um / xy_um
            print(f"   Voxel size (z,y,x) µm: ({z_um:.4f}, {y_um:.4f}, {x_um:.4f})  "
                  f"| actual anisotropy: {actual_anisotropy:.3f}  "
                  f"| model anisotropy: {model_anisotropy_z:.3f}")
            resample_factor = compute_resample_factor(
                z_um, xy_um, model_anisotropy_z, threshold=resample_threshold
            )
            if resample_factor is not None:
                direction = "upsampling" if resample_factor > 1 else "downsampling"
                print(f"   Anisotropy mismatch detected — {direction} z by factor "
                      f"{resample_factor:.3f} before segmentation.")
            else:
                print("   Anisotropy matches model training within tolerance — no resampling needed.")

        for channel in segmentation_channels:
            channel_stem = f"{stem}_channel{channel}"
            print(f"   Channel {channel}:")

            try:
                # 1. Load raw image (kept untouched — this is what gets saved as ch1)
                if file_type == "czi":
                    img_raw = load_czi(file_path, channel=channel)
                else:
                    img_raw = load_tif(file_path, channel=channel)

                print(f"         Raw shape: {img_raw.shape}  dtype: {img_raw.dtype}")

                # 2. Resample z (on a copy) to match the model's trained anisotropy
                orig_z = img_raw.shape[0]
                if resample_factor is not None:
                    img_for_seg = resample_z(img_raw, resample_factor)
                    print(f"         Resampled for segmentation: {img_raw.shape} → {img_for_seg.shape}")
                else:
                    img_for_seg = img_raw

                # 3. Normalise
                img_norm = normalize_image(img_for_seg, pmin=norm_pmin, pmax=norm_pmax)

                # 4. StarDist OPP segmentation
                # predict_instances(return_predict=True) always returns a
                # DENSE per-voxel distance array for the WHOLE volume,
                # regardless of n_tiles -- that array (not the tiled
                # prediction itself) is what OOMs on large full frames.
                # Estimate its size up front and route to blockwise
                # processing automatically if it would exceed the budget.
                grid = tuple(model.config.grid)
                n_rays = model.config.n_rays
                # pad_to_even may grow the shape slightly; estimate on the
                # padded size for an accurate (slightly conservative) check
                padded_shape = tuple(
                    s + ((g - s % g) % g) for s, g in zip(img_norm.shape, grid)
                )
                dense_gb = estimate_dense_array_gb(padded_shape, grid, n_rays)
                if dense_gb > max_dense_array_gb:
                    print(f"         Estimated dense array size ~{dense_gb:.1f} GiB exceeds "
                          f"{max_dense_array_gb:.1f} GiB budget — using blockwise processing.")
                    labels = run_stardist_opp_large(
                        model, img_norm,
                        prob_thresh=prob_thresh,
                        nms_thresh=nms_thresh,
                        max_full_overlaps=max_full_overlaps,
                        n_tiles=n_tiles,
                        small_volume_voxel_threshold=small_volume_voxel_threshold,
                        use_opp_fusion=use_opp_fusion,
                        max_dense_array_gb=max_dense_array_gb,
                        overlap_px=block_overlap_px,
                    )
                else:
                    labels = run_stardist_opp(
                        model, img_norm,
                        prob_thresh=prob_thresh,
                        nms_thresh=nms_thresh,
                        max_full_overlaps=max_full_overlaps,
                        n_tiles=n_tiles,
                        small_volume_voxel_threshold=small_volume_voxel_threshold,
                        use_opp_fusion=use_opp_fusion,
                    )

                # 5. Resample the label mask back down to the original z-slice
                #    count so it lines up with the raw image for BiofilmQ.
                if resample_factor is not None:
                    labels = resample_labels_to_shape(labels, orig_z)
                    print(f"         Resampled labels back to original shape: {labels.shape}")

                n_cells = int(labels.max())
                print(f"         Detected {n_cells} cells")

                # 6. Create a subfolder named after the source file and channel
                subfolder = output_dir / channel_stem
                subfolder.mkdir(parents=True, exist_ok=True)

                # 7. Save raw image as ch1 (original, unresampled)
                ch1_path = subfolder / f"{channel_stem}_ch1.tif"
                save_tiff(img_raw, ch1_path)
                print(f"         Saved → {ch1_path.relative_to(output_dir)}")

                # 8. Save label mask as ch2 (resampled back to match ch1)
                ch2_path = subfolder / f"{channel_stem}_{label_channel_name}.tif"
                save_tiff(labels, ch2_path)
                print(f"         Saved → {ch2_path.relative_to(output_dir)}\n")

            except Exception as exc:
                print(f"         ERROR processing {file_path.name} channel {channel}: {exc}\n")


def load_model(model_path: Path) -> StarDist3D:
    """Load a StarDist3D model from a local directory."""
    print(f"\nLoading StarDist model from: {model_path}")
    from stardist.rays3d import rays_from_json
    model = StarDist3D(None, name=model_path.name, basedir=str(model_path.parent))

    # Reconstruct the rays object from config — in some StarDist versions
    # model.rays is not set automatically but rays_json holds all the info needed.
    if not hasattr(model, 'rays') or model.rays is None:
        model.rays = rays_from_json(model.config.rays_json)
        print(f"         Reconstructed rays: {model.config.rays_json['name']} "
              f"(n={model.config.n_rays}, anisotropy={model.config.anisotropy})")

    return model


def prompt_path(message: str, must_exist: bool = True) -> Path:
    """Prompt the user for a folder path, stripping quotes if pasted from Explorer."""
    while True:
        raw = input(message).strip().replace('"', '').replace("'", '')
        path = Path(raw)
        if must_exist and not path.exists():
            print(f"  ! Path not found: {path} — please try again.")
        else:
            return path


def prompt_file_type() -> str:
    """Ask the user which image file type to process."""
    print("Select input file type:")
    print("  [1] CZI  (.czi  — Zeiss microscopy files)")
    print("  [2] TIF  (.tif / .tiff — TIFF stacks, including variable-size crops)")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return "czi"
        elif choice == "2":
            return "tif"
        else:
            print("  ! Please enter 1 or 2.")


def prompt_channels() -> list:
    """Ask the user which channels to segment."""
    print("Enter channel indices to segment (e.g. [0]  or  [0,1]  or  [0,1,2]):")
    while True:
        raw = input("Channels: ").strip()
        cleaned = raw.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        parts = [p.strip() for p in cleaned.replace(",", " ").split() if p.strip()]
        try:
            channels = [int(p) for p in parts]
            if channels:
                return channels
        except ValueError:
            pass
        print("  ! Please enter channel indices as integers, e.g. [0] or [0, 1].")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration defaults (edit these if you want fixed values)
# ─────────────────────────────────────────────────────────────────────────────

SEGMENTATION_CHANNELS = [0]       # 0-indexed channels to segment (e.g. [0] or [0, 1])
LABEL_CHANNEL_NAME    = "ch2"
NORM_PMIN            = 1.0
NORM_PMAX            = 99.8
# Set to None to use the model's own optimised thresholds from thresholds.json
# Override with a float (e.g. 0.5) if you want a custom value
PROB_THRESH          = None       # model default: 0.4537
NMS_THRESH           = None       # model default: 0.3
MAX_FULL_OVERLAPS    = 0          # OPP: max times a cell can be fully overlapped before suppression
N_TILES              = (4, 4, 4)  # (Z, Y, X) tiling — auto-clamped per image; increase if OOM
# Volumes at or below this many voxels are segmented in a single tile
# (n_tiles=(1,1,1)) regardless of N_TILES above, since tile seams can cut
# through cells on small crops. 2,000,000 voxels comfortably covers crops
# like 100x100x20 (200,000 voxels) with headroom for the z-resampled
# version (100x100x~30). Raise this only if you hit OOM on small crops.
SMALL_VOLUME_VOXEL_THRESHOLD = 2_000_000
# Whether to run OPP fusion (merging overlapping StarDist proposals) at all.
# OPP helps when cells are curved/bent enough to get split into multiple
# star-convex proposals, but on straighter rod-shaped cells it can instead
# wrongly merge separate touching cells (see diagnose_opp_fusion.py to check
# for your dataset). This is a per-dataset decision -- validate it on a
# representative subset before committing to it for a full batch; don't
# just leave the default without checking.
USE_OPP_FUSION = True

# ── Large-volume blockwise processing settings (NEW) ─────────────────────────
# predict_instances(return_predict=True) always returns a dense per-voxel
# distance array for the WHOLE volume regardless of tiling -- this is what
# OOMs on large full frames (e.g. full 1163x1163 acquisitions), not the
# prediction step itself. If a file's estimated dense array would exceed this
# many GiB, the pipeline automatically splits it into overlapping blocks,
# segments each block independently, and stitches the results into one
# seamless full-size output. Lower this if you still hit OOM (smaller blocks,
# more of them, slower but safer); raise it if you have plenty of RAM to spare
# and want fewer, larger blocks (faster).
MAX_DENSE_ARRAY_GB = 3.0
# Overlap (in pixels) added around each block's core region so cells sitting
# near a block boundary are fully captured in at least one block and never
# double-counted or cut. Must exceed the largest real cell diameter in your
# images. Default of 64px is generous for typical bacterial cells at the
# ~0.09 um/px resolution used in this pipeline's test data -- increase if
# your cells are notably larger or your xy pixel size is coarser.
BLOCK_OVERLAP_PX = 64

# ── Anisotropy correction settings (NEW) ─────────────────────────────────────
# Fraction of mismatch between actual and trained anisotropy that's tolerated
# before resampling kicks in (0.05 = 5%).
RESAMPLE_THRESHOLD   = 0.05
# Fallback voxel sizes (in microns) used ONLY when a file's metadata doesn't
# contain physical pixel sizes (this is common for plain TIFs). Leave as None
# to skip anisotropy correction for files with missing metadata.
MANUAL_Z_VOXEL_SIZE  = None        # e.g. 0.31
MANUAL_XY_VOXEL_SIZE = None        # e.g. 0.0916


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("============================================")
    print(" StarDist OPP → BiofilmQ Segmentation Tool ")
    print("============================================\n")

    # 1. Ask for file type first
    file_type = prompt_file_type()
    print()

    # 2. Ask which channels to segment
    channels = prompt_channels()
    print()

    # 3. Paths
    input_dir = prompt_path(f"Enter folder path containing .{file_type} files: ", must_exist=True)
    output_dir = input_dir.parent / f"{input_dir.name}_stardist_opp_output"
    model_dir  = prompt_path("Enter StarDist model folder path: ", must_exist=True)

    # 4. Load model and run
    model = load_model(model_dir)

    process_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        model=model,
        file_type=file_type,
        segmentation_channels=channels,
        label_channel_name=LABEL_CHANNEL_NAME,
        prob_thresh=PROB_THRESH,
        nms_thresh=NMS_THRESH,
        max_full_overlaps=MAX_FULL_OVERLAPS,
        norm_pmin=NORM_PMIN,
        norm_pmax=NORM_PMAX,
        n_tiles=N_TILES,
        resample_threshold=RESAMPLE_THRESHOLD,
        small_volume_voxel_threshold=SMALL_VOLUME_VOXEL_THRESHOLD,
        use_opp_fusion=USE_OPP_FUSION,
        max_dense_array_gb=MAX_DENSE_ARRAY_GB,
        block_overlap_px=BLOCK_OVERLAP_PX,
        manual_z_voxel_um=MANUAL_Z_VOXEL_SIZE,
        manual_xy_voxel_um=MANUAL_XY_VOXEL_SIZE,
    )
    print("Done!\n")
    print(f"Output folder: {output_dir}\n")
