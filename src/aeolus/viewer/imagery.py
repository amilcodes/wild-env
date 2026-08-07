"""Offline, georeferenced imagery loading for replay scenes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aeolus.viewer.config import ImageryConfig
from aeolus.viewer.model import ReplayModel


@dataclass(frozen=True)
class ImageryLayer:
    rgb: np.ndarray
    attribution: str | None
    source: Path


def _display_stretch(values: np.ndarray, gamma: float) -> np.ndarray:
    rgb = values.astype(np.float32)
    output = np.zeros_like(rgb)
    for channel in range(3):
        data = rgb[..., channel]
        finite = data[np.isfinite(data)]
        if not finite.size:
            continue
        lower, upper = np.percentile(finite, [2.0, 98.0])
        if upper <= lower:
            upper = lower + 1.0
        output[..., channel] = np.clip((data - lower) / (upper - lower), 0.0, 1.0)
    return np.power(output, 1.0 / gamma)


def load_imagery(model: ReplayModel, config: ImageryConfig) -> ImageryLayer | None:
    """Load PNG/JPEG/NPY or reproject a GeoTIFF onto the replay grid."""

    if config.path is None:
        return None
    path = Path(config.path)
    if not path.exists():
        raise FileNotFoundError(path)
    height, width = model.shape
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
    elif path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise ImportError("install aeolus-ia[render] for image backgrounds") from exc
        values = np.asarray(
            Image.open(path).convert("RGB").resize((width, height)),
            dtype=np.float32,
        )
    else:
        try:
            import rasterio
            from rasterio.enums import Resampling
            from rasterio.transform import Affine
            from rasterio.warp import reproject
        except ImportError as exc:  # pragma: no cover
            raise ImportError("install aeolus-ia[geo] for GeoTIFF imagery") from exc
        with rasterio.open(path) as source:
            if max(config.bands) > source.count:
                raise ValueError(f"imagery requests band {max(config.bands)}, source has {source.count}")
            source_values = source.read(config.bands).astype(np.float32)
            transform = model.spatial_reference.get("transform")
            crs = model.spatial_reference.get("crs")
            if transform and crs and source.crs:
                destination = np.zeros((3, height, width), dtype=np.float32)
                for channel in range(3):
                    reproject(
                        source=source_values[channel],
                        destination=destination[channel],
                        src_transform=source.transform,
                        src_crs=source.crs,
                        dst_transform=Affine(*transform),
                        dst_crs=crs,
                        resampling=Resampling.bilinear,
                    )
                values = np.moveaxis(destination, 0, -1)
            elif source.height == height and source.width == width:
                values = np.moveaxis(source_values, 0, -1)
            else:
                raise ValueError("GeoTIFF and replay require CRS/transform metadata or identical dimensions")
    values = np.asarray(values)
    if values.shape[:2] != (height, width) or values.ndim != 3 or values.shape[2] < 3:
        raise ValueError(f"imagery shape {values.shape} does not match replay grid {(height, width, 3)}")
    return ImageryLayer(
        rgb=_display_stretch(values[..., :3], config.gamma),
        attribution=config.attribution,
        source=path.resolve(),
    )
