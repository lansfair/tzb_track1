import warnings

import numpy as np
from mmcv.transforms import BaseTransform

from mmrotate.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadGeoTiffRGB(BaseTransform):
    """Load the RGB bands of a GeoTIFF with GDAL.

    The Tianzhibei files store RGB plus an unassociated alpha band.  The
    transform validates that alpha is opaque, preserves georeferencing in the
    sample metadata, and returns BGR so MMDetection's standard
    ``bgr_to_rgb=True`` preprocessor remains correct.
    """

    def __init__(self,
                 to_float32: bool = False,
                 alpha_policy: str = 'warn') -> None:
        if alpha_policy not in ('ignore', 'warn', 'error'):
            raise ValueError('alpha_policy must be ignore, warn, or error')
        self.to_float32 = to_float32
        self.alpha_policy = alpha_policy

    def transform(self, results: dict) -> dict:
        try:
            from osgeo import gdal
        except ImportError as exc:
            raise ImportError(
                'LoadGeoTiffRGB requires GDAL Python bindings. Install a '
                'GDAL build matching the native GDAL library.') from exc

        dataset = gdal.Open(results['img_path'], gdal.GA_ReadOnly)
        if dataset is None:
            raise FileNotFoundError(results['img_path'])
        if dataset.RasterCount < 3:
            raise ValueError(
                f'{results["img_path"]} has only {dataset.RasterCount} bands')

        # Read all three color bands in one GDAL/SWIG call and convert CxHxW
        # to HxWxC. This removes two native-call round trips per sample.
        rgb = dataset.ReadAsArray(band_list=[1, 2, 3])
        if rgb is None or rgb.ndim != 3 or rgb.shape[0] != 3:
            raise ValueError(
                f'Failed to read RGB bands from {results["img_path"]}')
        rgb = np.moveaxis(rgb, 0, -1)
        if dataset.RasterCount >= 4 and self.alpha_policy != 'ignore':
            alpha = dataset.GetRasterBand(4).ReadAsArray()
            opaque = bool(alpha.min() == 255 and alpha.max() == 255)
            if not opaque:
                message = (
                    f'Non-opaque alpha band in {results["img_path"]}: '
                    f'min={alpha.min()}, max={alpha.max()}')
                if self.alpha_policy == 'error':
                    raise ValueError(message)
                warnings.warn(message)

        image = np.ascontiguousarray(rgb[..., ::-1])
        if self.to_float32:
            image = image.astype(np.float32)
        results['img'] = image
        results['img_shape'] = image.shape[:2]
        results['ori_shape'] = image.shape[:2]
        results['geo_transform'] = dataset.GetGeoTransform()
        results['projection'] = dataset.GetProjectionRef()
        dataset = None
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(to_float32={self.to_float32}, '
                f'alpha_policy={self.alpha_policy!r})')


@TRANSFORMS.register_module()
class ResizeAndPad(BaseTransform):
    """Downscale only when needed, then pad to a fixed square.

    Unlike a normal fixed-scale Resize, 600x800 and 800x600 inputs retain
    their native pixel scale. This is important for the median 18x8-pixel
    vehicles in this dataset.
    """

    def __init__(self,
                 size: int = 1024,
                 pad_val=(104, 116, 124),
                 interpolation: str = 'bilinear') -> None:
        self.size = size
        self.resize = TRANSFORMS.build(
            dict(
                type='mmdet.Resize',
                scale=(size, size),
                keep_ratio=True,
                interpolation=interpolation))
        self.pad = TRANSFORMS.build(
            dict(
                type='mmdet.Pad',
                size=(size, size),
                pad_val=dict(img=pad_val)))

    def transform(self, results: dict) -> dict:
        height, width = results['img_shape'][:2]
        if max(height, width) > self.size:
            results = self.resize(results)
        else:
            # ``mmdet.Resize`` normally creates this metadata.  Native-scale
            # images skip Resize by design, so provide the identity transform
            # expected by PackDetInputs and downstream prediction rescaling.
            results['scale_factor'] = (1.0, 1.0)
        return self.pad(results)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size})'


@TRANSFORMS.register_module()
class RandomGaussianNoise(BaseTransform):
    """Add weak Gaussian sensor noise without changing the image dtype."""

    def __init__(self, prob: float = 0.1, sigma_range=(1.0, 4.0)) -> None:
        self.prob = prob
        self.sigma_range = sigma_range

    def transform(self, results: dict) -> dict:
        if np.random.rand() >= self.prob:
            return results
        image = results['img']
        sigma = np.random.uniform(*self.sigma_range)
        noise = np.random.normal(0.0, sigma, image.shape)
        results['img'] = np.clip(
            image.astype(np.float32) + noise, 0, 255).astype(image.dtype)
        return results


@TRANSFORMS.register_module()
class RandomGaussianBlur(BaseTransform):
    """Apply a low-probability 3x3 Gaussian blur."""

    def __init__(self, prob: float = 0.05, sigma_range=(0.1, 0.8)) -> None:
        self.prob = prob
        self.sigma_range = sigma_range

    def transform(self, results: dict) -> dict:
        if np.random.rand() >= self.prob:
            return results
        import cv2
        sigma = np.random.uniform(*self.sigma_range)
        results['img'] = cv2.GaussianBlur(results['img'], (3, 3), sigma)
        return results
