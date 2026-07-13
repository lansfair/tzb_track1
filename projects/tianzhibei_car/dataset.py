import os.path as osp
import xml.etree.ElementTree as ET
from typing import List

import cv2
import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import list_from_file

from mmrotate.registry import DATASETS


@DATASETS.register_module()
class TianzhibeiCarDataset(BaseDataset):
    """FAIR1M-style XML dataset for the Tianzhibei vehicle track.

    ``ann_file`` is a text file containing one image id per line. Images and
    XML files are resolved from ``data_prefix.img_path`` and
    ``data_prefix.ann_path`` respectively.

    Degenerate quadrilaterals are dropped before they reach qbox-to-rbox
    conversion or Gaussian regression losses.
    """

    METAINFO = dict(
        classes=(
            'Small Car', 'Van', 'Dump Truck', 'Cargo Truck',
            'other-vehicle', 'Bus', 'Truck Tractor', 'Excavator',
            'Trailer', 'Tractor'),
        palette=[
            (59, 105, 106), (246, 0, 122), (119, 0, 170), (153, 69, 1),
            (178, 90, 62), (147, 186, 208), (128, 76, 255), (0, 165, 120),
            (95, 54, 80), (196, 172, 0)
        ])

    def __init__(self,
                 img_suffix: str = '.tif',
                 min_box_area: float = 1.0,
                 min_box_side: float = 1.0,
                 drop_invalid: bool = True,
                 boundary_mode: str = 'keep',
                 min_visible_ratio: float = 0.5,
                 backend_args: dict = None,
                 **kwargs) -> None:
        if boundary_mode not in ('keep', 'refit', 'ignore'):
            raise ValueError('boundary_mode must be keep, refit, or ignore')
        self.img_suffix = img_suffix
        self.min_box_area = min_box_area
        self.min_box_side = min_box_side
        self.drop_invalid = drop_invalid
        self.boundary_mode = boundary_mode
        self.min_visible_ratio = min_visible_ratio
        self.backend_args = backend_args
        super().__init__(**kwargs)

    @property
    def ann_path(self) -> str:
        return self.data_prefix.get('ann_path', '')

    @staticmethod
    def _geometry(points: np.ndarray) -> tuple:
        shifted = np.roll(points, -1, axis=0)
        area = 0.5 * abs(np.sum(points[:, 0] * shifted[:, 1] -
                                shifted[:, 0] * points[:, 1]))
        edges = np.linalg.norm(shifted - points, axis=1)
        side_a = 0.5 * (edges[0] + edges[2])
        side_b = 0.5 * (edges[1] + edges[3])
        return float(area), float(min(side_a, side_b))

    def _handle_boundary(self, points: np.ndarray, width: int,
                         height: int):
        inside = ((points[:, 0] >= 0) & (points[:, 0] <= width - 1) &
                  (points[:, 1] >= 0) & (points[:, 1] <= height - 1)).all()
        if inside or self.boundary_mode == 'keep':
            return points, False
        if self.boundary_mode == 'ignore':
            return points, True

        image_polygon = np.asarray(
            [[0, 0], [width - 1, 0], [width - 1, height - 1],
             [0, height - 1]], dtype=np.float32)
        source = cv2.convexHull(points).reshape(-1, 2)
        intersection_area, intersection = cv2.intersectConvexConvex(
            source, image_polygon)
        source_area = abs(cv2.contourArea(source))
        visible_ratio = intersection_area / max(source_area, 1e-6)
        if (intersection is None or len(intersection) < 3 or
                visible_ratio < self.min_visible_ratio):
            return points, True
        rectangle = cv2.minAreaRect(intersection.reshape(-1, 2))
        return cv2.boxPoints(rectangle).astype(np.float32), False

    def load_data_list(self) -> List[dict]:
        class_to_label = {
            name: idx for idx, name in enumerate(self.metainfo['classes'])
        }
        image_ids = [
            line.strip().lstrip('\ufeff') for line in list_from_file(
                self.ann_file, backend_args=self.backend_args)
            if line.strip().lstrip('\ufeff')
        ]
        data_list = []
        for image_id in image_ids:
            image_id = osp.splitext(osp.basename(image_id))[0]
            xml_path = osp.join(self.ann_path, f'{image_id}.xml')
            root = ET.parse(xml_path).getroot()
            width = int(float(root.findtext('size/width', '0')))
            height = int(float(root.findtext('size/height', '0')))

            instances = []
            for obj in root.findall('objects/object'):
                class_name = (obj.findtext('possibleresult/name', '') or '').strip()
                if class_name not in class_to_label:
                    continue
                point_nodes = obj.findall('points/point')[:4]
                if len(point_nodes) != 4:
                    continue
                points = []
                valid = True
                for node in point_nodes:
                    try:
                        x, y = (node.text or '').split(',')[:2]
                        points.append((float(x), float(y)))
                    except (TypeError, ValueError):
                        valid = False
                        break
                if not valid:
                    continue
                points = np.asarray(points, dtype=np.float32)
                if not np.isfinite(points).all() or len(np.unique(points, axis=0)) < 4:
                    valid = False
                area, short_side = self._geometry(points) if valid else (0.0, 0.0)
                invalid_geometry = (
                    not valid or area <= self.min_box_area or
                    short_side < self.min_box_side)
                if invalid_geometry and self.drop_invalid:
                    continue
                boundary_ignored = False
                if not invalid_geometry:
                    points, boundary_ignored = self._handle_boundary(
                        points, width, height)
                if boundary_ignored and self.drop_invalid:
                    continue
                instances.append(
                    dict(
                        bbox=points.reshape(-1).tolist(),
                        bbox_label=class_to_label[class_name],
                        ignore_flag=int(invalid_geometry or boundary_ignored)))

            data_list.append(
                dict(
                    img_id=image_id,
                    file_name=f'{image_id}{self.img_suffix}',
                    img_path=osp.join(self.data_prefix['img_path'],
                                      f'{image_id}{self.img_suffix}'),
                    xml_path=xml_path,
                    width=width,
                    height=height,
                    instances=instances))
        return data_list

    def filter_data(self) -> List[dict]:
        if self.test_mode:
            return self.data_list
        filter_empty = bool(
            self.filter_cfg.get('filter_empty_gt', False)) \
            if self.filter_cfg else False
        if not filter_empty:
            return self.data_list
        return [item for item in self.data_list if item['instances']]

    def get_cat_ids(self, idx: int) -> List[int]:
        return [
            instance['bbox_label']
            for instance in self.get_data_info(idx)['instances']
            if not instance.get('ignore_flag', 0)
        ]
