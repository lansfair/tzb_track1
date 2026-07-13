import math
import random
from pathlib import Path

from mmengine.dist import get_dist_info, sync_random_seed
from torch.utils.data import Sampler

from mmrotate.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class TianzhibeiBalancedSampler(Sampler):
    """Mix ordinary, rare-class, and hard/dense images in each epoch.

    ``hard_ids_file`` may be replaced between epochs by an external hard
    example miner. If it is absent, dense images serve as the hard pool.
    """

    def __init__(self,
                 dataset,
                 ordinary_fraction: float = 0.5,
                 rare_fraction: float = 0.3,
                 hard_fraction: float = 0.2,
                 rare_labels=(5, 6, 7, 8, 9),
                 dense_threshold: int = 118,
                 hard_ids_file: str = None,
                 phase_schedule=None,
                 seed: int = None,
                 round_up: bool = True) -> None:
        fractions = ordinary_fraction + rare_fraction + hard_fraction
        if abs(fractions - 1.0) > 1e-6:
            raise ValueError('sampling fractions must sum to one')
        self.dataset = dataset
        self.ordinary_fraction = ordinary_fraction
        self.rare_fraction = rare_fraction
        self.hard_fraction = hard_fraction
        self.rare_labels = set(rare_labels)
        self.dense_threshold = dense_threshold
        self.hard_ids_file = Path(hard_ids_file) if hard_ids_file else None
        self.phase_schedule = sorted(
            phase_schedule or [], key=lambda item: item['begin_epoch'])
        self.seed = sync_random_seed() if seed is None else seed
        self.epoch = 0
        self.rank, self.world_size = get_dist_info()
        self.round_up = round_up
        self.size = len(dataset)
        self.num_samples = (math.ceil(self.size / self.world_size)
                            if round_up else
                            math.ceil((self.size - self.rank) / self.world_size))
        self.total_size = self.num_samples * self.world_size

        self.id_to_index = {}
        self.rare_pool = []
        self.dense_pool = []
        for index in range(self.size):
            info = dataset.get_data_info(index)
            self.id_to_index[str(info['img_id'])] = index
            labels = {
                item['bbox_label'] for item in info['instances']
                if not item.get('ignore_flag', 0)
            }
            count = sum(
                not item.get('ignore_flag', 0) for item in info['instances'])
            if labels & self.rare_labels:
                self.rare_pool.append(index)
            if count >= dense_threshold:
                self.dense_pool.append(index)

    @staticmethod
    def _draw(pool, count, generator, replacement=True):
        if not pool:
            return []
        if not replacement and count <= len(pool):
            return generator.sample(pool, count)
        return [pool[generator.randrange(len(pool))] for _ in range(count)]

    def _phase(self):
        phase = dict(
            ordinary_fraction=self.ordinary_fraction,
            rare_fraction=self.rare_fraction,
            hard_fraction=self.hard_fraction,
            hard_ids_file=self.hard_ids_file)
        for candidate in self.phase_schedule:
            if self.epoch >= candidate['begin_epoch']:
                phase.update(candidate)
        return phase

    def _hard_pool(self, hard_ids_file):
        path = Path(hard_ids_file) if hard_ids_file else None
        if path and path.is_file():
            ids = path.read_text(
                encoding='utf-8-sig').splitlines()
            pool = [self.id_to_index[item.strip()] for item in ids
                    if item.strip() in self.id_to_index]
            if pool:
                return pool
        return self.dense_pool

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch)
        phase = self._phase()
        rare_count = round(self.size * phase['rare_fraction'])
        hard_count = round(self.size * phase['hard_fraction'])
        ordinary_count = self.size - rare_count - hard_count
        indices = self._draw(
            list(range(self.size)), ordinary_count, generator,
            replacement=False)
        indices += self._draw(self.rare_pool, rare_count, generator)
        indices += self._draw(
            self._hard_pool(phase.get('hard_ids_file')), hard_count,
            generator)
        generator.shuffle(indices)
        if self.round_up and len(indices) < self.total_size:
            indices += indices[:self.total_size - len(indices)]
        return iter(indices[self.rank:self.total_size:self.world_size])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
