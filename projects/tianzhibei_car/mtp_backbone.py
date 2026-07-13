"""Register the external MTP ViT+RVSA backbone with MMRotate.

Keeping path discovery in a normal Python module avoids using ``import``
statements inside legacy-style MMEngine config files, which would otherwise
make MMEngine interpret them as lazy-import configs.
"""

import importlib.util
import sys
from pathlib import Path


MODULE_NAME = '_tianzhibei_external_mtp_vit_rvsa'
DEFAULT_ROOT = Path(__file__).resolve().parents[3] / 'tianzhibei-inference'


def _load_external_module():
    if MODULE_NAME in sys.modules:
        return sys.modules[MODULE_NAME]

    root = DEFAULT_ROOT
    source = root / 'local_modules' / 'vit_rvsa_mtp_branches.py'
    if not source.is_file():
        raise FileNotFoundError(
            f'MTP backbone module not found: {source}. Keep the mmrotate and '
            'tianzhibei-inference repositories next to each other.')

    spec = importlib.util.spec_from_file_location(MODULE_NAME, source)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot create an import spec for {source}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


_external_module = _load_external_module()
RVSA_MTP_branches = _external_module.RVSA_MTP_branches

__all__ = ['RVSA_MTP_branches']
