"""africa-vc — fine-tune Seed-VC on African synthetic speech, and run it."""
from .config import DATASET, DEFAULT_CONFIG, SEEDVC_COMMIT, SEEDVC_REPO

__version__ = "0.1.0"
__all__ = ["DATASET", "DEFAULT_CONFIG", "SEEDVC_COMMIT", "SEEDVC_REPO",
           "prepare", "train", "convert"]


def prepare(*args, **kwargs):
    from .prepare import prepare as _p
    return _p(*args, **kwargs)


def train(*args, **kwargs):
    from .seedvc import train as _t
    return _t(*args, **kwargs)


def convert(*args, **kwargs):
    from .infer import convert as _c
    return _c(*args, **kwargs)
