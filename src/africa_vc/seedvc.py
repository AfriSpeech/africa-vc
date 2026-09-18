"""Fetch Seed-VC and run its trainer.

Seed-VC is a repository, not a package: training means calling its `train.py`
in its own working directory. This clones it once into the cache, installs its
requirements, and shells out — the alternative is vendoring a copy that goes
stale.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .config import SEEDVC_COMMIT, SEEDVC_REPO, cache_root

log = logging.getLogger(__name__)


def ensure(install_deps: bool = True) -> Path:
    dest = cache_root() / "seed-vc"
    if not (dest / "train.py").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("Cloning Seed-VC into %s", dest)
        subprocess.run(["git", "clone", SEEDVC_REPO, str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", SEEDVC_COMMIT], check=True)
        if install_deps:
            # Seed-VC owns the torch stack. Installing our own build alongside
            # it leaves a mismatched CUDA runtime and torchaudio fails with
            # "libcudart.so.12: cannot open shared object file".
            log.info("Installing Seed-VC requirements (this owns torch)")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                            str(dest / "requirements.txt")], check=True)
    return dest


def train(dataset_dir: Path, run_name: str, config: str, batch_size: int,
          max_steps: int, save_every: int, num_workers: int,
          max_epochs: Optional[int] = None, extra: Optional[List[str]] = None) -> Path:
    root = ensure()
    cmd = [sys.executable, "train.py",
           "--config", config,
           "--dataset-dir", str(dataset_dir.resolve()),
           "--run-name", run_name,
           "--batch-size", str(batch_size),
           "--max-steps", str(max_steps),
           "--save-every", str(save_every),
           "--num-workers", str(num_workers)]
    if max_epochs is not None:
        cmd += ["--max-epochs", str(max_epochs)]
    cmd += list(extra or [])
    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(root), check=True)
    return root / "runs" / run_name
