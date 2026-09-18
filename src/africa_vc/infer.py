"""Run a trained checkpoint over audio.

Voice conversion takes two inputs: the *source*, whose words survive, and the
*target*, whose voice is borrowed. It does not synthesise speech and cannot
change the language being spoken — to get Twi out, the source must already be
Twi. Fine-tuning on this dataset teaches timbre and exposes the model to
African phonology; it does not teach it to speak a language.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from .seedvc import ensure

log = logging.getLogger(__name__)

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus")


def convert(source: Path, target: Path, out_dir: Path, checkpoint: Path,
            config: Path, diffusion_steps: int = 50,
            length_adjust: float = 1.0, inference_cfg_rate: float = 0.7,
            f0_condition: bool = False) -> Path:
    """Convert one clip. Returns the directory Seed-VC wrote into."""
    root = ensure()
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "inference.py",
           "--source", str(source.resolve()),
           "--target", str(target.resolve()),
           "--output", str(out_dir.resolve()),
           "--checkpoint", str(checkpoint.resolve()),
           "--config", str(config.resolve()),
           "--diffusion-steps", str(diffusion_steps),
           "--length-adjust", str(length_adjust),
           "--inference-cfg-rate", str(inference_cfg_rate)]
    if f0_condition:
        cmd.append("--f0-condition")
    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(root), check=True)
    return out_dir


def convert_folder(source_dir: Path, target: Path, out_dir: Path,
                   checkpoint: Path, config: Path, **kwargs) -> int:
    """Convert every clip in a folder, one Seed-VC call each.

    Seed-VC loads its models per invocation, so this is slower than a batched
    engine; it is here because a folder is what people actually have. For bulk
    work over a whole dataset, ghana-vc already does it with the models held
    open.
    """
    clips = sorted(p for p in source_dir.iterdir()
                   if p.suffix.lower() in AUDIO_SUFFIXES)
    for i, clip in enumerate(clips, 1):
        convert(clip, target, out_dir, checkpoint, config, **kwargs)
        log.info("  [%d/%d] %s", i, len(clips), clip.name)
    return len(clips)
