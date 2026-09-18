"""Turn the Hub dataset into the folder of clips Seed-VC trains on.

Seed-VC wants a directory of audio files and nothing else: no manifest, no
split, no speaker labels. Everything interesting about this dataset — which
voice, which language — lives in columns, so choosing what to train on happens
here, at extraction, rather than in the trainer.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .config import DATASET, MAX_SECONDS, MIN_SECONDS

log = logging.getLogger(__name__)


def _keep(row, voices: Optional[Sequence[str]], languages: Optional[Sequence[str]]) -> bool:
    if voices and row["voice"] not in voices:
        return False
    if languages and row["language"] not in languages:
        return False
    return True


def prepare(out_dir: Path, voices: Optional[Sequence[str]] = None,
            languages: Optional[Sequence[str]] = None,
            limit: Optional[int] = None, dataset: str = DATASET,
            split: str = "train", token: Optional[str] = None) -> int:
    """Write the selected clips as WAV files under `out_dir`. Returns the count."""
    import soundfile as sf
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    # Streaming: the full set is 13 GB and materialising it to pick a subset
    # would download all of it to throw most away.
    ds = load_dataset(dataset, split=split, streaming=True, token=token)

    written = skipped_len = 0
    for row in ds:
        if not _keep(row, voices, languages):
            continue
        audio = row["audio"]
        seconds = len(audio["array"]) / audio["sampling_rate"]
        if not (MIN_SECONDS <= seconds <= MAX_SECONDS):
            # Seed-VC drops these silently; not writing them keeps the count
            # on disk equal to the count it trains on.
            skipped_len += 1
            continue
        name = f"{row['language']}_{row['voice']}_{written:06d}.wav"
        sf.write(out_dir / name, audio["array"], audio["sampling_rate"],
                 subtype="PCM_16")
        written += 1
        if written % 500 == 0:
            log.info("  %d clips written", written)
        if limit and written >= limit:
            break

    log.info("%d clips -> %s (%d outside %g-%gs, skipped)",
             written, out_dir, skipped_len, MIN_SECONDS, MAX_SECONDS)
    return written
