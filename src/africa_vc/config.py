"""Where Seed-VC comes from, and what this project trains on by default."""
from __future__ import annotations

import os
from pathlib import Path

# Seed-VC is pinned to a commit rather than a branch: its configs and training
# flags have moved before, and an unpinned clone turns a reproducible run into
# whatever upstream looked like that morning. This is the commit ghana-vc was
# built and measured against.
SEEDVC_REPO = "https://github.com/Plachtaa/seed-vc.git"
SEEDVC_COMMIT = "51383efd921027683c89e5348211d93ff12ac2a8"

DATASET = "AfriSpeech/multivoice-synthetic-speech"

# V1 offline VC. The tiny xlsr preset is for real-time and sounds it; the 44k
# f0 preset is for singing. This is the one to fine-tune for speech.
DEFAULT_CONFIG = "configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml"

# Seed-VC ignores anything outside 1-30 s, so clips it would silently drop are
# never written in the first place — the count you prepare is the count it
# trains on.
MIN_SECONDS = 1.0
MAX_SECONDS = 30.0


def cache_root() -> Path:
    root = os.environ.get("AFRICA_VC_HOME")
    return Path(root) if root else Path.home() / ".cache" / "africa-vc"
