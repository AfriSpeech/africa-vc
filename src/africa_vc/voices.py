"""The voice bank: which voices a converted clip can be given.

A single-speaker checkpoint carries its voice inside it — ghana-vc converts to
one Twi speaker and there is nothing to choose. Training on all 30 makes the
voice a *runtime* argument instead: the reference clip selects it, so all 30
stay reachable from one checkpoint.

The clips are not vendored. Each is fetched from the dataset on first use and
cached, the same way Seed-VC itself is: 30 WAVs is 10 MB of git history for
files that already live on the Hub.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .config import DATASET, cache_root

log = logging.getLogger(__name__)

# name -> the characteristic Google documents for it
VOICES: Dict[str, str] = {
    'Zephyr': 'Bright',
    'Puck': 'Upbeat',
    'Charon': 'Informative',
    'Kore': 'Firm',
    'Fenrir': 'Excitable',
    'Leda': 'Youthful',
    'Orus': 'Firm',
    'Aoede': 'Breezy',
    'Callirrhoe': 'Easy-going',
    'Autonoe': 'Bright',
    'Enceladus': 'Breathy',
    'Iapetus': 'Clear',
    'Umbriel': 'Easy-going',
    'Algieba': 'Smooth',
    'Despina': 'Smooth',
    'Erinome': 'Clear',
    'Algenib': 'Gravelly',
    'Rasalgethi': 'Informative',
    'Laomedeia': 'Upbeat',
    'Achernar': 'Soft',
    'Alnilam': 'Firm',
    'Schedar': 'Even',
    'Gacrux': 'Mature',
    'Pulcherrima': 'Forward',
    'Achird': 'Friendly',
    'Zubenelgenubi': 'Casual',
    'Vindemiatrix': 'Gentle',
    'Sadachbia': 'Lively',
    'Sadaltager': 'Knowledgeable',
    'Sulafat': 'Warm'
}

# Which language's clip stands in for a voice. Timbre is language-independent,
# but a reference clip still carries prosody, so the default is a widely spoken
# language the synthesiser handles well rather than an arbitrary one.
DEFAULT_REFERENCE_LANGUAGE = "swh"


def names() -> List[str]:
    return list(VOICES)


def describe(voice: str) -> str:
    return VOICES.get(voice, "")


def resolve(voice: str) -> str:
    """Accept a voice name case-insensitively, or say what was meant instead."""
    for name in VOICES:
        if name.lower() == voice.lower():
            return name
    raise SystemExit(
        f"Unknown voice {voice!r}. Run `africa-vc voices` for the {len(VOICES)} "
        f"this dataset covers."
    )


def reference(voice: str, language: str = DEFAULT_REFERENCE_LANGUAGE,
              dataset: str = DATASET, token: Optional[str] = None) -> Path:
    """The reference clip for a voice, downloaded and cached on first use."""
    from huggingface_hub import hf_hub_download

    voice = resolve(voice)
    cache = cache_root() / "voices"
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / f"{language}_{voice}.wav"
    if not local.exists():
        log.info("Fetching reference clip for %s (%s)", voice, language)
        path = hf_hub_download(dataset, f"audio/{language}/{voice}.wav",
                               repo_type="dataset", token=token)
        local.write_bytes(Path(path).read_bytes())
    return local
