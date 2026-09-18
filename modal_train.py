"""Fine-tune Seed-VC on Modal, detached.

    modal run --detach modal_train.py --voices Zephyr --max-steps 5000

`--detach` is the point: the run keeps going after the terminal closes or the
laptop sleeps. Without it, Modal stops the app when the client disconnects, and
a multi-hour training run dies with the SSH session.

Everything durable lives on a Volume — the prepared clips, the Seed-VC clone
and every checkpoint — so a preempted or crashed run resumes instead of
starting over, and the checkpoints survive the container.
"""
from __future__ import annotations

import os

import modal

SEEDVC_REPO = "https://github.com/Plachtaa/seed-vc.git"
SEEDVC_COMMIT = "51383efd921027683c89e5348211d93ff12ac2a8"
DATASET = "AfriSpeech/multivoice-synthetic-speech"
DEFAULT_CONFIG = "configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml"

VOL = "/vol"
app = modal.App("africa-vc-train")
volume = modal.Volume.from_name("africa-vc", create_if_missing=True)

# Seed-VC gets its own virtualenv, and this is not tidiness.
#
# Its requirements pull descript-audiotools, which pins protobuf<3.20. Modal's
# own runtime is injected into the container and needs protobuf>=5, so
# installing Seed-VC's stack into the same interpreter breaks Modal itself:
#   AttributeError: Enum VolumeFsVersion has no value defined for name 'ValueType'
# That fails at import, before any training code runs, and reads like a Modal
# bug rather than a dependency conflict. Forcing protobuf back up afterwards
# only moves the break to whichever side loses.
#
# So: Seed-VC lives in /opt/venv and is invoked as a subprocess. The Modal side
# keeps a Hub-compatible stack and the two never share an interpreter. Seed-VC
# also owns torch inside that venv — installing our own alongside leaves a
# mismatched CUDA runtime and torchaudio fails on "libcudart.so.12".
SEEDVC_PY = "/opt/venv/bin/python"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "build-essential", "libsndfile1", "python3-venv")
    .run_commands(
        f"git clone {SEEDVC_REPO} /opt/seed-vc",
        f"cd /opt/seed-vc && git checkout {SEEDVC_COMMIT}",
        "python -m venv /opt/venv",
        "/opt/venv/bin/pip install -q -U pip",
        "/opt/venv/bin/pip install -r /opt/seed-vc/requirements.txt",
    )
    # Modal-side only: streaming the dataset and writing clips. numpy stays on
    # 1.x and datasets below 3 to match what the Hub tooling expects here.
    .pip_install("datasets>=2.18,<3", "huggingface_hub>=0.28.1,<0.34",
                 "soundfile>=0.12", "numpy>=1.26,<2")
)


def _with_backoff(fn, what: str, attempts: int = 10, cap: float = 300.0):
    """Retry through the Hub's rate limiting.

    A 429 is a wait, not a failure, and it can arrive hours into a run or on
    the very first metadata call. Giving up on one throws away everything
    already downloaded, so this backs off and keeps going.
    """
    import random
    import time

    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            transient = any(m in str(exc) for m in ("429", "Too Many Requests",
                                                    "503", "504", "timed out"))
            if not transient or attempt == attempts:
                raise
            delay = min(cap, 2 ** attempt) * (0.5 + random.random())
            print(f"  {what}: {type(exc).__name__} — retry {attempt}/{attempts} "
                  f"in {delay:.0f}s", flush=True)
            time.sleep(delay)


def _from_shards(out, limit: int = 0) -> int:
    """Write every clip, reading the parquet shards rather than loose files."""
    import io
    from pathlib import Path

    import soundfile as sf
    from datasets import Audio, load_dataset

    ds = _with_backoff(
        lambda: load_dataset(DATASET, split="train", streaming=True,
                             token=os.environ["HF_TOKEN"]),
        "load_dataset")
    # decode=False hands back the stored WAV bytes untouched; decoding would
    # drag in librosa to rebuild a file we already have.
    ds = ds.cast_column("audio", Audio(decode=False))

    written = 0
    rows = iter(ds)
    while True:
        row = _with_backoff(lambda: next(rows, None), "stream")
        if row is None:
            break
        raw = row["audio"]["bytes"]
        info = sf.info(io.BytesIO(raw))
        if not (1.0 <= info.duration <= 30.0):
            continue
        name = f"{row['language']}_{row['voice']}_{written:06d}.wav"
        (Path(out) / name).write_bytes(raw)
        written += 1
        if written % 1000 == 0:
            print(f"  {written} clips", flush=True)
            volume.commit()
        if limit and written >= limit:
            break
    return written


@app.function(
    image=image,
    # No GPU: this streams rows and writes WAVs. Attaching one bills an idle
    # accelerator for the length of a download.
    volumes={VOL: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=6 * 60 * 60,
)
def prepare(voices: str = "", languages: str = "", limit: int = 0,
            name: str = "all") -> int:
    """Copy the selected clips onto the Volume. Idempotent: skips if present.

    The dataset ships the same audio twice and each packaging wins a different
    case, so this picks between them:

    * **A subset** (a voice, a few languages) comes from the loose WAVs at
      `audio/<language>/<voice>.wav`. Streaming the shards instead would pull
      ~13 GB of rows to keep 567 clips, since one voice is one row in thirty.
    * **Everything** comes from the parquet shards. Asking the Hub for 17,010
      individual files is 17,010 requests and earns a 429; the same audio is
      17 shard downloads.

    Either way the stored bytes are written verbatim — the dataset went out of
    its way not to re-encode, and decoding to re-encode would undo that.
    """
    import io
    from pathlib import Path

    import soundfile as sf
    from huggingface_hub import snapshot_download

    out = Path(VOL) / "data" / name
    done = Path(VOL) / "data" / f".{name}.done"
    if done.exists():
        n = int(done.read_text())
        print(f"{n} clips already prepared at {out}")
        return n

    want_v = [v.strip() for v in voices.split(",") if v.strip()]
    want_l = [l.strip() for l in languages.split(",") if l.strip()]
    out.mkdir(parents=True, exist_ok=True)

    if not want_v and not want_l:
        written = _from_shards(out, limit)
        done.write_text(str(written))
        volume.commit()
        print(f"{written} clips -> {out}")
        return written

    if want_v and want_l:
        patterns = [f"audio/{l}/{v}.wav" for l in want_l for v in want_v]
    elif want_v:
        patterns = [f"audio/*/{v}.wav" for v in want_v]
    elif want_l:
        patterns = [f"audio/{l}/*.wav" for l in want_l]
    else:
        patterns = ["audio/**"]
    print(f"fetching {patterns[:4]}{' …' if len(patterns) > 4 else ''}", flush=True)

    snap = _with_backoff(
        lambda: snapshot_download(DATASET, repo_type="dataset",
                                  allow_patterns=patterns,
                                  token=os.environ["HF_TOKEN"],
                                  cache_dir=f"{VOL}/hf", max_workers=4),
        "snapshot_download")

    written = skipped = 0
    for clip in sorted(Path(snap, "audio").rglob("*.wav")):
        raw = clip.read_bytes()
        # The WAV header alone gives the duration; no samples are read.
        info = sf.info(io.BytesIO(raw))
        # Seed-VC ignores anything outside 1-30 s, so not writing those keeps
        # the count on disk equal to the count it trains on.
        if not (1.0 <= info.duration <= 30.0):
            skipped += 1
            continue
        language, voice = clip.parent.name, clip.stem
        (out / f"{language}_{voice}_{written:06d}.wav").write_bytes(raw)
        written += 1
        if written % 1000 == 0:
            print(f"  {written} clips", flush=True)
            volume.commit()
        if limit and written >= limit:
            break

    done.write_text(str(written))
    volume.commit()
    print(f"{written} clips -> {out} ({skipped} outside 1-30s)")
    return written


@app.function(
    image=image,
    gpu=os.environ.get("AFRICA_VC_GPU", "A100-40GB"),
    volumes={VOL: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=24 * 60 * 60,
)
def train(name: str = "all", run_name: str = "multivoice",
          config: str = DEFAULT_CONFIG, batch_size: int = 2,
          max_steps: int = 5000, save_every: int = 500,
          num_workers: int = 4) -> str:
    """Run Seed-VC's trainer against the prepared clips on the Volume."""
    import shutil
    import subprocess
    from pathlib import Path

    data = Path(VOL) / "data" / name
    clips = len(list(data.glob("*.wav"))) if data.exists() else 0
    if not clips:
        raise SystemExit(f"No clips at {data} — run prepare first.")

    # Train inside the Volume so checkpoints survive the container, and so a
    # restarted run finds the ones already written.
    workdir = Path(VOL) / "seed-vc"
    if not (workdir / "train.py").exists():
        shutil.copytree("/opt/seed-vc", workdir, dirs_exist_ok=True)
        volume.commit()

    cmd = [SEEDVC_PY, "train.py",
           "--config", config,
           "--dataset-dir", str(data),
           "--run-name", run_name,
           "--batch-size", str(batch_size),
           "--max-steps", str(max_steps),
           "--save-every", str(save_every),
           "--num-workers", str(num_workers)]
    print(f"{clips} clips | {' '.join(cmd)}", flush=True)
    env = {**os.environ, "HF_HOME": f"{VOL}/hf"}
    proc = subprocess.run(cmd, cwd=str(workdir), env=env)
    volume.commit()
    if proc.returncode != 0:
        raise SystemExit(f"Seed-VC training exited {proc.returncode}")

    out = workdir / "runs" / run_name
    print(f"checkpoint: {out}/ft_model.pth")
    return str(out)


MODEL_CARD = """---
license: gpl-3.0
library_name: seed-vc
pipeline_tag: audio-to-audio
tags:
- voice-conversion
- seed-vc
- african-languages
- multilingual
datasets:
- AfriSpeech/multivoice-synthetic-speech
---

# africa-vc

Voice conversion for African languages, fine-tuned from
[Seed-VC](https://github.com/Plachtaa/seed-vc) on
[AfriSpeech/multivoice-synthetic-speech](https://huggingface.co/datasets/AfriSpeech/multivoice-synthetic-speech)
— 17,010 clips, **567 African languages**, **30 voices**, 38.7 hours.

## What it does

It converts **who is speaking**, not what is said. The words and the language
come from the source audio; only the voice is replaced. It cannot make a model
speak a language — to get Twi out, the source must already be Twi.

Because it was trained on all 30 voices rather than one, **the target voice is
a runtime choice**: the reference clip selects it, and every one of the 30 is
reachable from this single checkpoint.

## Use

```bash
pip install git+https://github.com/AfriSpeech/africa-vc
africa-vc voices
africa-vc convert --source speech.wav --voice Sulafat --checkpoint ft_model.pth
```

## Training

| | |
|---|---|
| Base | Seed-VC `{commit}` |
| Config | `{config}` |
| Data | {clips} clips, {languages} languages, {voices} voices |
| Steps | {steps} |
| Batch size | {batch} |
| Hardware | Modal {gpu} |

## Caveats

The training audio is **synthetic**, generated with Google Gemini. The model
learns those voices, and where the synthesiser mispronounced a language it
learns that too. With 30 speakers from one TTS family there is a real risk of
overfitting to synthetic timbre and generalising less well to human reference
clips — check against real speech before relying on it.

Quality varies enormously by language. The voices were built for widely spoken
languages and were asked to read hundreds of others.
"""


@app.function(
    image=image,
    volumes={VOL: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=2 * 60 * 60,
)
def publish(run_name: str = "multivoice", repo: str = "AfriSpeech/africa-vc",
            name: str = "all", steps: int = 0, batch: int = 0,
            gpu: str = "", config: str = DEFAULT_CONFIG) -> str:
    """Push the checkpoint, its config and a card to the Hub."""
    from pathlib import Path

    from huggingface_hub import HfApi

    run_dir = Path(VOL) / "seed-vc" / "runs" / run_name
    ckpts = sorted(run_dir.glob("*.pth"))
    if not ckpts:
        raise SystemExit(f"No checkpoint under {run_dir}")
    # ft_model.pth is the one Seed-VC's inference expects; fall back to the
    # newest step checkpoint if a run was cut short before writing it.
    final = run_dir / "ft_model.pth"
    if not final.exists():
        final = max(ckpts, key=lambda p: p.stat().st_mtime)
        print(f"no ft_model.pth; publishing {final.name}")

    data = Path(VOL) / "data" / name
    clips = len(list(data.glob("*.wav")))
    languages = len({f.name.split("_")[0] for f in data.glob("*.wav")})
    voices = len({f.name.split("_")[1] for f in data.glob("*.wav")})

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True, private=False)
    api.upload_file(path_or_fileobj=str(final), path_in_repo="ft_model.pth",
                    repo_id=repo, repo_type="model")
    cfg = Path(VOL) / "seed-vc" / config
    if cfg.exists():
        api.upload_file(path_or_fileobj=str(cfg), path_in_repo=Path(config).name,
                        repo_id=repo, repo_type="model")
    card = MODEL_CARD.format(commit=SEEDVC_COMMIT[:12], config=Path(config).name,
                             clips=f"{clips:,}", languages=languages,
                             voices=voices, steps=f"{steps:,}", batch=batch,
                             gpu=gpu or "A100-40GB")
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=repo, repo_type="model")
    url = f"https://huggingface.co/{repo}"
    print(f"published: {url}")
    return url


@app.local_entrypoint()
def main(voices: str = "", languages: str = "", limit: int = 0,
         name: str = "all", run_name: str = "multivoice",
         batch_size: int = 2, max_steps: int = 5000, save_every: int = 500,
         config: str = DEFAULT_CONFIG):
    """Prepare then train. Run with `modal run --detach` to survive the terminal."""
    n = prepare.remote(voices=voices, languages=languages, limit=limit, name=name)
    print(f"prepared {n} clips")
    out = train.remote(name=name, run_name=run_name, config=config,
                       batch_size=batch_size, max_steps=max_steps,
                       save_every=save_every)
    print(f"trained: {out}")
    url = publish.remote(run_name=run_name, name=name, steps=max_steps,
                         batch=batch_size, config=config,
                         gpu=os.environ.get("AFRICA_VC_GPU", "A100-40GB"))
    print(f"published: {url}")
