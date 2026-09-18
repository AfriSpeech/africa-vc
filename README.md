# africa-vc

**Fine-tune [Seed-VC](https://github.com/Plachtaa/seed-vc) on African synthetic
speech, and convert with the result.** Training and inference in one place.

The training data is
[`AfriSpeech/multivoice-synthetic-speech`](https://huggingface.co/datasets/AfriSpeech/multivoice-synthetic-speech):
**17,010 clips · 567 African languages · 30 voices · 38.7 hours**, one distinct
sentence per voice per language, 24 kHz mono WAV.

## What voice conversion does, and does not

Seed-VC takes two inputs: a **source**, whose words survive, and a **target**,
whose voice is borrowed. It converts *who is speaking*, never *what is said* or
*in which language*.

So this cannot make a model speak Twi. To get Twi out, the source must already
be Twi. What fine-tuning here buys you is two things:

1. **The voices.** ~1.29 h per voice across 567 languages. Timbre is
   language-independent, so every clip of a voice contributes to learning it,
   and the phonetic variety is far wider than an hour of one language.
2. **African phonology.** If the content encoder currently garbles Ghanaian or
   Ethiopian phonemes, exposure should make conversions of African speech hold
   up better. This is the part worth measuring.

**Try zero-shot first.** Seed-VC clones from a single reference clip with no
training at all. Fine-tune only once you have heard it fall short — and let
what it gets wrong decide which languages to weight.

## Install

```bash
pip install -e .
```

Seed-VC itself is cloned on first use into `~/.cache/africa-vc/seed-vc`, pinned
to a commit, and it installs its own torch stack. Do not pin torch here — see
the comment in `pyproject.toml` for what breaks when you do.

## 1 · Prepare

Seed-VC trains on a directory of audio files and nothing else: no manifest, no
speaker labels. Everything interesting about this dataset lives in columns, so
the choice of what to train on happens at extraction.

```bash
africa-vc prepare --out data/zephyr --voices Zephyr          # one voice, 567 langs
africa-vc prepare --out data/ghana  --languages twi,ewe,dag  # one region, all voices
africa-vc prepare --out data/small  --limit 500              # a quick trial
```

Streams from the Hub, so a subset does not download 13 GB to throw most away.
Clips outside 1–30 s are dropped rather than written, because Seed-VC ignores
them silently — the count on disk is the count it trains on.

## 2 · Train

```bash
africa-vc train --dataset-dir data/zephyr --run-name zephyr \
  --batch-size 2 --max-steps 1000 --save-every 500
```

Defaults to `config_dit_mel_seed_uvit_whisper_small_wavenet.yml`, the offline
speech preset. The tiny `xlsr` preset is for real-time and sounds like it; the
44 kHz `f0` preset is for singing.

The checkpoint lands at `~/.cache/africa-vc/seed-vc/runs/<run-name>/ft_model.pth`.

Seed-VC's own guidance: minimum 1 utterance per speaker and 100 steps, "the
more data you have, the better", and fine-tuning "will largely improve speaker
similarity on particular speakers, but may slightly increase WER". With content
spanning 567 languages, expect that WER cost at the higher end — worth
measuring rather than assuming.

## 3 · Convert

```bash
africa-vc voices                                              # the 30 on offer
africa-vc convert --source speech.wav --voice Sulafat --run multivoice
africa-vc convert --source clips/ --voice Kore --run multivoice --output out/
africa-vc convert --source speech.wav --target my_own.wav --run multivoice
```

`--source` is the audio whose words you keep. The voice comes either from the
**bank** (`--voice`) or from any clip you supply (`--target`).

`--diffusion-steps` defaults to 50: 25 is audibly rough, 100 buys little.

### The voice bank

A single-speaker checkpoint carries its voice inside it — ghana-vc converts to
one Twi speaker and there is nothing to choose. Training on all 30 makes the
voice a **runtime argument**: the reference clip selects it, so every voice
stays reachable from one checkpoint.

That is the reason to train **once on the full set** rather than a checkpoint
per voice. A per-voice fine-tune sharpens one timbre and loses the other 29.

Reference clips are fetched from the dataset on first use and cached under
`~/.cache/africa-vc/voices/`, not vendored — 30 WAVs is 10 MB of git history
for files already on the Hub. `--voice-language` picks which language's clip
stands in for a voice (default `swh`): timbre is language-independent, but a
reference still carries prosody.

For bulk conversion over a whole Hub dataset with the models held open, use
[ghana-vc](https://github.com/GhanaNLP/ghana-vc) instead — it loads once rather
than per clip.

## Caveats worth keeping in view

The training audio is **synthetic**. The model will learn Gemini's voices and,
where Gemini mispronounces a language, that mispronunciation as ground truth.
With 30 speakers all from one TTS family, there is also a real risk of
overfitting to synthetic timbre and generalising worse to human references.
Hold out some real speech to check against.

## Licence

GPL-3.0-or-later, following Seed-VC.
