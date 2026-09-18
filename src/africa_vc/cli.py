"""africa-vc — fine-tune Seed-VC on African synthetic speech, and run it.

    africa-vc prepare --out data/zephyr --voices Zephyr
    africa-vc train   --dataset-dir data/zephyr --run-name zephyr
    africa-vc convert --source in.wav --voice Sulafat --run multivoice
    africa-vc voices
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import config as cfg


def _split(value):
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def cmd_prepare(args) -> int:
    from .prepare import prepare
    n = prepare(Path(args.out), voices=_split(args.voices),
                languages=_split(args.languages), limit=args.limit,
                dataset=args.dataset, split=args.split)
    if not n:
        print("No clips matched. Check --voices / --languages against the "
              "dataset's own columns.", file=sys.stderr)
        return 1
    print(f"\n{n} clips in {args.out}\nNext: africa-vc train --dataset-dir {args.out} "
          f"--run-name <name>")
    return 0


def cmd_train(args) -> int:
    from .seedvc import train
    run = train(Path(args.dataset_dir), args.run_name, args.config,
                args.batch_size, args.max_steps, args.save_every,
                args.num_workers, args.max_epochs)
    print(f"\nCheckpoint: {run}/ft_model.pth")
    return 0


def cmd_voices(args) -> int:
    from .voices import VOICES, DEFAULT_REFERENCE_LANGUAGE
    print(f"{len(VOICES)} voices. Any of these can be the target of a "
          f"conversion:\n")
    for name, character in VOICES.items():
        print(f"  {name:<16} {character}")
    print(f"\n  africa-vc convert --source in.wav --voice Sulafat --run <name>",
          file=sys.stderr)
    print(f"  Reference clips come from the dataset, language "
          f"{DEFAULT_REFERENCE_LANGUAGE} by default (--voice-language).",
          file=sys.stderr)
    return 0


def cmd_convert(args) -> int:
    from .infer import convert, convert_folder

    # Argument checks come before anything that touches Seed-VC. Resolving a
    # run name clones the repo and installs its torch stack, which is minutes
    # of work to then report a missing flag.
    if args.voice:
        from .voices import reference
        target = reference(args.voice, args.voice_language)
    elif args.target:
        target = Path(args.target)
    else:
        print("Give a voice to convert to: --voice <name> (see `africa-vc "
              "voices`) or --target <reference.wav>", file=sys.stderr)
        return 1
    if not args.run and not args.checkpoint:
        print("Give a checkpoint: --run <name> or --checkpoint <ft_model.pth>",
              file=sys.stderr)
        return 1
    source = Path(args.source)
    if not source.exists():
        print(f"No such source: {source}", file=sys.stderr)
        return 1

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        run_dir = checkpoint.parent
    else:
        from .seedvc import ensure
        run_dir = ensure() / "runs" / args.run
        checkpoint = run_dir / "ft_model.pth"
    if not checkpoint.exists():
        print(f"No checkpoint at {checkpoint}", file=sys.stderr)
        return 1
    if args.config:
        config = Path(args.config)
    else:
        configs = sorted(run_dir.glob("*.yml"))
        if not configs:
            print(f"No config .yml beside {checkpoint}; pass --config",
                  file=sys.stderr)
            return 1
        config = configs[0]

    kwargs = dict(diffusion_steps=args.diffusion_steps,
                  length_adjust=args.length_adjust,
                  inference_cfg_rate=args.inference_cfg_rate,
                  f0_condition=args.f0_condition)
    if source.is_dir():
        n = convert_folder(source, target, Path(args.output), checkpoint,
                           config, **kwargs)
        print(f"\n{n} clips -> {args.output}")
    else:
        convert(source, target, Path(args.output), checkpoint, config, **kwargs)
        print(f"\nConverted -> {args.output}")
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(prog="africa-vc", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("prepare", help="Pull clips from the Hub into a training folder")
    pre.add_argument("--out", required=True)
    pre.add_argument("--voices", help="Comma-separated, e.g. Zephyr,Puck")
    pre.add_argument("--languages", help="Comma-separated ISO codes, e.g. twi,ewe")
    pre.add_argument("--limit", type=int)
    pre.add_argument("--dataset", default=cfg.DATASET)
    pre.add_argument("--split", default="train")
    pre.set_defaults(func=cmd_prepare)

    tr = sub.add_parser("train", help="Fine-tune Seed-VC on a prepared folder")
    tr.add_argument("--dataset-dir", required=True)
    tr.add_argument("--run-name", required=True)
    tr.add_argument("--config", default=cfg.DEFAULT_CONFIG)
    tr.add_argument("--batch-size", type=int, default=2)
    tr.add_argument("--max-steps", type=int, default=1000)
    tr.add_argument("--max-epochs", type=int)
    tr.add_argument("--save-every", type=int, default=500)
    tr.add_argument("--num-workers", type=int, default=0)
    tr.set_defaults(func=cmd_train)

    cv = sub.add_parser("convert", help="Convert audio with a trained checkpoint")
    cv.add_argument("--source", required=True, help="File or folder: the words")
    cv.add_argument("--voice", help="Voice from the bank, e.g. Sulafat")
    cv.add_argument("--voice-language", default=None,
                    help="Which language's clip stands in for the voice")
    cv.add_argument("--target", help="Your own reference clip instead of --voice")
    cv.add_argument("--output", default="out")
    cv.add_argument("--run", help="Run name under the Seed-VC cache")
    cv.add_argument("--checkpoint", help="Explicit ft_model.pth instead of --run")
    cv.add_argument("--config", help="Explicit config yml instead of --run")
    cv.add_argument("--diffusion-steps", type=int, default=50)
    cv.add_argument("--length-adjust", type=float, default=1.0)
    cv.add_argument("--inference-cfg-rate", type=float, default=0.7)
    cv.add_argument("--f0-condition", action="store_true")
    cv.set_defaults(func=cmd_convert)

    vo = sub.add_parser("voices", help="List the voices a clip can be converted to")
    vo.set_defaults(func=cmd_voices)

    args = p.parse_args(argv)
    if getattr(args, "voice_language", None) is None and hasattr(args, "voice"):
        from .voices import DEFAULT_REFERENCE_LANGUAGE
        args.voice_language = DEFAULT_REFERENCE_LANGUAGE
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
