"""africa-vc — fine-tune Seed-VC on African synthetic speech, and run it.

    africa-vc prepare --out data/zephyr --voices Zephyr
    africa-vc train   --dataset-dir data/zephyr --run-name zephyr
    africa-vc convert --source in.wav --target ref.wav --run zephyr
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


def cmd_convert(args) -> int:
    from .infer import convert, convert_folder
    from .seedvc import ensure
    run_dir = Path(args.checkpoint).parent if args.checkpoint else \
        ensure() / "runs" / args.run
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "ft_model.pth"
    config = Path(args.config) if args.config else next(run_dir.glob("*.yml"))
    if not checkpoint.exists():
        print(f"No checkpoint at {checkpoint}", file=sys.stderr)
        return 1
    source = Path(args.source)
    kwargs = dict(diffusion_steps=args.diffusion_steps,
                  length_adjust=args.length_adjust,
                  inference_cfg_rate=args.inference_cfg_rate,
                  f0_condition=args.f0_condition)
    if source.is_dir():
        n = convert_folder(source, Path(args.target), Path(args.output),
                           checkpoint, config, **kwargs)
        print(f"\n{n} clips -> {args.output}")
    else:
        convert(source, Path(args.target), Path(args.output), checkpoint,
                config, **kwargs)
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
    cv.add_argument("--target", required=True, help="Reference clip: the voice")
    cv.add_argument("--output", default="out")
    cv.add_argument("--run", help="Run name under the Seed-VC cache")
    cv.add_argument("--checkpoint", help="Explicit ft_model.pth instead of --run")
    cv.add_argument("--config", help="Explicit config yml instead of --run")
    cv.add_argument("--diffusion-steps", type=int, default=50)
    cv.add_argument("--length-adjust", type=float, default=1.0)
    cv.add_argument("--inference-cfg-rate", type=float, default=0.7)
    cv.add_argument("--f0-condition", action="store_true")
    cv.set_defaults(func=cmd_convert)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
