from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .infer import package_submission, run_inference, verify_submission
from .prepare import prepare_archive
from .train import train_baseline

DEFAULT_CONFIG = "configs/baseline.toml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Street Falcon Vehicle ReID baseline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="проверить и распаковать датасет")
    prepare.add_argument("--archive", required=True)
    prepare.add_argument("--data-dir", required=True)
    prepare.add_argument("--config", default=DEFAULT_CONFIG)

    train = subparsers.add_parser("train", help="обучить baseline и подобрать порог")
    train.add_argument("--data-dir", required=True)
    train.add_argument("--run-dir", required=True)
    train.add_argument("--config", default=DEFAULT_CONFIG)

    infer = subparsers.add_parser("infer", help="сформировать submission")
    infer.add_argument("--data-dir", required=True)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--output-dir", required=True)
    infer.add_argument("--config", default=DEFAULT_CONFIG)

    verify = subparsers.add_parser("verify", help="проверить и упаковать submission")
    verify.add_argument("--data-dir", required=True)
    verify.add_argument("--output-dir", required=True)

    run = subparsers.add_parser("run", help="выполнить полный baseline-конвейер")
    run.add_argument("--archive", required=True)
    run.add_argument("--data-dir", required=True)
    run.add_argument("--run-dir", required=True)
    run.add_argument("--config", default=DEFAULT_CONFIG)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        config = load_config(args.config)
        report = prepare_archive(
            args.archive,
            args.data_dir,
            expected_sha256=str(config["data"]["archive_sha256"]),
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.command == "train":
        config = load_config(args.config)
        checkpoint = train_baseline(
            args.data_dir,
            args.run_dir,
            config,
            config_path=args.config,
        )
        print(checkpoint)
        return

    if args.command == "infer":
        config = load_config(args.config)
        output = run_inference(args.data_dir, args.checkpoint, args.output_dir, config)
        report = verify_submission(
            args.data_dir,
            output,
            top_k=int(config["inference"]["top_k"]),
        )
        archive = package_submission(output)
        print(json.dumps({"verification": report, "archive": str(archive)}, ensure_ascii=False))
        return

    if args.command == "verify":
        report = verify_submission(args.data_dir, args.output_dir)
        archive = package_submission(args.output_dir)
        print(json.dumps({"verification": report, "archive": str(archive)}, ensure_ascii=False))
        return

    if args.command == "run":
        config = load_config(args.config)
        prepare_archive(
            args.archive,
            args.data_dir,
            expected_sha256=str(config["data"]["archive_sha256"]),
        )
        checkpoint = train_baseline(
            args.data_dir,
            args.run_dir,
            config,
            config_path=args.config,
        )
        output = Path(args.run_dir) / "submission"
        run_inference(args.data_dir, checkpoint, output, config)
        report = verify_submission(
            args.data_dir,
            output,
            top_k=int(config["inference"]["top_k"]),
        )
        archive = package_submission(output)
        print(json.dumps({"verification": report, "archive": str(archive)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
