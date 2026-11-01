"""Command line for audit, split, train, calibrate, evaluate, and the local demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .utils import ProjectError


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset directory, or its parent.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m open_breast_ultrasound.cli",
        description="Open Breast Ultrasound research prototype. Not a medical device.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Inventory images. Does not delete source files or masks.")
    _add_common(audit)
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument("--id-map", type=Path, default=None, help="Optional CSV with relative_path,patient_id,case_id.")
    audit.add_argument("--perceptual-max-distance", type=int, default=6)
    audit.add_argument(
        "--group-perceptual",
        action="store_true",
        help="Also link perceptual-hash candidates. Off by default. This is not patient identity.",
    )
    audit.add_argument("--overwrite", action="store_true")

    check = subparsers.add_parser("check-review", help="Exit with an error if any included row is unreviewed.")
    check.add_argument("--inventory", type=Path, required=True)

    split = subparsers.add_parser("split", help="Write a grouped 60/20/20-style split manifest.")
    split.add_argument("--inventory", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--train", type=float, default=0.6)
    split.add_argument("--validation", type=float, default=0.2)
    split.add_argument("--test", type=float, default=0.2)
    split.add_argument("--overwrite", action="store_true")

    train = subparsers.add_parser("train", help="Train the baseline. Requires a reviewed inventory.")
    train.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    _add_common(train)
    train.add_argument("--inventory", type=Path, required=True)
    train.add_argument("--split", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument(
        "--confirm-review",
        action="store_true",
        help="Confirm a person reviewed the inventory. reviewed=true is not a medical validation.",
    )
    train.add_argument("--overwrite", action="store_true")
    train.add_argument("--verbose", type=int, default=1)

    calibrate = subparsers.add_parser(
        "calibrate",
        help="Fit temperature on validation predictions. Does not use the test split.",
    )
    _add_common(calibrate)
    calibrate.add_argument("--experiment", type=Path, required=True)
    calibrate.add_argument("--inventory", type=Path, required=True)
    calibrate.add_argument("--split", type=Path, required=True)
    calibrate.add_argument("--overwrite", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="Score the locked model on the test split.")
    _add_common(evaluate)
    evaluate.add_argument("--experiment", type=Path, required=True)
    evaluate.add_argument("--inventory", type=Path, required=True)
    evaluate.add_argument("--split", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, default=None)
    evaluate.add_argument("--bootstrap", type=int, default=200)
    evaluate.add_argument("--bootstrap-seed", type=int, default=0)
    evaluate.add_argument("--overwrite", action="store_true")

    external = subparsers.add_parser(
        "evaluate-external",
        help="Score a separate manifest. Does not retrain, recalibrate, or change the threshold.",
    )
    _add_common(external)
    external.add_argument("--experiment", type=Path, required=True)
    external.add_argument("--manifest", type=Path, required=True)
    external.add_argument("--output-dir", type=Path, required=True)
    external.add_argument("--bootstrap", type=int, default=200)
    external.add_argument("--bootstrap-seed", type=int, default=0)
    external.add_argument("--overwrite", action="store_true")

    predict = subparsers.add_parser("predict", help="Print model scores for one image.")
    predict.add_argument("--experiment", type=Path, required=True)
    predict.add_argument("--image", type=Path, required=True)

    demo = subparsers.add_parser("demo", help="Local Gradio demo. Public sharing is off unless requested.")
    demo.add_argument("--experiment", type=Path, required=True)
    demo.add_argument("--host", default="127.0.0.1")
    demo.add_argument("--port", type=int, default=7860)
    demo.add_argument(
        "--allow-public-share",
        action="store_true",
        help="Create a public Gradio link. Off by default. Only for non-identifying research images.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = _dispatch(args)
    except ProjectError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _dispatch(args: argparse.Namespace):
    if args.command == "audit":
        from .audit import run_audit

        return run_audit(
            args.data_root,
            args.output_dir,
            id_map=args.id_map,
            perceptual_max_distance=args.perceptual_max_distance,
            group_perceptual=args.group_perceptual,
            overwrite=args.overwrite,
        )
    if args.command == "check-review":
        from .audit import assert_reviewed_for_training, included_rows, load_inventory
        from .utils import parse_bool

        rows = load_inventory(args.inventory)
        assert_reviewed_for_training(rows, confirm_review=True)
        return {
            "reviewed_included_rows": sum(1 for row in included_rows(rows) if parse_bool(row["reviewed"])),
            "medical_validation": False,
        }
    if args.command == "split":
        from .splits import run_split

        return run_split(
            args.inventory,
            args.output_dir,
            seed=args.seed,
            train=args.train,
            validation=args.validation,
            test=args.test,
            overwrite=args.overwrite,
        )
    if args.command == "train":
        from .train import run_training

        return run_training(
            config=args.config,
            data_root=args.data_root,
            inventory_path=args.inventory,
            split_path=args.split,
            output_dir=args.output_dir,
            confirm_review=args.confirm_review,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
    if args.command == "calibrate":
        from .calibration import run_calibration

        return run_calibration(
            args.experiment,
            args.data_root,
            args.inventory,
            args.split,
            overwrite=args.overwrite,
        )
    if args.command == "evaluate":
        from .evaluate import run_evaluation

        return run_evaluation(
            args.experiment,
            args.data_root,
            args.inventory,
            args.split,
            output_dir=args.output_dir,
            n_bootstrap=args.bootstrap,
            bootstrap_seed=args.bootstrap_seed,
            overwrite=args.overwrite,
        )
    if args.command == "evaluate-external":
        from .evaluate import run_external_evaluation

        return run_external_evaluation(
            args.experiment,
            args.data_root,
            args.manifest,
            args.output_dir,
            n_bootstrap=args.bootstrap,
            bootstrap_seed=args.bootstrap_seed,
            overwrite=args.overwrite,
        )
    if args.command == "predict":
        from .inference import format_scores, load_runtime, predict_image

        runtime = load_runtime(args.experiment)
        scores = predict_image(runtime, args.image)
        print(format_scores(scores, runtime))
        return {"scores": scores, "diagnosis": False}
    if args.command == "demo":
        from .demo import launch_demo

        launch_demo(
            args.experiment,
            host=args.host,
            port=args.port,
            allow_public_share=args.allow_public_share,
        )
        return None
    raise ProjectError(f"Unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
