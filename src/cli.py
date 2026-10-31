import argparse
import sys
from pathlib import Path

import yaml

from src.augmentation import run_augmentation_demo
from src.cleaning import run_cleaning
from src.data.loader import load_cub, summarize
from src.eda import run_caption_eda, run_class_eda, run_image_eda
from src.evaluation.report import run_evaluation
from src.experiment import load_processed, synthetic_run, train_run, training_code
from src.transfer import run_transfer
from src.tuning import make_objective, retrain_best, run_tuning, synthetic_objective
from src.training.checkpoints import checkpoint_directory


def add_training_arguments(command) -> None:
    command.add_argument("--data-dir", default="data/processed", help="Cleaned splits saved by the clean command")
    command.add_argument("--checkpoint-dir", help="Optional subfolder inside root models/ (default: root models/)")
    command.add_argument("--limit", type=int, help="Use only the first N images of each split, for a quick check")
    command.add_argument("--log-dir", help="TensorBoard log root (default: root outputs/tensorboard)")
    command.add_argument("--num-workers", type=int, help="Data loader workers (0 avoids shared-memory restrictions)")
    command.add_argument("--dry-run", action="store_true",
                         help="Made-up scores instead of training, to test the pipeline in seconds")
    command.add_argument("--fig-dir", help="Where to save figures (default outputs/figures)")
    command.add_argument("--table-dir", help="Where to save results (default outputs/tables)")


def output_dirs(args) -> tuple:
    """Keep preview reports separate; every real training run still saves weights under models/."""
    test_run = args.dry_run or args.limit
    defaults = ["outputs/dry_run"] * 2 if test_run else ["outputs/tables", "outputs/figures"]
    reports = tuple(Path(given or default) for given, default in
                    zip([args.table_dir, args.fig_dir], defaults))
    return (*reports, checkpoint_directory(args.checkpoint_dir))


def training_runner(args, parser):
    """train_run on the cleaned data (loaded at the first run, so a report of finished runs needs no data),
    or synthetic_run with --dry-run."""
    if args.dry_run:
        return synthetic_run
    try:
        training_code()
    except ImportError as error:
        parser.error(str(error))
    data = {}

    def train_fn(run, **kwargs):
        if not data:
            data["splits"] = load_processed(Path(args.data_dir), args.limit)
        return train_run(run, data["splits"], tensorboard_dir=args.log_dir, **kwargs)
    return train_fn


def main() -> None:
    # captions contain characters like U+FFFD; print them as � instead of crashing on
    # consoles or redirected output that cannot encode them (e.g. cp1252 on Windows)
    sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    eda = sub.add_parser("eda", help="Load the dataset and run exploratory data analysis")
    eda.add_argument("--cache-dir", default="data/hf_cache", help="Where Hugging Face stores the dataset")
    eda.add_argument("--part", choices=["summary", "images", "classes", "captions"], default="summary", help="Which part of the EDA to run")
    eda.add_argument("--fig-dir", default="outputs/figures", help="Where to save figures")
    eda.add_argument("--table-dir", default="outputs/tables", help="Where to save CSV tables")

    clean = sub.add_parser("clean", help="Clean the captions, build the validation split, plot and save the result")
    clean.add_argument("--cache-dir", default="data/hf_cache", help="Where Hugging Face stores the dataset")
    clean.add_argument("--val-fraction", type=float, default=0.15, help="Share of train held out for validation (0 = no validation split)")
    clean.add_argument("--seed", type=int, default=42, help="Seed of the stratified train/validation split")
    clean.add_argument("--out-dir", default="data/processed", help="Where to save the cleaned splits")
    clean.add_argument("--fig-dir", default="outputs/figures", help="Where to save figures")
    clean.add_argument("--table-dir", default="outputs/tables", help="Where to save CSV tables")
    clean.add_argument("--duplicates", default="outputs/tables/duplicate_pairs.csv",
                       help="Duplicate image pairs from `eda --part images`; their train copies are dropped")
    clean.add_argument("--no-plots", action="store_true", help="Skip the figures")

    augment = sub.add_parser("augment", help="Save a figure of train images before and after the train augmentation")
    augment.add_argument("--cache-dir", default="data/hf_cache", help="Where Hugging Face stores the dataset")
    augment.add_argument("--index", type=int, nargs="+", help="Train image index(es) to show; random images if omitted")
    augment.add_argument("--n-images", type=int, default=4, help="How many random images to show when --index is not given")
    augment.add_argument("--n-augmented", type=int, default=4, help="Augmented versions shown per image")
    augment.add_argument("--image-size", type=int, default=224, help="Side of the augmented square image")
    augment.add_argument("--seed", type=int, default=0, help="Seed for the image choice and the random augmentation")
    augment.add_argument("--fig-dir", default="outputs/figures", help="Where to save the figure")

    transfer = sub.add_parser("transfer", help="Train scratch and pretrained encoders, then compare validation scores")
    transfer.add_argument("--config", default="configs/transfer.yaml", help="Shared settings and the list of runs")
    transfer.add_argument("--runs", nargs="+", help="Only these runs (names from the config), e.g. to split them "
                                                    "between Colab and Kaggle; finished runs are always skipped")
    add_training_arguments(transfer)

    tune = sub.add_parser("tune", help="Grid search and random search over the classifier hyperparameters, then compare them")
    tune.add_argument("--config", default="configs/tuning.yaml", help="Encoder, epochs per trial and search spaces")
    tune.add_argument("--encoder", help="Override the encoder in the config")
    tune.add_argument("--retrain-best", action="store_true",
                      help="After tuning: train the best setting fully and compare validation scores with the untuned run")
    add_training_arguments(tune)

    gradcam = sub.add_parser("gradcam", help="Explain trained classifiers with Grad-CAM")
    gradcam.add_argument("--checkpoints", nargs="+", required=True)
    gradcam.add_argument("--data-dir", default="data/processed")
    gradcam.add_argument("--split", choices=["test", "validation"], default="test")
    gradcam.add_argument("--out-dir", default="outputs/grad_cam")
    gradcam.add_argument("--examples", type=int, default=3, help="Examples per correct/wrong category")
    gradcam.add_argument("--max-images", type=int, help="Optional maximum number of images to scan")
    gradcam.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    evaluation = sub.add_parser("evaluate", help="Score saved checkpoints on validation and one selected run on test")
    evaluation.add_argument("--data-dir", default="data/processed")
    evaluation.add_argument("--runs-dir", default="outputs/tables/transfer_runs")
    evaluation.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    evaluation.add_argument("--table-dir", default="outputs/tables/evaluation")
    evaluation.add_argument("--fig-dir", default="outputs/figures/evaluation")
    evaluation.add_argument("--runs", nargs="+", help="Evaluate only these saved runs")
    evaluation.add_argument("--test-run", help="Run the final test evaluation for this one validation-selected model")
    evaluation.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()
    if args.command == "gradcam":
        from src.grad_cam import run_reports
        run_reports(args)
    elif args.command == "eda":
        splits = load_cub(args.cache_dir)
        if args.part == "summary":
            summarize(splits)
        elif args.part == "images":
            run_image_eda(splits, Path(args.fig_dir), Path(args.table_dir))
        elif args.part == "classes":
            run_class_eda(splits, Path(args.fig_dir), Path(args.table_dir))
        elif args.part == "captions":
            run_caption_eda(splits, Path(args.fig_dir), Path(args.table_dir))
    elif args.command == "clean":
        run_cleaning(load_cub(args.cache_dir), args.val_fraction, args.seed, Path(args.out_dir),
                     Path(args.fig_dir), Path(args.table_dir), Path(args.duplicates), plots=not args.no_plots)
    elif args.command == "augment":
        run_augmentation_demo(load_cub(args.cache_dir), Path(args.fig_dir), args.index, args.n_images,
                              args.n_augmented, args.image_size, args.seed)
    elif args.command == "transfer":
        config = yaml.safe_load(Path(args.config).read_text())
        if args.num_workers is not None:
            config["defaults"]["num_workers"] = args.num_workers
        table_dir, fig_dir, checkpoint_dir = output_dirs(args)
        run_transfer(config, training_runner(args, parser), table_dir, fig_dir, checkpoint_dir, args.runs)
    elif args.command == "tune":
        config = yaml.safe_load(Path(args.config).read_text())
        if args.num_workers is not None:
            config["num_workers"] = args.num_workers
        if args.encoder:
            config["encoder"] = args.encoder
        table_dir, fig_dir, checkpoint_dir = output_dirs(args)
        train_fn = training_runner(args, parser)
        if args.retrain_best:
            retrain_best(config, train_fn, table_dir, checkpoint_dir)
        else:
            objective = synthetic_objective if args.dry_run else make_objective(config, train_fn, checkpoint_dir)
            run_tuning(config, objective, table_dir, fig_dir)
    elif args.command == "evaluate":
        run_evaluation(Path(args.data_dir), Path(args.runs_dir), Path(args.checkpoint_dir),
                       Path(args.table_dir), Path(args.fig_dir), args.runs, args.test_run, args.num_workers)


if __name__ == "__main__":
    main()
