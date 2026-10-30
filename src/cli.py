import argparse
from pathlib import Path

from src.cleaning import run_cleaning
from src.data.loader import load_cub, summarize
from src.eda import run_caption_eda, run_class_eda, run_image_eda


def main() -> None:
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
    clean.add_argument("--no-plots", action="store_true", help="Skip the figures")

    args = parser.parse_args()
    if args.command == "eda":
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
                     Path(args.fig_dir), Path(args.table_dir), plots=not args.no_plots)


if __name__ == "__main__":
    main()
