import argparse
from pathlib import Path

from src.augmentation import run_augmentation_demo
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

    augment = sub.add_parser("augment", help="Save a figure of train images before and after the train augmentation")
    augment.add_argument("--cache-dir", default="data/hf_cache", help="Where Hugging Face stores the dataset")
    augment.add_argument("--index", type=int, nargs="+", help="Train image index(es) to show; random images if omitted")
    augment.add_argument("--n-images", type=int, default=4, help="How many random images to show when --index is not given")
    augment.add_argument("--n-augmented", type=int, default=4, help="Augmented versions shown per image")
    augment.add_argument("--image-size", type=int, default=224, help="Side of the augmented square image")
    augment.add_argument("--seed", type=int, default=0, help="Seed for the image choice and the random augmentation")
    augment.add_argument("--fig-dir", default="outputs/figures", help="Where to save the figure")

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
    elif args.command == "augment":
        run_augmentation_demo(load_cub(args.cache_dir), Path(args.fig_dir), args.index, args.n_images,
                              args.n_augmented, args.image_size, args.seed)


if __name__ == "__main__":
    main()
