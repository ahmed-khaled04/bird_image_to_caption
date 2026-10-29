import argparse
from pathlib import Path

from src.data.loader import load_cub, summarize
from src.eda import run_image_eda


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    eda = sub.add_parser("eda", help="Load the dataset and run exploratory data analysis")
    eda.add_argument("--cache-dir", default="data/hf_cache", help="Where Hugging Face stores the dataset")
    eda.add_argument("--part", choices=["summary", "images"], default="summary", help="Which part of the EDA to run")
    eda.add_argument("--fig-dir", default="outputs/figures", help="Where to save figures")
    eda.add_argument("--table-dir", default="outputs/tables", help="Where to save CSV tables")

    args = parser.parse_args()
    if args.command == "eda":
        splits = load_cub(args.cache_dir)
        if args.part == "summary":
            summarize(splits)
        elif args.part == "images":
            run_image_eda(splits, Path(args.fig_dir), Path(args.table_dir))


if __name__ == "__main__":
    main()
