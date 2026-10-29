import argparse

from src.data.loader import load_cub, summarize


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    eda = sub.add_parser("eda", help="Load the dataset and run exploratory data analysis")
    eda.add_argument("--cache-dir", default="data/hf_cache", help="Where Hugging Face stores the dataset")

    args = parser.parse_args()
    if args.command == "eda":
        summarize(load_cub(args.cache_dir))


if __name__ == "__main__":
    main()
