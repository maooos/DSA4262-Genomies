import argparse
import json
from pathlib import Path

from src.data_parser import parse_to_parquet


def main():
    parser = argparse.ArgumentParser(
        description="Convert RNA signal JSON into per-read Parquet."
    )

    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--max-sites", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=100_000)

    args = parser.parse_args()

    audit = parse_to_parquet(
        signal_path=args.signals,
        output_path=args.output,
        label_path=args.labels,
        batch_size=args.batch_size,
        max_sites=args.max_sites,
    )

    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()