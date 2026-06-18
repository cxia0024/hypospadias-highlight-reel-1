import argparse
from pathlib import Path

import yaml

from src.pipeline import HighlightPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Generate a highlight reel from a hypospadias repair video"
    )
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default="output/", help="Output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pipeline = HighlightPipeline(config)
    reel_path, report = pipeline.run(args.input, args.output)
    print(f"\nDone. Highlight reel at: {reel_path}")


if __name__ == "__main__":
    main()
