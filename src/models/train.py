"""Run local baseline experiments: python -m src.models.train --help."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.data.preprocessor import load_config
from src.models.experiments import run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--experiment-config", help="YAML overlay containing only training settings"
    )
    parser.add_argument("--mode", choices=["baseline", "ablations", "sequential"])
    parser.add_argument("--stages", nargs="+", choices=["A", "B", "C"])
    parser.add_argument(
        "--output-dir", help="New directory; existing runs are never overwritten"
    )
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="Evaluate selected model on held-out test after CV selection",
    )
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.experiment_config:
        overlay = yaml.safe_load(
            Path(args.experiment_config).read_text(encoding="utf-8")
        )
        if set(overlay) != {"training"}:
            raise ValueError("Experiment overlay may only change training settings")
        config["training"].update(overlay["training"])
        config["training"]["experiment_config_path"] = args.experiment_config
    if args.mode:
        config["training"]["mode"] = args.mode
    if args.stages:
        if config["training"].get("mode") != "sequential":
            raise ValueError("--stages requires sequential mode")
        config["training"].setdefault("search", {})["stages"] = args.stages
    output = args.output_dir or str(
        Path(config["training"]["output_dir"])
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    report = run(config, args.config, load_config(args.config), output, args.final_test)
    print(f"Selected: {report['selected']}; artifacts: {output}")
    return report


if __name__ == "__main__":
    main()
