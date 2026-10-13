import argparse

from wave_rl.run import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a controller on a task, then sweep it over the object counts.")
    parser.add_argument("task", help="a task under configs/tasks/, e.g. gantry_towers")
    parser.add_argument("overrides", nargs="*", help="config overrides, e.g. train.seed=42")
    args = parser.parse_args()
    run(args.task, args.overrides)


if __name__ == "__main__":
    main()
