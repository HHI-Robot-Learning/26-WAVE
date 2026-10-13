import argparse
from pathlib import Path

from wave_rl.run import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep a trained head over the object counts of a task.")
    parser.add_argument("task", help="a task under configs/tasks/, e.g. gantry_towers")
    parser.add_argument("head", help="the trained head (.pt) to evaluate")
    parser.add_argument("overrides", nargs="*", help="config overrides, e.g. eval.stop=15")
    args = parser.parse_args()
    run(args.task, ["train.enabled=false", f"train.save_path={args.head}",
                    f"run_name=eval_{args.task}_{Path(args.head).stem}", *args.overrides])


if __name__ == "__main__":
    main()
