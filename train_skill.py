import argparse

import omegaconf

from wave_rl.config import CONFIGS, SkillTrainConfig
from wave_rl.sop.trainer import train_skill


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a single-object skill (the frozen policy WAVE warps) with TD7.")
    parser.add_argument("embodiment", help="a config under configs/skills/, e.g. gantry_gripper")
    parser.add_argument("overrides", nargs="*", help="config overrides, e.g. seed=1 T_end=50000")
    parser.add_argument("--name", help="output name under data/skills/ (default: "
                                       "<embodiment>_seed<seed>)")
    args = parser.parse_args()
    cfg = omegaconf.OmegaConf.to_object(omegaconf.OmegaConf.merge(
        omegaconf.OmegaConf.structured(SkillTrainConfig),
        omegaconf.OmegaConf.load(CONFIGS / "skills" / f"{args.embodiment}.yaml"),
        omegaconf.OmegaConf.from_dotlist(args.overrides)))
    train_skill(cfg, args.name or f"{args.embodiment}_seed{cfg.seed}")


if __name__ == "__main__":
    main()
