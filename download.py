import argparse
import tarfile
import urllib.request
from pathlib import Path

RELEASE = "https://github.com/HHI-Robot-Learning/26-WAVE/releases/download/v1.0"
#: what each archive unpacks into the repository root
ASSETS = {
    "skills": ("skills.tar", "skills/*.torch: the three frozen single-object skills"),
    "pools": ("pools.tar", "data/pools, data/sop_pools: the layout pools of every task"),
    "heads": ("heads.tar", "data/heads: the WAVE heads of every paper result, three seeds"),
    "ablations": ("heads_ablations.tar", "data/heads/ablations: the ablation heads, two seeds"),
}
ROOT = Path(__file__).resolve().parent


def fetch(name: str) -> None:
    archive, what = ASSETS[name]
    target = ROOT / "data" / "downloads" / archive
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"-- {name}: {what}\n   {RELEASE}/{archive}")
    urllib.request.urlretrieve(f"{RELEASE}/{archive}", target)
    with tarfile.open(target) as tar:
        tar.extractall(ROOT, filter="data")
    target.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the released skills, layout pools "
                                                 "and WAVE heads into the repository.")
    parser.add_argument("what", nargs="*", choices=[*ASSETS, "all"], default=["all"],
                        help="which archives (default: all)")
    args = parser.parse_args()
    for name in (list(ASSETS) if "all" in args.what else args.what):
        fetch(name)


if __name__ == "__main__":
    main()
