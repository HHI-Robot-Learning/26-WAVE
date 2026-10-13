from .base import Algorithm, AlgorithmConfig
from .baselines.crop.finetune import FineTuneCrop, FineTuneCropConfig
from .baselines.crop.hrl import HRL, HRLConfig
from .baselines.crop.td7 import TD7Crop, TD7CropConfig
from .baselines.eit.algorithm import EIT
from .baselines.eit.config import EITConfig
from .baselines.finetune import FineTune, FineTuneConfig
from .baselines.jsrl import JSRL, JSRLConfig
from .heuristic import SkillAlone
from .wave.algorithm import WaveAlgorithm
from .wave.config import WaveConfig

#: ``name -> (Algorithm, AlgorithmConfig)``; a task config selects one by ``algorithm`` and
#: validates ``algorithm_cfg`` against its config class.
ALGORITHMS: dict[str, tuple[type[Algorithm], type[AlgorithmConfig]]] = {
    "wave": (WaveAlgorithm, WaveConfig),
    "skill_alone": (SkillAlone, AlgorithmConfig),
    "finetune": (FineTune, FineTuneConfig),
    "finetune_crop": (FineTuneCrop, FineTuneCropConfig),
    "td7_crop": (TD7Crop, TD7CropConfig),
    "hrl": (HRL, HRLConfig),
    "eit": (EIT, EITConfig),
    "jsrl": (JSRL, JSRLConfig),
}
