import colorsys
import copy
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

SCENES = Path(__file__).resolve().parents[2] / "scenes"


@dataclass(frozen=True)
class ObjectGeometry:
    """The manipulated object's shape. `half_extent` (h) is the length unit every
    algorithm-side distance is expressed in: a crop length, a closeness scale, a selector
    radius and a warp quantum are all multiples of h."""

    half_extent: float
    half_xy: tuple[float, float]
    half_z: float
    rest_z: float                  # object-centre z when resting flat on the support


@dataclass(frozen=True)
class Scene:
    """A compiled-ready MJCF scene and the facts the environment reads off it."""

    name: str
    xml: str
    robot: str
    obj_names: tuple[str, ...]
    ref_name: str
    object: ObjectGeometry | None
    #: (2, nq) joint limits in qpos order; a free joint's 7 entries bound its position and
    #: the quaternion it is drawn with
    limits: np.ndarray
    #: (nq,) the pose the simulator starts in before the first reset
    home: np.ndarray


def set_contact(root: ET.Element, friction: str, solref: str, solimp: str) -> None:
    """The contact parameters every geom of the `geom_contact` class shares. They live on the
    class rather than on individual geoms because mujoco mixes a pair's parameters by solmix,
    so setting one side only buys half the change on every pair it takes part in."""
    geom = root.find("default/default[@class='geom_contact']/geom")
    geom.set("friction", friction)
    geom.set("solref", solref)
    geom.set("solimp", solimp)


def _object_colors(n: int) -> list[str]:
    out = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / n, 0.85, 0.9)
        out.append(" ".join(str(float(f"{c:.2f}")) for c in (r, g, b)) + " 1")
    return out


def _spread_x(n: int) -> list[float]:
    if n == 1:
        return [0.0]
    span = min(0.6, (n - 1) * 0.25)
    return [float(f"{x:.3f}") for x in np.linspace(-span / 2, span / 2, n)]


def multi_object_scene(name: str, n_objects: int, friction: str, solref: str,
                       solimp: str) -> Scene:
    """The base scene `name` with `n_objects` copies of its object, `obj_0` .. `obj_{n-1}`,
    each with a geometry-free goal body `obj_k_goal` under `obj_base`."""
    meta = json.loads((SCENES / f"{name}.json").read_text())
    root = ET.fromstring((SCENES / f"{name}.xml").read_text())
    set_contact(root, friction, solref, solimp)
    world = root.find("worldbody")
    template = world.find("body[@name='obj_0']")
    base = world.find("body[@name='obj_base']")
    goal_template = base.find("body[@name='obj_0_goal']")
    at = list(world).index(template)
    world.remove(template)
    base.remove(goal_template)
    for k, color in enumerate(_object_colors(n_objects)):
        body = copy.deepcopy(template)
        body.set("name", f"obj_{k}")
        body.find("geom").set("rgba", color)
        world.insert(at + k, body)
        goal = copy.deepcopy(goal_template)
        goal.set("name", f"obj_{k}_goal")
        base.append(goal)

    obj_home = np.array([[x, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0] for x in _spread_x(n_objects)])
    limits = np.concatenate([np.asarray(meta["robot_limits"]),
                             np.tile(np.asarray(meta["object_limits"]), (1, n_objects))], axis=1)
    o = meta["object"]
    return Scene(
        name=name,
        xml=ET.tostring(root, encoding="unicode"),
        robot=meta["robot"],
        obj_names=tuple(f"obj_{k}" for k in range(n_objects)),
        ref_name="obj_base",
        object=ObjectGeometry(o["half_extent"], tuple(o["half_xy"]), o["half_z"], o["rest_z"]),
        limits=limits,
        home=np.concatenate([np.asarray(meta["robot_home"]), obj_home.reshape(-1)]),
    )


def skill_scene(name: str, friction: str, solref: str, solimp: str) -> Scene:
    """The single-object scene a skill trains in: one object `obj` with its corner markers
    `goal1..4`, and its goal body `obj_goal`."""
    meta = json.loads((SCENES / "skills" / f"{name}.json").read_text())
    root = ET.fromstring((SCENES / "skills" / f"{name}.xml").read_text())
    set_contact(root, friction, solref, solimp)
    return Scene(
        name=name,
        xml=ET.tostring(root, encoding="unicode"),
        robot="",
        obj_names=("obj",),
        ref_name="obj_base",
        object=None,
        limits=np.asarray(meta["limits"]),
        home=np.asarray(meta["home"]),
    )


def compile_model(xml: str) -> mujoco.MjModel:
    """Compile an MJCF string whose mesh files are relative to the scenes directory."""
    return mujoco.MjModel.from_xml_string(xml.replace('file="meshes/', f'file="{SCENES}/meshes/'))


def arena_bound(model) -> float | None:
    """Outer face of the outermost static containment, in xy. An active object past it can
    never be recovered, so the episode ends as unachievable rather than burning its clock."""
    extent = 0.0
    for g in range(int(model.ngeom)):
        if int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0:
            continue
        b = int(model.geom_bodyid[g])
        if int(model.body_weldid[b]) != 0:
            continue
        gn = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()
        bn = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").lower()
        if "floor" in gn or "floor" in bn:
            continue
        px = float(model.body_pos[b][0]) + float(model.geom_pos[g][0])
        py = float(model.body_pos[b][1]) + float(model.geom_pos[g][1])
        extent = max(extent, abs(px) + float(model.geom_size[g][0]),
                     abs(py) + float(model.geom_size[g][1]))
    return extent if extent > 0.0 else None
