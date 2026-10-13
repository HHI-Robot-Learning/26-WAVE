import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import mujoco
import numpy as np
import torch
from PIL import Image

from wave_rl.sim.scene import SCENES, Scene

#: half the jaw pad's y-thickness. A pad's grasping face sits this far inside its joint
#: value, while the Franka finger mesh's grasping face sits exactly at its own, so the
#: finger has to trail the pad by this much for the two faces to coincide.
JAW_PAD_HALF_Y = 0.004
#: rai's viewer, which drew every training video of the paper: its ground plane colour, its
#: light position and the colour of the scenes' goal frames
GROUND_RGBA = (0.4, 0.45, 0.5, 1.0)
LIGHT_POS = np.array([-3.0, 2.0, 3.0])
GOAL_RGBA = np.array([0.0, 1.0, 0.5, 0.4])


class EpisodeVideoCapture:
    """Rolling qpos capture of a few training scenes, for rendering out of band.

    Stores nothing but joint states, on device, every ``stride`` ticks — the render itself
    is CPU/GL work that :func:`render_payload` below does from qpos alone, with no physics,
    no policy and no env. That is what lets the montage be produced by a
    separate process while training continues.

    Only complete episodes are handed out: the buffer for a scene is frozen into a
    ``last complete`` slot when that scene terminates, so a montage never shows a fragment
    that happens to straddle an auto-reset.
    """

    def __init__(self, scenes: list[int], max_objects: int, max_ticks: int,
                 device, *, stride: int = 4) -> None:
        self.scenes = scenes
        self.stride = stride
        self.slots = max_ticks // stride + 2
        K = len(self.scenes)
        self._dev = device
        self._idx = torch.as_tensor(self.scenes, dtype=torch.long, device=device)
        # The joint-state buffers are sized on the first observation rather than here: the
        # per-scene qpos width is only known once the env has reset.
        self._q = None
        self._done_q = None
        self._t = torch.zeros(K, device=device, dtype=torch.long)
        self._goals = torch.zeros((K, max_objects, 7), device=device, dtype=torch.float32)
        self._active = torch.zeros((K, max_objects), device=device, dtype=torch.bool)
        self._done_len = torch.zeros(K, device=device, dtype=torch.long)
        self._done_goals = torch.zeros_like(self._goals)
        self._done_active = torch.zeros_like(self._active)
        self._have = torch.zeros((), device=device, dtype=torch.long)
        self._tick = 0

    @torch.no_grad()
    def observe(self, q_live: torch.Tensor, goals_qpos: torch.Tensor, active: torch.Tensor,
                done: torch.Tensor) -> None:
        """Append this tick's joint state and retire any render scene that just finished."""
        if self._q is None:
            self._q = torch.zeros((len(self.scenes), self.slots, q_live.shape[1]),
                                  device=self._dev, dtype=torch.float32)
            self._done_q = torch.zeros_like(self._q)
        if self._tick % self.stride == 0:
            write = self._t.clamp(max=self.slots - 1)
            self._q[torch.arange(len(self.scenes), device=self._dev), write] = \
                q_live.index_select(0, self._idx).float()
            self._t = (self._t + 1).clamp(max=self.slots)
        self._tick += 1

        d = done.index_select(0, self._idx)
        if bool(d.any()):
            d1, d2 = d.unsqueeze(1), d.unsqueeze(1).unsqueeze(2)
            self._done_q = torch.where(d2, self._q, self._done_q)
            self._done_len = torch.where(d, self._t, self._done_len)
            self._done_goals = torch.where(d2, self._goals, self._done_goals)
            self._done_active = torch.where(d1, self._active, self._done_active)
            self._have = self._have + d.long().sum()
            self._t = torch.where(d, torch.zeros_like(self._t), self._t)
        # goals/active are re-read every tick: after the env's auto-reset these already
        # describe the NEW episode, which is exactly what the next capture needs.
        self._goals = goals_qpos.index_select(0, self._idx).float()
        self._active = active.index_select(0, self._idx)

    @torch.no_grad()
    def take(self) -> list[dict]:
        """The last complete episode of each render scene, as host arrays. ONE sync."""
        if not self._have.item():
            return []
        lens = self._done_len.to("cpu").numpy()
        q = self._done_q.to("cpu").numpy()
        goals = self._done_goals.to("cpu").numpy()
        active = self._done_active.to("cpu").numpy()
        out = []
        for k, scene in enumerate(self.scenes):
            n = int(lens[k])
            if n <= 1:
                continue
            out.append({"scene": scene, "qpos": q[k, :n].copy(),
                        "goals": goals[k].copy(), "active": active[k].copy(),
                        "count": int(active[k].sum()), "stride": self.stride})
        return out


class BackgroundMontageRenderer:
    """Renders training montages in a SEPARATE PROCESS so training never waits on GL.

    The parent only writes a few MB of joint states and forks this module as a worker; the
    mp4 is picked up whenever it happens to be ready, and a late video is simply logged at a
    later step.

    At most one render is in flight — they are 20M steps apart, so a second one queueing up
    means rendering is slower than training and the right response is to skip, not to pile
    processes onto the node.
    """

    def __init__(self, out_dir: str, scene: Scene, *, fps: int = 12,
                 max_frames: int = 400) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.xml_path = self.dir / "scene.xml"
        self.xml_path.write_text(scene.xml)
        self.half_size = [*scene.object.half_xy, scene.object.half_z]
        self.fps = fps
        self.max_frames = max_frames
        self._pending: list[dict] = []

    def submit(self, episodes: list[dict], step: int) -> None:
        """Fork a render of ``episodes``, unless one is still running."""
        if not episodes or self._pending:
            return
        arrays_path = self.dir / f"capture_{step:012d}.npz"
        manifest_path = self.dir / f"capture_{step:012d}.json"
        out_path = self.dir / f"montage_{step:012d}.mp4"
        arrays = {}
        for e, ep in enumerate(episodes):
            arrays[f"qpos_{e}"] = ep["qpos"]
            arrays[f"goals_{e}"] = ep["goals"]
            arrays[f"active_{e}"] = ep["active"]
        np.savez_compressed(arrays_path, **arrays)
        manifest = {
            "xml_path": str(self.xml_path),
            "half_size": self.half_size,
            "arrays_path": str(arrays_path),
            "out_path": str(out_path),
            "fps": self.fps,
            "max_frames": self.max_frames,
            "episodes": [{"scene": ep["scene"], "count": ep["count"]} for ep in episodes],
        }
        manifest_path.write_text(json.dumps(manifest))
        proc = subprocess.Popen([sys.executable, "-m", "wave_rl.video", str(manifest_path)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                env={"MUJOCO_GL": "egl", **os.environ})
        self._pending.append({"proc": proc, "step": step, "out": out_path,
                              "arrays": arrays_path, "manifest": manifest_path,
                              "counts": [ep["count"] for ep in episodes]})

    def collect(self) -> list[dict]:
        """Finished renders, as ``{path, step, counts}``. Non-blocking."""
        done, still = [], []
        for job in self._pending:
            if job["proc"].poll() is None:
                still.append(job)
                continue
            err = job["proc"].stderr.read().decode(errors="replace")
            job["arrays"].unlink(missing_ok=True)
            job["manifest"].unlink(missing_ok=True)
            if job["proc"].returncode == 0 and job["out"].exists():
                done.append({"path": str(job["out"]), "step": job["step"],
                             "counts": job["counts"]})
            else:
                print(f"-- WARNING: montage render failed at step {job['step']} "
                      f"(rc={job['proc'].returncode}): {err.strip()[-500:]}")
        self._pending = still
        return done

    def close(self) -> None:
        for job in self._pending:
            job["proc"].kill()
        self._pending = []


def _montage(cells: list[list]) -> list:
    """Side-by-side montage from per-scene frame lists of unequal length: each shorter
    scene HOLDS its last (terminal) frame until the longest one ends."""
    cells = [c for c in cells if c]
    if not cells:
        return []
    T = max(len(c) for c in cells)
    return [np.concatenate([c[min(i, len(c) - 1)] for c in cells], axis=1) for i in range(T)]


def render_model(xml: str) -> mujoco.MjModel:
    """The scene as rai's viewer drew it, for pictures only: a white sky over a 10 m ground
    plane, one shadow-casting light from above the back left, and every mesh that ships rai's
    vertex colours split into one geom per colour. The physics model keeps its single-colour
    meshes, which also carry the links' inertia."""
    spec = mujoco.MjSpec.from_string(xml)
    spec.meshdir = str(SCENES)
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=[1.0, 1.0, 1.0],
                     rgb2=[1.0, 1.0, 1.0], width=8, height=8)
    # rai lights from the viewer as well: a face turned to the camera reads near white, the
    # floor in shadow at 0.72 of its colour and in light at its full colour
    spec.visual.headlight.ambient = [0.47, 0.47, 0.47]
    spec.visual.headlight.diffuse = [0.48, 0.48, 0.48]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.worldbody.add_light(pos=LIGHT_POS, dir=-LIGHT_POS / np.linalg.norm(LIGHT_POS),
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=True,
                             diffuse=[0.44, 0.44, 0.44], specular=[0.3, 0.3, 0.3])
    # just below z=0, where the Franka's table top sits
    spec.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5.0, 5.0, 0.1],
                            pos=[0.0, 0.0, -0.003], rgba=GROUND_RGBA, contype=0, conaffinity=0)
    for mesh in list(spec.meshes):
        colored = SCENES / Path(mesh.file).with_suffix(".h5")
        if not colored.exists():
            continue
        with h5py.File(colored) as f:
            vert, face, rgb = (f["mesh/vertices"][:], f["mesh/faces"][:].astype(np.int32),
                               f["mesh/colors"][:])
        face_rgb = rgb[face[:, 0]]
        users = [g for g in spec.geoms if g.meshname == mesh.name]
        for k, c in enumerate(np.unique(face_rgb, axis=0)):
            part = spec.add_mesh(name=f"{mesh.name}_{k}", uservert=vert.ravel(),
                                 userface=face[(face_rgb == c).all(axis=1)].ravel(),
                                 inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
            for g in users:
                g.parent.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=part.name,
                                  pos=g.pos, quat=g.quat, rgba=[*(c / 255.0), 1.0],
                                  contype=0, conaffinity=0)
        for g in users:
            spec.delete(g)
        spec.delete(mesh)
    return spec.compile()


def render_payload(payload: dict, arrays: dict) -> list:
    """Replay every captured episode through :func:`render_model` and return montage frames.

    Deliberately built from **joint states only** — no physics, no policy, no env, and no
    CUDA context. That is what makes this safe to run in a separate process alongside a
    training job that owns the GPU. Each active object's goal is drawn as a translucent box.
    """
    model = render_model(Path(payload["xml_path"]).read_text())
    data = mujoco.MjData(model)
    width, height = model.cam_resolution[model.camera("cam0").id]
    renderer = mujoco.Renderer(model, height=int(height), width=int(width))
    half = np.asarray(payload["half_size"], dtype=np.float64)
    ground = model.geom("ground").id
    # The Franka's finger meshes carry no joint: they follow the jaw pads for the picture.
    fingers = ([model.body("panda_finger_joint1").id, model.body("panda_finger_joint2").id,
                int(model.joint("jawL").qposadr[0])]
               if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "jawL") >= 0 else None)

    cells = []
    for e, ep in enumerate(payload["episodes"]):
        qpos = arrays[f"qpos_{e}"]
        goals = arrays[f"goals_{e}"]
        active = arrays[f"active_{e}"]
        cell = []
        for t in range(len(qpos)):
            data.qpos[:] = qpos[t]
            if fingers is not None:
                q = float(qpos[t][fingers[2]]) - JAW_PAD_HALF_Y
                model.body_pos[fingers[0]] = [0.0, q, 0.0]
                model.body_pos[fingers[1]] = [0.0, -q, 0.0]
            mujoco.mj_kinematics(model, data)
            mujoco.mj_camlight(model, data)
            renderer.update_scene(data, camera="cam0")
            # translucent geoms cast no shadow in rai, and the ground has nothing under it to
            # shadow but itself, which only draws acne; the shadow pass skips decor geoms
            for g in renderer.scene.geoms[:renderer.scene.ngeom]:
                if g.rgba[3] < 1.0 or (g.objtype == mujoco.mjtObj.mjOBJ_GEOM
                                       and g.objid == ground):
                    g.category = mujoco.mjtCatBit.mjCAT_DECOR
            for j in np.flatnonzero(active):
                mat = np.zeros(9)
                mujoco.mju_quat2Mat(mat, goals[j, 3:].astype(np.float64))
                geom = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX, half,
                                    goals[j, :3].astype(np.float64), mat, GOAL_RGBA)
                renderer.scene.ngeom += 1
            cell.append(renderer.render().copy())
        cells.append(cell)
    return _montage(cells)


def write_mp4(frames: list, fps: int, path: str) -> None:
    """Encode RGB frames to an H.264 mp4 through ffmpeg."""
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "warning", "-f", "image2pipe", "-vcodec", "png",
         "-r", str(fps), "-i", "-", "-vcodec", "libx264", "-r", str(fps), "-pix_fmt", "yuv420p",
         path], stdin=subprocess.PIPE)
    for frame in frames:
        Image.fromarray(frame, "RGB").save(ffmpeg.stdin, "PNG")
    ffmpeg.stdin.close()
    ffmpeg.wait()


def main() -> int:
    manifest_path = sys.argv[1]
    with open(manifest_path) as fh:
        payload = json.load(fh)
    arrays = dict(np.load(payload["arrays_path"]))
    frames = render_payload(payload, arrays)
    if not frames:
        print("-- render worker: nothing to render")
        return 0
    cap = payload["max_frames"]
    if len(frames) > cap:
        frames = frames[::-(-len(frames) // cap)]
    write_mp4(frames, payload["fps"], payload["out_path"])
    print(f"-- render worker: wrote {payload['out_path']} ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
