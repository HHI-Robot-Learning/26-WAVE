import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from wave_rl.sim.scene import Scene, compile_model


class Sim:
    """`num_scenes` parallel copies of one MJCF scene, stepped on the GPU by mujoco_warp.

    State lives in mujoco_warp's arrays; the torch properties are zero-copy views of them."""

    def __init__(self, scene: Scene, cfg, num_scenes: int):
        self.scene = scene
        self.model = compile_model(scene.xml)
        self.tau_sim = cfg.tau_sim
        self.model.opt.timestep = self.tau_sim
        self.model.opt.impratio = cfg.impratio
        if cfg.box_box_primitive:
            self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_NATIVECCD
        self.wp_model = mjw.put_model(self.model)
        # Contacts and constraint rows are plain per-world allocations, sized to the object
        # count: a plant that runs out drops contacts, which reads as a policy failing to
        # generalise with count.
        nconmax = cfg.contacts_per_object * len(scene.obj_names)
        self.data = mjw.make_data(self.model, nworld=num_scenes, nconmax=nconmax,
                                  njmax=6 * nconmax + 8)
        self.num_scenes = num_scenes
        self.njmax = int(self.data.njmax)
        print(f"-- mujoco_warp: nv={self.model.nv} ngeom={self.model.ngeom} "
              f"solver={int(self.model.opt.iterations)}/{int(self.model.opt.ls_iterations)} "
              f"nconmax={nconmax} njmax={self.njmax} worlds={num_scenes}")

        self.ctrl_indices = np.array(
            [int(self.model.joint(int(self.model.actuator(i).trnid[0])).qposadr[0])
             for i in range(self.model.nu)], dtype=np.int32)

        self._wp_device = wp.get_cuda_device()
        self._torch_device = torch.device("cuda", self._wp_device.ordinal)
        self._wp_stream = wp.stream_from_torch(torch.cuda.current_stream(self._torch_device))
        self._qpos_t = wp.to_torch(self.data.qpos)
        self._qvel_t = wp.to_torch(self.data.qvel)
        self._act_t = wp.to_torch(self.data.actuator_force)
        self._xpos_t = wp.to_torch(self.data.xpos)
        self._step_graph = None
        self._step_graph_key = None
        self._nefc_peak = None
        self.set_state(torch.as_tensor(np.tile(scene.home, (num_scenes, 1)), dtype=torch.float32,
                                       device=self._torch_device))

    def step(self, reference: torch.Tensor) -> None:
        """Advance every world by one control tick, following `reference` (substeps + 1,
        worlds, controls) one physics substep at a time (a captured CUDA graph)."""
        sim_steps = reference.shape[0] - 1
        buffer = wp.from_torch(reference)
        key = (sim_steps, int(buffer.ptr), int(self.data.ctrl.ptr), int(self.data.qpos.ptr))
        if self._step_graph is None or self._step_graph_key != key:
            wp.copy(self.data.ctrl, buffer[0])
            mjw.step(self.wp_model, self.data)
            mjw.forward(self.wp_model, self.data)
            wp.synchronize_device(self._wp_device)
            with wp.ScopedCapture(device=self._wp_device, force_module_load=True) as cap:
                for k in range(sim_steps):
                    wp.copy(self.data.ctrl, buffer[k])
                    mjw.step(self.wp_model, self.data)
                mjw.forward(self.wp_model, self.data)
            self._step_graph = cap.graph
            self._step_graph_key = key
        self._wp_device.stream.wait_stream(self._wp_stream)
        wp.capture_launch(self._step_graph)
        self._wp_stream.wait_stream(self._wp_device.stream)
        nefc = wp.to_torch(self.data.nefc).max()
        self._nefc_peak = nefc.clone() if self._nefc_peak is None \
            else torch.maximum(self._nefc_peak, nefc)

    def forward(self) -> None:
        mjw.forward(self.wp_model, self.data)

    def write(self, dst: wp.array, src: torch.Tensor) -> None:
        """Copy `src` into the warp array `dst` without rebinding it, so cached views stay valid."""
        wp.copy(dst, wp.from_torch(src.detach().contiguous()))

    def set_state(self, qpos: torch.Tensor, qvel: torch.Tensor | None = None,
                  act: torch.Tensor | None = None, kinematics_only: bool = False) -> None:
        with wp.ScopedStream(self._wp_stream):
            self.write(self.data.qpos, qpos.reshape(self.num_scenes, -1).to(self._torch_device))
            if qvel is None:
                self._qvel_t.zero_()
            else:
                self.write(self.data.qvel, qvel.reshape(self.num_scenes, -1))
            if act is None:
                self._act_t.zero_()
            else:
                self.write(self.data.actuator_force, act.reshape(self.num_scenes, -1))
            if kinematics_only:
                mjw.kinematics(self.wp_model, self.data)
            else:
                mjw.forward(self.wp_model, self.data)

    def clear_warmstart(self, mask: torch.Tensor) -> None:
        """Zero the solver's `qacc_warmstart` for the scenes in `mask`.

        The warmstart is solver state that survives an episode boundary; without this every
        episode after the first inherits the previous episode's terminal accelerations."""
        wp.to_torch(self.data.qacc_warmstart)[mask] = 0.

    def heal(self, mask: torch.Tensor) -> None:
        """Zero the accelerations of the scenes in `mask`, whose state went non-finite, so
        the solver does not start the next substep from them."""
        self.clear_warmstart(mask)
        wp.to_torch(self.data.qacc)[mask] = 0.

    def drain_constraint_peak(self) -> dict:
        """The per-tick peak of constraint rows since the last drain, against `njmax`. Rows
        past it are dropped by the solver, so a plant that ran out reads as starved instead of
        as a policy that failed."""
        peak = float(self._nefc_peak.item())
        self._nefc_peak = None
        if peak > self.njmax:
            print(f"!! PLANT STARVED: {peak:.0f} constraint rows exceeded njmax {self.njmax}; "
                  f"rows past the cap were dropped. Raise contacts_per_object.")
        return {"simulator/peak_constraint_rows_nefc": peak,
                "simulator/constraint_headroom_fraction_used": peak / self.njmax}

    def body_ids(self, names) -> list[int]:
        """Body indices whose names end with each of `names`, in order."""
        out = []
        for name in names:
            for i in range(self.model.nbody):
                if self.model.body(i).name.endswith(name):
                    out.append(i)
        return out

    @property
    def qpos_t(self) -> torch.Tensor:
        return self._qpos_t

    @property
    def qvel_t(self) -> torch.Tensor:
        return self._qvel_t

    @property
    def act_t(self) -> torch.Tensor:
        return self._act_t

    @property
    def xpos_t(self) -> torch.Tensor:
        """(nworld, nbody, 3) body world positions."""
        return self._xpos_t

    @property
    def torch_device(self):
        return self._torch_device

    @property
    def ctrl_dim(self) -> int:
        return self.model.nu
