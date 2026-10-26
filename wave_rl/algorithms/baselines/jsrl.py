import dataclasses
from collections import deque
from dataclasses import dataclass, field

import torch

from wave_rl.algorithms.base import AlgorithmConfig
from wave_rl.algorithms.baselines.finetune import SCENE_VALUE_CLIP, ScriptedLifter
from wave_rl.env.observation import BatchedObservation
from wave_rl.td7.agent import TD7Agent


@dataclass
class JSRLConfig(AlgorithmConfig):
    """Jump-Start Reinforcement Learning (Uchendu et al. 2023), JSRL-Curriculum.

    A frozen guide (the skill, on the scripted selection's object) rolls the first fraction of
    every training episode, that fraction taken of the scene's OWN episode length; a separate
    TD7 exploration policy, seeded from the guide's weights, takes over for the rest. Only the
    exploration policy's own transitions enter its replay buffer, so it learns the episode
    tail first. The fraction steps down a linear schedule to 0 each time the COMBINED
    guide-then-explore policy's success holds up against the best it has reached so far,
    measured over episodes run entirely at the current stage; a stage that never holds up
    still ends after `budget / n_curriculum` env steps, so the schedule reaches the
    exploration policy alone within the budget. At eval the exploration policy acts alone; it
    is what ``save``/``load`` persist.

    Every TD7 hyperparameter not named here is the guide skill's own saved one."""

    learning_rate: float = 1e-4
    exploration_noise: float = 0.1
    #: the first stage's hand-off point, as a fraction of each scene's own episode length
    guide_horizon_frac: float = 0.9
    #: stages, the fraction stepped linearly from guide_horizon_frac down to 0
    n_curriculum: int = 8
    #: full-episode successes of the combined policy averaged for the advance test
    curriculum_window: int = 500
    #: a stage advances once the window mean is >= the best window mean so far - this
    curriculum_tolerance: float = 0.05
    #: gradient updates per env step, once out of warmup
    utd: int = 1
    #: transitions of guide-action rollout before any gradient update
    warmup_steps: int = 10_000
    #: the critic's bootstrap range on the scene reward
    value_clip: list[float] = field(default_factory=lambda: list(SCENE_VALUE_CLIP))


class JSRL(ScriptedLifter):
    """JSRL controller over the scripted selection. ``act_batched`` (eval) runs the
    exploration policy alone.

    Like ``finetune``, an eval-only run needs ``train.enabled=true`` (to train and save the
    policy) or a ``train.save_path`` pointing at a saved exploration policy; with neither, eval
    runs the guide-seeded exploration policy untrained."""

    def __init__(self, skill, env, cfg: JSRLConfig) -> None:
        super().__init__(skill, env, cfg)
        #: built on first use, so the guide's weights are what it starts from
        self._explore: TD7Agent | None = None
        frac0 = cfg.guide_horizon_frac
        n = cfg.n_curriculum
        self._frac_schedule = [frac0 * (n - 1 - i) / (n - 1) for i in range(n)]
        self._stage = 0
        self._frac = self._frac_schedule[0]
        self._stage_steps = 0
        self._best = -float("inf")
        self._curr_window = deque(maxlen=cfg.curriculum_window)
        #: (B,) each scene's hand-off step in its current episode
        self._h_scene = torch.zeros(env.num_scenes, dtype=torch.long, device=self.device)
        #: (B,) the stage each scene's current episode started in; only episodes run entirely
        #: at the current stage measure it
        self._episode_stage = torch.zeros(env.num_scenes, dtype=torch.long, device=self.device)

    def _build_explore(self) -> None:
        """The exploration :class:`TD7Agent`: the guide's dims and hyperparameters with this
        config's overrides, and the guide's weights."""
        if self._explore is not None:
            return
        hp = dataclasses.replace(self.skill.cfg.td7, learning_rate=self.cfg.learning_rate,
                                 exploration_noise=self.cfg.exploration_noise,
                                 value_clip=list(self.cfg.value_clip))
        self._explore = TD7Agent(state_dim=self.skill.obs_dim, action_dim=self.skill.action_dim,
                                 hp=hp)
        self._explore.load_state_dict(self.skill.agent.state_dict())

    def _episode_len(self, active_count: torch.Tensor) -> torch.Tensor:
        """(B,) each scene's episode length in env steps, from its own active object count."""
        ep_time = self.env.time_per_object * active_count.float()
        return torch.ceil(ep_time / self.env.tau_step).clamp(min=1).long()

    def _resample_horizons(self, mask: torch.Tensor, active_count: torch.Tensor) -> None:
        """A fresh hand-off step for the scenes in ``mask``, at episode reset: the stage
        fraction of that scene's own episode length."""
        ep_len = self._episode_len(active_count)
        frac = torch.full(ep_len.shape, self._frac, device=self.device)
        draw = torch.round(frac * ep_len.float()).long()
        self._h_scene[mask] = draw[mask]

    def _advance_curriculum(self, active_count: torch.Tensor) -> None:
        cfg = self.cfg
        if self._stage >= len(self._frac_schedule) - 1:
            return
        self._stage_steps += self.env.num_scenes
        win = self._curr_window
        # the first episodes to finish in a stage are its fastest; once the stage has run as
        # long as its longest episode, the finishing stream is the stage's own outcome mix
        settled = self._stage_steps >= int(self._episode_len(active_count).max()) * self.env.num_scenes
        ready = settled and len(win) >= win.maxlen
        mean = (sum(win) / len(win)) if win else 0.0
        perf_ok = ready and mean >= self._best - cfg.curriculum_tolerance
        if ready:
            self._best = max(self._best, mean)
        time_forced = self._stage_steps >= self.budget // cfg.n_curriculum
        if perf_ok or time_forced:
            self._stage += 1
            self._frac = self._frac_schedule[self._stage]
            self._stage_steps = 0
            win.clear()
            # in-flight episodes pick up the lower horizon immediately, from their own length
            ep_len = self._episode_len(active_count)
            self._h_scene[:] = torch.round(self._frac * ep_len.float()).long()
            why = "perf" if perf_ok else "time-forced"
            print(f"-- JSRL: curriculum -> stage {self._stage}/{len(self._frac_schedule) - 1}, "
                  f"guide_horizon_frac={self._frac:.3f} ({why}, window_mean={mean:.3f})")

    def act_batched(self, obs: BatchedObservation, dones=None) -> torch.Tensor:
        self._build_explore()
        return self._explore.select_action(self.skill.present(obs, self._select(obs)),
                                           use_exploration=False)

    def learn(self, n_steps: int) -> None:
        """Train for ``n_steps`` more env steps; re-entrant (see :meth:`Algorithm.learn`)."""
        self._build_explore()
        env = self.env
        cfg = self.cfg
        guide = self.skill.agent
        explore = self._explore
        b = env.num_scenes

        if self._obs is None:
            print(f"-- JSRL: guide_horizon_frac schedule {self._frac_schedule}, a fraction of "
                  "each scene's own episode length")
            self._obs = env.reset()
            self.reward.reset(self._obs)
            self._resample_horizons(torch.ones(b, dtype=torch.bool, device=self.device),
                                    self._obs.active_mask.sum(dim=1))
            self._ep_reward = torch.zeros(b, device=self.device)
            self._steps_since_reset = torch.zeros(b, dtype=torch.long, device=self.device)

        target = self.total_steps + n_steps
        while self.total_steps < target:
            obs = self._obs
            k = self._select(obs)
            base_obs = self.skill.present(obs, k)

            use_explore = self._steps_since_reset >= self._h_scene
            allow_train = explore.replay_buffer.size >= cfg.warmup_steps

            a_guide = guide.select_action(base_obs, use_exploration=False)
            # warmup: guide-action rollout only
            a_explore = (explore.select_action(base_obs, use_exploration=True) if allow_train
                         else a_guide)
            action = torch.where(use_explore.unsqueeze(-1), a_explore, a_guide)

            next_obs, next_obs_true, terminated, truncated, info = env.step(action)
            rewards = self.reward.step(next_obs_true, next_obs, terminated, truncated, info)
            discount = self.reward.discount * (~terminated).float()
            self.reward.refresh(next_obs)
            dones = terminated | truncated
            self._ep_reward = self._ep_reward + rewards

            # Only the exploration policy's own transitions, with the true pre-reset next
            # state presented for the SAME object k the step drove, so (s, a, s') describes one
            # object's transition rather than straddling a re-selection. Terminal excludes
            # timeouts, so bootstrapping isn't capped.
            self.logger.add("train/mean_reward", rewards, b)
            self.logger.add("jsrl/explore_step_frac", use_explore.float(), b)
            if bool(use_explore.any()):
                m = use_explore
                next_base_true = self.skill.present(next_obs_true, k)
                explore.replay_buffer.add(
                    base_obs[m], action[m], next_base_true[m], rewards[m], discount[m])

            self._steps_since_reset = self._steps_since_reset + 1
            if bool(dones.any()):
                self.logger.add("train/episode_return", self._ep_reward * dones, dones.float())
                # every episode run entirely at this stage measures it, handed off or not
                finished = (dones & (self._episode_stage == self._stage)).cpu()
                self._curr_window.extend(terminated.cpu()[finished].float().tolist())
                self._ep_reward = torch.where(dones, torch.zeros_like(self._ep_reward), self._ep_reward)
                self._steps_since_reset = torch.where(
                    dones, torch.zeros_like(self._steps_since_reset), self._steps_since_reset)
                self._episode_stage[dones] = self._stage
                self._resample_horizons(dones, info["active_count"])

            if allow_train:
                for _ in range(cfg.utd):
                    explore.train(self.logger)

            self._advance_curriculum(info["active_count"])
            self._obs = next_obs
            self.total_steps += b

        self.logger.add("train/buffer_size", explore.replay_buffer.size)
        self.logger.add("jsrl/guide_horizon", self._h_scene.float().mean())
        self.logger.add("jsrl/stage_frac", self._frac)
        self.logger.add("jsrl/stage", self._stage)
        if self._curr_window:
            self.logger.add("jsrl/curr_window_mean", sum(self._curr_window) / len(self._curr_window))

    def save(self, path: str) -> None:
        self._build_explore()
        self._explore.save(path)

    def load(self, path: str) -> None:
        self._build_explore()
        self._explore.load(path)
