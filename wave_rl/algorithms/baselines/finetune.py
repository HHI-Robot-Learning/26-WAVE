from dataclasses import dataclass

import torch

from wave_rl.algorithms.base import AlgorithmConfig
from wave_rl.algorithms.heuristic import scripted_selection
from wave_rl.algorithms.lifter import Lifter
from wave_rl.env.observation import BatchedObservation
from wave_rl.td7.buffer import LAP


@dataclass
class FineTuneConfig(AlgorithmConfig):
    """Keeps training the frozen skill's own actor/critic/encoder on the multi-object task;
    every TD7 hyperparameter not named here is the skill's own saved one."""

    #: of actor, critic and encoder, small so the skill does not forget itself
    learning_rate: float = 2e-5
    exploration_noise: float = 0.1
    #: gradient updates per env step
    updates_per_step: int = 1
    #: env steps of still-frozen rollout before any gradient update
    warmup_steps: int = 5_000


class ScriptedLifter(Lifter):
    """A lifter that works the scripted selection's object through the skill's own
    single-object observation of the absolute scene (`localize=False`)."""

    def __init__(self, skill, env, cfg) -> None:
        super().__init__(skill, env, cfg)
        self.device = env.device
        #: the observation the training rollout acts from next, set by the first `learn`
        self._obs = None

    def _select(self, obs: BatchedObservation) -> torch.Tensor:
        return scripted_selection(self.measure, obs)

    def _present(self, obs, idx) -> torch.Tensor:
        return self.skill.present(obs, idx, localize=False)


class FineTune(ScriptedLifter):
    """The paper's `fine tuning mu` row.

    Each step drives one object through the skill's own single-object observation and
    action, and feeds both straight back into the skill's own replay buffer. There is no
    release/retract behaviour: that is hand-coded skill knowledge, and this ablation is the
    skill's own projection plus the skill, nothing else.

    It presents the absolute scene. Fine-tuning changes the skill's weights; presenting it in
    a shifted frame at the same time would confound the two, and the question this row
    answers is what happens to the skill itself.
    """

    def __init__(self, skill, env, cfg: FineTuneConfig) -> None:
        super().__init__(skill, env, cfg)
        # the skill is unsealed here, which is the whole arm: its parameters were frozen at
        # load, so the optimizers below would otherwise step nothing
        agent = skill.agent
        for module in (agent.actor, agent.critic, agent.encoder):
            module.train()
            for p in module.parameters():
                p.requires_grad_(True)
        agent.hp.learning_rate = cfg.learning_rate
        agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=cfg.learning_rate)
        agent.critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=cfg.learning_rate)
        agent.encoder_optimizer = torch.optim.Adam(agent.encoder.parameters(), lr=cfg.learning_rate)
        agent.hp.exploration_noise = cfg.exploration_noise
        # the skill loads without replay memory: it is only run until this arm trains it
        agent.replay_buffer = LAP(agent.actor.l0.in_features, agent.actor.l3.out_features,
                                  agent.device, max_size=agent.hp.buffer_size,
                                  batch_size=agent.hp.batch_size)

    def act_batched(self, obs: BatchedObservation, dones=None) -> torch.Tensor:
        return self.skill.act(self._present(obs, self._select(obs)))

    def learn(self, n_steps: int) -> None:
        env, c, agent = self.env, self.cfg, self.skill.agent
        B = env.num_scenes
        if self._obs is None:
            self._obs = env.reset()
            self.reward.reset(self._obs)

        target = self.total_steps + n_steps
        while self.total_steps < target:
            obs = self._obs
            k = self._select(obs)
            presented = self._present(obs, k)
            training = agent.replay_buffer.size >= c.warmup_steps
            action = agent.select_action(presented, use_exploration=training)

            next_obs, next_true, terminated, truncated, info = env.step(action)
            rewards = self.reward.step(next_true, next_obs, terminated, truncated, info)

            next_presented = self._present(next_true, k)
            own_goal = torch.gather(
                next_true.satisfied_mask.bool() & next_true.active_mask.bool(),
                1, k.unsqueeze(1)).squeeze(1)
            agent.replay_buffer.add(presented, action, next_presented, rewards,
                                    self.reward.discount * (~(terminated | own_goal)).float())
            self.reward.refresh(next_obs)
            self.logger.add("train/mean_reward", rewards, B)
            self.logger.add("train/driven_object_terminal_fraction", own_goal.float(), B)

            if training:
                for _ in range(c.updates_per_step):
                    agent.train(self.logger)

            self._obs = next_obs
            self.total_steps += B
        self.logger.add("train/buffer_size", agent.replay_buffer.size)

    def save(self, path: str) -> None:
        self.skill.agent.save(path)

    def load(self, path: str) -> None:
        self.skill.agent.load(path)
