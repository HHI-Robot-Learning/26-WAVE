from dataclasses import dataclass, field

import torch

from wave_rl.algorithms.base import AlgorithmConfig
from wave_rl.algorithms.heuristic import scripted_selection
from wave_rl.algorithms.lifter import Lifter
from wave_rl.env.observation import BatchedObservation
from wave_rl.td7.buffer import LAP

#: the bootstrap range of a critic trained on the scene reward, as the crop arms use it: the
#: success bonus alone is worth w*gamma ~ 4, above the skill's own [0, 2]
SCENE_VALUE_CLIP = [-10.0, 30.0]


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
    #: env steps during which only the critic trains and the actor stays exactly the skill:
    #: the skill's critic was fitted to the skill's own reward and horizon, not the scene's
    critic_warmup_steps: int = 10_000_000
    #: the critic's bootstrap range on the scene reward
    value_clip: list[float] = field(default_factory=lambda: list(SCENE_VALUE_CLIP))


class ScriptedLifter(Lifter):
    """A lifter that works the scripted selection's object through the skill's own
    single-object observation, in the frame the skill was trained in."""

    def __init__(self, skill, env, cfg) -> None:
        super().__init__(skill, env, cfg)
        self.device = env.device
        #: the observation the training rollout acts from next, set by the first `learn`
        self._obs = None

    def _select(self, obs: BatchedObservation) -> torch.Tensor:
        return scripted_selection(self.measure, obs)


class FineTune(ScriptedLifter):
    """The paper's `fine tuning mu` row.

    Each step drives one object through the skill's own single-object observation and
    action, and feeds both straight back into the skill's own replay buffer. There is no
    release/retract behaviour: that is hand-coded skill knowledge, and this ablation is the
    skill's own projection plus the skill, nothing else.

    The skill's critic is first refitted to the scene reward under the unchanged skill
    (`critic_warmup_steps`); only then does the actor follow it.
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
        agent.hp.value_clip = list(cfg.value_clip)
        low, high = cfg.value_clip
        for bound, value in ((agent.min, low), (agent.max, high),
                             (agent.min_target, low), (agent.max_target, high)):
            bound.fill_(value)
        # the skill loads without replay memory: it is only run until this arm trains it
        agent.replay_buffer = LAP(agent.actor.l0.in_features, agent.actor.l3.out_features,
                                  agent.device, max_size=agent.hp.buffer_size,
                                  batch_size=agent.hp.batch_size)

    def act_batched(self, obs: BatchedObservation, dones=None) -> torch.Tensor:
        return self.skill.act(self.skill.present(obs, self._select(obs)))

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
            presented = self.skill.present(obs, k)
            training = agent.replay_buffer.size >= c.warmup_steps
            train_actor = self.total_steps >= c.critic_warmup_steps
            action = agent.select_action(presented, use_exploration=training)

            next_obs, next_true, terminated, truncated, info = env.step(action)
            rewards = self.reward.step(next_true, next_obs, terminated, truncated, info)

            next_presented = self.skill.present(next_true, k)
            own_goal = torch.gather(
                next_true.satisfied_mask.bool() & next_true.active_mask.bool(),
                1, k.unsqueeze(1)).squeeze(1)
            agent.replay_buffer.add(presented, action, next_presented, rewards,
                                    self.reward.discount * (~(terminated | own_goal)).float())
            self.reward.refresh(next_obs)
            self.logger.add("train/mean_reward", rewards, B)
            self.logger.add("train/driven_object_terminal_fraction", own_goal.float(), B)
            self.logger.add("train/critic_warmup", float(not train_actor))

            if training:
                agent.train(self.logger, c.updates_per_step, train_actor)

            self._obs = next_obs
            self.total_steps += B
        self.logger.add("train/buffer_size", agent.replay_buffer.size)

    def save(self, path: str) -> None:
        self.skill.agent.save(path)

    def load(self, path: str) -> None:
        self.skill.agent.load(path)
