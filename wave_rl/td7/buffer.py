import torch


class LAP:
    """A ring replay buffer with loss-adjusted prioritised sampling (Fujimoto et al., 2020).

    Every row also records the buffer index of its predecessor in the same episode (-1 at an
    episode start), so an episode can be walked back for hindsight relabelling."""

    def __init__(self, state_dim, action_dim, device, max_size=1_000_000, batch_size=256):
        max_size = int(max_size)
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device
        self.batch_size = int(batch_size)

        self.state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.action = torch.zeros((max_size, action_dim), dtype=torch.float32, device=device)
        self.next_state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.reward = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        #: the transition's bootstrap discount: gamma, or 0 on a true terminal
        self.discount = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.prev_index = torch.full((max_size,), -1, dtype=torch.int64, device=device)
        self.priority = torch.zeros(max_size, dtype=torch.float32, device=device)
        self.max_priority = torch.ones((), dtype=torch.float32, device=device)

    def add(self, state, action, next_state, reward, discount, prev_index=None):
        """Append a batch of transitions, `discount` being each one's bootstrap factor
        gamma * (1 - terminal); returns the buffer index of each row."""
        f = dict(dtype=torch.float32, device=self.device)
        state = torch.as_tensor(state, **f).reshape(-1, self.state.shape[1])
        n = state.shape[0]
        rows = {
            "state": state,
            "action": torch.as_tensor(action, **f).reshape(n, -1),
            "next_state": torch.as_tensor(next_state, **f).reshape(n, -1),
            "reward": torch.as_tensor(reward, **f).reshape(n, 1),
            "discount": torch.as_tensor(discount, **f).reshape(n, 1),
            "prev_index": (torch.full((n,), -1, dtype=torch.int64, device=self.device)
                           if prev_index is None else
                           torch.as_tensor(prev_index, dtype=torch.int64,
                                           device=self.device).reshape(n)),
        }
        if n > self.max_size:
            rows = {k: v[-self.max_size:] for k, v in rows.items()}
            n = self.max_size
        indices = (self.ptr + torch.arange(n, device=self.device)) % self.max_size
        for k, v in rows.items():
            getattr(self, k)[indices] = v
        self.priority[indices] = self.max_priority
        self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)
        return indices

    def __len__(self):
        return self.size

    def sample(self):
        csum = torch.cumsum(self.priority[: self.size], 0)
        val = torch.rand(self.batch_size, device=self.device) * csum[-1]
        indices = torch.searchsorted(csum, val).clamp_(max=self.size - 1)
        return (indices, self.state[indices], self.action[indices], self.next_state[indices],
                self.reward[indices], self.discount[indices])

    def update_priority(self, priority, indices):
        prio = priority.reshape(-1).detach()
        self.priority[indices] = prio
        self.max_priority = torch.maximum(prio.max(), self.max_priority)

    def reset_max_priority(self):
        self.max_priority = self.priority[: self.size].max().detach().clone()

    def get_episode_trace(self, i, max_len=None):
        """Walk the predecessor chain back from `i` to the episode start.

        ``max_len`` bounds the walk: a ring buffer that has wrapped past the episode leaves
        `prev_index` pointing at slots since overwritten by other data, and the chain can then
        cycle forever. Returns None once the chain is longer than ``max_len``, i.e. it is not
        this episode's trace."""
        i = int(i)
        indices = []
        while i != -1:
            indices.append(i)
            if max_len is not None and len(indices) > max_len:
                return None
            i = int(self.prev_index[i].item())
        indices.reverse()
        return torch.tensor(indices, dtype=torch.int64, device=self.device)

    def get_sub(self, indices):
        idx = torch.as_tensor(indices, dtype=torch.int64, device=self.device).reshape(-1)
        return (self.state[idx], self.action[idx], self.next_state[idx], self.reward[idx],
                self.discount[idx])
