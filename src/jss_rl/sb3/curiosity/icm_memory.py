import random

import numpy as np


class Memory:

    def __init__(self, capacity, elem_shape, replacement_strategy: str = "fifo"):
        if replacement_strategy not in ["random", "fifo"]:
            raise ValueError(f"replacement_strategy={replacement_strategy} is an invalid argument. "
                             f"valid options are 'random' and 'fifo'")
        self.replacement_strategy = replacement_strategy
        self._capacity = capacity
        self._memory = np.zeros(shape=(capacity, *elem_shape))
        self._size = 0

    def add(self, value):
        _len = len(self)
        if _len < self._capacity:
            # fill memory if space is available
            index = self._size
        elif self.replacement_strategy == 'fifo':
            index = self._size % self._capacity
        else:
            index = random.randint(0, _len - 1)  # both values are includes, therefore len -1

        # print(f"strat: {self.replacement_strategy}, {index=}, len={len(self)}, size={self._size}")
        self._memory[index] = value
        self._size += 1

    def get_all_entries(self):
        return self._memory[:len(self)]

    def __len__(self):
        return min(self._size, len(self._memory))


class IcmMemory:

    def __len__(self):
        # assert len(self.prev_obs_memory) == len(self.obs_memory)
        # assert len(self.obs_memory) == len(self.action_memory)
        return len(self.prev_obs_memory)

    def __init__(self, capacity, obs_shape, action_shape):
        self._capacity = capacity

        self.prev_obs_memory = Memory(capacity=self._capacity, elem_shape=obs_shape)
        self.obs_memory = Memory(capacity=self._capacity, elem_shape=obs_shape)
        self.action_memory = Memory(capacity=self._capacity, elem_shape=action_shape)

        # self._size = 0

    def add_single_entry(self, prev_obs, obs, action):
        self.prev_obs_memory.add(prev_obs)
        self.obs_memory.add(obs)
        self.action_memory.add(action)
        # self._size += 1

    def add_multiple_entries(self, prev_obs: np.ndarray, obs: np.ndarray, actions: np.ndarray) -> None:
        assert len(prev_obs) == len(obs)
        assert len(obs) == len(actions)
        for po, o, a in zip(prev_obs, obs, actions):
            a = np.array(a)
            self.add_single_entry(po, o, a)

    def sample(self, shuffle: bool = True, batch_size=None):
        if not batch_size:
            batch_size = len(self)

        all_prev_obs = self.prev_obs_memory.get_all_entries()
        all_obs = self.prev_obs_memory.get_all_entries()
        all_actions = self.action_memory.get_all_entries()

        if shuffle:
            from sklearn.utils import shuffle
            all_prev_obs, all_obs, all_actions = shuffle(all_prev_obs, all_obs, all_actions)

        return all_prev_obs[:batch_size], all_actions[:batch_size], all_obs[:batch_size]
